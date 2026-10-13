# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import os

from humanoidverse.agents.evaluations.humanoidverse_mjlab import (
    HumanoidVerseMjlabTrackingEvaluation,
    HumanoidVerseMjlabTrackingEvaluationConfig,
)
from humanoidverse.agents.envs.expert_motion_loader import load_expert_trajectories_from_motion_lib
from humanoidverse.agents.envs.humanoidverse_mjlab import HumanoidVerseMjlabConfig

os.environ["OMP_NUM_THREADS"] = "1"

import torch

torch.set_float32_matmul_precision("high")

import json
import time
import typing as tp
import warnings
from pathlib import Path
from typing import Dict, List
from torch.utils._pytree import tree_map

import gymnasium
import numpy as np
import pydantic
import torch  # better to use scoped import if we use processes
import wandb
from packaging.version import Version
from tqdm import tqdm


from humanoidverse.agents.base import BaseConfig
from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.agents.buffers.transition import DictBuffer, dtype_numpytotorch_lower_precision
from humanoidverse.agents.fb_cpr.agent import FBcprAgentConfig
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.misc.loggers import CSVLogger
from humanoidverse.agents.tldr_dist_aux.agent import TldrDistAuxAgentConfig
from humanoidverse.agents.utils import EveryNStepsChecker, get_local_workdir, set_seed_everywhere
from humanoidverse.distributed import average_metrics, barrier, broadcast_agent_state, broadcast_object, module_sync_report, sync_floating_buffers

TRAIN_LOG_FILENAME = "train_log.txt"
REWARD_EVAL_LOG_FILENAME = "reward_eval_log.csv"
TRACKING_EVAL_LOG_FILENAME = "tracking_eval_log.csv"

CHECKPOINT_DIR_NAME = "checkpoint"


Evaluation = tp.Annotated[
    tp.Union[
        HumanoidVerseMjlabTrackingEvaluationConfig,
    ],
    pydantic.Field(discriminator="name"),
]

Agent = FBcprAgentConfig | FBcprAuxAgentConfig | TldrDistAuxAgentConfig


def _trajectory_output_keys(agent: Agent) -> list[str]:
    keys = [
        "observation",
        "action",
        "z",
        "terminated",
        "truncated",
        "step_count",
        "reward",
    ]
    if getattr(agent, "aux_rewards", None):
        keys.append("aux_rewards")
    return keys


def _accumulate_metrics(
    total_metrics: dict[str, torch.Tensor] | None,
    metric_update_counts: dict[str, int],
    metrics: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    if total_metrics is None:
        total_metrics = {}
    for key, value in metrics.items():
        value = value.float()
        if key in total_metrics:
            total_metrics[key] = total_metrics[key] + value
        else:
            total_metrics[key] = value.clone()
        metric_update_counts[key] = metric_update_counts.get(key, 0) + 1
    return total_metrics, metric_update_counts


class TrainConfig(BaseConfig):
    # The "pydantic.Field" field is used to explicitely tell which field is the discriminative
    # feature
    agent: Agent = pydantic.Field(discriminator="name")
    motions: str | None = None
    motions_root: str | None = None

    env: HumanoidVerseMjlabConfig = pydantic.Field(discriminator="name")

    work_dir: str = pydantic.Field(default_factory=lambda: get_local_workdir("g1mujoco_train"))

    seed: int = 0
    online_parallel_envs: int = 50
    # Note: this is in env steps (multiples of online_parallel_envs)
    log_every_updates: int = 100_000
    num_env_steps: int = 30_000_000
    # Note: this is in env steps (multiples of online_parallel_envs)
    update_agent_every: int = 500
    # Note: this is in env steps (multiples of online_parallel_envs)
    num_seed_steps: int = 50_000
    num_agent_updates: int = 50
    # Note: this is in env steps (multiples of online_parallel_envs)
    checkpoint_every_steps: int = 5_000_000
    checkpoint_buffer: bool = True
    prioritization: bool = False
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 5
    prioritization_scale: float = 2
    prioritization_mode: str = "bin"  # ["bin", "exp", "lin"]
    padding_beginning: int = 0
    padding_end: int = 0

    # Buffer
    use_trajectory_buffer: bool = False
    buffer_size: int = 5_000_000

    # WANDB
    use_wandb: bool = False
    wandb_ename: str | None = None
    wandb_gname: str | None = None
    wandb_pname: str | None = None
    wandb_run_name: str | None = None

    # misc
    load_expert_data_from_motion_lib: bool = True
    buffer_device: str = "cpu"
    # Default to True; otherwise you will spam the console with tqdm
    disable_tqdm: bool = True
    log_torso_contact_forces: bool = True
    torso_contact_force_threshold: float = 1.0
    distributed_rank: int = 0
    distributed_world_size: int = 1
    rank0_only_writes: bool = True
    checkpoint_rank_buffers: bool = True
    distributed_sync: bool = True
    distributed_global_steps: bool = True
    distributed_average_metrics: bool = True
    fail_on_nonfinite: bool = True
    nonfinite_check_model_every_updates: int = 0
    nonfinite_check_rollout_every_local_steps: int = 0

    # If you want to add more available evaluations, Update "Evaluations" type above
    evaluations: Dict[str, Evaluation] | List[Evaluation] = pydantic.Field(default_factory=lambda: [])
    # Note: this is in env steps (multiples of online_parallel_envs)
    eval_every_steps: int = 1_000_000

    tags: dict = pydantic.Field(default_factory=lambda: {})

    infra: dict = pydantic.Field(default_factory=dict)

    def model_post_init(self, context):
        # TODO prioritization needs tracking eval to work, but this is bit hacky to check for it
        if self.load_expert_data_from_motion_lib and not isinstance(self.env, HumanoidVerseMjlabConfig):
            raise ValueError("Loading expert data from motion library is only supported for HumanoidVerseMjlabConfig")

        if self.prioritization:
            has_prioritization_eval = False
            for eval_type in self.evaluations:
                if isinstance(eval_type, HumanoidVerseMjlabTrackingEvaluationConfig):
                    has_prioritization_eval = True
                    break
            if not has_prioritization_eval:
                raise ValueError("Prioritization requires tracking evaluation to be enabled")


        if self.motions is None or self.motions_root is None:
            if self.prioritization:
                raise ValueError("Prioritization requires expert data to be provided (motions and motions_root)")
            elif self.agent == FBcprAgentConfig:
                # TODO how to do checks like these in pydantic or more systematically?
                raise ValueError("FBcprAgent requires expert data to be provided (motions and motions_root)")

        # Ensure all evaluations have unique log names
        if isinstance(self.evaluations, list):
            log_names = set()
            for eval_cfg in self.evaluations:
                if eval_cfg.name_in_logs in log_names:
                    raise ValueError(
                        f"Duplicate evaluation name_in_logs found: {eval_cfg.name}. These should be unique so we do not overwrite any logs"
                    )
                log_names.add(eval_cfg.name_in_logs)

    def build(self):
        return Workspace(self)


def create_agent_or_load_checkpoint(work_dir: Path, cfg: TrainConfig, agent_build_kwargs: dict[str, tp.Any]):
    checkpoint_dir = work_dir / CHECKPOINT_DIR_NAME
    train_status_path = checkpoint_dir / "train_status.json"
    checkpoint_status = _initial_train_status(cfg)
    if train_status_path.exists():
        with train_status_path.open("r") as f:
            train_status = json.load(f)
        checkpoint_status = _normalize_train_status(train_status, cfg)

        print(
            f"Loading the agent at local_time={checkpoint_status['local_time']} "
            f"global_time={checkpoint_status['global_time']} optimizer_steps={checkpoint_status['optimizer_steps']}"
        )
        agent = cfg.agent.object_class.load(checkpoint_dir, device=cfg.agent.model.device)
    else:
        agent = cfg.agent.build(**agent_build_kwargs)
    return agent, cfg, checkpoint_status


def _global_step_scale(cfg: TrainConfig) -> int:
    if cfg.distributed_sync and cfg.distributed_global_steps and int(cfg.distributed_world_size) > 1:
        return int(cfg.distributed_world_size)
    return 1


def _effective_batch_size(cfg: TrainConfig) -> int:
    return int(cfg.agent.train.batch_size) * _global_step_scale(cfg)


def _num_envs_per_rank(cfg: TrainConfig) -> int:
    return int(cfg.online_parallel_envs)


def _global_parallel_envs(cfg: TrainConfig) -> int:
    return _num_envs_per_rank(cfg) * _global_step_scale(cfg)


def _replay_capacity_per_rank(cfg: TrainConfig) -> int:
    return int(cfg.buffer_size)


def _effective_replay_capacity(cfg: TrainConfig) -> int:
    return _replay_capacity_per_rank(cfg) * _global_step_scale(cfg)


def _trajectory_steps_per_rank(cfg: TrainConfig) -> int:
    if cfg.use_trajectory_buffer:
        return int(cfg.buffer_size) // max(_num_envs_per_rank(cfg), 1)
    return int(cfg.buffer_size)


def _tensor_nonfinite_summary(value: tp.Any) -> str | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0 or not torch.is_floating_point(value):
            return None
        finite = torch.isfinite(value)
        if bool(finite.all().item()):
            return None
        bad = ~finite
        bad_count = int(bad.sum().item())
        with torch.no_grad():
            finite_values = value[finite]
            finite_min = float(finite_values.min().item()) if finite_values.numel() > 0 else None
            finite_max = float(finite_values.max().item()) if finite_values.numel() > 0 else None
            bad_indices = bad.nonzero(as_tuple=False)[:5].detach().cpu().tolist()
            bad_values = value[bad][:5].detach().cpu().tolist()
        return (
            f"type=torch shape={tuple(value.shape)} dtype={value.dtype} bad_count={bad_count} "
            f"finite_min={finite_min} finite_max={finite_max} bad_indices={bad_indices} bad_values={bad_values}"
        )
    if isinstance(value, np.ndarray):
        if value.size == 0 or not np.issubdtype(value.dtype, np.floating):
            return None
        finite = np.isfinite(value)
        if bool(finite.all()):
            return None
        bad = ~finite
        bad_count = int(bad.sum())
        finite_values = value[finite]
        finite_min = float(finite_values.min()) if finite_values.size > 0 else None
        finite_max = float(finite_values.max()) if finite_values.size > 0 else None
        bad_indices = np.argwhere(bad)[:5].tolist()
        bad_values = value[bad][:5].tolist()
        return (
            f"type=numpy shape={value.shape} dtype={value.dtype} bad_count={bad_count} "
            f"finite_min={finite_min} finite_max={finite_max} bad_indices={bad_indices} bad_values={bad_values}"
        )
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return None
        return f"type=scalar value={value}"
    return None


def _iter_nonfinite(value: tp.Any, prefix: str = "") -> tp.Iterator[tuple[str, str]]:
    summary = _tensor_nonfinite_summary(value)
    if summary is not None:
        yield prefix or "<root>", summary
        return
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_nonfinite(item, child_prefix)
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from _iter_nonfinite(item, child_prefix)


def _assert_finite(value: tp.Any, *, label: str, rank: int, local_time: int, global_time: int, optimizer_steps: int) -> None:
    bad = list(_iter_nonfinite(value, label))
    if not bad:
        return
    details = "\n".join(f"  - {path}: {summary}" for path, summary in bad[:20])
    more = "" if len(bad) <= 20 else f"\n  ... {len(bad) - 20} more non-finite fields"
    raise FloatingPointError(
        "Non-finite value detected "
        f"(rank={rank}, local_time={local_time}, global_time={global_time}, optimizer_steps={optimizer_steps}):\n"
        f"{details}{more}"
    )


def _assert_model_finite(module: torch.nn.Module, *, rank: int, local_time: int, global_time: int, optimizer_steps: int) -> None:
    for name, param in module.named_parameters():
        _assert_finite(
            param,
            label=f"model.parameters.{name}",
            rank=rank,
            local_time=local_time,
            global_time=global_time,
            optimizer_steps=optimizer_steps,
        )
    for name, buffer in module.named_buffers():
        _assert_finite(
            buffer,
            label=f"model.buffers.{name}",
            rank=rank,
            local_time=local_time,
            global_time=global_time,
            optimizer_steps=optimizer_steps,
        )


def _distributed_loss_mode(cfg: TrainConfig) -> str:
    if cfg.distributed_sync and int(cfg.distributed_world_size) > 1:
        return "local_loss_average"
    return "single_rank"


def _initial_train_status(cfg: TrainConfig) -> dict[str, tp.Any]:
    return {
        "local_time": 0,
        "global_time": 0,
        "optimizer_steps": 0,
        "world_size": int(cfg.distributed_world_size),
        "loss_mode": _distributed_loss_mode(cfg),
        "effective_batch_size": _effective_batch_size(cfg),
    }


def _normalize_train_status(train_status: dict[str, tp.Any], cfg: TrainConfig) -> dict[str, tp.Any]:
    status = _initial_train_status(cfg)
    scale = _global_step_scale(cfg)
    current_world_size = int(cfg.distributed_world_size)
    if "world_size" not in train_status and current_world_size > 1:
        raise RuntimeError(
            "Cannot safely resume this checkpoint because checkpoint/train_status.json does not record world_size. "
            "Distributed checkpoints contain rank-local replay buffer shards; start a fresh work_dir or migrate buffers explicitly."
        )
    checkpoint_world_size = int(train_status.get("world_size", current_world_size))
    if checkpoint_world_size != current_world_size:
        raise RuntimeError(
            "Cannot resume checkpoint with a different distributed world_size: "
            f"checkpoint world_size={checkpoint_world_size}, current world_size={current_world_size}. "
            "Rank-local replay buffer shards cannot be automatically migrated; use a matching GPU count, "
            "start a fresh work_dir, or perform an explicit buffer migration."
        )
    if "local_time" in train_status:
        status["local_time"] = int(train_status["local_time"])
        status["global_time"] = int(train_status.get("global_time", status["local_time"] * scale))
    else:
        legacy_time = int(train_status.get("time", 0))
        status["global_time"] = legacy_time
        status["local_time"] = (legacy_time + scale - 1) // scale
    status["optimizer_steps"] = int(train_status.get("optimizer_steps", 0))
    status["world_size"] = checkpoint_world_size
    status["loss_mode"] = str(train_status.get("loss_mode", _distributed_loss_mode(cfg)))
    status["effective_batch_size"] = int(train_status.get("effective_batch_size", _effective_batch_size(cfg)))
    return status


def _make_train_status(cfg: TrainConfig, *, local_time: int, global_time: int, optimizer_steps: int) -> dict[str, tp.Any]:
    return {
        "time": int(global_time),
        "local_time": int(local_time),
        "global_time": int(global_time),
        "optimizer_steps": int(optimizer_steps),
        "world_size": int(cfg.distributed_world_size),
        "loss_mode": _distributed_loss_mode(cfg),
        "effective_batch_size": _effective_batch_size(cfg),
    }


def init_wandb(cfg: TrainConfig):
    from pathlib import Path
    exp_name = cfg.wandb_run_name if cfg.wandb_run_name else Path(cfg.work_dir).name
    wandb_config = cfg.model_dump()
    wandb.init(entity=cfg.wandb_ename, project=cfg.wandb_pname, group=cfg.wandb_gname, name=exp_name, config=wandb_config, dir="./_wandb")


class Workspace:
    def __init__(self, cfg: TrainConfig) -> None:
        # 传入一个TrainConfig配置对象cfg，把训练配置保存到实例上，之后所有的方法
        # 都可以通过self.cfg.agent或者self.cfg.env等方法访问配置
        self.cfg = cfg
        # 保存分布式训练信息，包括当前rank和总的rank数，比如8个gpu就是8个rank编号
        self.distributed_rank = int(self.cfg.distributed_rank)
        # 保存参与训练的进程总数
        self.distributed_world_size = int(self.cfg.distributed_world_size)
        # 决定当前进程是否负责写公共文件和日志，如果是rank0_only_writes为True，就rank0写日志和文件，否则不写
        # 否则如果配置为false，所有rank都会写日志和文件
        self._write_shared_artifacts = (not self.cfg.rank0_only_writes) or self.distributed_rank == 0

        # MJLab environments are created once and shared with evaluation.
        # 如果配置环境是HumanoidVerseMjlabConfig类型，就创建训练环境和训练环境信息，并保存观测空间和动作空间
        # 默认配置就是这个环境类型
        if isinstance(cfg.env, HumanoidVerseMjlabConfig):
            from omegaconf import OmegaConf
            # 根据环境配置创建并行环境，返回两个对象，一个是训练环境对象，一个是训练环境附加信息对象
            self.train_env, self.train_env_info = cfg.env.build(num_envs=cfg.online_parallel_envs)
            # 取得单个环境的观测空间描述，就是描述观测有哪些字段以及每个字段的形状和类型
            # 实际观测得调用reset()或step()方法获得
            # 当前配置的观测字段包括state、privileged_state、last_action
            # history_actor、time
            self.obs_space = self.train_env.single_observation_space
            # 取得单个环境的动作空间描述
            self.action_space = self.train_env.single_action_space
        else:
            sample_env, _ = cfg.env.build(num_envs=1)
            self.obs_space = sample_env.observation_space
            self.action_space = sample_env.action_space
        # 要求单环境动作是一维向量，也就是没有batch这东西
        assert "time" in self.obs_space.keys(), "Observation space must contain 'obs' and 'time' (TimeAwareObservation wrapper)"
        assert len(self.action_space.shape) == 1, "Only 1D action space is supported (first dim should be vector env)"
        # TODO for backwards consistency, we do not pass "time" to the agent, so we remove it from the obs_space we pass to the agent/model
        #      but would we need it at some point?
        # 去除其中的“times”字段，因为这部分观测不会扔给网络
        del self.obs_space.spaces["time"]

        self.action_dim = self.action_space.shape[0]

        print(f"Workdir: {self.cfg.work_dir}")
        # 把路径转换成Path对象，方便拼接路径创建目录
        self.work_dir = Path(self.cfg.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)
        # 如果当前进程负责公共输出，而且使用MJLab，就保存环境配置
        if self._write_shared_artifacts and isinstance(cfg.env, HumanoidVerseMjlabConfig):
            with open(self.work_dir / "config.yaml", "w") as file:
                OmegaConf.save(self.train_env_info["unresolved_conf"], file)
        # 如果当前进程允许写公共输出，就创建CSV日志记录器，否则设为None
        self.train_logger = CSVLogger(filename=self.work_dir / TRAIN_LOG_FILENAME) if self._write_shared_artifacts else None
        # 设置随机种子
        set_seed_everywhere(self.cfg.seed)
        # 得到的self.agent就是算法实例，self.cfg就是配置对象
        self.agent, self.cfg, self._checkpoint_status = create_agent_or_load_checkpoint(
            self.work_dir, self.cfg, agent_build_kwargs=dict(obs_space=self.obs_space, action_dim=self.action_dim)
        )
        self._checkpoint_local_time = int(self._checkpoint_status["local_time"])
        self._checkpoint_global_time = int(self._checkpoint_status["global_time"])
        self._optimizer_steps = int(self._checkpoint_status["optimizer_steps"])
        if self.cfg.distributed_sync:
            broadcast_agent_state(self.agent, src=0)
        self.agent._model.train()

        if isinstance(self.cfg.evaluations, list):
            self.evaluations = {eval_cfg.name_in_logs: eval_cfg.build() for eval_cfg in self.cfg.evaluations}
        else:
            self.evaluations = {eval_cfg: eval_cfg.build() for name, eval_cfg in self.cfg.evaluations.items()}
        self.evaluate = len(self.evaluations) > 0

        self.eval_loggers = {
            name: CSVLogger(filename=self.work_dir / f"{name}.csv") for name in self.evaluations.keys()
        } if self._write_shared_artifacts else {}

        if self._write_shared_artifacts and self.cfg.use_wandb:
            init_wandb(self.cfg)

        if self._write_shared_artifacts:
            with (self.work_dir / "config.json").open("w") as f:
                f.write(self.cfg.model_dump_json(indent=4))

        self.priorization_eval_name = None
        if self.cfg.prioritization:
            for name, evaluation in self.evaluations.items():
                if isinstance(evaluation.cfg, HumanoidVerseMjlabTrackingEvaluationConfig):
                    self.priorization_eval_name = name
                    break
            if self.priorization_eval_name is None:
                raise ValueError("Prioritization requires tracking evaluation to be enabled")

        self.training_with_expert_data = True

        self.manager = None

    def _checkpoint_buffer_path(self, checkpoint_dir: Path) -> Path:
        if self.cfg.checkpoint_rank_buffers and self.distributed_world_size > 1:
            return checkpoint_dir / "buffers" / f"train_rank_{self.distributed_rank}"
        return checkpoint_dir / "buffers" / "train"
    # 启动训练
    def train(self):
        self.start_time = time.time()
        self.train_online()

    def _get_torso_contact_force_metrics(self, train_env) -> dict[str, float]:
        if not self.cfg.log_torso_contact_forces:
            return {}

        raw_env = getattr(train_env, "_env", train_env)
        simulator = getattr(raw_env, "simulator", None)
        torso_index = getattr(raw_env, "torso_index", None)
        if simulator is None or torso_index is None:
            return {}

        with torch.no_grad():
            torso_force = simulator.contact_forces[:, torso_index, :].float()
            torso_norm = torch.linalg.vector_norm(torso_force, dim=-1)
            torso_mean = torso_force.mean(dim=0)
            torso_force_z = torso_force[:, 2]
            torso_z = simulator._rigid_body_pos[:, torso_index, 2].float()
            contact_mask = torso_norm > self.cfg.torso_contact_force_threshold
            contact_count = contact_mask.sum()
            num_envs = max(torso_norm.numel(), 1)

            metrics = {
                "torso_contact_force_mean": np.round(torso_norm.mean().item(), 6),
                "torso_contact_force_max": np.round(torso_norm.max().item(), 6),
                "torso_contact_force_p95": np.round(torch.quantile(torso_norm, 0.95).item(), 6),
                "torso_contact_force_contact_count": int(contact_count.item()),
                "torso_contact_force_contact_frac": np.round((contact_count.float() / num_envs).item(), 6),
                "torso_contact_force_mean_x": np.round(torso_mean[0].item(), 6),
                "torso_contact_force_mean_y": np.round(torso_mean[1].item(), 6),
                "torso_contact_force_mean_z": np.round(torso_mean[2].item(), 6),
                "torso_contact_force_z_mean": np.round(torso_force_z.mean().item(), 6),
                "torso_contact_force_z_max": np.round(torso_force_z.max().item(), 6),
                "torso_contact_force_z_p95": np.round(torch.quantile(torso_force_z, 0.95).item(), 6),
                "torso_z_mean": np.round(torso_z.mean().item(), 6),
                "torso_z_min": np.round(torso_z.min().item(), 6),
                "torso_z_p05": np.round(torch.quantile(torso_z, 0.05).item(), 6),
            }
            if contact_count.item() > 0:
                torso_z_contact = torso_z[contact_mask]
                metrics.update(
                    {
                        "torso_z_contact_mean": np.round(torso_z_contact.mean().item(), 6),
                        "torso_z_contact_min": np.round(torso_z_contact.min().item(), 6),
                        "torso_z_contact_p05": np.round(torch.quantile(torso_z_contact, 0.05).item(), 6),
                    }
                )
            else:
                metrics.update(
                    {
                        "torso_z_contact_mean": 0.0,
                        "torso_z_contact_min": 0.0,
                        "torso_z_contact_p05": 0.0,
                    }
                )
            return metrics

    def train_online(self) -> None:
        # 判断这次训练是否需要专家数据，当前的配置是训练的时候需要专家数据
        if self.training_with_expert_data:
            if self.cfg.load_expert_data_from_motion_lib:
                # 判断是否从环境的运动裤MotionLib中加载专家轨迹
                expert_buffer = load_expert_trajectories_from_motion_lib(self.train_env._env, self.cfg.agent, device=self.cfg.buffer_device)
            else:
                raise RuntimeError(
                    "This MJLab-focused build only supports expert data loaded from the motion library. "
                    "Set load_expert_data_from_motion_lib=True."
                )
        print("Creating the training environment")

        if isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
            train_env = self.train_env
            train_env_info = self.train_env_info
        else:
            train_env, train_env_info = self.cfg.env.build(num_envs=self.cfg.online_parallel_envs)

        print("Allocating buffers")
        replay_buffer = {}
        checkpoint_dir = self.work_dir / CHECKPOINT_DIR_NAME
        checkpoint_buffer_dir = self._checkpoint_buffer_path(checkpoint_dir)
        # 加载replay_buffer, 这里的replay_buffer是一个字典，里面有train和expert_slicer两个键
        # 其中训练时用的放到replay_buffer["train"]，专家数据放到replay_buffer["expert_slicer"]
        if checkpoint_buffer_dir.exists():
            print("Loading checkpointed buffer")
            if self.cfg.use_trajectory_buffer:
                replay_buffer["train"] = TrajectoryDictBufferMultiDim.load(checkpoint_buffer_dir, device=self.cfg.buffer_device)
            else:
                replay_buffer["train"] = DictBuffer.load(checkpoint_buffer_dir, device=self.cfg.buffer_device)
            print(f"Loaded buffer of size {len(replay_buffer['train'])}")
        else:
            if self.cfg.use_trajectory_buffer:
                replay_buffer["train"] = TrajectoryDictBufferMultiDim(
                    capacity=self.cfg.buffer_size // self.cfg.online_parallel_envs,  # make sure to divide by num_envs
                    device=self.cfg.buffer_device,
                    n_dim=2,
                    end_key="truncated",
                    output_key_t=_trajectory_output_keys(self.cfg.agent),
                    output_key_tp1=["observation", "terminated"],
                )
            else:
                replay_buffer["train"] = DictBuffer(capacity=self.cfg.buffer_size, device=self.cfg.buffer_device)
        if self.training_with_expert_data:
            replay_buffer["expert_slicer"] = expert_buffer

        print("Starting training")
        global_step_scale = _global_step_scale(self.cfg)
        local_step_increment = self.cfg.online_parallel_envs
        global_step_increment = local_step_increment * global_step_scale
        max_local_time = (self.cfg.num_env_steps + global_step_scale - 1) // global_step_scale
        if self._write_shared_artifacts:
            print(
                "[INFO] Step accounting: "
                f"num_envs_per_rank={_num_envs_per_rank(self.cfg)}, global_parallel_envs={_global_parallel_envs(self.cfg)}, "
                f"local_step_increment={local_step_increment}, global_step_increment={global_step_increment}, "
                f"num_env_steps_global={self.cfg.num_env_steps}, num_seed_steps_local={self.cfg.num_seed_steps}, "
                f"update_agent_every_local={self.cfg.update_agent_every}, world_size={self.distributed_world_size}, "
                f"loss_mode={_distributed_loss_mode(self.cfg)}, effective_batch_size={_effective_batch_size(self.cfg)}, "
                f"replay_capacity_per_rank={_replay_capacity_per_rank(self.cfg)}, "
                f"effective_replay_capacity={_effective_replay_capacity(self.cfg)}, "
                f"trajectory_steps_per_rank={_trajectory_steps_per_rank(self.cfg)}, "
                f"compile={self.cfg.agent.compile}"
            )
        # 初始化进度条和环境观测
        # 当前td通常是
        # {
        #     "state": numpy数组,
        #     "privileged_state": numpy数组,
        #     "last_action": numpy数组,
        #     "history_actor": numpy数组,
        #     "time": numpy数组,
        # }
        progb = tqdm(
            total=self.cfg.num_env_steps,
            initial=min(self._checkpoint_global_time, self.cfg.num_env_steps),
            disable=self.cfg.disable_tqdm,
        )
        # 重置环境，获得td，这个td就是观测
        td, info = train_env.reset()
        # 如果启用了异常数值检查，那么其检查td中是否有非有限值（NaN或Inf）
        if self.cfg.fail_on_nonfinite:
            _assert_finite(
                td,
                label="env.reset.obs",
                rank=self.distributed_rank,
                local_time=self._checkpoint_local_time,
                global_time=self._checkpoint_global_time,
                optimizer_steps=self._optimizer_steps,
            )
        # see https://farama.org/Vector-Autoreset-Mode
        # 初始化结束标记，包括每个环境是否真正终止，初始为false
        # 以及每个环境是否截断初始为false
        terminated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        truncated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        done = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        total_metrics, context = None, None
        metric_update_counts: dict[str, int] = {}
        start_time = time.time()
        fps_start_time = time.time()
        checkpoint_time_checker = EveryNStepsChecker(self._checkpoint_global_time, self.cfg.checkpoint_every_steps)
        eval_time_checker = EveryNStepsChecker(self._checkpoint_global_time, self.cfg.eval_every_steps)
        update_agent_time_checker = EveryNStepsChecker(self._checkpoint_local_time, self.cfg.update_agent_every)
        log_time_checker = EveryNStepsChecker(self._checkpoint_global_time, self.cfg.log_every_updates)
        # 检查是否存在MJLAB跟踪评估器
        eval_instances = []
        # 遍历评估器名称
        for evaluation_name in self.evaluations.keys():
            # 取得对应评估器
            evaluation = self.evaluations[evaluation_name]
            eval_instances.append(isinstance(evaluation, HumanoidVerseMjlabTrackingEvaluation))
        # 只要存在一个MJLab跟踪评估器，就设为true
        uses_humanoidverse_eval = True if any(eval_instances) else False
        # 训练循环，处理预算和检查点
        for local_time in range(self._checkpoint_local_time, max_local_time + local_step_increment, local_step_increment):
            global_time = local_time * global_step_scale
            if global_time > self.cfg.num_env_steps:
                break
            if (local_time != self._checkpoint_local_time) and checkpoint_time_checker.check(global_time):
                checkpoint_time_checker.update_last_step(global_time)
                self.save(local_time=local_time, global_time=global_time, optimizer_steps=self._optimizer_steps, replay_buffer=replay_buffer)

            if global_time >= self.cfg.num_env_steps:
                break
            # 判断是否需要评估
            if (self.evaluate and eval_time_checker.check(global_time)) or (
                self.evaluate and global_time == self._checkpoint_global_time
            ):
                # 初始化评估结果
                eval_metrics = {}
                run_eval_on_this_rank = (not self.cfg.distributed_sync) or self.distributed_rank == 0
                if run_eval_on_this_rank:
                    eval_metrics = self.eval(global_time, replay_buffer=replay_buffer)
                if self.cfg.distributed_sync:
                    barrier()
                eval_time_checker.update_last_step(global_time)
                # 如果是评估阶段，就走这段代码
                if uses_humanoidverse_eval:
                    # reset if there is a humanoidverse evaluation
                    # 重新重置训练环境，重新获得训练初始观测td
                    td, info = train_env.reset()
                    # 依旧如果开启异常检测，则检查观测是否有出现Naf之类的问题
                    if self.cfg.fail_on_nonfinite:
                        _assert_finite(
                            td,
                            label="env.post_eval_reset.obs",
                            rank=self.distributed_rank,
                            local_time=local_time,
                            global_time=global_time,
                            optimizer_steps=self._optimizer_steps,
                        )
                    terminated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                    truncated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                    done = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                # 如果开启优先采样时，根据评估结果更新动作权重
                if self.cfg.prioritization:
                    # priorities
                    # 初始化等待同步的权重数据
                    priority_payload = None
                    if run_eval_on_this_rank:
                        # 检查评估动作数量与专家动作数量一致
                        assert len(eval_metrics[self.priorization_eval_name]) == len(replay_buffer["expert_slicer"].motion_ids), (
                            "Mismatch in number of motions returned by the eval"
                        )
                        # 创建动作 ID 到索引、文件名的映射
                        index_in_buffer, name_in_buffer = {}, {}
                        for i, motion_id in enumerate(replay_buffer["expert_slicer"].motion_ids):
                            index_in_buffer[motion_id] = i
                            if hasattr(replay_buffer["expert_slicer"], "file_names"):
                                name_in_buffer[motion_id] = replay_buffer["expert_slicer"].file_names[i]
                        motions_id, priorities, idxs = [], [], []
                        # 把跟踪误差转换成采样权重
                        for _, metr in eval_metrics[self.priorization_eval_name].items():
                            motions_id.append(metr["motion_id"])
                            priorities.append(metr["emd"])
                            idxs.append(index_in_buffer[metr["motion_id"]])
                        priorities = (
                            torch.clamp(
                                torch.tensor(priorities, dtype=torch.float32, device=self.agent.device),
                                min=self.cfg.prioritization_min_val,
                                max=self.cfg.prioritization_max_val,
                            )
                            * self.cfg.prioritization_scale
                        )

                        if self.cfg.prioritization_mode == "lin":
                            pass
                        elif self.cfg.prioritization_mode == "exp":
                            priorities = 2**priorities
                        elif self.cfg.prioritization_mode == "bin":
                            bins = torch.floor(priorities)
                            for i in range(int(bins.min().item()), int(bins.max().item()) + 1):
                                mask = bins == i
                                n = mask.sum().item()
                                if n > 0:
                                    priorities[mask] = 1 / n
                        else:
                            raise ValueError(f"Unsupported prioritization mode {self.cfg.prioritization_mode}")
                        priority_payload = {
                            "priorities": priorities.detach().cpu(),
                            "idxs": idxs,
                            "file_name": name_in_buffer,
                        }
                    if self.cfg.distributed_sync:
                        priority_payload = broadcast_object(priority_payload, src=0)
                    # 同步并应用新的采样权重
                    if priority_payload is None:
                        raise RuntimeError("Prioritization requires evaluation metrics, but no priority payload was produced.")
                    priorities = priority_payload["priorities"].to(self.agent.device)
                    idxs = priority_payload["idxs"]
                    name_in_buffer = priority_payload["file_name"]

                    train_env._env._motion_lib.update_sampling_weight_by_id(
                        priorities=list(priorities), motions_id=idxs, file_name=name_in_buffer
                    )

                    replay_buffer["expert_slicer"].update_priorities(
                        priorities=priorities.to(self.cfg.buffer_device), idxs=torch.tensor(np.array(idxs), device=self.cfg.buffer_device)
                    )

            if global_time + global_step_increment > self.cfg.num_env_steps:
                if self._write_shared_artifacts:
                    print(
                        "[INFO] Stopping before next rollout to avoid exceeding global sample budget: "
                        f"current_global_time={global_time}, next_global_time={global_time + global_step_increment}, "
                        f"num_env_steps_global={self.cfg.num_env_steps}"
                    )
                break
            # 采集动作时不建立梯度计算图
            with torch.no_grad():
                # 根据前面获得的观测td值转到tensor上并取名为obs
                # 这个obs是一个字典，包括的key有state、privileged_state、history_actor、time等，都是tensor类型
                obs = tree_map(lambda x: torch.tensor(x, dtype=dtype_numpytotorch_lower_precision(x.dtype), device=self.agent.device), td)
                # TODO consistency with obs_space: remove time assigned by TimeAwareObservationWrapper
                # 将其中的time键从obs中移除，并赋值给step_count
                step_count = obs.pop("time")
                # 默认没有独立历史编码
                history_context = None
                if "history" in obs:
                    # this works in inference mode
                    if len(obs["history"]["action"]) == 0:
                        history_context = self.agent._model._context_encoder.get_initial_context(self.cfg.online_parallel_envs)
                    else:
                        history_context = self.agent.history_inference(obs=obs["history"]["observation"], action=obs["history"]["action"])[
                            :, -1
                        ].clone()
                # 首次调用的时候context是None，FB会为每个环境采样一个归一化的随机z
                # 后续调用的时候每个环境按自己的step_count和obs来更新自己的context
                # 这个context就是归一化后采样的z形状为[N,z_dim]
                # 也就是这里是用上一轮使用的任务向量和每个环境当前episode也就是step_count
                # 以及replay_buffer来获得任务向量z
                context = self.agent.maybe_update_rollout_context(z=context, step_count=step_count, replay_buffer=replay_buffer)
                # 当local_time小于num_seed_steps时，先用随机动作收集初始数据
                # 也就是训练初期，动作是随机探索得到的
                if local_time < self.cfg.num_seed_steps:
                    # 先用随机动作收集初始数据
                    action = train_env.action_space.sample().astype(np.float32)
                else:
                    # 后续就是由actor网络获得，actor网络输入obs和z
                    # this works in inference mode
                    if history_context is not None:
                        action = self.agent.act(obs=obs, z=context, context=history_context, mean=False)
                    else:
                        action = self.agent.act(obs=obs, z=context, mean=False)
                    # TODO a bit hard-coded -- just to avoid moving stuff from cpu to cuda
                    # Tensor动作从gpu传给cpu变成numpy，准备传给环境
                    if isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
                        action = action.cpu().detach().numpy()
                # 判断本论是否检查异常数值
                check_rollout_nonfinite = (
                    self.cfg.fail_on_nonfinite
                    and self.cfg.nonfinite_check_rollout_every_local_steps > 0
                    and local_time % self.cfg.nonfinite_check_rollout_every_local_steps == 0
                )
                if check_rollout_nonfinite:
                    _assert_finite(
                        obs,
                        label="rollout.obs",
                        rank=self.distributed_rank,
                        local_time=local_time,
                        global_time=global_time,
                        optimizer_steps=self._optimizer_steps,
                    )
                    _assert_finite(
                        context,
                        label="rollout.context",
                        rank=self.distributed_rank,
                        local_time=local_time,
                        global_time=global_time,
                        optimizer_steps=self._optimizer_steps,
                    )
                    _assert_finite(
                        history_context,
                        label="rollout.history_context",
                        rank=self.distributed_rank,
                        local_time=local_time,
                        global_time=global_time,
                        optimizer_steps=self._optimizer_steps,
                    )
                    _assert_finite(
                        action,
                        label="rollout.action",
                        rank=self.distributed_rank,
                        local_time=local_time,
                        global_time=global_time,
                        optimizer_steps=self._optimizer_steps,
                    )
            # 执行动作，推进环境，从action得到新的观测td、奖励、终止标记、截断标记和info
            new_td, new_reward, new_terminated, new_truncated, new_info = train_env.step(action)
            # 检查恶道的新的观测和奖励是否有异常
            if check_rollout_nonfinite:
                _assert_finite(
                    new_td,
                    label="env.step.obs",
                    rank=self.distributed_rank,
                    local_time=local_time,
                    global_time=global_time,
                    optimizer_steps=self._optimizer_steps,
                )
                _assert_finite(
                    new_reward,
                    label="env.step.reward",
                    rank=self.distributed_rank,
                    local_time=local_time,
                    global_time=global_time,
                    optimizer_steps=self._optimizer_steps,
                )

            # we check if at the next iteration we will evaluate
            # 提前处理下一轮评估的轨迹边界
            next_local_time = local_time + local_step_increment
            next_global_time = next_local_time * global_step_scale
            if (self.evaluate and eval_time_checker.check(next_global_time)) or (
                self.evaluate and next_global_time == self._checkpoint_global_time
            ):
                if isinstance(self.cfg.env, HumanoidVerseMjlabConfig) and uses_humanoidverse_eval:
                    # make sure we set truncated since at the next iteration we are forced to reset the environment
                    # after the evaluation. This is because we share the environment with the evaluation
                    new_truncated = np.ones_like(new_truncated, dtype=bool)
                    truncated = np.ones_like(new_truncated, dtype=bool)
            # 检查gymnasium版本，构造轨迹数据
            if Version(gymnasium.__version__) >= Version("1.0"):
                # 如果有使用轨迹缓冲区
                if self.cfg.use_trajectory_buffer:
                    data = {
                        "observation": tree_map(lambda x: x[None, ...], obs),
                        "action": action[None, ...],
                        "terminated": terminated[None, ..., None],
                        "truncated": truncated[None, ..., None],
                        "step_count": step_count[None, ..., None],
                        "reward": new_reward[None, ..., None],
                    }
                    data["observation"].pop("history", None)
                    if context is not None:
                        data["z"] = context[None, ...]
                    if history_context is not None:
                        data["history_context"] = history_context[None, ...]
                    if "qpos" in info:
                        data["qpos"] = info["qpos"][None, ...]
                    if "qvel" in info:
                        data["qvel"] = info["qvel"][None, ...]
                    if "aux_rewards" in new_info:
                        data["aux_rewards"] = {k: v[None, ..., None] for k, v in new_info["aux_rewards"].items() if not k.startswith("_")}
                # 如果没有使用轨迹缓冲区，那么使用普通缓冲区
                else:
                    # We add only transitions corresponding to environments that have not reset in the previous step.
                    # For environments that have reset in the previous step, the new observation corresponds to the state after reset.
                    indexes = ~done
                    # float64 降为 float32，再筛选有效环境
                    real_next_obs = tree_map(lambda x: x.astype(np.float32 if x.dtype == np.float64 else x.dtype)[indexes], new_td)
                    # TODO again, we need to remove "time" from the observation (to stay consistent with obs_space)
                    _ = real_next_obs.pop("time")
                    _ = real_next_obs.pop("history", None)
                    # 普通缓冲区分支就得显示构造转移
                    data = {
                        "observation": tree_map(lambda x: x[indexes], obs),
                        "action": action[indexes],
                        "step_count": step_count[indexes],
                        "reward": new_reward[indexes].reshape(-1, 1),
                        "next": {
                            "observation": real_next_obs,
                            "terminated": new_terminated[indexes].reshape(-1, 1),
                            "truncated": new_truncated[indexes].reshape(-1, 1),
                        },
                    }
                    data["observation"].pop("history", None)
                    if context is not None:
                        data["z"] = context[indexes]
                    if history_context is not None:
                        data["history_context"] = history_context[indexes]
                    if "qpos" in info:
                        data["qpos"] = info["qpos"][indexes]
                        data["next"]["qpos"] = new_info["qpos"][indexes]
                    if "qvel" in info:
                        data["qvel"] = info["qvel"][indexes]
                        data["next"]["qvel"] = new_info["qvel"][indexes]
                    if "aux_rewards" in new_info:
                        data["aux_rewards"] = {
                            k: v[indexes].reshape(-1, 1) for k, v in new_info["aux_rewards"].items() if not k.startswith("_")
                        }
            else:
                raise NotImplementedError("still some work to do for gymnasium < 1.0")
            if check_rollout_nonfinite:
                _assert_finite(
                    data,
                    label="replay.extend.data",
                    rank=self.distributed_rank,
                    local_time=local_time,
                    global_time=global_time,
                    optimizer_steps=self._optimizer_steps,
                )
            replay_buffer["train"].extend(data)
            # 判断是否更新网络，必须满足三哪个条件，首先缓冲区必须非空
            # 并且超过了初始随机探索阶段，必须到达算法更新间隔
            if len(replay_buffer["train"]) > 0 and local_time > self.cfg.num_seed_steps and update_agent_time_checker.check(local_time):
                # 记录这次更新触发的位置
                update_agent_time_checker.update_last_step(local_time)
                # 连续执行配置数量的算法更新
                for _ in range(self.cfg.num_agent_updates):
                    # 从缓冲区采样，计算损失，反向传播并更新网络参数，返回指标
                    metrics = self.agent.update(replay_buffer, local_time)
                    self._optimizer_steps += 1
                    # 对指标进行检查看看数值是否异常
                    if self.cfg.fail_on_nonfinite:
                        _assert_finite(
                            metrics,
                            label="agent.update.metrics",
                            rank=self.distributed_rank,
                            local_time=local_time,
                            global_time=global_time,
                            optimizer_steps=self._optimizer_steps,
                        )
                        if (
                            self.cfg.nonfinite_check_model_every_updates > 0
                            and self._optimizer_steps % self.cfg.nonfinite_check_model_every_updates == 0
                        ):
                            _assert_model_finite(
                                self.agent._model,
                                rank=self.distributed_rank,
                                local_time=local_time,
                                global_time=global_time,
                                optimizer_steps=self._optimizer_steps,
                            )
                    # 同步模型统计量，累计训练指标
                    if self.cfg.distributed_sync:
                        sync_floating_buffers(self.agent._model)
                    if self.cfg.distributed_sync and self.cfg.distributed_average_metrics:
                        metrics = average_metrics(metrics)
                    total_metrics, metric_update_counts = _accumulate_metrics(
                        total_metrics,
                        metric_update_counts,
                        metrics,
                    )
            # 达到日志间隔时计算平均指标
            if log_time_checker.check(global_time) and total_metrics is not None:
                # 记录本次日志进度
                log_time_checker.update_last_step(global_time)
                # 创建最终输出的日志字典
                m_dict = {}
                for k in sorted(list(total_metrics.keys())):
                    tmp = total_metrics[k] / metric_update_counts[k]
                    m_dict[k] = np.round(tmp.mean().item(), 6)
                m_dict.update(self._get_torso_contact_force_metrics(train_env))
                m_dict["duration [minutes]"] = (time.time() - start_time) / 60
                m_dict["FPS"] = (1 if global_time == 0 else self.cfg.log_every_updates) / (time.time() - fps_start_time)
                if self.cfg.distributed_sync and self.distributed_world_size > 1:
                    m_dict["distributed/world_size"] = self.distributed_world_size
                    m_dict["distributed/loss_mode"] = _distributed_loss_mode(self.cfg)
                    m_dict["distributed/effective_batch_size"] = _effective_batch_size(self.cfg)
                    m_dict["distributed/num_envs_per_rank"] = _num_envs_per_rank(self.cfg)
                    m_dict["distributed/global_parallel_envs"] = _global_parallel_envs(self.cfg)
                    m_dict["distributed/replay_capacity_per_rank"] = _replay_capacity_per_rank(self.cfg)
                    m_dict["distributed/effective_replay_capacity"] = _effective_replay_capacity(self.cfg)
                    m_dict["distributed/trajectory_steps_per_rank"] = _trajectory_steps_per_rank(self.cfg)
                    m_dict["distributed/compile"] = int(bool(self.cfg.agent.compile))
                m_dict["distributed/local_env_steps"] = int(local_time)
                m_dict["distributed/global_env_steps"] = int(global_time)
                m_dict["distributed/optimizer_steps"] = int(self._optimizer_steps)
                if self._write_shared_artifacts and self.cfg.use_wandb:
                    wandb.log(
                        {f"train/{k}": v for k, v in m_dict.items()},
                        step=global_time,
                    )
                if self._write_shared_artifacts:
                    print(m_dict)
                total_metrics = None
                metric_update_counts = {}
                fps_start_time = time.time()
                m_dict["timestep"] = global_time
                m_dict["local_timestep"] = local_time
                if self.train_logger is not None:
                    self.train_logger.log(m_dict)
            # 把新观测变成下一轮的当前观测
            progb.update(global_step_increment)
            # 下一轮使用这次环境返回的新观测
            td = new_td
            # 下一轮保留这次返回的终止标记
            terminated = new_terminated
            # 下一轮保留这次返回的截断标记
            truncated = new_truncated
            done = np.logical_or(new_terminated.ravel(), new_truncated.ravel())
            info = new_info
        train_env.close()

    def eval(self, t, replay_buffer):
        print(f"Starting evaluation at time {t}")
        evaluation_results = {}

        # This will contain the results, mapping evaluation.cfg.name --> dict of metrics
        evaluation_results = {}
        for evaluation_name in self.evaluations.keys():
            logger = self.eval_loggers.get(evaluation_name)
            evaluation = self.evaluations[evaluation_name]

            # NOTE we have this inside the loop so that the agent is not moved to cpu if we don't evaluate
            if not isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
                self.agent._model.to("cpu")
            self.agent._model.train(False)

            if isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
                # Pass train env
                evaluation_metrics, wandb_dict = evaluation.run(
                    timestep=t, agent_or_model=self.agent, replay_buffer=replay_buffer, logger=logger, env=self.train_env
                )
            else:
                evaluation_metrics, wandb_dict = evaluation.run(
                    timestep=t,
                    agent_or_model=self.agent,
                    replay_buffer=replay_buffer,
                    logger=logger,
                )
            # For wandb dict, put it on wandb
            if self._write_shared_artifacts and self.cfg.use_wandb and wandb_dict is not None:
                wandb.log(
                    {f"eval/{evaluation_name}/{k}": v for k, v in wandb_dict.items()},
                    step=t,
                )

            evaluation_results[evaluation_name] = evaluation_metrics

        # ---------------------------------------------------------------
        # this is important, move back the agent to cuda and
        # restart the training
        if not isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
            self.agent._model.to(self.cfg.agent.model.device)
        self.agent._model.train()

        return evaluation_results

    def save(self, *, local_time: int, global_time: int, optimizer_steps: int, replay_buffer: Dict[str, tp.Any]) -> None:
        checkpoint_dir = self.work_dir / CHECKPOINT_DIR_NAME
        sync_report = None
        if self.cfg.distributed_sync:
            barrier()
            sync_report = module_sync_report(self.agent._model, src=0)
            if sync_report["max_abs_diff_from_rank0"] > 1.0e-5:
                raise RuntimeError(f"Distributed model state diverged before checkpoint: {sync_report}")
        if self._write_shared_artifacts:
            print(f"Checkpointing at local_time={local_time} global_time={global_time} optimizer_steps={optimizer_steps}")
            self.agent.save(str(checkpoint_dir))
        if self.cfg.checkpoint_buffer:
            replay_buffer["train"].save(self._checkpoint_buffer_path(checkpoint_dir))
        if self.cfg.distributed_sync:
            barrier()
        if self._write_shared_artifacts:
            if sync_report is not None:
                with (checkpoint_dir / "distributed_sync.json").open("w+") as f:
                    json.dump(sync_report, f, indent=4)
            with (checkpoint_dir / "train_status.json").open("w+") as f:
                json.dump(
                    _make_train_status(
                        self.cfg,
                        local_time=local_time,
                        global_time=global_time,
                        optimizer_steps=optimizer_steps,
                    ),
                    f,
                    indent=4,
                )
        if self.cfg.distributed_sync:
            barrier()


def train_bfm_zero():
    raise RuntimeError(
        "Legacy train_bfm_zero entrypoint is disabled in this MJLab build. "
        "Use humanoidverse.train or ./run_train.sh."
    )

if __name__ == "__main__":
    # This is the bare minimum CLI interface to launch experiments, but ideally you should
    # launch your experiments from Python code (e.g., see under "scripts")
    train_bfm_zero()

# uv run --no-cache -m humanoidverse.meta_online_entry_point
