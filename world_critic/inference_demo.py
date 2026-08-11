from __future__ import annotations

import argparse
import bisect
import json
import os
from pathlib import Path
from typing import Any

import torch

from .checkpoint import CHECKPOINT_SCHEMA_VERSION, load_checkpoint_payload
from .config import apply_runtime_overrides, validate_train_config
from .data import (
    WorldCriticCollator,
    _to_image_tensor,
    build_processor,
    infer_feature_dim,
    load_lerobot_dataset,
    task_for_sample,
)
from .distributed import DistributedContext
from .model import WorldCriticModel
from .training import (
    autocast_context,
    config_from_checkpoint_payload,
    move_batch_to_device,
    seed_everything,
)


DEFAULT_DATASET_ROOT = "/home/wenchaoxu/phs/datasets/libero_plus_lerobot"


class LeRobotV21Reader:
    """Small read-only adapter for displaying v2.1 data with modern LeRobot installed."""

    def __init__(self, root: str | Path, image_keys: list[str]) -> None:
        self.root = Path(root)
        info = json.loads((self.root / "meta" / "info.json").read_text(encoding="utf-8"))
        self.features = info["features"]
        self.image_keys = image_keys
        self.total_frames = int(info["total_frames"])
        episode_lines = (self.root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        self.episodes = [json.loads(line) for line in episode_lines if line.strip()]
        self.episode_ends = []
        running_total = 0
        for episode in self.episodes:
            running_total += int(episode["length"])
            self.episode_ends.append(running_total)
        if running_total != self.total_frames:
            raise ValueError(
                f"Episode lengths total {running_total}, metadata reports {self.total_frames}."
            )
        task_lines = (self.root / "meta" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
        self.tasks = {
            int(item["task_index"]): str(item["task"])
            for item in (json.loads(line) for line in task_lines if line.strip())
        }
        self._cached_episode_index: int | None = None
        self._cached_rows: list[dict[str, Any]] | None = None
        self._cached_decoders: dict[str, Any] = {}

    def __len__(self) -> int:
        return self.total_frames

    def global_index(self, episode_index: int, frame_index: int) -> int:
        if episode_index < 0 or episode_index >= len(self.episodes):
            raise IndexError(
                f"episode_index={episode_index} is outside [0, {len(self.episodes) - 1}]."
            )
        episode = self.episodes[episode_index]
        actual_episode_index = int(episode["episode_index"])
        if actual_episode_index != episode_index:
            raise ValueError(
                "Episode metadata is not ordered by episode_index: "
                f"position={episode_index}, value={actual_episode_index}."
            )
        episode_length = int(episode["length"])
        if frame_index < 0 or frame_index >= episode_length:
            raise IndexError(
                f"frame_index={frame_index} is outside episode {episode_index} "
                f"length {episode_length}."
            )
        episode_start = 0 if episode_index == 0 else self.episode_ends[episode_index - 1]
        return episode_start + frame_index

    def _load_episode(self, episode_index: int) -> None:
        if self._cached_episode_index == episode_index:
            return
        try:
            import pyarrow.parquet as pq
            from torchcodec.decoders import VideoDecoder
        except ImportError as exc:
            raise ImportError("Reading LeRobot v2.1 requires pyarrow and torchcodec.") from exc

        chunk = episode_index // 1000
        parquet_path = (
            self.root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        )
        self._cached_rows = pq.read_table(parquet_path).to_pylist()
        self._cached_decoders = {}
        for key in self.image_keys:
            video_path = (
                self.root
                / "videos"
                / f"chunk-{chunk:03d}"
                / key
                / f"episode_{episode_index:06d}.mp4"
            )
            self._cached_decoders[key] = VideoDecoder(str(video_path))
        self._cached_episode_index = episode_index

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_position = bisect.bisect_right(self.episode_ends, index)
        episode = self.episodes[episode_position]
        episode_index = int(episode["episode_index"])
        episode_start = 0 if episode_position == 0 else self.episode_ends[episode_position - 1]
        local_index = index - episode_start
        self._load_episode(episode_index)
        assert self._cached_rows is not None
        sample = dict(self._cached_rows[local_index])
        sample["task"] = self.tasks[int(sample["task_index"])]
        for key, decoder in self._cached_decoders.items():
            sample[key] = decoder[local_index]
        return sample


def _load_dataset(config) -> Any:
    info_path = Path(config.data.root) / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if str(info.get("codebase_version")) == "v2.1":
        print("[demo] using the read-only LeRobot v2.1 adapter", flush=True)
        return LeRobotV21Reader(config.data.root, config.data.image_keys)
    return load_lerobot_dataset(config.data)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run WCM inference on real LeRobot observations.")
    result.add_argument("--checkpoint", required=True, help="WCM deploy.pt or full checkpoint.")
    result.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    result.add_argument(
        "--dataset-repo-id",
        default="Sylvest/libero_plus_lerobot",
        help="LeRobot dataset identity; --dataset-root supplies the local files.",
    )
    result.add_argument("--episode-index", type=int, default=0)
    result.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="First frame of the inference history window within the selected episode.",
    )
    result.add_argument("--seed", type=int, default=3072)
    result.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device (default: CUDA when available).",
    )
    result.add_argument(
        "--precision",
        choices=["fp32", "bf16"],
        help="Override the precision stored in the checkpoint.",
    )
    return result


def _resolve_device(requested: str) -> torch.device:
    force_cpu = os.environ.get("WCM_FORCE_CPU", "0") == "1"
    if requested == "cpu" or force_cpu:
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda", 0)
    return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


def _scalar_int(value: Any, name: str) -> int:
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be scalar, got shape={tuple(tensor.shape)}.")
    return int(tensor.item())


def _make_real_sample(dataset: Any, config, start_index: int) -> dict[str, Any]:
    """Build one inference window with the same layout as the training dataset wrapper."""

    window = config.data.history_size + config.data.prediction_horizon
    if start_index < 0 or start_index + window > len(dataset):
        raise IndexError(
            f"Window [{start_index}, {start_index + window}) is outside dataset length {len(dataset)}."
        )
    samples = [dataset[row] for row in range(start_index, start_index + window)]
    episode_ids = [_scalar_int(sample["episode_index"], "episode_index") for sample in samples]
    if len(set(episode_ids)) != 1:
        raise ValueError(f"The requested window crosses an episode boundary: {episode_ids}.")
    frame_indices = [_scalar_int(sample["frame_index"], "frame_index") for sample in samples]
    if any(right != left + 1 for left, right in zip(frame_indices, frame_indices[1:])):
        raise ValueError(f"The requested window has non-consecutive frames: {frame_indices}.")

    instruction = task_for_sample(dataset, samples[0])
    if any(task_for_sample(dataset, sample) != instruction for sample in samples):
        raise ValueError("The requested window changes task instruction.")

    current = samples[:-1]
    actions = torch.stack(
        [torch.as_tensor(sample[config.data.action_key], dtype=torch.float32).reshape(-1) for sample in current]
    )
    if config.data.normalize_action:
        if config.data.action_mean is None or config.data.action_std is None:
            raise ValueError("Checkpoint config is missing the fitted action normalization statistics.")
        mean = torch.as_tensor(config.data.action_mean, dtype=actions.dtype)
        std = torch.as_tensor(config.data.action_std, dtype=actions.dtype)
        actions = (actions - mean) / std

    result = {
        "images": [
            [_to_image_tensor(sample[key]) for key in config.data.image_keys]
            for sample in samples
        ],
        "actions": actions,
        "instruction": instruction,
        "valid_mask": torch.ones(len(current), dtype=torch.bool),
        "episode_id": episode_ids[0],
        "frame_indices": torch.as_tensor(frame_indices[:-1], dtype=torch.long),
        "sample_id": f"{episode_ids[0]}:{frame_indices[0]}",
    }
    if all(config.data.return_key in sample for sample in current):
        result["return_targets"] = torch.stack(
            [
                torch.as_tensor(sample[config.data.return_key], dtype=torch.float32).reshape(1)
                for sample in current
            ]
        )
    return result


def run() -> None:
    args = parser().parse_args()
    if args.episode_index < 0:
        raise ValueError("--episode-index cannot be negative.")
    if args.frame_index < 0:
        raise ValueError("--frame-index cannot be negative.")

    device = _resolve_device(args.device)
    ctx = DistributedContext(rank=0, local_rank=0, world_size=1, device=device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    payload = load_checkpoint_payload(checkpoint_path, ctx)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema in {checkpoint_path}.")
    if payload.get("artifact_type") not in {"full_resume", "deploy"}:
        raise ValueError(f"Unsupported artifact type: {payload.get('artifact_type')!r}.")
    if "model" not in payload or "config" not in payload:
        raise KeyError("Checkpoint artifact is missing model or config.")

    config = apply_runtime_overrides(config_from_checkpoint_payload(payload))
    config.data.repo_id = args.dataset_repo_id
    config.data.root = str(Path(args.dataset_root).expanduser().resolve())
    config.data.revision = None
    if args.precision is not None:
        config.precision = args.precision
    validate_train_config(config)
    seed_everything(args.seed, config.deterministic)

    print(f"[demo] loading real dataset from {config.data.root!r}...", flush=True)
    dataset = _load_dataset(config)
    action_dim = infer_feature_dim(dataset, config.data.action_key)
    if config.model.action_dim != action_dim:
        raise ValueError(
            f"Checkpoint action_dim={config.model.action_dim}, dataset action_dim={action_dim}."
        )
    if not isinstance(dataset, LeRobotV21Reader):
        raise ValueError(
            "Selecting an episode/frame is currently implemented for the configured LeRobot v2.1 dataset."
        )
    start_index = dataset.global_index(args.episode_index, args.frame_index)

    print(f"[demo] loading processor: vision={config.model.vision.model_name!r}, "
          f"language={config.model.language.model_name!r}", flush=True)
    processor = build_processor(config.model)
    collator = WorldCriticCollator(
        processor,
        config.model.vision.image_size,
        config.model.language.max_length,
    )
    samples = [_make_real_sample(dataset, config, start_index)]
    batch = collator(samples)

    # The WCM artifact contains the complete state_dict, including both HF
    # backbones.  Construct only their architectures from config so inference
    # does not download the original ViT/CLIP model.safetensors just to
    # overwrite them immediately below.
    config.model.vision.pretrained = False
    config.model.language.pretrained = False
    print(
        "[demo] constructing model from HF configs only (no original backbone weights)...",
        flush=True,
    )
    model = WorldCriticModel(config.model)
    print("[demo] loading complete WCM checkpoint weights...", flush=True)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    batch = move_batch_to_device(batch, device)

    with torch.inference_mode(), autocast_context(device, config.precision):
        output = model(
            images=batch["images"],
            actions=batch["actions"],
            instruction_input_ids=batch["instruction_input_ids"],
            instruction_attention_mask=batch["instruction_attention_mask"],
            valid_mask=batch["valid_mask"],
        )

    values = output.value.detach().float().cpu()[..., 0]
    summary = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": config.data.root,
        "device": str(device),
        "precision": config.precision,
        "samples": [
            {
                "sample_id": sample["sample_id"],
                "instruction": sample["instruction"],
                "frame_indices": sample["frame_indices"].tolist(),
                "predicted_values": values[index].tolist(),
                "return_targets": (
                    sample["return_targets"][:, 0].tolist()
                    if "return_targets" in sample
                    else None
                ),
            }
            for index, sample in enumerate(samples)
        ],
        "input_shapes": {
            "images": list(batch["images"].shape),
            "actions": list(batch["actions"].shape),
            "instruction_input_ids": list(batch["instruction_input_ids"].shape),
            "valid_mask": list(batch["valid_mask"].shape),
        },
        "output_shapes": {
            "value": list(output.value.shape),
            "next_state_pred": list(output.next_state_pred.shape),
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run()
