from __future__ import annotations

"""Action-free Python inference interface for WCM value estimates."""

import os
from pathlib import Path
from typing import Any, Sequence

import torch

from .checkpoint import CHECKPOINT_SCHEMA_VERSION, load_checkpoint_payload
from .config import TrainConfig, apply_runtime_overrides, validate_train_config
from .data import ValueOnlyCollator, _to_image_tensor, build_processor
from .distributed import DistributedContext
from .model import WorldCriticModel
from .training import autocast_context, config_from_checkpoint_payload, move_batch_to_device


def resolve_inference_device(requested: str | torch.device = "auto") -> torch.device:
    if isinstance(requested, torch.device):
        return requested
    force_cpu = os.environ.get("WCM_FORCE_CPU", "0") == "1"
    if requested == "cpu" or force_cpu:
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda", 0)
    if requested != "auto":
        raise ValueError(f"Unsupported inference device {requested!r}; use auto, cuda, or cpu.")
    return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


def _history_steps(history_images: Any, expected_views: int) -> list[list[torch.Tensor]]:
    if torch.is_tensor(history_images):
        if history_images.ndim == 4:
            steps = [[history_images[index]] for index in range(history_images.size(0))]
        elif history_images.ndim == 5:
            steps = [list(history_images[index]) for index in range(history_images.size(0))]
        else:
            raise ValueError(
                "Tensor history_images must be [T,C,H,W] or [T,V,C,H,W], "
                f"got {history_images.shape}."
            )
    else:
        try:
            raw_steps = list(history_images)
        except TypeError as exc:
            raise TypeError("history_images must be a non-empty image sequence.") from exc
        steps = []
        for step in raw_steps:
            if expected_views == 1 and not isinstance(step, (list, tuple)):
                steps.append([_to_image_tensor(step)])
            else:
                try:
                    views = list(step)
                except TypeError as exc:
                    raise TypeError(
                        f"Each history timestep must contain {expected_views} camera image(s)."
                    ) from exc
                steps.append([_to_image_tensor(image) for image in views])
    if not steps:
        raise ValueError("history_images must contain at least one timestep.")
    if any(len(step) != expected_views for step in steps):
        counts = [len(step) for step in steps]
        raise ValueError(f"Expected {expected_views} camera image(s) per timestep, got {counts}.")
    return [[_to_image_tensor(image) for image in step] for step in steps]


class ValuePredictor:
    """Load a WCM checkpoint and predict the value of the latest observation."""

    def __init__(
        self,
        model: WorldCriticModel,
        collator: ValueOnlyCollator,
        config: TrainConfig,
        device: torch.device,
    ) -> None:
        if config.data is None:
            raise ValueError("ValuePredictor requires checkpoint data settings.")
        self.model = model
        self.collator = collator
        self.config = config
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "auto",
        precision: str | None = None,
    ) -> "ValuePredictor":
        resolved_device = resolve_inference_device(device)
        ctx = DistributedContext(rank=0, local_rank=0, world_size=1, device=resolved_device)
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        payload = load_checkpoint_payload(checkpoint_path, ctx)
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported checkpoint schema in {checkpoint_path}.")
        if payload.get("artifact_type") not in {"full_resume", "deploy"}:
            raise ValueError(f"Unsupported artifact type: {payload.get('artifact_type')!r}.")
        if "model" not in payload or "config" not in payload:
            raise KeyError("Checkpoint artifact is missing model or config.")

        config = apply_runtime_overrides(config_from_checkpoint_payload(payload))
        if precision is not None:
            if precision not in {"fp32", "bf16"}:
                raise ValueError("precision must be fp32 or bf16.")
            config.precision = precision
        validate_train_config(config)
        processor = build_processor(config.model)
        collator = ValueOnlyCollator(
            processor,
            config.model.vision.image_size,
            config.model.language.max_length,
        )

        # The checkpoint contains both backbone weights. Construct only the
        # architectures before loading the complete state dict.
        config.model.vision.pretrained = False
        config.model.language.pretrained = False
        model = WorldCriticModel(config.model)
        model.load_state_dict(payload["model"], strict=True)
        model.to(resolved_device).eval().requires_grad_(False)
        return cls(model, collator, config, resolved_device)

    def _sample(
        self,
        history_images: Any,
        instruction: str,
        valid_mask: Sequence[bool] | torch.Tensor | None,
    ) -> dict[str, Any]:
        assert self.config.data is not None
        instruction = str(instruction).strip()
        if not instruction:
            raise ValueError("instruction must be a non-empty string.")
        images = _history_steps(history_images, len(self.config.data.image_keys))
        time = len(images)
        if time > self.config.model.max_history:
            raise ValueError(
                f"History length {time} exceeds model.max_history={self.config.model.max_history}."
            )
        if valid_mask is None:
            mask = torch.ones(time, dtype=torch.bool)
        else:
            mask = torch.as_tensor(valid_mask, dtype=torch.bool)
        if mask.shape != (time,):
            raise ValueError(f"valid_mask must have shape [{time}], got {tuple(mask.shape)}.")
        if not mask.any():
            raise ValueError("valid_mask must contain at least one valid observation.")
        if not bool(mask[-1]):
            raise ValueError("The latest/current observation must be valid (valid_mask[-1] == True).")
        return {"images": images, "instruction": instruction, "valid_mask": mask}

    def predict_batch(
        self,
        history_images: Sequence[Any],
        instructions: Sequence[str],
        valid_masks: Sequence[Sequence[bool] | torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        histories = list(history_images)
        instruction_list = list(instructions)
        if len(histories) != len(instruction_list):
            raise ValueError(
                f"History and instruction batch sizes differ: {len(histories)} vs {len(instruction_list)}."
            )
        if not histories:
            raise ValueError("Cannot predict an empty batch.")
        masks = [None] * len(histories) if valid_masks is None else list(valid_masks)
        if len(masks) != len(histories):
            raise ValueError(f"History and valid-mask batch sizes differ: {len(histories)} vs {len(masks)}.")
        samples = [
            self._sample(images, instruction, mask)
            for images, instruction, mask in zip(histories, instruction_list, masks, strict=True)
        ]
        batch = move_batch_to_device(self.collator(samples), self.device)
        with torch.inference_mode(), autocast_context(self.device, self.config.precision):
            values = self.model.forward_value(
                images=batch["images"],
                instruction_input_ids=batch["instruction_input_ids"],
                instruction_attention_mask=batch["instruction_attention_mask"],
                valid_mask=batch["valid_mask"],
            )
        endpoints = values[:, -1, 0].detach().float().cpu()
        if not torch.isfinite(endpoints).all():
            raise ValueError("Value inference produced a non-finite current value.")
        return endpoints

    def predict(
        self,
        history_images: Any,
        instruction: str,
        valid_mask: Sequence[bool] | torch.Tensor | None = None,
    ) -> float:
        return float(self.predict_batch([history_images], [instruction], [valid_mask])[0].item())
