from __future__ import annotations

"""Infer an action-free value curve for every frame of one LeRobot episode."""

import argparse
import csv
import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import torch

from .data import _to_image_tensor, task_for_sample
from .inference_demo import LeRobotV21Reader, _scalar_int
from .training import seed_everything
from .value_inference import ValuePredictor


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Infer an action-free value for every frame of one LeRobot v2.1/v3.0 episode."
    )
    result.add_argument("--checkpoint", required=True, help="WCM deploy.pt or full checkpoint.")
    result.add_argument("--dataset-root", required=True)
    result.add_argument(
        "--dataset-repo-id",
        help="LeRobot dataset identity (defaults to the checkpoint config).",
    )
    result.add_argument("--dataset-revision")
    result.add_argument("--episode-index", type=int, required=True)
    result.add_argument(
        "--instruction",
        help="Override the episode task instruction; otherwise use LeRobot task metadata.",
    )
    result.add_argument(
        "--output",
        required=True,
        help="Output episode_curves.json path; sibling CSV and PNG files are also written.",
    )
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument(
        "--reverse-episode",
        action="store_true",
        help=(
            "Play the episode from its last frame to its first frame before building each "
            "history window. Output frame_indices are reverse-playback steps; "
            "source_frame_indices record the corresponding original frame indices."
        ),
    )
    result.add_argument("--seed", type=int, default=3072)
    result.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    result.add_argument("--precision", choices=["fp32", "bf16"])
    return result


def _dataset_version(root: Path) -> str:
    info_path = root / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"LeRobot metadata does not exist: {info_path}") from exc
    version = str(info.get("codebase_version", "")).strip().lower()
    if version not in {"v2.1", "v3.0"}:
        raise ValueError(
            f"Unsupported LeRobot codebase_version={version!r} in {info_path}; expected v2.1 or v3.0."
        )
    return version


def _load_episode(
    *,
    root: Path,
    repo_id: str,
    revision: str | None,
    image_keys: list[str],
    episode_index: int,
) -> tuple[Any, int, Callable[[int], dict[str, Any]], str]:
    version = _dataset_version(root)
    if version == "v2.1":
        dataset = LeRobotV21Reader(root, image_keys)
        start = dataset.global_index(episode_index, 0)
        length = int(dataset.episodes[episode_index]["length"])
        return dataset, length, lambda frame: dataset[start + frame], version

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError("LeRobot v3.0 inference requires lerobot>=0.5.1,<0.6.") from exc
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=[episode_index],
        revision=revision,
    )
    length = len(dataset)
    if length < 1:
        raise ValueError(f"LeRobot v3.0 episode_index={episode_index} contains no frames.")
    return dataset, length, dataset.__getitem__, version


def _episode_histories(
    *,
    read_frame: Callable[[int], dict[str, Any]],
    frame_indices: range,
    frame_order: Sequence[int] | None = None,
    history_size: int,
    image_keys: list[str],
    episode_index: int,
    instruction_override: str | None,
    dataset: Any,
) -> tuple[list[list[list[torch.Tensor]]], list[torch.Tensor], str]:
    if history_size < 1:
        raise ValueError("history_size must be positive.")

    @lru_cache(maxsize=None)
    def checked_frame(frame_index: int) -> dict[str, Any]:
        sample = read_frame(frame_index)
        actual_episode = _scalar_int(sample["episode_index"], "episode_index")
        actual_frame = _scalar_int(sample["frame_index"], "frame_index")
        if actual_episode != episode_index or actual_frame != frame_index:
            raise ValueError(
                "LeRobot episode/frame metadata is not aligned: "
                f"requested=({episode_index},{frame_index}), actual=({actual_episode},{actual_frame})."
            )
        missing = [key for key in image_keys if key not in sample]
        if missing:
            raise KeyError(f"LeRobot frame is missing checkpoint image features: {missing}.")
        return sample

    override = instruction_override.strip() if instruction_override is not None else None
    if instruction_override is not None and not override:
        raise ValueError("--instruction cannot be empty.")
    if frame_order is None:
        frame_order = range(max(frame_indices.stop, 1))
    if not frame_order:
        raise ValueError("frame_order cannot be empty.")
    if frame_indices.start < 0 or frame_indices.stop > len(frame_order):
        raise ValueError(
            f"Playback steps {frame_indices.start}:{frame_indices.stop} are outside "
            f"frame_order length {len(frame_order)}."
        )

    first_sample = checked_frame(frame_order[0])
    instruction = override or task_for_sample(dataset, first_sample)

    histories: list[list[list[torch.Tensor]]] = []
    masks: list[torch.Tensor] = []
    for playback_step in frame_indices:
        first_real = max(0, playback_step - history_size + 1)
        real_indices = list(frame_order[first_real : playback_step + 1])
        pad = history_size - len(real_indices)
        padded_indices = [frame_order[0]] * pad + real_indices
        history = [
            [_to_image_tensor(checked_frame(index)[key]) for key in image_keys]
            for index in padded_indices
        ]
        mask = torch.tensor([False] * pad + [True] * len(real_indices), dtype=torch.bool)
        if override is None:
            for index in set(real_indices):
                sample_instruction = task_for_sample(dataset, checked_frame(index))
                if sample_instruction != instruction:
                    raise ValueError(
                        f"Episode {episode_index} changes task instruction at frame_index={index}."
                    )
        histories.append(history)
        masks.append(mask)
    return histories, masks, instruction


def _write_curve_artifacts(
    output_path: Path,
    *,
    episode_index: int,
    frame_indices: list[int],
    values: list[float],
    source_frame_indices: list[int] | None = None,
) -> tuple[Path, Path, Path]:
    output_path = output_path.expanduser().resolve()
    if output_path.suffix.lower() != ".json":
        raise ValueError(f"--output must be a .json path, got {output_path}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")
    png_path = output_path.with_suffix(".png")
    curve_entry: dict[str, Any] = {
        "episode_id": episode_index,
        "frame_indices": frame_indices,
        "values": values,
    }
    if source_frame_indices is not None:
        if len(source_frame_indices) != len(frame_indices):
            raise ValueError("source_frame_indices must align one-to-one with frame_indices.")
        curve_entry["playback_direction"] = "reverse"
        curve_entry["source_frame_indices"] = source_frame_indices
    curve = [curve_entry]

    temporary_json = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_json.write_text(json.dumps(curve, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary_json.replace(output_path)

    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if source_frame_indices is None:
            writer.writerow(["episode_id", "frame_index", "value"])
            for frame_index, value in zip(frame_indices, values, strict=True):
                writer.writerow([episode_index, frame_index, value])
        else:
            writer.writerow(["episode_id", "frame_index", "source_frame_index", "value"])
            for frame_index, source_frame_index, value in zip(
                frame_indices, source_frame_indices, values, strict=True
            ):
                writer.writerow([episode_index, frame_index, source_frame_index, value])
    temporary_csv.replace(csv_path)

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Writing the value curve PNG requires matplotlib.") from exc
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(frame_indices, values, color="#2563eb", linewidth=2)
    direction = "reverse" if source_frame_indices is not None else "forward"
    axis.set_title(f"WCM predicted value — episode {episode_index} ({direction})")
    axis.set_xlabel("reverse playback step" if source_frame_indices is not None else "frame_index")
    axis.set_ylabel("value")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    temporary_png = png_path.with_suffix(png_path.suffix + ".tmp")
    figure.savefig(temporary_png, format="png", dpi=160)
    plt.close(figure)
    temporary_png.replace(png_path)
    return output_path, csv_path, png_path


def run() -> None:
    args = parser().parse_args()
    if args.episode_index < 0:
        raise ValueError("--episode-index cannot be negative.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")

    predictor = ValuePredictor.from_checkpoint(
        args.checkpoint,
        device=args.device,
        precision=args.precision,
    )
    config = predictor.config
    assert config.data is not None
    seed_everything(args.seed, config.deterministic)
    root = Path(args.dataset_root).expanduser().resolve()
    repo_id = args.dataset_repo_id or config.data.repo_id
    revision = args.dataset_revision if args.dataset_revision is not None else config.data.revision
    dataset, episode_length, read_frame, version = _load_episode(
        root=root,
        repo_id=repo_id,
        revision=revision,
        image_keys=config.data.image_keys,
        episode_index=args.episode_index,
    )
    history_size = config.data.history_size
    if history_size > config.model.max_history:
        raise ValueError(
            f"Checkpoint history_size={history_size} exceeds model.max_history={config.model.max_history}."
        )

    frame_order = (
        list(range(episode_length - 1, -1, -1))
        if args.reverse_episode
        else list(range(episode_length))
    )
    frame_indices: list[int] = []
    values: list[float] = []
    resolved_instruction: str | None = None
    for first in range(0, episode_length, args.batch_size):
        last = min(first + args.batch_size, episode_length)
        batch_frames = range(first, last)
        histories, masks, instruction = _episode_histories(
            read_frame=read_frame,
            frame_indices=batch_frames,
            frame_order=frame_order,
            history_size=history_size,
            image_keys=config.data.image_keys,
            episode_index=args.episode_index,
            instruction_override=args.instruction,
            dataset=dataset,
        )
        if resolved_instruction is None:
            resolved_instruction = instruction
        elif instruction != resolved_instruction:
            raise ValueError("Episode task instruction changed between inference batches.")
        batch_values = predictor.predict_batch(
            histories,
            [instruction] * len(histories),
            masks,
        )
        frame_indices.extend(batch_frames)
        values.extend(float(value) for value in batch_values.tolist())
        print(f"[episode-value] frames={last}/{episode_length}", flush=True)

    if len(frame_indices) != episode_length or len(values) != episode_length:
        raise RuntimeError(
            f"Expected {episode_length} curve points, got {len(frame_indices)}/{len(values)}."
        )
    json_path, csv_path, png_path = _write_curve_artifacts(
        Path(args.output),
        episode_index=args.episode_index,
        frame_indices=frame_indices,
        values=values,
        source_frame_indices=frame_order if args.reverse_episode else None,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
                "dataset_root": str(root),
                "dataset_version": version,
                "episode_id": args.episode_index,
                "playback_direction": "reverse" if args.reverse_episode else "forward",
                "instruction": resolved_instruction,
                "history_size": history_size,
                "episode_frames": episode_length,
                "curve_points": len(values),
                "outputs": {
                    "json": str(json_path),
                    "csv": str(csv_path),
                    "png": str(png_path),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
