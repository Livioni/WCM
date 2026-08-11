#!/usr/bin/env bash
set -euo pipefail

# Usage: bash 4_gen_video.sh [episode_index]
# Example: bash 4_gen_video.sh 123
EPISODE_ID="${1:-0}"
DATASET_REPO_ID="Sylvest/libero_plus_lerobot"
DATASET_ROOT="/home/wenchaoxu/phs/datasets/libero_plus_lerobot"
CHECKPOINT="checkpoints/WCM_LIBEROplus/best.pt"
CAMERA_KEY="observation.images.front"
BATCH_SIZE=16
SPEED="2.0"
DEVICE="${DEVICE:-cuda}"              # Override with DEVICE=cpu when needed.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="${PYTHON:-$SCRIPT_DIR/.venv/bin/python}"

if [[ ! "$EPISODE_ID" =~ ^[0-9]+$ ]]; then
  echo "episode_index must be a non-negative integer (got: $EPISODE_ID)" >&2
  exit 2
fi
EPISODE_ID="$((10#$EPISODE_ID))"
CURVE_DIR="outputs/wcm/inference/episode-${EPISODE_ID}/episode_curves"
CURVE_PATH="$CURVE_DIR/episode_curves.json"
VIDEO_OUTPUT_DIR="outputs/wcm/inference/episode-${EPISODE_ID}/episode_value_video"
EPISODE_CHUNK="$(printf '%03d' "$((EPISODE_ID / 1000))")"
SOURCE_VIDEO="$DATASET_ROOT/videos/chunk-$EPISODE_CHUNK/$CAMERA_KEY/episode_$(printf '%06d' "$EPISODE_ID").mp4"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable does not exist: $PYTHON" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint does not exist: $CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_VIDEO" ]]; then
  echo "Episode video does not exist: $SOURCE_VIDEO" >&2
  exit 2
fi

"$PYTHON" -u -m world_critic.infer_episode_curve \
  --checkpoint "$CHECKPOINT" \
  --dataset-root "$DATASET_ROOT" \
  --dataset-repo-id "$DATASET_REPO_ID" \
  --episode-index "$EPISODE_ID" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --output "$CURVE_PATH"

"$PYTHON" -u -m episode_value_video render \
  --curves "$CURVE_PATH" \
  --video-template "$SOURCE_VIDEO" \
  --history-size 3 \
  --source-fps 20 \
  --episode-id "$EPISODE_ID" \
  --speed "$SPEED" \
  --output-dir "$VIDEO_OUTPUT_DIR" \
  --overwrite
