DATASET_REPO_ID="lerobot_with_return_val"   # empty = use the checkpoint's dataset id
DATASET_ROOT="/path/to/lerobot_with_return_val"     
VISION_MODEL_NAME="google/vit-base-patch16-224-in21k"  # Local path / hf name supported
LANGUAGE_MODEL_NAME="openai/clip-vit-base-patch32"   # Local path / hf name supported

export WCM_DATASET_ROOT="$DATASET_ROOT"
export WCM_DATASET_REVISION="$DATASET_REVISION"
export WCM_VISION_MODEL_NAME="$VISION_MODEL_NAME"
export WCM_LANGUAGE_MODEL_NAME="$LANGUAGE_MODEL_NAME"

python -m episode_value_video render \
  --curves outputs/wcm/eval/episode_curves/episode_curves.json \
  --checkpoint outputs/wcm/checkpoints/best.pt \
  --speed 2.0 \
  --output-dir outputs/wcm/eval/episode_value_videos_2x