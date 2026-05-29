# PathMNIST Project Structure

This `model/` directory is organized by experiment role.

## Shared Implementation

- `src/`: shared Python implementation copied from the official DistDiff code.
  - `src/generate_data.py`: Stable Diffusion / DistDiff synthetic generation.
  - `src/train.py`: original-data classifier training.
  - `src/train_expanded_data_concat_original.py`: original + synthetic classifier training.
  - `src/train_transform.py`: transformation-augmentation trainer.
  - `src/model_utils.py`: ResNet and CLIP-ViT-B/32 model factory.
  - `src/parse_logs.py`: result aggregation.

All launcher scripts `cd` back to the `model/` root before calling files in `src/`, so existing
relative paths such as `data/`, `checkpoint/`, `logs/`, and `save/` remain valid.

## Synthetic Data Generation

- `make_synthetic_data/DistDiff/`
  - `run_scale_distdiff_pathmnist.sh`: generate DistDiff synthetic data for 1x/2x/5x/10x.
  - `monitor_scale_distdiff_pathmnist.sh`: 30-second live monitor.
  - `results/`: symlinks to generated DistDiff PNG folders.
  - `legacy/`: older exploratory/stage scripts kept for reference.

- `make_synthetic_data/Naive_SD/`
  - `run_scale_naive_sd_pathmnist.sh`: generate naive Stable Diffusion synthetic data for 1x/2x/5x/10x.
  - `monitor_scale_naive_sd_pathmnist.sh`: 30-second live monitor.
  - `results/`: symlinks to generated naive SD PNG folders.
  - `legacy/`: older exploratory/stage scripts kept for reference.

## Classifier Training

- `classifier/ResNet50_scratch/`
  - `run_scale_resnet50_pathmnist.sh`: train ResNet-50 from scratch on Original, Naive SD 1x/2x/5x/10x, and DistDiff 1x/2x/5x/10x.
  - `monitor_scale_resnet50_pathmnist.sh`: 30-second live monitor.
  - `results/`: symlinks to ResNet-50 scratch checkpoint folders.
  - `legacy/`: older exploratory/stage scripts kept for reference.

- `classifier/CLIP_ViT_B32_pretrained/`
  - `run_scale_clip_pathmnist.sh`: train CLIP-ViT-B/32 pretrained on Original, Naive SD 1x/2x/5x/10x, and DistDiff 1x/2x/5x/10x.
  - `monitor_scale_clip_pathmnist.sh`: 30-second live monitor.
  - `run_table2_clip_pathmnist.sh`: Table 2-style 5x-only CLIP-ViT-B/32 fine-tuning, kept for paper-specific comparison.
  - `monitor_table2_clip_pathmnist.sh`: monitor for the 5x-only CLIP experiment.
  - `results/`: symlinks to CLIP checkpoint folders.

## Persistent Data and Outputs

- `data/`: PathMNIST symlinks and synthetic image storage.
- `checkpoint/`: actual model checkpoints and `results.yaml` files.
- `logs/`: actual run logs.
- `debug/`: analysis notebooks, CSV summaries, and figures.
- `save/`: cached VAE embeddings and optional local pretrained weights.
