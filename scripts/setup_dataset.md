# Dataset Setup

## 1. Get the Data
You need a Face Mask Detection dataset in YOLO format. You can find many on Kaggle or Roboflow Universe.

**Example Source:** [Roboflow Universe - Face Mask Detection](https://universe.roboflow.com/search?q=face%20mask)

## 2. Prepare Directory Structure
Inside the repository root, create a `datasets` folder if it doesn't exist, and extract your data into `datasets/face-mask-yolo`.

Your structure **MUST** look like this for the default `data.yaml` to work:

```
repo-root/
├── datasets/
│   └── face-mask-yolo/
│       ├── data.yaml  (optional, we use the one in repo root usually, but keep it if present)
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
├── data.yaml (The one we use for training)
├── train.py
...
```

## 3. Verify
Ensure that `data.yaml` in the root correctly points to `../datasets/face-mask-yolo` (relative to where `train.py` runs, or use absolute paths).

If your dataset is organized efficiently, you are ready to train!
