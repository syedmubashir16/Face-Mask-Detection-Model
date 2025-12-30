# Face Mask Detection (YOLOv8n)

## Objective
To build a lightweight, deployable Face Mask Detection system using the Ultralytics YOLOv8n model. The system detects two classes: `mask` and `no_mask`. The project includes training scripts, a CLI inference tool, and a Streamlit web application for easy demonstration and deployment.

## Dataset
The project expects a dataset in standard YOLO format:
- **Classes**: `mask`, `no_mask`
- **Structure**:
  ```
  datasets/face-mask-yolo/
  ├── images/
  │   ├── train/
  │   ├── val/
  │   └── test/
  └── labels/
      ├── train/
      ├── val/
      └── test/
  ```
See `scripts/setup_dataset.md` for details on preparing the data.

## Methodology
- **Model**: YOLOv8n (Nano) for minimal size (~6MB) and fast inference.
- **Training**: 640x640 resolution, 50 epochs, batch size 16.
- **Artifacts**: Best weights are saved as `artifacts/best.pt` (PyTorch) and `artifacts/best.onnx` (ONNX) for portability.

## Quick Start
1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd <repo-name>
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the model** (ensure dataset is in `datasets/face-mask-yolo/`):
    ```bash
    python train.py
    ```

4.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

## Model Artifacts
- **`artifacts/best.pt`**: ~6.2 MB (Primary for Streamlit)
- **`artifacts/best.onnx`**: ~12 MB (Portable)
*Note: These are small enough to be tracked by standard Git without LFS.*

## Deploy to Streamlit Cloud
1.  Push this repository to GitHub.
2.  Log in to [Streamlit Cloud](https://streamlit.io/cloud).
3.  Click **New App**.
4.  Select your repository, branch (`main`), and main file (`app.py`).
5.  Click **Deploy**.
For detailed steps, see `scripts/deploy_streamlit_cloud.md`.

## Troubleshooting
- **CUDA/GPU**: If you have a GPU, ensure you have the correct PyTorch version installed: [pytorch.org](https://pytorch.org/).
- **Requirements**: If `ultralytics` conflicts, try a fresh virtual environment.
- **Paths**: Ensure your `datasets/` folder is in the project root if running locally.

## License
MIT
