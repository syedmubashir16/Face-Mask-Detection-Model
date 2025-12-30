# Run Locally

## Prerequisites
- Python 3.8+
- Virtual Environment (recommended)

## 1. Setup Environment
```bash
# Create venv
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## 2. Train the Model
This will download `yolov8n.pt`, fine-tune it on your data, and save the best model to `artifacts/best.pt`.
```bash
python train.py --epochs 50 --batch 16
```
*Note: If you have a GPU, it will be used automatically. If not, it will run on CPU (slower).*

## 3. Run the App
Launch the Streamlit interface:
```bash
streamlit run app.py
```
Open the URL provided in the terminal (usually `http://localhost:8501`).

## 4. Run CLI Inference
To run on a single image without the UI:
```bash
python infer.py --source path/to/your/image.jpg --conf 0.5
```
Check `outputs/` for the result.
