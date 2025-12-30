# Deploy to Streamlit Cloud

Streamlit Cloud allows you to host your app for free directly from GitHub.

## 1. Prepare GitHub Repo
1.  **Commit your code**:
    ```bash
    git add .
    git commit -m "Initial commit for Face Mask App"
    ```
2.  **Push to GitHub**:
    Ensure your repo is public (or private if you have a Streamlit account connected).
3.  **Verify Artifacts**:
    Make sure `artifacts/best.pt` is in your repo. We specifically allowed it in `.gitignore` because it is small enough (<10MB).

## 2. Deploy
1.  Go to [share.streamlit.io](https://share.streamlit.io/).
2.  Click **New App**.
3.  **Repository**: Select your `face-mask-detection` repo.
4.  **Branch**: `main` (or `master`).
5.  **Main file path**: `app.py`.
6.  Click **Deploy!**

## 3. Configuration (Optional)
On the deployment screen, if you see "Manage app":
- You can add **Secrets** if you need (not required for this simple app).
- You can check **Logs** to see installation progress.

## 4. Performance Tips
- Since we use YOLOv8n (nano), it should fit within the resource limits (1GB RAM usually).
- If you hit memory limits, try reducing image resolution in `app.py` or using `best.onnx` with `onnxruntime` (which is even lighter) though `ultralytics` package is quite optimized.
