# hand_gesture_recognition

結合立體相機（雙鏡頭）深度資訊的手勢辨識研究/開發專案，涵蓋相機校正、視差（深度）計算與 CNN 分類模型訓練。為 [`hand-gesture-recognition`](https://github.com/oilotaku/hand-gesture-recognition) 專案的開發／實驗版本。

A research and development repository for stereo-camera (dual-lens) depth-based hand gesture recognition, covering camera calibration, disparity/depth computation, and CNN classifier training. This is the development/experimental counterpart of the [`hand-gesture-recognition`](https://github.com/oilotaku/hand-gesture-recognition) project.

## 功能 / Features

- **雙鏡頭立體校正**：`aa.py`、`camera_L.py`、`camera_R.py` 進行左右相機的棋盤格校正，取得內外部參數。
  Stereo calibration (`aa.py`, `camera_L.py`, `camera_R.py`) using chessboard patterns to compute left/right camera intrinsics and extrinsics.
- **視差/深度計算**：`bb.py` 計算左右影像視差圖並換算實際距離，並支援匯出至 Excel（`openpyxl`）。
  Disparity/depth computation (`bb.py`) that converts stereo disparity into real-world distance and can export results to Excel via `openpyxl`.
- **CNN 手勢分類模型**：`cnn.py` 以 TensorFlow/Keras 讀取 RGB + 深度影像訓練分類模型。
  CNN gesture classifier (`cnn.py`) built with TensorFlow/Keras, trained on combined RGB + depth image data.
- **即時辨識主程式**：`main.py` 整合雙鏡頭讀取、視差計算與已訓練模型（`.h5`/`keras_metadata.pb`）進行即時手勢辨識。
  Real-time recognition pipeline (`main.py`) that combines stereo capture, disparity computation, and the trained model for live gesture inference.

## 技術 / Tech Stack

Python、OpenCV (`cv2`)、TensorFlow/Keras、NumPy、`openpyxl`、Matplotlib

## 檔案結構 / File Overview

| 檔案 / File | 說明 / Description |
| --- | --- |
| `main.py` | 即時辨識主程式 Real-time recognition entry point |
| `aa.py` | 立體相機校正 Stereo camera calibration |
| `bb.py` | 視差/距離計算與滑鼠互動量測 Disparity/distance computation with mouse interaction |
| `cnn.py` | CNN 模型訓練 CNN model training |
| `camera_L.py`, `camera_R.py` | 左右相機參數/擷取 Left/right camera parameter handling and capture |
| `cmpL.npz`, `cmpR.npz` | 已校正的相機參數資料 Saved calibration parameter data |
| `keras_metadata.pb` | 已訓練 Keras 模型的中繼資料 Metadata of the trained Keras model |

## 使用方式 / Usage

```bash
pip install opencv-python numpy tensorflow openpyxl matplotlib
python main.py
```

> 需搭配雙鏡頭（立體相機）硬體與對應的已校正參數檔（`cmpL.npz` / `cmpR.npz`）方可正常運作。
> Requires stereo (dual-lens) camera hardware and the corresponding calibration parameter files (`cmpL.npz` / `cmpR.npz`) to run correctly.
