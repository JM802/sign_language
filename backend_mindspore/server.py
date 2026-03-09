from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
ICT_ROOT = WORKSPACE_ROOT / "ict"

if str(ICT_ROOT) not in sys.path:
    sys.path.insert(0, str(ICT_ROOT))

from src.inference import RealTimeSignRecognizer, translate_sign  # type: ignore  # noqa: E402


class HealthResponse(BaseModel):
    status: str
    model_ready: bool


class InferenceResponse(BaseModel):
    text: str
    raw_text: str
    confidence: float
    confidence_text: str
    frames: int


class FileSignLanguageService:
    def __init__(self) -> None:
        self.recognizer = RealTimeSignRecognizer()

    def predict_video(self, file_path: str) -> dict[str, Any]:
        ext = Path(file_path).suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp"}:
            raise ValueError("此模型需要时序动作，请上传视频文件而不是图片。")

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError("无法读取上传的视频文件。")

        frame_buffer: list[np.ndarray] = []
        self.recognizer.smoother.prev_landmarks = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.recognizer.holistic.process(rgb)
                raw_feat = self.recognizer.extract_features(results)
                smooth_feat = self.recognizer.smoother.smooth(raw_feat)

                if smooth_feat is not None:
                    frame_buffer.append(smooth_feat)
        finally:
            cap.release()

        if not frame_buffer:
            raise ValueError("未能提取到有效骨架数据，请确保视频中包含清晰的人体和手部动作。")

        inp = self.recognizer.preprocess(frame_buffer)
        if inp is None:
            raise ValueError("视频过短或动作不完整，无法识别。")

        with torch.no_grad():
            out = self.recognizer.model(inp)
            probs = torch.softmax(out, dim=1)
            conf, idx = torch.max(probs, dim=1)

        idx_val = idx.item()
        conf_val = float(conf.item())
        raw_text = self.recognizer.idx2name.get(idx_val, f"unknown_{idx_val}")
        text = translate_sign(raw_text)

        return {
            "text": text,
            "raw_text": raw_text,
            "confidence": conf_val,
            "confidence_text": f"{conf_val:.2%}",
            "frames": len(frame_buffer),
        }


service: FileSignLanguageService | None = None


def get_service() -> FileSignLanguageService:
    global service
    if service is None:
        service = FileSignLanguageService()
    return service


app = FastAPI(title="Sign Language Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        get_service()
        return HealthResponse(status="ok", model_ready=True)
    except Exception:
        return HealthResponse(status="error", model_ready=False)


@app.post("/api/inference", response_model=InferenceResponse)
async def infer_video(video: UploadFile = File(...)) -> InferenceResponse:
    suffix = Path(video.filename or "upload.webm").suffix or ".webm"
    temp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            while chunk := await video.read(1024 * 1024):
                tmp.write(chunk)

        result = get_service().predict_video(temp_path)
        return InferenceResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"推理服务异常: {exc}") from exc
    finally:
        await video.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
