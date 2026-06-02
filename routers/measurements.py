import time
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from services.measurement_calculator import MeasurementCalculator
from services.pose_estimator import PoseEstimator

router = APIRouter()
_estimator = PoseEstimator()
_calculator = MeasurementCalculator()


@router.post("/measurements/estimate")
async def estimate_measurements(
    front_image:    Optional[UploadFile] = File(None),
    upper_image:    Optional[UploadFile] = File(None),
    lower_image:    Optional[UploadFile] = File(None),
    diagonal_image: Optional[UploadFile] = File(None),
    side_image:     Optional[UploadFile] = File(None),
    scan_mode:   str   = Form("full"),
    height_cm:   float = Form(..., gt=50, lt=250),
    weight_kg:   Optional[float] = Form(None),
    gender:      Optional[str]   = Form("unknown"),
    foot_size_cm: Optional[float] = Form(None),
):
    t0 = time.time()

    async def read_and_estimate(file: Optional[UploadFile]):
        if file is None:
            return None, None
        data = await file.read()
        return _estimator.estimate(data)

    front_lm,   front_shape   = await read_and_estimate(front_image)
    upper_lm,   upper_shape   = await read_and_estimate(upper_image)
    lower_lm,   lower_shape   = await read_and_estimate(lower_image)
    diag_lm,    diag_shape    = await read_and_estimate(diagonal_image)
    side_lm,    side_shape    = await read_and_estimate(side_image)

    # モードに応じてメイン画像を決定
    if scan_mode == "upper":
        main_lm, main_shape = upper_lm, upper_shape
    elif scan_mode == "lower":
        main_lm, main_shape = lower_lm, lower_shape
    else:
        main_lm, main_shape = front_lm, front_shape

    if main_lm is None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "POSE_NOT_DETECTED",
                "message": "人物のポーズを検出できませんでした。全身が映っている写真を使用し、背景がシンプルな場所で撮影してください。",
            },
        )

    measurements = _calculator.calculate(
        main_landmarks=main_lm,
        main_shape=main_shape,
        scan_mode=scan_mode,
        height_cm=height_cm,
        weight_kg=weight_kg,
        gender=gender or "unknown",
        foot_size_cm=foot_size_cm,
        diagonal_landmarks=diag_lm,
        diagonal_shape=diag_shape,
        side_landmarks=side_lm,
        side_shape=side_shape,
    )

    return {
        "measurements": measurements,
        "has_side_view": side_lm is not None,
        "processing_time_ms": int((time.time() - t0) * 1000),
    }
