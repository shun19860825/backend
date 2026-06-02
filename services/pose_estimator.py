import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float


class PoseEstimator:
    def __init__(self):
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )

    def estimate(
        self, image_bytes: bytes
    ) -> Tuple[Optional[List[Landmark]], Optional[Tuple[int, int, int]]]:
        arr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None, None

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)

        if not result.pose_landmarks:
            return None, bgr.shape

        landmarks = [
            Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
            for lm in result.pose_landmarks.landmark
        ]
        return landmarks, bgr.shape

    def __del__(self):
        if hasattr(self, "_pose"):
            self._pose.close()
