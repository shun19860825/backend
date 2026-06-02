import math
from typing import List, Optional, Tuple

from services.pose_estimator import Landmark

_LM = {
    "NOSE": 0,
    "LEFT_EAR": 7, "RIGHT_EAR": 8,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
}

_DEPTH_RATIO = {
    "female": {
        "bust": 0.78, "under_bust": 0.74, "waist": 0.68,
        "hip": 0.92, "thigh": 0.82, "upper_arm": 0.92,
        "knee": 0.82, "neck": 0.88, "wrist": 0.85,
    },
    "male": {
        "bust": 0.84, "under_bust": 0.80, "waist": 0.78,
        "hip": 0.82, "thigh": 0.80, "upper_arm": 0.92,
        "knee": 0.82, "neck": 0.88, "wrist": 0.85,
    },
    "unknown": {
        "bust": 0.80, "under_bust": 0.76, "waist": 0.72,
        "hip": 0.88, "thigh": 0.81, "upper_arm": 0.92,
        "knee": 0.82, "neck": 0.88, "wrist": 0.85,
    },
}


def _mid(a: Landmark, b: Landmark) -> Landmark:
    return Landmark(
        x=(a.x + b.x) / 2, y=(a.y + b.y) / 2,
        z=(a.z + b.z) / 2, visibility=min(a.visibility, b.visibility),
    )


def _h_dist(a: Landmark, b: Landmark, img_w: int) -> float:
    return abs(a.x - b.x) * img_w


def _dist(a: Landmark, b: Landmark, img_w: int, img_h: int) -> float:
    dx = (a.x - b.x) * img_w
    dy = (a.y - b.y) * img_h
    return math.sqrt(dx * dx + dy * dy)


def _ellipse_circ(width: float, depth: float) -> float:
    a = width / 2
    b = depth / 2
    if a < b:
        a, b = b, a
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))


def _circ(width_cm: float, depth_ratio: float) -> float:
    return _ellipse_circ(width_cm, width_cm * depth_ratio)


def _scale_cm_per_px(lm: List[Landmark], img_h: int, height_cm: float,
                     scan_mode: str = "full") -> float:
    """
    scan_mode に応じてスケール校正方法を切り替える。
    full  : 鼻〜足首の距離（最も安定）
    upper : 鼻〜肩中点の距離（上半身のみ）
    lower : 股関節〜足首の距離（下半身のみ）
    """
    if scan_mode == "upper":
        nose = lm[_LM["NOSE"]]
        mid_sh = _mid(lm[_LM["LEFT_SHOULDER"]], lm[_LM["RIGHT_SHOULDER"]])
        nose_to_shoulder_px = max(abs(mid_sh.y - nose.y) * img_h, 1.0)
        # 鼻〜肩 ≈ 身長の 9%
        return (height_cm * 0.09) / nose_to_shoulder_px
    elif scan_mode == "lower":
        mid_hip = _mid(lm[_LM["LEFT_HIP"]], lm[_LM["RIGHT_HIP"]])
        mid_ankle = _mid(lm[_LM["LEFT_ANKLE"]], lm[_LM["RIGHT_ANKLE"]])
        hip_to_ankle_px = max(abs(mid_ankle.y - mid_hip.y) * img_h, 1.0)
        # 股関節〜足首 ≈ 身長の 48%
        return (height_cm * 0.48) / hip_to_ankle_px
    else:
        nose = lm[_LM["NOSE"]]
        ankle_y = (_mid(lm[_LM["LEFT_ANKLE"]], lm[_LM["RIGHT_ANKLE"]])).y
        nose_to_ankle_px = max((ankle_y - nose.y) * img_h, 1.0)
        body_height_px = nose_to_ankle_px / 0.89
        return height_cm / body_height_px


class MeasurementCalculator:

    def _depth_from_diagonal(
        self,
        diag_landmarks: List[Landmark],
        diag_shape: Tuple[int, int, int],
        front_shoulder_w_cm: float,
        height_cm: float,
    ) -> Optional[float]:
        """
        45度斜め写真から体の奥行き(cm)を推定する。
        45度回転時: projected_width = (W + D) / sqrt(2)
        よって: D ≈ projected_width * sqrt(2) - W
        """
        import math
        lm = diag_landmarks
        img_h, img_w = diag_shape[:2]
        sc = _scale_cm_per_px(lm, img_h, height_cm)
        l_sh = lm[_LM["LEFT_SHOULDER"]]
        r_sh = lm[_LM["RIGHT_SHOULDER"]]
        diag_shoulder_w_cm = _h_dist(l_sh, r_sh, img_w) * sc
        # 45度投影: projected = (front_width + depth) / sqrt(2)
        estimated_depth = diag_shoulder_w_cm * math.sqrt(2) - front_shoulder_w_cm
        # 物理的にありえない値をクリップ
        return max(min(estimated_depth, front_shoulder_w_cm * 1.2), front_shoulder_w_cm * 0.4)

    def calculate(
        self,
        main_landmarks: List[Landmark],
        main_shape: Tuple[int, int, int],
        scan_mode: str = "full",
        height_cm: float = 160,
        weight_kg: Optional[float] = None,
        gender: str = "unknown",
        foot_size_cm: Optional[float] = None,
        diagonal_landmarks: Optional[List[Landmark]] = None,
        diagonal_shape: Optional[Tuple[int, int, int]] = None,
        side_landmarks: Optional[List[Landmark]] = None,
        side_shape: Optional[Tuple[int, int, int]] = None,
    ) -> List[dict]:

        lm = main_landmarks
        img_h, img_w = main_shape[:2]
        sc = _scale_cm_per_px(lm, img_h, height_cm, scan_mode)
        dr = _DEPTH_RATIO.get(gender, _DEPTH_RATIO["unknown"])
        has_side = side_landmarks is not None
        has_diag = diagonal_landmarks is not None

        # 斜め写真から実測の体深度を計算し、depth_ratioを補正
        if has_diag and diagonal_shape is not None:
            shoulder_w_cm_temp = _h_dist(
                lm[_LM["LEFT_SHOULDER"]], lm[_LM["RIGHT_SHOULDER"]], img_w
            ) * _scale_cm_per_px(lm, img_h, height_cm)
            measured_depth = self._depth_from_diagonal(
                diagonal_landmarks, diagonal_shape, shoulder_w_cm_temp, height_cm
            )
            if measured_depth:
                actual_ratio = measured_depth / shoulder_w_cm_temp
                # 実測値でdepth_ratioを全体補正（±20%以内に制限）
                for key in dr:
                    base = dr[key]
                    dr = dict(dr)
                    dr[key] = max(base * 0.8, min(base * 1.2, actual_ratio * (base / 0.80)))

        def m(key, name, value, unit, confidence, note=None):
            return {
                "key": key, "name_ja": name,
                "value": round(value, 1) if value is not None else None,
                "unit": unit, "confidence": confidence, "note": note,
            }

        l_sh = lm[_LM["LEFT_SHOULDER"]]
        r_sh = lm[_LM["RIGHT_SHOULDER"]]
        l_hip = lm[_LM["LEFT_HIP"]]
        r_hip = lm[_LM["RIGHT_HIP"]]
        l_ankle = lm[_LM["LEFT_ANKLE"]]
        r_ankle = lm[_LM["RIGHT_ANKLE"]]
        l_elbow = lm[_LM["LEFT_ELBOW"]]
        r_elbow = lm[_LM["RIGHT_ELBOW"]]
        l_wrist = lm[_LM["LEFT_WRIST"]]
        r_wrist = lm[_LM["RIGHT_WRIST"]]
        l_ear = lm[_LM["LEFT_EAR"]]
        r_ear = lm[_LM["RIGHT_EAR"]]

        mid_sh = _mid(l_sh, r_sh)
        mid_hip = _mid(l_hip, r_hip)
        mid_ankle = _mid(l_ankle, r_ankle)

        shoulder_w_cm = _h_dist(l_sh, r_sh, img_w) * sc
        hip_joint_w_cm = _h_dist(l_hip, r_hip, img_w) * sc
        hip_actual_w_cm = hip_joint_w_cm * 1.15
        waist_y = mid_hip.y - (mid_hip.y - mid_sh.y) * 0.35
        waist_w_cm = min(shoulder_w_cm, hip_actual_w_cm) * 0.78
        neck_base_y = mid_sh.y - (2.0 / height_cm)
        crotch_y = mid_hip.y + (3.0 / height_cm)

        # ── 上半身項目 ────────────────────────────────────────────
        upper_items = []
        upper_items.append(m("height", "身長", height_cm, "cm", "manual", "ユーザー入力値"))
        upper_items.append(m("shoulder_width", "肩幅", shoulder_w_cm, "cm", "high"))
        upper_items.append(m("back_width", "背巾", shoulder_w_cm * 0.78, "cm", "medium"))

        bust_w = shoulder_w_cm * 0.92
        upper_items.append(m(
            "bust", "バスト", _circ(bust_w, dr["bust"]), "cm",
            "medium" if (has_side or has_diag) else "low",
            "推定値" + ("" if (has_side or has_diag) else " ※斜め/横向き写真で精度向上"),
        ))
        under_w = bust_w * 0.88
        upper_items.append(m("under_bust", "アンダーバスト", _circ(under_w, dr["under_bust"]), "cm", "low", "バスト値から推定"))
        upper_items.append(m("waist", "ウエスト", _circ(waist_w_cm, dr["waist"]), "cm", "low", "推定値（誤差±5cm程度）"))
        upper_items.append(m("armhole", "アームホール", shoulder_w_cm * 0.28 * math.pi, "cm", "low"))

        back_len_px = abs(waist_y - neck_base_y) * img_h
        upper_items.append(m("back_length", "背丈", back_len_px * sc, "cm", "medium"))

        left_sleeve = _dist(l_sh, l_elbow, img_w, img_h) + _dist(l_elbow, l_wrist, img_w, img_h)
        right_sleeve = _dist(r_sh, r_elbow, img_w, img_h) + _dist(r_elbow, r_wrist, img_w, img_h)
        upper_items.append(m("sleeve_length", "袖丈", ((left_sleeve + right_sleeve) / 2) * sc, "cm", "medium", "腕を自然に下ろした状態"))
        upper_items.append(m("upper_arm", "二の腕周り", _circ(shoulder_w_cm * 0.21, dr["upper_arm"]), "cm", "low"))
        upper_items.append(m("wrist", "手首周り", _circ(shoulder_w_cm * 0.095, dr["wrist"]), "cm", "low"))

        neck_w = _h_dist(l_ear, r_ear, img_w) * sc * 0.88
        upper_items.append(m("neck", "首回り", _circ(neck_w, dr["neck"]), "cm", "low"))

        # ── 下半身項目 ────────────────────────────────────────────
        lower_items = []
        lower_items.append(m("hip", "ヒップ", _circ(hip_actual_w_cm, dr["hip"]), "cm", "medium" if has_side else "low"))
        thigh_w = hip_actual_w_cm * 0.62
        lower_items.append(m("thigh", "太もも", _circ(thigh_w, dr["thigh"]), "cm", "low"))
        lower_items.append(m("knee", "ひざ周り", _circ(hip_joint_w_cm * 0.32, dr["knee"]), "cm", "low"))
        inseam_px = abs(mid_ankle.y - crotch_y) * img_h
        lower_items.append(m("inseam", "股下", inseam_px * sc, "cm", "medium"))
        skirt_px = abs(mid_ankle.y - waist_y) * img_h
        lower_items.append(m("skirt_length", "スカート丈", skirt_px * sc, "cm", "medium"))
        total_px = abs(mid_ankle.y - neck_base_y) * img_h
        lower_items.append(m("total_length", "総丈", total_px * sc, "cm", "medium"))
        lower_items.append(m("crotch_depth", "PP", height_cm * 0.35, "cm", "low", "推定値"))

        # ── その他 ────────────────────────────────────────────────
        misc_items = [
            m("weight",    "体重",       weight_kg,    "kg", "manual", None if weight_kg    else "入力してください"),
            m("foot_size", "足のサイズ", foot_size_cm, "cm", "manual", None if foot_size_cm else "入力してください"),
        ]

        # scan_mode に応じて返す項目を選択
        if scan_mode == "upper":
            return upper_items + misc_items
        elif scan_mode == "lower":
            return lower_items + misc_items
        else:
            return upper_items + lower_items + misc_items
