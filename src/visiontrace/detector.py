# 영상의 움직임과 기준 화면 대비 변화를 구분해 이벤트를 만듭니다.
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import cv2
import numpy as np


# 감도, 최소 면적, 안정 대기 횟수, 감시 영역 설정입니다.
@dataclass
class DetectorSettings:
    pixel_threshold: int = 24
    min_contour_area: int = 350
    min_change_ratio: float = 0.0025
    motion_ratio: float = 0.0012
    stable_frames: int = 12
    cooldown_seconds: float = 1.5
    roi: dict[str, int] | None = None

    # 입력한 설정을 숫자로 변환하고 허용 범위 안으로 제한합니다.
    def validated(self) -> "DetectorSettings":
        self.pixel_threshold = max(5, min(int(self.pixel_threshold), 100))
        self.min_contour_area = max(20, min(int(self.min_contour_area), 100_000))
        self.min_change_ratio = max(0.0001, min(float(self.min_change_ratio), 0.5))
        self.motion_ratio = max(0.0001, min(float(self.motion_ratio), 0.5))
        self.stable_frames = max(3, min(int(self.stable_frames), 100))
        self.cooldown_seconds = max(0.0, min(float(self.cooldown_seconds), 30.0))
        if self.roi is not None:
            self.roi = {key: int(self.roi.get(key, 0)) for key in ("x", "y", "width", "height")}
            if self.roi["width"] <= 0 or self.roi["height"] <= 0:
                self.roi = None
        return self


# 기준 사진과 직전 프레임을 보관하는 변화 감지기입니다.
class ChangeDetector:
    """Event-oriented scene change detector.

    Frame-to-frame differences determine whether a person is still moving.
    Baseline-to-current differences determine whether the final workspace state
    changed. This distinction prevents one physical action from generating many
    events.
    """

    # 감지 설정과 비교에 사용할 상태를 초기화합니다.
    def __init__(self, settings: DetectorSettings | None = None) -> None:
        self.settings = (settings or DetectorSettings()).validated()
        self.armed = False
        self.phase = "stopped"
        self._baseline_frame: np.ndarray | None = None
        self._baseline_gray: np.ndarray | None = None
        self._last_gray: np.ndarray | None = None
        self._activity_seen = False
        self._stable_count = 0
        self._last_event_monotonic = 0.0
        self.last_metrics = {"motion_ratio": 0.0, "change_ratio": 0.0}

    # 전달받은 항목만 기존 감지 설정에 반영합니다.
    def configure(self, payload: dict[str, Any]) -> None:
        merged = asdict(self.settings)
        for key in merged:
            if key in payload:
                merged[key] = payload[key]
        self.settings = DetectorSettings(**merged).validated()

    # 현재 설정을 API 응답과 DB 저장에 쓸 사전으로 변환합니다.
    def settings_dict(self) -> dict[str, Any]:
        return asdict(self.settings)

    # 현재 프레임을 기준 사진으로 등록하고 감지를 시작합니다.
    def start(self, frame: np.ndarray) -> None:
        self._baseline_frame = frame.copy()
        self._baseline_gray = self._preprocess(self._crop(frame))
        self._last_gray = self._baseline_gray.copy()
        self._activity_seen = False
        self._stable_count = 0
        self._last_event_monotonic = 0.0
        self.armed = True
        self.phase = "watching"
        self.last_metrics = {"motion_ratio": 0.0, "change_ratio": 0.0}

    # 움직임 판단 상태를 초기화하고 감지를 중지합니다.
    def stop(self) -> None:
        self.armed = False
        self.phase = "stopped"
        self._activity_seen = False
        self._stable_count = 0

    # 움직임을 확인하고 화면이 안정되면 최종 변화 이벤트를 생성합니다.
    def process(self, frame: np.ndarray) -> dict[str, Any] | None:
        if not self.armed or self._baseline_gray is None or self._baseline_frame is None:
            return None

        current_gray = self._preprocess(self._crop(frame))
        if current_gray.shape != self._baseline_gray.shape:
            self.start(frame)
            return None

        # 직전 프레임 차이는 움직임, 기준 사진과의 차이는 최종 변화를 판단합니다.
        frame_mask, frame_ratio, _ = self._compare(self._last_gray, current_gray)
        baseline_mask, change_ratio, boxes = self._compare(self._baseline_gray, current_gray)
        del frame_mask, baseline_mask
        self.last_metrics = {
            "motion_ratio": round(frame_ratio, 6),
            "change_ratio": round(change_ratio, 6),
        }

        moving = frame_ratio >= self.settings.motion_ratio
        meaningful = change_ratio >= self.settings.min_change_ratio and bool(boxes)

        # 움직이는 동안 대기 횟수를 초기화하고, 멈추면 안정 횟수를 누적합니다.
        if moving:
            self._activity_seen = True
            self._stable_count = 0
            self.phase = "changing"
        elif self._activity_seen or meaningful:
            self._stable_count += 1
            self.phase = "settling"
        else:
            self._stable_count = 0
            self.phase = "watching"

        event = None
        # 연속 안정 횟수와 이벤트 간격 조건을 만족하면 변화 이벤트를 만듭니다.
        if self._stable_count >= self.settings.stable_frames:
            elapsed = time.monotonic() - self._last_event_monotonic
            if meaningful and elapsed >= self.settings.cooldown_seconds:
                full_frame_boxes = self._offset_boxes(boxes, frame)
                change_type, change_summary = self._classify_change(
                    self._baseline_frame, frame, full_frame_boxes
                )
                event = {
                    "occurred_at": datetime.now(),
                    "before": self._baseline_frame.copy(),
                    "after": frame.copy(),
                    "boxes": full_frame_boxes,
                    "change_ratio": change_ratio,
                    "change_type": change_type,
                    "change_summary": change_summary,
                }
                # 이벤트를 만든 현재 모습을 다음 비교 기준으로 갱신합니다.
                self._baseline_frame = frame.copy()
                self._baseline_gray = current_gray.copy()
                self._last_event_monotonic = time.monotonic()
                self.phase = "event"
            self._activity_seen = False
            self._stable_count = 0

        self._last_gray = current_gray
        return event

    # 분석할 감시 영역만 영상에서 잘라냅니다.
    def _crop(self, frame: np.ndarray) -> np.ndarray:
        x, y, width, height = self.roi_for_frame(frame)
        return frame[y : y + height, x : x + width]

    # 감시 영역 좌표를 실제 영상 크기 안으로 맞춥니다.
    def roi_for_frame(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        roi = self.settings.roi
        if not roi:
            return 0, 0, width, height
        x = max(0, min(int(roi["x"]), width - 1))
        y = max(0, min(int(roi["y"]), height - 1))
        roi_width = max(1, min(int(roi["width"]), width - x))
        roi_height = max(1, min(int(roi["height"]), height - y))
        return x, y, roi_width, roi_height

    # 영상을 흑백으로 바꾸고 흐림 처리로 작은 잡음을 줄입니다.
    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (7, 7), 0)

    # 밝기 보정과 차영상 분석으로 변화 비율과 사각형 영역을 구합니다.
    def _compare(
        self, first: np.ndarray | None, second: np.ndarray
    ) -> tuple[np.ndarray, float, list[tuple[int, int, int, int]]]:
        if first is None:
            empty = np.zeros_like(second)
            return empty, 0.0, []

        # Compensate for uniform illumination changes before calculating absdiff.
        median_shift = float(np.median(second.astype(np.int16) - first.astype(np.int16)))
        compensated = np.clip(second.astype(np.float32) - median_shift, 0, 255).astype(np.uint8)
        difference = cv2.absdiff(first, compensated)
        _, mask = cv2.threshold(
            difference, self.settings.pixel_threshold, 255, cv2.THRESH_BINARY
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple[int, int, int, int]] = []
        filtered_mask = np.zeros_like(mask)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.settings.min_contour_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            boxes.append((x, y, width, height))
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

        ratio = float(cv2.countNonZero(filtered_mask)) / float(filtered_mask.size)
        return filtered_mask, ratio, boxes

    # 잘라낸 영상의 변화 좌표를 원본 영상 좌표로 되돌립니다.
    def _offset_boxes(
        self, boxes: list[tuple[int, int, int, int]], frame: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        roi_x, roi_y, _, _ = self.roi_for_frame(frame)
        return [(x + roi_x, y + roi_y, width, height) for x, y, width, height in boxes]

    # 변화 영역의 윤곽량을 비교해 추가·사라짐·이동을 분류합니다.
    @staticmethod
    def _classify_change(
        before: np.ndarray,
        after: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
    ) -> tuple[str, str]:
        """Classify a stable scene change without persisting any coordinates.

        The changed rectangles are used only as a temporary mask.  Edge content
        inside the union of those areas increases when an object appears,
        decreases when it disappears, and remains similar when it is moved.
        """
        if not boxes:
            return "workspace_changed", "작업대 상태 변경"

        height, width = before.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        for x, y, box_width, box_height in boxes:
            padding = 6
            left = max(0, x - padding)
            top = max(0, y - padding)
            right = min(width, x + box_width + padding)
            bottom = min(height, y + box_height + padding)
            cv2.rectangle(mask, (left, top), (right, bottom), 255, thickness=cv2.FILLED)

        # 변화 영역 안에서 물체 구조를 나타내는 윤곽 픽셀 수를 셉니다.
        def structure_score(frame: np.ndarray) -> int:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(gray, 35, 110)
            return int(cv2.countNonZero(cv2.bitwise_and(edges, mask)))

        before_score = structure_score(before)
        after_score = structure_score(after)
        tolerance = max(24, int(max(before_score, after_score) * 0.15))

        if after_score - before_score > tolerance:
            return "object_added", "물체 추가"
        if before_score - after_score > tolerance:
            return "object_removed", "물체 사라짐"
        return "object_moved", "물체 이동·배치 변경"

