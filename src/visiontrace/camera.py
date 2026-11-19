# 카메라 연결과 영상 수집을 담당하고 감지 결과를 저장 서비스에 전달합니다.
from __future__ import annotations

import random
import sys
import threading
import time
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from .detector import ChangeDetector
from .events import EventService


# 실제 카메라 없이 기능을 확인할 수 있는 합성 작업대입니다.
class DemoWorkspace:
    """Synthetic workbench selected explicitly with CAMERA_SOURCE=demo."""

    # 데모 영상 크기와 처음 표시할 물체를 준비합니다.
    def __init__(self, width: int = 960, height: int = 540) -> None:
        self.width = width
        self.height = height
        self._parts = [
            {"kind": "gear", "x": 220, "y": 230, "size": 56, "color": (72, 102, 126)},
            {"kind": "bearing", "x": 475, "y": 195, "size": 46, "color": (118, 132, 143)},
            {"kind": "bolt", "x": 690, "y": 310, "size": 42, "color": (81, 96, 112)},
        ]
        self._next_id = 0

    # 데모 물체를 그려 카메라 프레임과 같은 배열을 만듭니다.
    def render(self) -> np.ndarray:
        frame = np.full((self.height, self.width, 3), (49, 58, 66), dtype=np.uint8)
        cv2.rectangle(frame, (35, 55), (self.width - 35, self.height - 35), (62, 76, 86), -1)
        for x in range(55, self.width - 35, 45):
            cv2.line(frame, (x, 55), (x, self.height - 35), (66, 80, 90), 1)
        for y in range(75, self.height - 35, 45):
            cv2.line(frame, (35, y), (self.width - 35, y), (66, 80, 90), 1)

        for index, part in enumerate(self._parts):
            x, y, size, color = part["x"], part["y"], part["size"], part["color"]
            if part["kind"] == "gear":
                cv2.circle(frame, (x, y), size, color, -1)
                cv2.circle(frame, (x, y), size // 3, (48, 58, 66), -1)
                for angle in range(0, 360, 45):
                    radians = np.deg2rad(angle)
                    px = int(x + np.cos(radians) * (size + 8))
                    py = int(y + np.sin(radians) * (size + 8))
                    cv2.rectangle(frame, (px - 8, py - 8), (px + 8, py + 8), color, -1)
            elif part["kind"] == "bearing":
                cv2.circle(frame, (x, y), size, color, -1)
                cv2.circle(frame, (x, y), size // 2, (48, 58, 66), -1)
            else:
                cv2.rectangle(frame, (x - size, y - 13), (x + size, y + 13), color, -1)
                cv2.circle(frame, (x - size, y), 19, color, -1)
                cv2.line(frame, (x + size - 22, y - 13), (x + size - 22, y + 13), (180, 188, 192), 2)
            cv2.putText(
                frame,
                f"P{index + 1}",
                (x - 13, y + size + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (190, 202, 210),
                1,
                cv2.LINE_AA,
            )
        return frame

    # 데모 물체의 추가·이동·제거 동작을 수행합니다.
    def change(self, action: str = "random") -> str:
        available = ["add", "move", "remove"] if self._parts else ["add"]
        if action == "random" or action not in available:
            action = random.choice(available)
        if action == "add":
            kinds = ["gear", "bearing", "bolt"]
            self._parts.append(
                {
                    "kind": random.choice(kinds),
                    "x": random.randint(130, self.width - 130),
                    "y": random.randint(130, self.height - 120),
                    "size": random.randint(35, 55),
                    "color": random.choice([(70, 116, 150), (118, 132, 143), (85, 112, 126)]),
                }
            )
        elif action == "move":
            part = random.choice(self._parts)
            part["x"] = random.randint(130, self.width - 130)
            part["y"] = random.randint(130, self.height - 120)
        elif action == "remove":
            self._parts.pop(random.randrange(len(self._parts)))
        return action


# 카메라 상태와 감시 작업을 관리하는 객체입니다.
class CameraManager:
    # 공유 상태를 초기화하고 영상 수집 스레드를 시작합니다.
    def __init__(
        self,
        detector: ChangeDetector,
        event_service: EventService,
        source: str = "0",
    ) -> None:
        self.detector = detector
        self.event_service = event_service
        self.requested_source = source
        self.source_mode = "starting"
        self.camera_connected = False
        self.monitoring = False
        self.last_error = ""
        self.last_event_at: str | None = None
        self.last_event_pending = False
        self._raw_frame: np.ndarray | None = None
        self._jpeg: bytes | None = None
        self._last_boxes: list[tuple[int, int, int, int]] = []
        self._last_boxes_until = 0.0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._demo = DemoWorkspace()
        self._pending_source: str | None = None
        self._first_frame: np.ndarray | None = None
        self.backend = ""
        self._capture: cv2.VideoCapture | None = None
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
        self._thread.start()

    # 수집 스레드에 종료를 요청하고 잠시 기다립니다.
    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        # VideoCapture는 수집 스레드의 finally에서만 해제합니다.

    # 카메라와 영역을 확인하고 최초 기준 사진 저장 후 감시를 시작합니다.
    def start_monitoring(self, roi: dict[str, int] | None = None) -> dict[str, Any]:
        with self._lock:
            if self.monitoring:
                raise RuntimeError("이미 검사 중입니다. 새 기준을 저장하려면 먼저 중지해 주세요.")
            if not self.camera_connected or self._raw_frame is None:
                raise RuntimeError("카메라 프레임을 아직 받을 수 없습니다.")
            selected = roi if roi is not None else self.detector.settings.roi
            if not isinstance(selected, dict):
                raise RuntimeError("화면에서 감시할 구역을 먼저 드래그해 주세요.")
            try:
                x, y, width, height = (int(selected[k]) for k in ('x', 'y', 'width', 'height'))
            except (KeyError, TypeError, ValueError):
                raise RuntimeError("감시 구역 좌표가 올바르지 않습니다.")
            frame_height, frame_width = self._raw_frame.shape[:2]
            if x < 0 or y < 0 or width < 10 or height < 10 or x + width > frame_width or y + height > frame_height:
                raise RuntimeError("현재 영상 안에서 감시 구역을 다시 드래그해 주세요.")
            frame = self._raw_frame.copy()
            # 파일 저장 실패 시 감시를 시작하지 않습니다. 최초 사진과 비교 기준은 동일합니다.
            # 최초 기준 사진과 감지기의 기준 프레임을 같은 영상으로 맞춥니다.
            saved = self.event_service.create_event({
                "occurred_at": datetime.now(), "before": frame, "after": frame,
                "boxes": [], "change_ratio": 0.0,
                "change_type": "baseline_saved", "change_summary": "검사 시작 · 최초 기준 저장",
            })
            self.detector.configure({"roi": {"x": x, "y": y, "width": width, "height": height}})
            self.detector.start(frame)
            self.monitoring = True
            self.last_event_at = saved["occurred_at"]
            self.last_event_pending = bool(saved["pending"])
            self.last_error = ""
            return saved

    # 변화 감시만 중지하고 실시간 영상은 유지합니다.
    def stop_monitoring(self) -> None:
        with self._lock:
            self.detector.stop()
            self.monitoring = False

    # 감시가 중지된 상태에서 감지 설정을 변경합니다.
    def update_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self.monitoring:
                raise RuntimeError("감시를 중지한 후 설정을 변경해 주세요.")
            self.detector.configure(payload)
            return self.detector.settings_dict()

    # MFC에 전송할 최신 JPEG 데이터를 반환합니다.
    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    # 카메라 연결, 감시 상태, 감지 설정 등을 사전으로 반환합니다.
    def get_status(self) -> dict[str, Any]:
        with self._lock:
            height, width = (0, 0) if self._raw_frame is None else self._raw_frame.shape[:2]
            return {
                "camera_control_version": 1,
                "requested_source": self._pending_source if self._pending_source is not None else str(self.requested_source),
                "backend": self.backend,
                "camera_connected": self.camera_connected,
                "source_mode": self.source_mode,
                "monitoring": self.monitoring,
                "detector_phase": self.detector.phase,
                "last_error": self.last_error,
                "last_event_at": self.last_event_at,
                "last_event_pending": self.last_event_pending,
                "frame_width": width,
                "frame_height": height,
                "metrics": self.detector.last_metrics,
                "settings": self.detector.settings_dict(),
            }

    # 데모 모드일 때만 작업대 물체 변경을 허용합니다.
    def demo_change(self, action: str) -> str:
        with self._lock:
            if self.source_mode != "demo":
                raise RuntimeError("데모 카메라에서만 사용할 수 있는 기능입니다.")
            return self._demo.change(action)

    # 실제 장치 변경은 수집 스레드가 처리하도록 요청을 남깁니다.
    def change_source(self, source: str) -> None:
        """연결 요청만 전달하고, 카메라 해제/열기는 수집 스레드에서 처리합니다."""
        source = str(source).strip()
        if source != "demo" and (not source.isascii() or not source.isdecimal() or not 0 <= int(source) <= 9):
            raise ValueError("카메라 번호는 0~9 사이로 입력해 주세요.")
        with self._lock:
            self._pending_source = source
            self._clear_camera_state("카메라 연결을 변경하는 중입니다.")
            self.source_mode = "connecting"

    # 연결 오류나 번호 변경 시 이전 프레임과 감시 상태를 비웁니다.
    def _clear_camera_state(self, error: str) -> None:
        with self._lock:
            self.camera_connected = False
            self.monitoring = False
            self.detector.stop()
            self._raw_frame = None
            self._jpeg = None
            self._first_frame = None
            self._last_boxes = []
            self._last_boxes_until = 0.0
            self.last_error = error
            self.backend = ""

    # 사용하던 OpenCV 카메라 장치를 해제합니다.
    def _release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._first_frame = None

    # 선택한 카메라를 열고 실제 첫 프레임을 받아 연결을 확인합니다.
    def _open_source(self) -> None:
        self._release_capture()
        if str(self.requested_source).lower() == "demo":
            with self._lock:
                self.source_mode = "demo"
                self.camera_connected = self._pending_source is None
                self.last_error = ""
            return
        with self._lock:
            self.source_mode = "connecting"
            self.camera_connected = False
        try:
            source: int | str = int(self.requested_source)
        except ValueError:
            source = self.requested_source
        # Windows USB 웹캠은 DirectShow 우선, 실패하면 Media Foundation 시도.
        backends = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "Media Foundation")] if sys.platform == "win32" and isinstance(source, int) else [(cv2.CAP_ANY, "Auto")]
        for backend_id, backend_name in backends:
            if self._stop.is_set():
                return
            capture = None
            try:
                capture = cv2.VideoCapture(source, backend_id)
                if not capture.isOpened():
                    continue
                # 장치 기본 설정으로 먼저 확인합니다. 특정 해상도/MJPG를 강제하지 않습니다.
                for _ in range(10):
                    if self._stop.is_set():
                        return
                    success, frame = capture.read()
                    if success and frame is not None and frame.size:
                        with self._lock:
                            if self._pending_source is not None:
                                return
                            self._capture = capture
                            self._first_frame = frame
                            self.source_mode = "webcam"
                            self.camera_connected = True
                            self.backend = backend_name
                            self.last_error = ""
                        capture = None  # 소유권을 수집 스레드에 넘깁니다.
                        return
                    if self._stop.wait(0.05):
                        return
            except cv2.error:
                pass
            finally:
                if capture is not None:
                    capture.release()
        self._clear_camera_state(
            "카메라 연결 실패: USB 연결, 카메라 번호, Windows 카메라 권한을 확인하고 "
            "카메라를 사용하는 다른 프로그램을 종료해 주세요. 2초 후 재시도합니다."
        )
        self.source_mode = "disconnected"

    # 실제 카메라 또는 데모에서 다음 영상 프레임을 가져옵니다.
    def _next_frame(self) -> np.ndarray:
        if self.source_mode == "demo":
            return self._demo.render()
        with self._lock:
            if self._first_frame is not None:
                frame, self._first_frame = self._first_frame, None
                return frame
        try:
            success, frame = self._capture.read() if self._capture is not None else (False, None)
            if not success or frame is None or not frame.size:
                raise RuntimeError("USB 카메라 영상 수신이 끊겼습니다. 연결을 다시 시도합니다.")
            return frame
        except (cv2.error, RuntimeError) as exc:
            self._clear_camera_state(str(exc))
            self.source_mode = "disconnected"
            self._release_capture()
            raise RuntimeError(str(exc)) from exc

    # 영상을 계속 수집하면서 정해진 간격으로 감지하고 이벤트를 저장합니다.
    def _run(self) -> None:
        retry_at = 0.0
        next_detection_at = 0.0
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                with self._lock:
                    changed = self._pending_source is not None
                    if changed:
                        self.requested_source = self._pending_source
                        self._pending_source = None
                        self._clear_camera_state("")
                        self.source_mode = "connecting"
                if changed:
                    self._release_capture()
                    retry_at = 0.0
                if self._capture is None and self.source_mode != "demo":
                    if started < retry_at:
                        self._stop.wait(0.1)
                        continue
                    self._open_source()
                    retry_at = time.monotonic() + 2.0
                    if not self.camera_connected:
                        self._stop.wait(0.1)
                        continue
                detection = None
                try:
                    frame = self._next_frame()
                    with self._lock:
                        if self._pending_source is not None:
                            continue  # 이전 카메라 프레임을 새 연결 화면에 표시하지 않습니다.
                        self._raw_frame = frame
                        if self.monitoring and time.monotonic() >= next_detection_at:
                            # 감지 빈도는 기존 10Hz를 유지해 안정화 설정의 의미를 보존합니다.
                            next_detection_at = time.monotonic() + 0.1
                            # 감지기가 이벤트를 반환한 경우에만 아래에서 사진 저장을 수행합니다.
                            detection = self.detector.process(frame)
                            if detection:
                                self._last_boxes = detection["boxes"]
                                self._last_boxes_until = time.monotonic() + 3.0
                        display = self._annotate(frame.copy())
                        success, encoded = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 82])
                        if success:
                            self._jpeg = encoded.tobytes()
                except Exception as exc:
                    with self._lock:
                        self.last_error = str(exc)
                    retry_at = time.monotonic() + 2.0
                    self._stop.wait(0.1)
                if detection:
                    try:
                        # 사진과 DB 저장을 요청하고 마지막 기록 상태를 갱신합니다.
                        saved = self.event_service.create_event(detection)
                        with self._lock:
                            self.last_event_at = saved["occurred_at"]
                            self.last_event_pending = bool(saved["pending"])
                    except Exception as exc:
                        with self._lock:
                            self.last_error = str(exc)
                elapsed = time.monotonic() - started
                self._stop.wait(max(0.001, 1.0 / 30.0 - elapsed))
        finally:
            self._release_capture()

    # MFC에 보낼 영상에 감시 영역, 상태, 변화 사각형을 표시합니다.
    def _annotate(self, frame: np.ndarray) -> np.ndarray:
        x, y, width, height = self.detector.roi_for_frame(frame)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (238, 175, 64), 2)
        cv2.putText(
            frame,
            "MONITORING" if self.monitoring else "READY",
            (18, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (75, 220, 134) if self.monitoring else (205, 210, 214),
            2,
            cv2.LINE_AA,
        )
        if time.monotonic() < self._last_boxes_until:
            for box_x, box_y, box_width, box_height in self._last_boxes:
                cv2.rectangle(
                    frame,
                    (box_x, box_y),
                    (box_x + box_width, box_y + box_height),
                    (36, 90, 245),
                    3,
                )
        return frame

