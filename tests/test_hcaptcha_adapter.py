from types import SimpleNamespace

import cv2
import numpy as np
from hcaptcha_challenger.models import PointCoordinate, SpatialPath

from extensions.hcaptcha_adapter import _correct_drag_source_points, _detect_task_canvas_origin


def _write_challenge_screenshot(path, *, canvas_y: int, canvas_height: int):
    image = np.full((470, 500, 3), 245, dtype=np.uint8)
    image[:108] = (143, 131, 0)
    image[canvas_y : canvas_y + canvas_height, 10:490] = (80, 120, 160)
    assert cv2.imwrite(str(path), image)


def test_drag_canvas_origin_supports_multi_shape_layout(tmp_path):
    screenshot = tmp_path / "challenge.png"
    _write_challenge_screenshot(screenshot, canvas_y=130, canvas_height=330)

    assert _detect_task_canvas_origin(screenshot) == (10, 130)


def test_payload_entity_centers_replace_invalid_model_sources(tmp_path):
    screenshot = tmp_path / "challenge.png"
    _write_challenge_screenshot(screenshot, canvas_y=130, canvas_height=330)
    payload = SimpleNamespace(
        tasklist=[
            SimpleNamespace(
                entities=[SimpleNamespace(coords=[416, 55]), SimpleNamespace(coords=[406, 219])]
            )
        ]
    )
    paths = [
        SpatialPath(
            start_point=PointCoordinate(x=819, y=323), end_point=PointCoordinate(x=533, y=323)
        ),
        SpatialPath(
            start_point=PointCoordinate(x=819, y=623), end_point=PointCoordinate(x=461, y=422)
        ),
    ]

    corrected = _correct_drag_source_points(
        paths,
        captcha_payload=payload,
        crumb_id=0,
        challenge_screenshot=screenshot,
        challenge_bbox={"x": 390, "y": 100, "width": 500, "height": 470},
    )

    assert [(path.start_point.x, path.start_point.y) for path in corrected] == [
        (816, 285),
        (806, 449),
    ]
    assert [(path.end_point.x, path.end_point.y) for path in corrected] == [(533, 323), (461, 422)]


def test_source_correction_requires_one_entity_per_model_path(tmp_path):
    screenshot = tmp_path / "challenge.png"
    _write_challenge_screenshot(screenshot, canvas_y=135, canvas_height=320)
    payload = SimpleNamespace(
        tasklist=[SimpleNamespace(entities=[SimpleNamespace(coords=[414, 60])])]
    )
    paths = [
        SpatialPath(
            start_point=PointCoordinate(x=800, y=300), end_point=PointCoordinate(x=500, y=300)
        ),
        SpatialPath(
            start_point=PointCoordinate(x=800, y=450), end_point=PointCoordinate(x=500, y=450)
        ),
    ]

    corrected = _correct_drag_source_points(
        paths,
        captcha_payload=payload,
        crumb_id=0,
        challenge_screenshot=screenshot,
        challenge_bbox={"x": 390, "y": 100, "width": 500, "height": 470},
    )

    assert [(path.start_point.x, path.start_point.y) for path in corrected] == [
        (800, 300),
        (800, 450),
    ]
