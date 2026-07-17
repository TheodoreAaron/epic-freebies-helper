# -*- coding: utf-8 -*-
from contextlib import suppress
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from hcaptcha_challenger.agent.challenger import RoboticArm
from loguru import logger
from playwright.async_api import TimeoutError as PlaywrightTimeoutError


def _longest_contiguous_run(values: np.ndarray) -> list[int]:
    runs: list[list[int]] = []
    for value in values.tolist():
        value = int(value)
        if not runs or value != runs[-1][-1] + 1:
            runs.append([value])
        else:
            runs[-1].append(value)
    return max(runs, key=len, default=[])


def _detect_task_canvas_origin(challenge_screenshot: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(challenge_screenshot))
    if image is None:
        return None

    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    colored_pixels = (hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 20)

    # The prompt header occupies the top of the challenge. The task canvas is the longest
    # colored run below it, regardless of whether hCaptcha renders the 320px or 330px layout.
    row_counts = colored_pixels.sum(axis=1)
    row_indexes = np.flatnonzero(
        (row_counts > width * 0.35) & (np.arange(height) >= int(height * 0.23))
    )
    task_rows = _longest_contiguous_run(row_indexes)
    if len(task_rows) < height * 0.45:
        return None

    task_mask = colored_pixels[task_rows[0] : task_rows[-1] + 1]
    column_indexes = np.flatnonzero(task_mask.sum(axis=0) > len(task_rows) * 0.05)
    if not len(column_indexes):
        return None

    return int(column_indexes.min()), task_rows[0]


def _entity_centers(captcha_payload: Any, crumb_id: int) -> list[tuple[int, int]]:
    tasklist = getattr(captcha_payload, "tasklist", None) or []
    if crumb_id < 0 or crumb_id >= len(tasklist):
        return []

    centers: list[tuple[int, int]] = []
    for entity in getattr(tasklist[crumb_id], "entities", None) or []:
        coords = getattr(entity, "coords", None) or []
        if len(coords) < 2:
            return []
        centers.append((int(coords[0]), int(coords[1])))
    return centers


def _correct_drag_source_points(
    paths: list[Any],
    *,
    captcha_payload: Any,
    crumb_id: int,
    challenge_screenshot: Path,
    challenge_bbox: dict[str, float] | None,
) -> list[Any]:
    centers = _entity_centers(captcha_payload, crumb_id)
    if not paths or len(centers) != len(paths) or not challenge_bbox:
        return paths

    canvas_origin = _detect_task_canvas_origin(challenge_screenshot)
    if canvas_origin is None:
        logger.warning("Could not locate hCaptcha drag canvas; keeping model source coordinates")
        return paths

    image = cv2.imread(str(challenge_screenshot))
    if image is None:
        return paths

    image_height, image_width = image.shape[:2]
    scale_x = float(challenge_bbox["width"]) / image_width
    scale_y = float(challenge_bbox["height"]) / image_height
    origin_x, origin_y = canvas_origin
    resolved_sources = [
        (
            int(round(float(challenge_bbox["x"]) + (origin_x + x) * scale_x)),
            int(round(float(challenge_bbox["y"]) + (origin_y + y) * scale_y)),
        )
        for x, y in centers
    ]

    path_order = sorted(range(len(paths)), key=lambda index: paths[index].start_point.y)
    source_order = sorted(resolved_sources, key=lambda point: point[1])
    for path_index, source in zip(path_order, source_order):
        path = paths[path_index]
        previous = (path.start_point.x, path.start_point.y)
        path.start_point.x, path.start_point.y = source
        logger.info("Corrected hCaptcha drag source from model={} to payload={}", previous, source)

    return paths


def apply_hcaptcha_drag_patch() -> None:
    if getattr(RoboticArm.challenge_image_drag_drop, "_epic_drag_source_patch", False):
        return

    async def patched_challenge_image_drag_drop(self: RoboticArm, job_type: Any):
        frame_challenge = await self.get_challenge_frame_locator()
        crumb_count = await self.check_crumb_count()
        cache_key = self.config.create_cache_key(self.captcha_payload)

        for cid in range(crumb_count):
            await self.page.wait_for_timeout(self.config.WAIT_FOR_CHALLENGE_VIEW_TO_RENDER_MS)
            raw, projection = await self._capture_spatial_mapping(frame_challenge, cache_key, cid)
            challenge_bbox = await frame_challenge.locator(
                "//div[@class='challenge-view']"
            ).bounding_box()
            user_prompt = self._match_user_prompt(job_type)

            response = await self._spatial_path_reasoner(
                challenge_screenshot=raw,
                grid_divisions=projection,
                auxiliary_information=user_prompt,
            )
            logger.debug(f'[{cid+1}/{crumb_count}]ToolInvokeMessage: {response.log_message}')
            self._spatial_path_reasoner.cache_response(
                path=cache_key.joinpath(f"{cache_key.name}_{cid}_model_answer.json")
            )

            paths = _correct_drag_source_points(
                response.paths,
                captcha_payload=self.captcha_payload,
                crumb_id=cid,
                challenge_screenshot=raw,
                challenge_bbox=challenge_bbox,
            )
            for path in paths:
                await self._perform_drag_drop(path)

            with suppress(PlaywrightTimeoutError):
                submit_btn = frame_challenge.locator("//div[@class='button-submit button']")
                await self.click_by_mouse(submit_btn)

    patched_challenge_image_drag_drop._epic_drag_source_patch = True
    RoboticArm.challenge_image_drag_drop = patched_challenge_image_drag_drop
    logger.info("hCaptcha drag source-coordinate patch loaded")
