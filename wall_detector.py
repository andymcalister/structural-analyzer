"""
wall_detector.py — OpenCV-based wall detection and overlay rendering.
Claude classifies walls; this module finds their actual pixel positions.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from typing import Optional


def detect_walls(img: Image.Image) -> dict:
    """
    Detect horizontal and vertical wall segments using morphological operations.
    Returns dict with 'horizontal' and 'vertical' lists of (x1,y1,x2,y2) bboxes.
    """
    img_np = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    w, h = img.size

    # Threshold: find dark lines
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    # Minimum wall length = 3% of image dimension
    min_h_len = max(40, int(w * 0.03))
    min_v_len = max(40, int(h * 0.03))

    # Horizontal walls: long horizontal runs
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_h_len, 1))
    h_walls_bin = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    h_dilated = cv2.dilate(h_walls_bin, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 18)))

    # Vertical walls: long vertical runs
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v_len))
    v_walls_bin = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    v_dilated = cv2.dilate(v_walls_bin, cv2.getStructuringElement(cv2.MORPH_RECT, (18, 1)))

    def extract_bboxes(mask, min_w, min_h, max_w, max_h):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if min_w <= cw <= max_w and min_h <= ch <= max_h:
                boxes.append((x, y, x + cw, y + ch))
        return boxes

    img_w, img_h = img.size
    h_boxes = extract_bboxes(h_dilated,
                              min_w=min_h_len, min_h=1,
                              max_w=img_w, max_h=int(img_h * 0.08))
    v_boxes = extract_bboxes(v_dilated,
                              min_w=1, min_h=min_v_len,
                              max_w=int(img_w * 0.08), max_h=img_h)

    return {"horizontal": h_boxes, "vertical": v_boxes}


def find_drawing_bounds(img: Image.Image) -> tuple[int, int, int, int]:
    """
    Find the bounding box of the main drawing area (largest rectangle).
    Returns (x1, y1, x2, y2) in image coordinates.
    """
    img_np = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    w, h = img.size

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    min_area = w * h * 0.08

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / ch if ch > 0 else 0
        if 0.2 < aspect < 5.0 and area > best_area:
            best = (x, y, x + cw, y + ch)
            best_area = area

    if best:
        pad = 15
        x1 = max(0, best[0] - pad)
        y1 = max(0, best[1] - pad)
        x2 = min(w, best[2] + pad)
        y2 = min(h, best[3] + pad)
        return x1, y1, x2, y2

    # Fallback: generous margins
    mx, my = int(w * 0.04), int(w * 0.04)
    return mx, my, w - mx, int(h * 0.78)


def match_wall_to_segment(wall: dict, h_boxes: list, v_boxes: list,
                           img_w: int, img_h: int,
                           draw_x1: int, draw_y1: int,
                           draw_x2: int, draw_y2: int) -> Optional[tuple]:
    """
    Match a Claude-described wall to the best detected OpenCV segment.
    Uses the wall's bbox hint as a search region, then finds the best
    detected segment within that region.
    """
    bbox = wall.get("bbox", {})
    if not bbox:
        return None

    # Claude's coords are relative to the cropped drawing area
    draw_w = draw_x2 - draw_x1
    draw_h = draw_y2 - draw_y1

    # Convert normalized bbox to full-image pixel coords
    bx1 = draw_x1 + bbox.get("x1", 0) * draw_w
    by1 = draw_y1 + bbox.get("y1", 0) * draw_h
    bx2 = draw_x1 + bbox.get("x2", 1) * draw_w
    by2 = draw_y1 + bbox.get("y2", 1) * draw_h

    # Expand search region generously to account for estimation error
    margin_x = max(60, (bx2 - bx1) * 0.5)
    margin_y = max(60, (by2 - by1) * 0.5)
    sx1 = max(0, bx1 - margin_x)
    sy1 = max(0, by1 - margin_y)
    sx2 = min(img_w, bx2 + margin_x)
    sy2 = min(img_h, by2 + margin_y)

    # Determine orientation from bbox aspect ratio
    bbox_w = bx2 - bx1
    bbox_h = by2 - by1
    prefer_horizontal = bbox_w >= bbox_h

    candidates = h_boxes if prefer_horizontal else v_boxes
    fallback = v_boxes if prefer_horizontal else h_boxes

    def overlap_score(box):
        """Score how well a detected box overlaps with the search region."""
        ox1, oy1, ox2, oy2 = box
        inter_x = max(0, min(ox2, sx2) - max(ox1, sx1))
        inter_y = max(0, min(oy2, sy2) - max(oy1, sy1))
        if inter_x == 0 or inter_y == 0:
            return 0
        inter_area = inter_x * inter_y
        box_area = (ox2 - ox1) * (oy2 - oy1)
        return inter_area / box_area if box_area > 0 else 0

    best_box = None
    best_score = 0
    for box in candidates + fallback:
        score = overlap_score(box)
        if score > best_score:
            best_score = score
            best_box = box

    # Only use if there's meaningful overlap
    if best_box and best_score > 0.05:
        return best_box

    # Final fallback: use Claude's bbox directly
    return (int(bx1), int(by1), int(bx2), int(by2))


def find_stacking_pairs(floor_results: list) -> dict:
    """
    Cross-reference walls across floors to find stacking pairs.
    floor_results: list of (label, result_dict) per floor
    Returns: {floor_idx: {wall_id: stacking_wall_id_on_other_floor}}
    """
    stacking = {i: {} for i in range(len(floor_results))}
    if len(floor_results) < 2:
        return stacking

    for i, (label_a, result_a) in enumerate(floor_results):
        for j, (label_b, result_b) in enumerate(floor_results):
            if i >= j:
                continue
            walls_a = result_a.get("walls", [])
            walls_b = result_b.get("walls", [])

            for wa in walls_a:
                if not wa.get("loadBearing"):
                    continue
                bbox_a = wa.get("bbox", {})
                if not bbox_a:
                    continue
                cx_a = (bbox_a.get("x1", 0) + bbox_a.get("x2", 0)) / 2
                cy_a = (bbox_a.get("y1", 0) + bbox_a.get("y2", 0)) / 2

                for wb in walls_b:
                    if not wb.get("loadBearing"):
                        continue
                    bbox_b = wb.get("bbox", {})
                    if not bbox_b:
                        continue
                    cx_b = (bbox_b.get("x1", 0) + bbox_b.get("x2", 0)) / 2
                    cy_b = (bbox_b.get("y1", 0) + bbox_b.get("y2", 0)) / 2

                    # Check if centerlines are close (within 10% of drawing)
                    if abs(cx_a - cx_b) < 0.10 and abs(cy_a - cy_b) < 0.10:
                        stacking[i][wa["id"]] = wb["id"]
                        stacking[j][wb["id"]] = wa["id"]

    return stacking


def render_overlay(full_img: Image.Image, cropped_img: Image.Image,
                   crop_box: tuple, walls: list, openings: list,
                   stacking_map: dict) -> Image.Image:
    """
    Render color-coded wall overlays using OpenCV-detected segments.
    stacking_map: {wall_id: partner_wall_id} for walls that stack across floors
    """
    draw_x1, draw_y1, draw_x2, draw_y2 = crop_box
    img_w, img_h = full_img.size

    # Detect actual wall segments in the cropped drawing
    detected = detect_walls(cropped_img)

    # Offset detected segments to full-image coords
    h_boxes_full = [(x1 + draw_x1, y1 + draw_y1, x2 + draw_x1, y2 + draw_y1)
                    for x1, y1, x2, y2 in detected["horizontal"]]
    v_boxes_full = [(x1 + draw_x1, y1 + draw_y1, x2 + draw_x1, y2 + draw_y1)
                    for x1, y1, x2, y2 in detected["vertical"]]

    result = full_img.convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font_size = max(14, img_w // 120)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    for wall in walls:
        wall_id = wall.get("id", "")
        is_load_bearing = wall.get("loadBearing", False)
        is_stacking = wall_id in stacking_map

        if is_stacking:
            fill = (138, 43, 226, 130)
            border = (100, 0, 200, 230)
        elif is_load_bearing:
            fill = (220, 50, 50, 120)
            border = (180, 20, 20, 230)
        else:
            fill = (120, 120, 120, 70)
            border = (80, 80, 80, 160)

        # Match to detected segment
        matched = match_wall_to_segment(
            wall, h_boxes_full, v_boxes_full,
            img_w, img_h, draw_x1, draw_y1, draw_x2, draw_y2
        )

        if matched:
            x1, y1, x2, y2 = matched
            # Ensure minimum thickness
            if abs(x2 - x1) < 8:
                cx = (x1 + x2) // 2
                x1, x2 = cx - 4, cx + 4
            if abs(y2 - y1) < 8:
                cy = (y1 + y2) // 2
                y1, y2 = cy - 4, cy + 4

            draw.rectangle([x1, y1, x2, y2], fill=fill, outline=border, width=2)

            # Label
            label = wall_id
            if is_stacking:
                partner = stacking_map[wall_id]
                label += f"↕{partner}"
            # White shadow then label
            draw.text((x1 + 4, y1 + 2), label, fill=(0, 0, 0, 160), font=font)
            draw.text((x1 + 3, y1 + 1), label, fill=(255, 255, 255, 240), font=font)

    # Openings
    for opening in openings:
        bbox = opening.get("bbox", {})
        if not bbox:
            continue
        draw_w = draw_x2 - draw_x1
        draw_h = draw_y2 - draw_y1
        ox1 = int(draw_x1 + bbox.get("x1", 0) * draw_w)
        oy1 = int(draw_y1 + bbox.get("y1", 0) * draw_h)
        ox2 = int(draw_x1 + bbox.get("x2", 1) * draw_w)
        oy2 = int(draw_y1 + bbox.get("y2", 1) * draw_h)
        if abs(ox2 - ox1) < 6:
            cx = (ox1 + ox2) // 2
            ox1, ox2 = cx - 3, cx + 3
        if abs(oy2 - oy1) < 6:
            cy = (oy1 + oy2) // 2
            oy1, oy2 = cy - 3, cy + 3
        draw.rectangle([ox1, oy1, ox2, oy2], fill=(255, 165, 0, 110), outline=(200, 120, 0, 210), width=2)

    composited = Image.alpha_composite(result, overlay).convert("RGB")
    return composited


def add_legend(img: Image.Image) -> Image.Image:
    legend_h = 64
    new_img = Image.new("RGB", (img.width, img.height + legend_h), (245, 245, 245))
    new_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    items = [
        ((220, 50, 50), "Load bearing"),
        ((138, 43, 226), "Stacking (multi-floor) ↕"),
        ((120, 120, 120), "Non-structural"),
        ((255, 165, 0), "Opening / header"),
    ]
    x = 16
    y_box = img.height + 14
    y_text = img.height + 17
    for color, label in items:
        draw.rectangle([x, y_box, x + 20, y_box + 20], fill=color, outline=(50, 50, 50))
        draw.text((x + 26, y_text), label, fill=(40, 40, 40), font=font)
        x += 210

    draw.text((16, img.height + 44),
              "⚠ Wall highlights use OpenCV line detection — positions are approximate, for preliminary review only",
              fill=(110, 70, 0), font=font)
    return new_img
