"""
wall_detector.py
Step 1: OpenCV detects wall segments with pixel-accurate positions
Step 2: Claude classifies each segment (load bearing / non-structural / stacking)
Step 3: Render using OpenCV coordinates — no guessing
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import json
import re


def find_plan_bounds(img: Image.Image) -> tuple[int,int,int,int]:
    """Find the main plan drawing area, skipping page borders."""
    img_np = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    w, h = img.size
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = sorted([(cv2.contourArea(c), c) for c in contours], key=lambda x: x[0], reverse=True)
    for area, cnt in areas[2:8]:
        if area < w * h * 0.05:
            break
        x, y, cw, ch = cv2.boundingRect(cnt)
        if 0.3 < (cw / ch if ch > 0 else 0) < 4.0:
            pad = 10
            return (max(0,x-pad), max(0,y-pad), min(w,x+cw+pad), min(h,y+ch+pad))
    # fallback
    mx, my = int(w*0.05), int(h*0.05)
    return mx, my, w-mx, int(h*0.85)


def detect_wall_segments(img: Image.Image) -> tuple[list, tuple]:
    """
    Detect wall line segments using morphological operations.
    Returns (wall_list, plan_box) where wall_list contains dicts with pixel coords
    and normalized position within the plan area.
    """
    plan_box = find_plan_bounds(img)
    px1, py1, px2, py2 = plan_box
    plan_w = px2 - px1
    plan_h = py2 - py1

    img_np = np.array(img.convert("RGB"))
    plan_gray = cv2.cvtColor(img_np[py1:py2, px1:px2], cv2.COLOR_RGB2GRAY)
    _, plan_bin = cv2.threshold(plan_gray, 110, 255, cv2.THRESH_BINARY_INV)

    min_h_len = max(30, plan_w // 18)
    min_v_len = max(30, plan_h // 18)

    # Horizontal walls
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_h_len, 1))
    h_det = cv2.morphologyEx(plan_bin, cv2.MORPH_OPEN, h_kernel)
    h_merged = cv2.dilate(h_det, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 22)))

    # Vertical walls
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v_len))
    v_det = cv2.morphologyEx(plan_bin, cv2.MORPH_OPEN, v_kernel)
    v_merged = cv2.dilate(v_det, cv2.getStructuringElement(cv2.MORPH_RECT, (22, 1)))

    def extract(mask, orient):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            if orient == "H":
                if cw < min_h_len or ch > plan_h * 0.08:
                    continue
            else:
                if ch < min_v_len or cw > plan_w * 0.08:
                    continue
            boxes.append((x, y, x+cw, y+ch))
        return boxes

    h_boxes = extract(h_merged, "H")
    v_boxes = extract(v_merged, "V")

    walls = []
    for i, (x1,y1,x2,y2) in enumerate(h_boxes):
        cx, cy = (x1+x2)//2, (y1+y2)//2
        walls.append({
            "id": f"H{i+1:02d}",
            "orient": "horizontal",
            "length_pct": round((x2-x1)/plan_w*100, 1),
            "x_pct": round(cx/plan_w*100, 1),
            "y_pct": round(cy/plan_h*100, 1),
            "at_edge": cy < plan_h*0.12 or cy > plan_h*0.88,
            "span_pct": f"{round(x1/plan_w*100,1)}-{round(x2/plan_w*100,1)}%",
            # Pixel coords in FULL image space
            "px": [px1+x1, py1+y1, px1+x2, py1+y2],
        })
    for i, (x1,y1,x2,y2) in enumerate(v_boxes):
        cx, cy = (x1+x2)//2, (y1+y2)//2
        walls.append({
            "id": f"V{i+1:02d}",
            "orient": "vertical",
            "length_pct": round((y2-y1)/plan_h*100, 1),
            "x_pct": round(cx/plan_w*100, 1),
            "y_pct": round(cy/plan_h*100, 1),
            "at_edge": cx < plan_w*0.12 or cx > plan_w*0.88,
            "span_pct": f"{round(y1/plan_h*100,1)}-{round(y2/plan_h*100,1)}%",
            "px": [px1+x1, py1+y1, px1+x2, py1+y2],
        })

    return walls, plan_box


def classify_walls(walls: list, img: Image.Image, plan_box: tuple,
                   params: dict, api_key: str, floor_label: str = "1st floor") -> list:
    """
    Send wall segment list + small plan image to Claude for classification.
    Returns walls with loadBearing, stacking, and confidence fields added.
    Fast — Claude only needs to classify a list, not generate coordinates.
    """
    import anthropic

    # Send a small version of the plan image for visual context
    px1, py1, px2, py2 = plan_box
    plan_crop = img.crop(plan_box)
    # Resize to max 1000px for speed
    plan_crop.thumbnail((1000, 1000), Image.LANCZOS)
    buf = io.BytesIO()
    plan_crop.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    # Build compact wall list (no pixel coords needed by Claude)
    wall_summary = []
    for w in walls:
        wall_summary.append({
            "id": w["id"],
            "orient": w["orient"],
            "length_pct": w["length_pct"],
            "x_pct": w["x_pct"],
            "y_pct": w["y_pct"],
            "at_edge": w["at_edge"],
            "span": w["span_pct"],
        })

    prompt = f"""You are a structural engineer reviewing a floor plan.

I have used computer vision to detect {len(walls)} wall segments in this {floor_label} plan.
Each wall has an ID, orientation, and position as % of the plan area (0=top-left, 100=bottom-right).

Your job: classify each wall as load-bearing or non-structural.

Rules:
- Exterior walls (at_edge=true, or spanning full width/height) are almost always load-bearing
- Long walls (length_pct > 40%) spanning most of the plan are likely load-bearing
- Short interior partitions are often non-structural
- Walls aligned with joist direction labels in the drawing may be non-structural (parallel to joists)
- Walls perpendicular to joist spans are load-bearing

Building: {params.get('building_type','residential')}, {params.get('stories','?')} stories,
Material: {params.get('wall_material','wood frame')}, Seismic: {params.get('seismic','')}.

Detected walls:
{json.dumps(wall_summary, indent=2)}

Return ONLY valid JSON — no markdown, no explanation:
{{
  "classifications": [
    {{"id": "H01", "loadBearing": true, "reason": "North exterior wall, full width"}},
    {{"id": "V03", "loadBearing": false, "reason": "Short interior partition"}}
  ]
}}

Classify ALL {len(walls)} walls. Be concise in reasons (max 8 words each)."""

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": prompt}
        ]}]
    )

    raw = "".join(b.text for b in response.content if hasattr(b, "text"))
    clean = raw.replace("```json","").replace("```","").strip()
    match = re.search(r'\{[\s\S]*\}', clean)
    if match:
        clean = match.group(0)
    result = json.loads(clean)

    # Merge classifications back into wall list
    classifications = {c["id"]: c for c in result.get("classifications", [])}
    for w in walls:
        cls = classifications.get(w["id"], {})
        w["loadBearing"] = cls.get("loadBearing", w["at_edge"])
        w["reason"] = cls.get("reason", "")

    return walls


def find_stacking_walls(floor_walls_list: list) -> list:
    """
    Cross-reference walls across floors to find stacking pairs.
    floor_walls_list: list of wall lists, one per floor (ordered ground→top)
    Returns same structure with 'stacksWith' field added where applicable.
    """
    if len(floor_walls_list) < 2:
        return floor_walls_list

    for fi, walls_a in enumerate(floor_walls_list):
        for fj, walls_b in enumerate(floor_walls_list):
            if fi >= fj:
                continue
            for wa in walls_a:
                if not wa.get("loadBearing"):
                    continue
                for wb in walls_b:
                    if not wb.get("loadBearing"):
                        continue
                    # Same orientation and close position
                    if wa["orient"] != wb["orient"]:
                        continue
                    if wa["orient"] == "horizontal":
                        same_pos = abs(wa["y_pct"] - wb["y_pct"]) < 8
                        overlap = not (float(wa["span_pct"].split("-")[1].rstrip("%")) < float(wb["span_pct"].split("-")[0].rstrip("%")) or
                                      float(wb["span_pct"].split("-")[1].rstrip("%")) < float(wa["span_pct"].split("-")[0].rstrip("%")))
                    else:
                        same_pos = abs(wa["x_pct"] - wb["x_pct"]) < 8
                        overlap = not (float(wa["span_pct"].split("-")[1].rstrip("%")) < float(wb["span_pct"].split("-")[0].rstrip("%")) or
                                      float(wb["span_pct"].split("-")[1].rstrip("%")) < float(wa["span_pct"].split("-")[0].rstrip("%")))

                    if same_pos and overlap:
                        wa["stacksWith"] = wb["id"]
                        wb["stacksWith"] = wa["id"]

    return floor_walls_list


def render_wall_overlay(img: Image.Image, walls: list, floor_label: str = "") -> Image.Image:
    """
    Render color-coded wall highlights using pixel-accurate OpenCV coordinates.
    """
    result = img.convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    iw, ih = img.size

    try:
        font_sz = max(13, iw // 110)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_sz)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(10, font_sz-3))
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    lw = max(4, iw // 180)  # line width scales with image

    for w in walls:
        x1, y1, x2, y2 = w["px"]
        is_lb = w.get("loadBearing", False)
        is_stack = "stacksWith" in w

        if is_stack:
            fill   = (100, 0, 200, 100)
            border = (100, 0, 200, 240)
        elif is_lb:
            fill   = (210, 40, 40, 100)
            border = (180, 20, 20, 240)
        else:
            fill   = (130, 130, 130, 55)
            border = (100, 100, 100, 160)

        # Ensure minimum thickness
        if abs(x2-x1) < lw*2:
            cx = (x1+x2)//2; x1,x2 = cx-lw, cx+lw
        if abs(y2-y1) < lw*2:
            cy = (y1+y2)//2; y1,y2 = cy-lw, cy+lw

        draw.rectangle([x1, y1, x2, y2], fill=fill, outline=border, width=lw)

        # Label — small, placed above wall
        label = w["id"]
        if is_stack:
            label += "↕"
        lx = x1 + 3
        ly = max(2, y1 - font_sz - 2)
        draw.text((lx+1, ly+1), label, fill=(0,0,0,180), font=small_font)
        draw.text((lx, ly), label, fill=(255,255,255,240), font=small_font)

    return Image.alpha_composite(result, overlay).convert("RGB")


def add_legend(img: Image.Image, floor_label: str = "") -> Image.Image:
    legend_h = 56
    out = Image.new("RGB", (img.width, img.height + legend_h), (248, 248, 248))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    items = [
        ((210,40,40),  "Load bearing"),
        ((100,0,200),  "Stacking ↕ (multi-floor)"),
        ((130,130,130),"Non-structural"),
    ]
    x, yb, yt = 16, img.height+14, img.height+17
    for color, label in items:
        draw.rectangle([x, yb, x+18, yb+18], fill=color, outline=(60,60,60))
        draw.text((x+24, yt), label, fill=(30,30,30), font=font)
        x += 200

    note = f"Wall detection: OpenCV morphological analysis{' — ' + floor_label if floor_label else ''}"
    draw.text((16, img.height+38), note, fill=(100,60,0), font=font)
    return out
