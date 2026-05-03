"""
wall_detector.py
Step 1: Morphological detection of line runs + double-line pairing
Step 2: Claude classifies each confirmed wall segment  
Step 3: Render using pixel-accurate OpenCV coordinates
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io, base64, json, re


def find_plan_bounds(img: Image.Image) -> tuple[int,int,int,int]:
    """Find the main plan drawing area, skipping page borders."""
    img_np = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    w, h = img.size
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = sorted([(cv2.contourArea(c), c) for c in contours],
                   key=lambda x: x[0], reverse=True)
    for area, cnt in areas[2:8]:
        if area < w * h * 0.05:
            break
        x, y, cw, ch = cv2.boundingRect(cnt)
        if 0.3 < (cw / ch if ch > 0 else 0) < 4.0:
            pad = 10
            return (max(0,x-pad), max(0,y-pad), min(w,x+cw+pad), min(h,y+ch+pad))
    mx, my = int(w*0.04), int(h*0.04)
    return mx, my, w-mx, int(h*0.88)


def detect_wall_segments(img: Image.Image) -> tuple[list, tuple]:
    """
    Detect walls using morphological line detection + double-line pairing.
    Real walls appear as two parallel lines (double-line convention).
    Returns (wall_list, plan_box).
    """
    plan_box = find_plan_bounds(img)
    px1, py1, px2, py2 = plan_box
    plan_w, plan_h = px2-px1, py2-py1

    img_np = np.array(img.convert("RGB"))
    plan_gray = cv2.cvtColor(img_np[py1:py2, px1:px2], cv2.COLOR_RGB2GRAY)
    _, plan_bin = cv2.threshold(plan_gray, 110, 255, cv2.THRESH_BINARY_INV)

    min_h = int(plan_w * 0.04)
    min_v = int(plan_h * 0.04)

    # --- Detect individual line runs ---
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_h, 1))
    h_det = cv2.morphologyEx(plan_bin, cv2.MORPH_OPEN, h_kernel)
    h_cnts, _ = cv2.findContours(h_det, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_lines = []  # (y_center, x1, x2, height)
    for c in h_cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w >= min_h and h <= 18:
            h_lines.append((y + h//2, x, x+w, h))

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v))
    v_det = cv2.morphologyEx(plan_bin, cv2.MORPH_OPEN, v_kernel)
    v_cnts, _ = cv2.findContours(v_det, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_lines = []  # (x_center, y1, y2, width)
    for c in v_cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h >= min_v and w <= 18:
            v_lines.append((x + w//2, y, y+h, w))

    # --- Pair parallel lines into walls ---
    def pair_parallel(lines, orient, gap_min=3, gap_max=28, overlap_thresh=0.50):
        """
        Find pairs of parallel lines that form double-line walls.
        lines: list of (pos, span_start, span_end, thickness)
        pos = perpendicular coordinate (y for H, x for V)
        """
        lines_sorted = sorted(enumerate(lines), key=lambda x: x[1][0])
        used = set()
        walls = []
        for i, (orig_i, line_a) in enumerate(lines_sorted):
            if orig_i in used:
                continue
            pos_a, s_a, e_a, _ = line_a
            best, best_gap = None, gap_max + 1
            for orig_j, line_b in lines_sorted[i+1:]:
                if orig_j in used:
                    continue
                pos_b, s_b, e_b, _ = line_b
                gap = abs(pos_b - pos_a)
                if gap > gap_max:
                    break
                if gap < gap_min:
                    continue
                overlap = min(e_a, e_b) - max(s_a, s_b)
                span = max(e_a, e_b) - min(s_a, s_b)
                if span > 0 and overlap / span >= overlap_thresh and gap < best_gap:
                    best_gap = gap
                    best = (orig_j, line_b)
            if best:
                orig_j, line_b = best
                used.add(orig_i)
                used.add(orig_j)
                pos_b, s_b, e_b, _ = line_b
                walls.append((
                    min(s_a, s_b), max(e_a, e_b),
                    min(pos_a, pos_b), max(pos_a, pos_b)
                ))
        return walls  # (span_start, span_end, pos_min, pos_max)

    h_walls_raw = pair_parallel(h_lines, "H")
    v_walls_raw = pair_parallel(v_lines, "V")

    # --- Cluster filter: remove stair/hatch clusters ---
    def filter_clusters(walls, orient, plan_main, plan_cross,
                        nearby_thresh=4, len_thresh=0.15, region=0.08):
        result = []
        for i, (s, e, p1, p2) in enumerate(walls):
            length = e - s
            pos = (p1 + p2) // 2
            mid = (s + e) // 2
            nearby = 0
            for j, (os, oe, op1, op2) in enumerate(walls):
                if i == j:
                    continue
                opos = (op1 + op2) // 2
                omid = (os + oe) // 2
                if (abs(pos - opos) / plan_cross < region and
                        abs(mid - omid) / plan_main < 0.12):
                    nearby += 1
            if nearby >= nearby_thresh and length / plan_main < len_thresh:
                continue
            result.append((s, e, p1, p2))
        return result

    h_walls_raw = filter_clusters(h_walls_raw, "H", plan_w, plan_h)
    v_walls_raw = filter_clusters(v_walls_raw, "V", plan_h, plan_w)

    # --- Convert to wall dicts with full image pixel coords ---
    walls = []
    for i, (x1, x2, y1, y2) in enumerate(h_walls_raw):
        cx, cy = (x1+x2)//2, (y1+y2)//2
        walls.append({
            "id": f"H{i+1:02d}",
            "orient": "horizontal",
            "length_pct": float(round((x2-x1)/plan_w*100, 1)),
            "x_pct": float(round(cx/plan_w*100, 1)),
            "y_pct": float(round(cy/plan_h*100, 1)),
            "at_edge": bool(cy < plan_h*0.14 or cy > plan_h*0.86),
            "span_pct": f"{round(x1/plan_w*100,1)}-{round(x2/plan_w*100,1)}",
            "px": [int(px1+x1), int(py1+y1), int(px1+x2), int(py1+y2)],
        })
    for i, (y1, y2, x1, x2) in enumerate(v_walls_raw):
        cx, cy = (x1+x2)//2, (y1+y2)//2
        walls.append({
            "id": f"V{i+1:02d}",
            "orient": "vertical",
            "length_pct": float(round((y2-y1)/plan_h*100, 1)),
            "x_pct": float(round(cx/plan_w*100, 1)),
            "y_pct": float(round(cy/plan_h*100, 1)),
            "at_edge": bool(cx < plan_w*0.14 or cx > plan_w*0.86),
            "span_pct": f"{round(y1/plan_h*100,1)}-{round(y2/plan_h*100,1)}",
            "px": [int(px1+x1), int(py1+y1), int(px1+x2), int(py1+y2)],
        })

    return walls, plan_box


def classify_walls(walls: list, img: Image.Image, plan_box: tuple,
                   params: dict, api_key: str, floor_label: str = "") -> list:
    """Send wall list + cropped plan image to Claude for classification."""
    import anthropic

    plan_crop = img.crop(plan_box)
    plan_crop.thumbnail((900, 900), Image.LANCZOS)
    buf = io.BytesIO()
    plan_crop.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    wall_summary = [{
        "id": str(w["id"]),
        "orient": str(w["orient"]),
        "length_pct": float(w["length_pct"]),
        "x_pct": float(w["x_pct"]),
        "y_pct": float(w["y_pct"]),
        "at_edge": bool(w["at_edge"]),
        "span": str(w["span_pct"]),
    } for w in walls]

    prompt = f"""You are a licensed structural engineer reviewing a floor plan.

Computer vision has detected {len(walls)} wall segments. Each has:
- id, orient (horizontal/vertical)
- position as % of plan area (0=top-left corner, 100=bottom-right)
- at_edge: true if near the plan boundary (likely exterior wall)
- span: the range it covers as % of plan width/height

Classify each as load-bearing or non-structural:
1. Exterior walls (at_edge=true) → almost always load-bearing
2. Long walls spanning >40% of plan → likely load-bearing
3. Walls perpendicular to joist direction (visible in plan) → load-bearing
4. Short interior partitions <15% span → likely non-structural
5. Walls at mid-span supporting beams → load-bearing

Building: {params.get('building_type','residential')}, {params.get('stories','?')} stories
Material: {params.get('wall_material','wood frame')}, Seismic: {params.get('seismic','')}

Walls:
{json.dumps(wall_summary, indent=2)}

Return ONLY valid JSON, no markdown, no extra text:
{{"classifications":[{{"id":"H01","loadBearing":true,"reason":"North exterior wall"}},{{"id":"V03","loadBearing":false,"reason":"Short interior partition"}}]}}

Classify ALL {len(walls)} walls. Keep reasons under 8 words each."""

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
             "media_type": "image/png", "data": b64}},
            {"type": "text", "text": prompt}
        ]}]
    )

    raw = "".join(b.text for b in response.content if hasattr(b, "text"))
    clean = raw.replace("```json", "").replace("```", "").strip()
    match = re.search(r'\{[\s\S]*\}', clean)
    if match:
        clean = match.group(0)

    try:
        result = json.loads(clean)
    except json.JSONDecodeError:
        clean = re.sub(r',\s*([}\]])', r'\1', clean)
        clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', clean)
        try:
            result = json.loads(clean)
        except json.JSONDecodeError:
            found = re.findall(
                r'\{"id"\s*:\s*"([^"]+)"\s*,\s*"loadBearing"\s*:\s*(true|false)\s*,\s*"reason"\s*:\s*"([^"]*?)"\s*\}',
                clean
            )
            result = {"classifications": [
                {"id": m[0], "loadBearing": m[1] == "true", "reason": m[2]}
                for m in found
            ]}

    cls_map = {c["id"]: c for c in result.get("classifications", [])}
    for w in walls:
        cls = cls_map.get(w["id"], {})
        w["loadBearing"] = bool(cls.get("loadBearing", w["at_edge"]))
        w["reason"] = str(cls.get("reason", ""))

    return walls


def find_stacking_walls(floor_walls_list: list) -> list:
    """Cross-reference load-bearing walls across floors for stacking pairs."""
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
                    if wa["orient"] != wb["orient"]:
                        continue
                    try:
                        if wa["orient"] == "horizontal":
                            same = abs(wa["y_pct"] - wb["y_pct"]) < 8
                        else:
                            same = abs(wa["x_pct"] - wb["x_pct"]) < 8
                        if same:
                            wa["stacksWith"] = wb["id"]
                            wb["stacksWith"] = wa["id"]
                    except Exception:
                        pass
    return floor_walls_list


def render_wall_overlay(img: Image.Image, walls: list, floor_label: str = "") -> Image.Image:
    """Render color-coded highlights using pixel-accurate coordinates."""
    result = img.convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    iw, ih = img.size

    try:
        fsz = max(13, iw // 110)
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fsz)
        sfont = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(10, fsz-3))
    except Exception:
        font = sfont = ImageFont.load_default()

    lw = max(3, iw // 180)

    for w in walls:
        x1, y1, x2, y2 = w["px"]
        is_lb = w.get("loadBearing", False)
        is_stack = "stacksWith" in w

        if is_stack:
            fill = (100, 0, 200, 110)
            border = (100, 0, 200, 235)
        elif is_lb:
            fill = (210, 40, 40, 110)
            border = (180, 20, 20, 235)
        else:
            fill = (130, 130, 130, 55)
            border = (100, 100, 100, 150)

        if abs(x2-x1) < lw*2:
            cx = (x1+x2)//2; x1, x2 = cx-lw, cx+lw
        if abs(y2-y1) < lw*2:
            cy = (y1+y2)//2; y1, y2 = cy-lw, cy+lw

        draw.rectangle([x1, y1, x2, y2], fill=fill, outline=border, width=lw)

        label = w["id"] + ("↕" if is_stack else "")
        lx = x1 + 3
        ly = max(2, y1-fsz-2) if y1 > fsz+4 else y2+2
        draw.text((lx+1, ly+1), label, fill=(0, 0, 0, 180), font=sfont)
        draw.text((lx, ly), label, fill=(255, 255, 255, 240), font=sfont)

    return Image.alpha_composite(result, overlay).convert("RGB")


def add_legend(img: Image.Image, floor_label: str = "") -> Image.Image:
    lh = 56
    out = Image.new("RGB", (img.width, img.height+lh), (248, 248, 248))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    items = [
        ((210, 40, 40),   "Load bearing"),
        ((100, 0, 200),   "Stacking ↕ (multi-floor)"),
        ((130, 130, 130), "Non-structural"),
    ]
    x, yb, yt = 16, img.height+14, img.height+17
    for color, label in items:
        draw.rectangle([x, yb, x+18, yb+18], fill=color, outline=(60, 60, 60))
        draw.text((x+24, yt), label, fill=(30, 30, 30), font=font)
        x += 210

    note = f"Wall detection: OpenCV double-line pairing{' — '+floor_label if floor_label else ''}"
    draw.text((16, img.height+40), note, fill=(100, 60, 0), font=font)
    return out
