"""
wall_detector.py
Step 1: OpenCV detects wall segments using parallel line pairing
Step 2: Claude classifies each segment with visual context
Step 3: Render using pixel-accurate coordinates
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io, base64, json, re
from typing import Optional


def find_plan_bounds(img: Image.Image) -> tuple[int,int,int,int]:
    """Find the main plan area, skipping page borders."""
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
    Detect wall segments using parallel line pairing.
    Real walls = two parallel lines close together (double-line convention).
    Returns (wall_list, plan_box).
    """
    plan_box = find_plan_bounds(img)
    px1, py1, px2, py2 = plan_box
    plan_w, plan_h = px2-px1, py2-py1

    img_np = np.array(img.convert("RGB"))
    plan_gray = cv2.cvtColor(img_np[py1:py2, px1:px2], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(plan_gray, 25, 90)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                             threshold=45, minLineLength=35, maxLineGap=10)

    h_segs, v_segs = [], []
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            if abs(y2-y1) < abs(x2-x1)*0.25:   # horizontal
                cy = (y1+y2)//2
                lx1, lx2 = min(x1,x2), max(x1,x2)
                if lx2-lx1 > plan_w*0.025:
                    h_segs.append((lx1, lx2, cy))
            elif abs(x2-x1) < abs(y2-y1)*0.25:  # vertical
                cx = (x1+x2)//2
                ly1, ly2 = min(y1,y2), max(y1,y2)
                if ly2-ly1 > plan_h*0.025:
                    v_segs.append((ly1, ly2, cx))

    def pair_lines(segs, orient, plan_dim_main, plan_dim_cross,
                   gap_min=3, gap_max=24, overlap_thresh=0.45):
        """Find pairs of parallel lines that form wall double-lines."""
        # segs format: (start, end, pos) where pos is perpendicular coordinate
        segs_sorted = sorted(segs, key=lambda s: s[2])
        pairs = []
        used = set()
        for i, seg_a in enumerate(segs_sorted):
            if i in used: continue
            a_start, a_end, a_pos = seg_a
            best, best_gap = None, gap_max+1
            for j, seg_b in enumerate(segs_sorted[i+1:], start=i+1):
                if j in used: continue
                b_start, b_end, b_pos = seg_b
                gap = abs(b_pos - a_pos)
                if gap > gap_max: break
                if gap < gap_min: continue
                # Check span overlap
                overlap = min(a_end, b_end) - max(a_start, b_start)
                span = max(a_end, b_end) - min(a_start, b_start)
                if span > 0 and overlap/span >= overlap_thresh and gap < best_gap:
                    best_gap = gap; best = j, seg_b
            if best:
                j, seg_b = best
                used.add(i); used.add(j)
                b_start, b_end, b_pos = seg_b
                pairs.append((
                    min(a_start, b_start), max(a_end, b_end),
                    min(a_pos, b_pos), max(a_pos, b_pos)
                ))
        return pairs  # (main_start, main_end, cross_min, cross_max)

    h_pairs = pair_lines(h_segs, "H", plan_w, plan_h)
    v_pairs = pair_lines(v_segs, "V", plan_h, plan_w)

    # Find building envelope using the longest detected walls
    def get_building_envelope(h_pairs, v_pairs, plan_w, plan_h):
        if not h_pairs or not v_pairs:
            return 0, 0, plan_w, plan_h
        # Sort by length
        h_by_len = sorted(h_pairs, key=lambda p: p[1]-p[0], reverse=True)
        v_by_len = sorted(v_pairs, key=lambda p: p[1]-p[0], reverse=True)
        # Top N long walls define boundary
        n = min(4, len(h_by_len))
        h_ys = sorted([(p[2]+p[3])//2 for p in h_by_len[:n]])
        n = min(4, len(v_by_len))
        v_xs = sorted([(p[2]+p[3])//2 for p in v_by_len[:n]])
        pad = 60
        by1 = max(0, h_ys[0] - pad)
        by2 = min(plan_h, h_ys[-1] + pad)
        bx1 = max(0, v_xs[0] - pad)
        bx2 = min(plan_w, v_xs[-1] + pad)
        return bx1, by1, bx2, by2

    bx1, by1, bx2, by2 = get_building_envelope(h_pairs, v_pairs, plan_w, plan_h)

    def filter_cluster(pairs, orient, plan_w, plan_h):
        """Remove stair/hatch clusters: many short parallels in a small area."""
        result = []
        for i, pa in enumerate(pairs):
            pa_start, pa_end, pa_cmin, pa_cmax = pa
            pa_len = pa_end - pa_start
            pa_pos = (pa_cmin+pa_cmax)//2
            nearby = 0
            for j, pb in enumerate(pairs):
                if i==j: continue
                pb_start, pb_end, pb_cmin, pb_cmax = pb
                pb_pos = (pb_cmin+pb_cmax)//2
                pos_gap = abs(pa_pos - pb_pos)
                if orient=="H":
                    if pos_gap/plan_h < 0.10 and abs((pa_start+pa_end)//2 - (pb_start+pb_end)//2)/plan_w < 0.15:
                        nearby += 1
                else:
                    if pos_gap/plan_w < 0.10 and abs((pa_start+pa_end)//2 - (pb_start+pb_end)//2)/plan_h < 0.15:
                        nearby += 1
            len_pct = pa_len / (plan_w if orient=="H" else plan_h)
            if nearby >= 3 and len_pct < 0.22:
                continue
            result.append(pa)
        return result

    h_pairs = filter_cluster(h_pairs, "H", plan_w, plan_h)
    v_pairs = filter_cluster(v_pairs, "V", plan_w, plan_h)

    # Convert to wall dicts; filter to building envelope
    walls = []
    for i, (x_start, x_end, y_min, y_max) in enumerate(h_pairs):
        cx, cy = (x_start+x_end)//2, (y_min+y_max)//2
        # Must be within building envelope
        if not (bx1-40 <= cx <= bx2+40 and by1-40 <= cy <= by2+40):
            continue
        length = x_end - x_start
        walls.append({
            "id": f"H{i+1:02d}",
            "orient": "horizontal",
            "length_pct": round(length/plan_w*100, 1),
            "x_pct": round(cx/plan_w*100, 1),
            "y_pct": round(cy/plan_h*100, 1),
            "at_edge": cy < plan_h*0.14 or cy > plan_h*0.86,
            "span_pct": f"{round(x_start/plan_w*100,1)}-{round(x_end/plan_w*100,1)}",
            "px": [px1+x_start, py1+y_min, px1+x_end, py1+y_max],
        })

    for i, (y_start, y_end, x_min, x_max) in enumerate(v_pairs):
        cx, cy = (x_min+x_max)//2, (y_start+y_end)//2
        if not (bx1-40 <= cx <= bx2+40 and by1-40 <= cy <= by2+40):
            continue
        length = y_end - y_start
        walls.append({
            "id": f"V{i+1:02d}",
            "orient": "vertical",
            "length_pct": round(length/plan_h*100, 1),
            "x_pct": round(cx/plan_w*100, 1),
            "y_pct": round(cy/plan_h*100, 1),
            "at_edge": cx < plan_w*0.14 or cx > plan_w*0.86,
            "span_pct": f"{round(y_start/plan_h*100,1)}-{round(y_end/plan_h*100,1)}",
            "px": [px1+x_min, py1+y_start, px1+x_max, py1+y_end],
        })

    return walls, plan_box


def classify_walls(walls: list, img: Image.Image, plan_box: tuple,
                   params: dict, api_key: str, floor_label: str = "") -> list:
    """
    Send detected wall list + small plan image to Claude for classification.
    Claude classifies each wall as load-bearing or non-structural.
    """
    import anthropic

    px1, py1, px2, py2 = plan_box
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
- position as % of plan (0=top-left corner)
- at_edge: true if near the plan boundary (likely exterior)
- span: the range it covers as % of plan width/height

Classify each wall as load-bearing or non-structural based on:
1. Exterior walls (at_edge=true) → almost always load-bearing
2. Long walls spanning >40% of plan → likely load-bearing  
3. Walls aligned with joist direction (note "2x_ JOISTS @ _" labels in drawing) → may be non-structural if parallel
4. Walls perpendicular to joist span → load-bearing
5. Short interior partitions (<15% span) → likely non-structural
6. Walls supporting beams or at mid-span → load-bearing

Building: {params.get('building_type','residential')}, {params.get('stories','?')} stories
Material: {params.get('wall_material','wood frame')}
Seismic: {params.get('seismic','')}

Walls to classify:
{json.dumps(wall_summary, indent=2)}

Return ONLY valid JSON, no markdown:
{{
  "classifications": [
    {{"id": "H01", "loadBearing": true, "reason": "North exterior wall full width"}},
    {{"id": "V03", "loadBearing": false, "reason": "Short interior partition"}}
  ]
}}

Classify ALL {len(walls)} walls. Keep reasons under 8 words."""

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
             "media_type": "image/png", "data": b64}},
            {"type": "text", "text": prompt}
        ]}]
    )

    raw = "".join(b.text for b in response.content if hasattr(b, "text"))
    clean = raw.replace("```json","").replace("```","").strip()
    match = re.search(r'\{[\s\S]*\}', clean)
    if match: clean = match.group(0)
    try:
        result = json.loads(clean)
    except json.JSONDecodeError:
        clean = re.sub(r',\s*([}\]])', r'\1', clean)
        result = json.loads(clean)

    cls_map = {c["id"]: c for c in result.get("classifications", [])}
    for w in walls:
        cls = cls_map.get(w["id"], {})
        w["loadBearing"] = cls.get("loadBearing", w["at_edge"])
        w["reason"] = cls.get("reason", "")

    return walls


def find_stacking_walls(floor_walls_list: list) -> list:
    """Cross-reference load-bearing walls across floors to find stacking pairs."""
    if len(floor_walls_list) < 2:
        return floor_walls_list
    for fi, walls_a in enumerate(floor_walls_list):
        for fj, walls_b in enumerate(floor_walls_list):
            if fi >= fj: continue
            for wa in walls_a:
                if not wa.get("loadBearing"): continue
                for wb in walls_b:
                    if not wb.get("loadBearing"): continue
                    if wa["orient"] != wb["orient"]: continue
                    try:
                        if wa["orient"] == "horizontal":
                            same = abs(wa["y_pct"]-wb["y_pct"]) < 8
                        else:
                            same = abs(wa["x_pct"]-wb["x_pct"]) < 8
                        if same:
                            wa["stacksWith"] = wb["id"]
                            wb["stacksWith"] = wa["id"]
                    except Exception:
                        pass
    return floor_walls_list


def render_wall_overlay(img: Image.Image, walls: list, floor_label: str = "") -> Image.Image:
    """Render color-coded highlights using pixel-accurate coordinates."""
    result = img.convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    iw, ih = img.size

    try:
        fsz = max(13, iw//110)
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fsz)
        sfont = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(10, fsz-3))
    except Exception:
        font = sfont = ImageFont.load_default()

    lw = max(3, iw//180)

    for w in walls:
        x1, y1, x2, y2 = w["px"]
        is_lb = w.get("loadBearing", False)
        is_stack = "stacksWith" in w

        if is_stack:
            fill   = (100, 0, 200, 110)
            border = (100, 0, 200, 235)
        elif is_lb:
            fill   = (210, 40, 40, 110)
            border = (180, 20, 20, 235)
        else:
            fill   = (130, 130, 130, 55)
            border = (100, 100, 100, 150)

        # Ensure minimum visible thickness
        if abs(x2-x1) < lw*2:
            cx=(x1+x2)//2; x1,x2 = cx-lw, cx+lw
        if abs(y2-y1) < lw*2:
            cy=(y1+y2)//2; y1,y2 = cy-lw, cy+lw

        draw.rectangle([x1,y1,x2,y2], fill=fill, outline=border, width=lw)

        label = w["id"] + ("↕" if is_stack else "")
        lx = x1+3
        ly = max(2, y1-fsz-2) if y1 > fsz+4 else y2+2
        draw.text((lx+1,ly+1), label, fill=(0,0,0,180), font=sfont)
        draw.text((lx, ly),   label, fill=(255,255,255,240), font=sfont)

    return Image.alpha_composite(result, overlay).convert("RGB")


def add_legend(img: Image.Image, floor_label: str = "") -> Image.Image:
    lh = 56
    out = Image.new("RGB", (img.width, img.height+lh), (248,248,248))
    out.paste(img, (0,0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    items = [
        ((210,40,40),   "Load bearing"),
        ((100,0,200),   "Stacking ↕ (multi-floor)"),
        ((130,130,130), "Non-structural"),
    ]
    x, yb, yt = 16, img.height+14, img.height+17
    for color, label in items:
        draw.rectangle([x,yb,x+18,yb+18], fill=color, outline=(60,60,60))
        draw.text((x+24,yt), label, fill=(30,30,30), font=font)
        x += 210

    note = f"Wall detection: OpenCV parallel line pairing{' — '+floor_label if floor_label else ''}"
    draw.text((16, img.height+40), note, fill=(100,60,0), font=font)
    return out
