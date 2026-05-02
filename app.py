import streamlit as st
import anthropic
import base64
import json
import re
import io
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(
    page_title="Structural Wall Analyzer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .disclaimer {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 13px;
        color: #5d4037;
        margin-bottom: 1rem;
    }
    h1 { font-size: 1.6rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
</style>
""", unsafe_allow_html=True)


def get_api_key():
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        import os
        return os.environ.get("ANTHROPIC_API_KEY", "")


def pdf_to_images(pdf_bytes: bytes, dpi: int = 150) -> list[Image.Image]:
    """Convert all PDF pages to PIL Images."""
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for i in range(len(doc)):
        pg = doc[i]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = pg.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def encode_image(img: Image.Image) -> tuple[str, str]:
    """Encode PIL image to base64 PNG."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/png"


def identify_floor_plan_pages(images: list[Image.Image], api_key: str) -> list[int]:
    """Ask Claude which pages contain floor plans."""
    client = anthropic.Anthropic(api_key=api_key)

    content = []
    for i, img in enumerate(images):
        b64, media_type = encode_image(img)
        content.append({
            "type": "text",
            "text": f"Page {i} (0-indexed):"
        })
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64}
        })

    content.append({
        "type": "text",
        "text": """Review all the pages above. Identify which page numbers contain floor plans (plan views showing room layouts, wall positions, dimensions from above).

Do NOT include: elevations, sections, details, foundation plans, roof framing plans, title sheets, or schedules.

Return ONLY valid JSON like this:
{
  "floorPlanPages": [0, 2],
  "pageDescriptions": {
    "0": "Floor plan — main level",
    "1": "South and East elevations",
    "2": "Floor plan — upper level",
    "3": "Foundation plan"
  }
}"""
    })

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": content}]
    )

    raw = "".join(b.text for b in response.content if hasattr(b, "text"))
    clean = raw.replace("```json", "").replace("```", "").strip()
    match = re.search(r'\{[\s\S]*\}', clean)
    if match:
        clean = match.group(0)
    result = json.loads(clean)
    return result.get("floorPlanPages", [0]), result.get("pageDescriptions", {})


def build_system_prompt(params: dict) -> str:
    return f"""You are a structural engineering analysis assistant. A licensed structural engineer will review all your outputs. You are performing PRELIMINARY analysis only.

Analyze the uploaded architectural floor plan and perform preliminary structural wall identification and load calculations.

IMPORTANT: For each wall, estimate its bounding box as normalized coordinates (0.0 to 1.0) relative to the image dimensions:
- x1, y1 = top-left corner (0,0 is top-left of image)
- x2, y2 = bottom-right corner (1,1 is bottom-right of image)

Return ONLY valid JSON with this exact structure:
{{
  "drawingDescription": "detailed description of what you see",
  "sheetsIdentified": ["Floor Plan — Main Level"],
  "walls": [
    {{
      "id": "W1",
      "description": "North exterior wall, full building width",
      "location": "North exterior",
      "loadBearing": true,
      "stacksWithWall": null,
      "estimatedLength": 48,
      "estimatedHeight": 9,
      "tributaryWidth": 6,
      "deadLoad": 15,
      "liveLoad": 20,
      "totalLoadPsf": 35,
      "totalLoadPlf": 210,
      "flag": "Verify rafter tie connection",
      "flagSeverity": "info",
      "bbox": {{"x1": 0.1, "y1": 0.05, "x2": 0.9, "y2": 0.08}}
    }}
  ],
  "openings": [
    {{
      "wallId": "W4",
      "type": "Sliding door",
      "size": "6-0 x 6-8",
      "headerRequired": "4x12 minimum",
      "notes": "Verify jack stud count and bearing length",
      "bbox": {{"x1": 0.5, "y1": 0.1, "x2": 0.6, "y2": 0.15}}
    }}
  ],
  "summary": {{
    "totalWalls": 10,
    "loadBearingCount": 6,
    "nonLoadBearingCount": 4,
    "roofDeadLoad": 15,
    "roofLiveLoad": 20,
    "floorDeadLoad": 10,
    "floorLiveLoad": 40,
    "governingLoad": "Gravity",
    "governingCombo": "1.2D + 1.6L",
    "criticalWall": "W5",
    "criticalWallLoad": 900,
    "houseArea": 0,
    "garageArea": 0
  }},
  "recommendations": [
    {{
      "id": "R1",
      "priority": "high",
      "title": "Recommendation title",
      "detail": "Detailed explanation."
    }}
  ],
  "engineerNarrative": "2-3 paragraph overall narrative for the reviewing engineer."
}}

For stacksWithWall: if a wall aligns with a wall on another floor, set this to that wall's ID. Otherwise null.
flagSeverity: critical, warning, or info
priority: high, medium, or low

Building parameters:
- Type: {params['building_type']}
- Stories: {params['stories']}
- Wall material: {params['wall_material']}
- Floor system: {params['floor_system']}
- Roof: {params['roof_type']}
- Seismic: {params['seismic']}
- Wind: {params['wind']}
- Snow load: {params['snow']} psf
- Code: ASCE 7 / IBC"""


def run_analysis(floor_img: Image.Image, params: dict, api_key: str) -> dict:
    """Analyze a single floor plan image."""
    b64, media_type = encode_image(floor_img)
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=build_system_prompt(params),
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
            {"type": "text", "text": "Analyze this floor plan and return the structural analysis JSON including accurate bbox coordinates for every wall."}
        ]}]
    )

    raw = "".join(b.text for b in response.content if hasattr(b, "text"))
    clean = raw.replace("```json", "").replace("```", "").strip()
    match = re.search(r'\{[\s\S]*\}', clean)
    if match:
        clean = match.group(0)
    return json.loads(clean)


def draw_wall_overlays(base_img: Image.Image, walls: list, openings: list) -> Image.Image:
    img = base_img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(12, w // 80))
    except Exception:
        font = ImageFont.load_default()

    stacking_ids = set()
    for wall in walls:
        if wall.get("stacksWithWall"):
            stacking_ids.add(wall["id"])
            stacking_ids.add(wall["stacksWithWall"])

    for wall in walls:
        bbox = wall.get("bbox")
        if not bbox:
            continue
        x1 = int(bbox["x1"] * w)
        y1 = int(bbox["y1"] * h)
        x2 = int(bbox["x2"] * w)
        y2 = int(bbox["y2"] * h)

        if abs(x2 - x1) < 6:
            cx = (x1 + x2) // 2
            x1, x2 = cx - 3, cx + 3
        if abs(y2 - y1) < 6:
            cy = (y1 + y2) // 2
            y1, y2 = cy - 3, cy + 3

        wall_id = wall.get("id", "")
        is_load_bearing = wall.get("loadBearing", False)
        is_stacking = wall_id in stacking_ids

        if is_stacking and is_load_bearing:
            fill = (138, 43, 226, 120)
            border = (138, 43, 226, 220)
        elif is_load_bearing:
            fill = (220, 50, 50, 110)
            border = (200, 30, 30, 220)
        else:
            fill = (100, 100, 100, 70)
            border = (80, 80, 80, 160)

        draw.rectangle([x1, y1, x2, y2], fill=fill, outline=border, width=2)
        label = wall_id + (" ⬆" if is_stacking else "")
        draw.text((x1 + 4, y1 + 2), label, fill=(255, 255, 255, 240), font=font)

    for opening in openings:
        bbox = opening.get("bbox")
        if not bbox:
            continue
        ox1 = int(bbox["x1"] * w)
        oy1 = int(bbox["y1"] * h)
        ox2 = int(bbox["x2"] * w)
        oy2 = int(bbox["y2"] * h)
        if abs(ox2 - ox1) < 4:
            cx = (ox1 + ox2) // 2
            ox1, ox2 = cx - 3, cx + 3
        if abs(oy2 - oy1) < 4:
            cy = (oy1 + oy2) // 2
            oy1, oy2 = cy - 3, cy + 3
        draw.rectangle([ox1, oy1, ox2, oy2], fill=(255, 165, 0, 100), outline=(200, 130, 0, 200), width=2)

    return Image.alpha_composite(img, overlay).convert("RGB")


def add_legend(img: Image.Image) -> Image.Image:
    legend_h = 60
    new_img = Image.new("RGB", (img.width, img.height + legend_h), (245, 245, 245))
    new_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    items = [
        ((220, 50, 50), "Load bearing"),
        ((138, 43, 226), "Stacking (multi-floor)"),
        ((100, 100, 100), "Non-structural"),
        ((255, 165, 0), "Opening / header"),
    ]
    x = 20
    y_box = img.height + 15
    y_text = img.height + 18
    for color, label in items:
        draw.rectangle([x, y_box, x + 20, y_box + 20], fill=color, outline=(50, 50, 50))
        draw.text((x + 26, y_text), label, fill=(40, 40, 40), font=font)
        x += 185
    draw.text((20, img.height + 42), "⚠ Wall positions are approximate — for preliminary review only", fill=(120, 80, 0), font=font)
    return new_img


def render_floor_analysis(label: str, floor_img: Image.Image, result: dict, params: dict):
    """Render results for a single floor plan."""
    summary = result.get("summary", {})
    walls = result.get("walls", [])
    openings = result.get("openings", [])
    recs = result.get("recommendations", [])
    stacking_count = sum(1 for w in walls if w.get("stacksWithWall"))

    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Walls identified", summary.get("totalWalls", len(walls)))
    col2.metric("Load bearing", summary.get("loadBearingCount", sum(1 for w in walls if w.get("loadBearing"))))
    col3.metric("Stacking walls", stacking_count)
    col4.metric("Peak load", f"{summary.get('criticalWallLoad', '—')} plf")
    col5.metric("Critical wall", summary.get("criticalWall", "—"))

    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Annotated drawing", "📋 Overview", "🧱 Wall analysis",
        "🚪 Openings", "⚠️ Recommendations", "📝 Engineer narrative"
    ])

    with tab0:
        annotated = draw_wall_overlays(floor_img, walls, openings)
        annotated = add_legend(annotated)
        st.image(annotated, caption=f"Annotated — {label}", use_column_width=True)
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        st.download_button(
            f"⬇ Download annotated drawing — {label}",
            data=buf.getvalue(),
            file_name=f"structural_annotated_{label.replace(' ', '_')}.png",
            mime="image/png",
            key=f"dl_{label}"
        )

    with tab1:
        st.subheader("Drawing interpretation")
        st.write(result.get("drawingDescription", ""))
        sheets = result.get("sheetsIdentified", [])
        if sheets:
            st.write("**Sheets identified:**", ", ".join(sheets))
        st.subheader("Load assumptions")
        lcol1, lcol2 = st.columns(2)
        with lcol1:
            st.write("**Roof dead load:**", f"{summary.get('roofDeadLoad', 15)} psf")
            st.write("**Roof live load:**", f"{summary.get('roofLiveLoad', 20)} psf")
            st.write("**Floor dead load:**", f"{summary.get('floorDeadLoad', 10)} psf")
            st.write("**Floor live load:**", f"{summary.get('floorLiveLoad', 40)} psf")
        with lcol2:
            st.write("**Ground snow:**", f"{params['snow']} psf")
            st.write("**Governing combo:**", summary.get("governingCombo", "D + L"))
            st.write("**Seismic:**", params["seismic"])
            st.write("**Wind:**", params["wind"])

    with tab2:
        st.subheader("Wall classification & preliminary loads")
        if walls:
            import pandas as pd
            rows = []
            for w in walls:
                severity = w.get("flagSeverity", "")
                flag_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(severity, "")
                rows.append({
                    "ID": w.get("id", ""),
                    "Location / description": w.get("description", ""),
                    "Load bearing": "✅ Yes" if w.get("loadBearing") else "— No",
                    "Stacks with": w.get("stacksWithWall") or "—",
                    "Length (ft)": w.get("estimatedLength", ""),
                    "Trib. width (ft)": w.get("tributaryWidth", "—") if w.get("loadBearing") else "—",
                    "Total load (plf)": w.get("totalLoadPlf", "—") if w.get("loadBearing") else "—",
                    "Flag": f"{flag_icon} {w.get('flag', '')}" if w.get("flag") else "",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No wall data returned.")
        critical = next((w for w in walls if w.get("id") == summary.get("criticalWall")), None)
        if critical:
            st.warning(
                f"**Critical wall — {critical.get('id')}: {critical.get('description')}** | "
                f"Total load: {critical.get('totalLoadPlf')} plf | "
                f"Tributary width: {critical.get('tributaryWidth')} ft"
            )
        if stacking_count:
            st.info(f"**{stacking_count} stacking wall(s) identified** — shown in purple on the annotated drawing. Verify continuous load path from roof to foundation.")

    with tab3:
        st.subheader("Openings & headers")
        if openings:
            import pandas as pd
            st.dataframe(pd.DataFrame([{
                "Wall": o.get("wallId", ""), "Type": o.get("type", ""),
                "Size": o.get("size", ""), "Header required": o.get("headerRequired", ""),
                "Notes": o.get("notes", "")
            } for o in openings]), use_container_width=True, hide_index=True)
        else:
            st.info("No specific openings flagged.")

    with tab4:
        st.subheader("Preliminary recommendations")
        priority_order = {"high": 0, "medium": 1, "low": 2}
        for rec in sorted(recs, key=lambda r: priority_order.get(r.get("priority", "low"), 2)):
            priority = rec.get("priority", "low")
            icon = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(priority, "🔵")
            label_str = {"high": "High priority", "medium": "Medium priority", "low": "Low priority"}.get(priority, "")
            with st.expander(f"{icon} {rec.get('id', '')} — {rec.get('title', '')} ({label_str})"):
                st.write(rec.get("detail", ""))

    with tab5:
        st.subheader("Engineering narrative")
        st.write(result.get("engineerNarrative", ""))
        st.markdown("---")
        st.markdown(f"*Parameters: {params['wall_material']} · {params['floor_system']} · {params['seismic']} · {params['wind']} · Snow: {params['snow']} psf · ASCE 7 / IBC*")
        st.markdown("> ⚠️ **Preliminary analysis only.** All values must be reviewed and stamped by a licensed structural engineer before use in construction documents.")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏗️ Structural Wall Analyzer")
    st.caption("Preliminary analysis tool for structural engineers")
    st.markdown("""
    <div class="disclaimer">
    <strong>Engineering review required.</strong> Preliminary analysis only.
    Must be reviewed and stamped by a licensed structural engineer.
    </div>
    """, unsafe_allow_html=True)

    api_key_input = get_api_key()

    st.subheader("Project parameters")
    building_type = st.selectbox("Building type", ["Single-family residential", "Multi-family residential", "Light commercial", "Mixed use"])
    stories = st.selectbox("Number of stories", ["1", "2", "3", "4+"])
    wall_material = st.selectbox("Wall material", [
        "Wood frame (2×6 @ 16\" OC)", "Wood frame (2×4 @ 16\" OC)",
        "CMU (8\" block)", "CMU (12\" block)",
        "Steel stud (3-5/8\")", "Steel stud (6\")",
        "ICF (6\" core)", "Poured concrete (6\")"
    ])
    floor_system = st.selectbox("Floor system above", [
        "Wood joists (2×10 @ 16\")", "Wood joists (2×12 @ 16\")",
        "TJI / engineered joists", "Concrete slab (5\")", "Concrete slab (8\")", "None (roof only)"
    ])
    roof_type = st.selectbox("Roof type", ["Gable (asphalt shingle)", "Hip roof", "Flat / low-slope", "Shed roof", "Complex / custom"])
    seismic = st.selectbox("Seismic design category", ["Low (SDC A/B)", "Moderate (SDC C)", "High (SDC D)", "Very high (SDC E/F)"])
    wind = st.selectbox("Wind exposure", ["Exposure B (suburban)", "Exposure C (open terrain)", "Exposure D (coastal)"])
    snow = st.number_input("Ground snow load (psf)", min_value=0, max_value=150, value=0, step=5)


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("Structural Wall Analyzer")
st.caption("Upload a PDF drawing set — the app automatically identifies floor plan pages and analyzes each one.")

uploaded_file = st.file_uploader(
    "Upload drawing set (PDF)",
    type=["pdf", "png", "jpg", "jpeg"],
    help="PDF preferred. Multi-sheet drawing sets supported — floor plan pages are detected automatically."
)

if uploaded_file:
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name.lower()

    with st.spinner("Loading PDF…"):
        try:
            if file_name.endswith(".pdf"):
                all_images = pdf_to_images(file_bytes)
                st.info(f"📄 **{uploaded_file.name}** — {len(all_images)} page(s) loaded")
            else:
                img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                all_images = [img]
                st.info(f"🖼️ **{uploaded_file.name}** loaded")
        except Exception as e:
            st.error(f"Could not load file: {e}")
            st.stop()

    if st.button("▶ Run structural analysis", type="primary", use_container_width=True):
        if not api_key_input:
            st.error("API key not configured. Please contact the app administrator.")
        else:
            params = {
                "building_type": building_type, "stories": stories,
                "wall_material": wall_material, "floor_system": floor_system,
                "roof_type": roof_type, "seismic": seismic,
                "wind": wind, "snow": snow
            }

            # Step 1 — identify floor plan pages
            if len(all_images) > 1:
                with st.spinner(f"Scanning {len(all_images)} pages to identify floor plans…"):
                    try:
                        floor_pages, page_descriptions = identify_floor_plan_pages(all_images, api_key_input)
                    except Exception as e:
                        st.warning(f"Page detection failed ({e}), defaulting to page 0.")
                        floor_pages = [0]
                        page_descriptions = {"0": "Floor plan (page 0)"}

                if not floor_pages:
                    st.error("No floor plan pages detected in this PDF. Please check the drawing set.")
                    st.stop()

                st.success(f"✅ Floor plan page(s) detected: {', '.join(str(p) for p in floor_pages)}")

                with st.expander("📋 All pages identified"):
                    for pg_idx, desc in page_descriptions.items():
                        icon = "🏠" if int(pg_idx) in floor_pages else "📐"
                        st.write(f"{icon} **Page {pg_idx}:** {desc}")
            else:
                floor_pages = [0]
                page_descriptions = {"0": "Floor plan"}

            # Step 2 — analyze each floor plan page
            all_results = []
            for pg in floor_pages:
                label = page_descriptions.get(str(pg), f"Floor plan — page {pg}")
                with st.spinner(f"Analyzing: {label}…"):
                    try:
                        result = run_analysis(all_images[pg], params, api_key_input)
                        all_results.append((label, all_images[pg], result))
                    except Exception as e:
                        st.error(f"Analysis failed for page {pg}: {e}")

            st.session_state["all_results"] = all_results
            st.session_state["last_params"] = params

if "all_results" in st.session_state and st.session_state["all_results"]:
    st.markdown("---")
    st.success("Analysis complete — review all findings with your structural engineer.")
    results = st.session_state["all_results"]
    params = st.session_state["last_params"]

    if len(results) == 1:
        label, floor_img, result = results[0]
        st.subheader(f"📐 {label}")
        render_floor_analysis(label, floor_img, result, params)
    else:
        # Multiple floor plans — show as tabs
        floor_tabs = st.tabs([f"📐 {label}" for label, _, _ in results])
        for tab, (label, floor_img, result) in zip(floor_tabs, results):
            with tab:
                render_floor_analysis(label, floor_img, result, params)
else:
    if not uploaded_file:
        st.markdown("---")
        st.markdown("""
        **How it works:**
        1. Upload a PDF drawing set using the uploader above
        2. Set your project parameters in the sidebar
        3. Click **Run structural analysis**
        4. The app automatically scans all pages, identifies floor plans, and analyzes each one

        **Color legend on annotated drawings:**
        - 🔴 Red — load bearing walls
        - 🟣 Purple — stacking walls (continuous load path across floors)
        - ⬜ Gray — non-structural partitions
        - 🟠 Orange — openings requiring headers

        *All analysis performed by Claude (claude-sonnet-4-6) and must be reviewed by a licensed structural engineer.*
        """)
