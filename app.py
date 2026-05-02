import streamlit as st
import anthropic
import base64
import json
import re
import io
from PIL import Image
from wall_detector import render_overlay, add_legend, find_drawing_bounds, find_stacking_pairs

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
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/png"


def resize_for_detection(img: Image.Image, max_px: int = 1500) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_px:
        return img
    scale = max_px / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def parse_json_response(raw: str) -> dict:
    clean = raw.replace("```json", "").replace("```", "").strip()
    match = re.search(r'\{[\s\S]*\}', clean)
    if match:
        clean = match.group(0)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        clean = re.sub(r',\s*([}\]])', r'\1', clean)
        clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', clean)
        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            last_brace = clean.rfind('}')
            if last_brace > 0:
                try:
                    return json.loads(clean[:last_brace + 1])
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Could not parse response as JSON: {e}")


def identify_floor_plan_pages(images: list[Image.Image], api_key: str) -> tuple[list[int], dict]:
    client = anthropic.Anthropic(api_key=api_key)
    content = []
    for i, img in enumerate(images):
        small = resize_for_detection(img)
        b64, media_type = encode_image(small)
        content.append({"type": "text", "text": f"Page {i} (0-indexed):"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}})
    content.append({"type": "text", "text": """Review all pages. Identify which contain floor plans (plan views showing room layouts and wall positions viewed from above).
Do NOT include elevations, sections, details, foundation plans, roof framing plans, title sheets, or schedules.
Return ONLY valid JSON:
{"floorPlanPages": [0, 2], "pageDescriptions": {"0": "1st floor plan", "1": "South elevation", "2": "2nd floor plan"}}"""})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": content}]
    )
    raw = "".join(b.text for b in response.content if hasattr(b, "text"))
    result = parse_json_response(raw)
    return result.get("floorPlanPages", [0]), result.get("pageDescriptions", {})


def build_system_prompt(params: dict) -> str:
    return f"""You are a structural engineering analysis assistant. A licensed structural engineer will review all your outputs. Preliminary analysis only.

Analyze this architectural floor plan image. For each wall, provide a bbox as normalized coordinates (0.0–1.0) relative to the image you see:
- x1,y1 = top-left, x2,y2 = bottom-right of the wall segment

Return ONLY valid JSON:
{{
  "drawingDescription": "description of drawing",
  "sheetsIdentified": ["1st Floor Plan"],
  "walls": [
    {{
      "id": "W1",
      "description": "North exterior wall",
      "location": "North exterior",
      "loadBearing": true,
      "stacksWithWall": null,
      "estimatedLength": 32,
      "estimatedHeight": 9,
      "tributaryWidth": 8,
      "deadLoad": 15,
      "liveLoad": 40,
      "totalLoadPsf": 55,
      "totalLoadPlf": 440,
      "flag": "Verify connection at ridge",
      "flagSeverity": "info",
      "bbox": {{"x1": 0.05, "y1": 0.05, "x2": 0.95, "y2": 0.09}}
    }}
  ],
  "openings": [
    {{
      "wallId": "W1",
      "type": "Window",
      "size": "3-0 x 4-0",
      "headerRequired": "4x8",
      "notes": "Verify header bearing",
      "bbox": {{"x1": 0.3, "y1": 0.05, "x2": 0.4, "y2": 0.09}}
    }}
  ],
  "summary": {{
    "totalWalls": 8,
    "loadBearingCount": 5,
    "nonLoadBearingCount": 3,
    "roofDeadLoad": 15,
    "roofLiveLoad": 20,
    "floorDeadLoad": 10,
    "floorLiveLoad": 40,
    "governingLoad": "Gravity",
    "governingCombo": "1.2D + 1.6L",
    "criticalWall": "W1",
    "criticalWallLoad": 600,
    "houseArea": 0,
    "garageArea": 0
  }},
  "recommendations": [
    {{"id": "R1", "priority": "high", "title": "Title", "detail": "Detail."}}
  ],
  "engineerNarrative": "Narrative for reviewing engineer."
}}

Be precise with bbox — place them where walls actually appear in this image.
flagSeverity: critical/warning/info. priority: high/medium/low.

Parameters: Type={params['building_type']}, Stories={params['stories']},
Material={params['wall_material']}, Floor={params['floor_system']},
Roof={params['roof_type']}, Seismic={params['seismic']},
Wind={params['wind']}, Snow={params['snow']}psf, Code=ASCE 7/IBC"""


def run_analysis(full_img: Image.Image, params: dict, api_key: str) -> tuple[dict, Image.Image, tuple]:
    """Analyze floor plan. Returns (result, cropped_img, crop_box)."""
    crop_box = find_drawing_bounds(full_img)
    x1, y1, x2, y2 = crop_box
    cropped_img = full_img.crop(crop_box)

    b64, media_type = encode_image(cropped_img)
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8096,
        system=build_system_prompt(params),
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
            {"type": "text", "text": "Analyze this floor plan. Return the structural analysis JSON with precise bbox coordinates for every wall as seen in this image."}
        ]}]
    )
    raw = "".join(b.text for b in response.content if hasattr(b, "text"))
    result = parse_json_response(raw)
    return result, cropped_img, crop_box


def render_floor_analysis(label: str, full_img: Image.Image, cropped_img: Image.Image,
                           crop_box: tuple, result: dict, params: dict, stacking_map: dict):
    summary = result.get("summary", {})
    walls = result.get("walls", [])
    openings = result.get("openings", [])
    recs = result.get("recommendations", [])
    stacking_count = len(stacking_map)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Walls identified", summary.get("totalWalls", len(walls)))
    col2.metric("Load bearing", summary.get("loadBearingCount", sum(1 for w in walls if w.get("loadBearing"))))
    col3.metric("Stacking walls", stacking_count)
    col4.metric("Peak load", f"{summary.get('criticalWallLoad', '—')} plf")
    col5.metric("Critical wall", summary.get("criticalWall", "—"))

    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Annotated drawing", "📋 Overview",
        "🧱 Wall analysis", "🚪 Openings",
        "⚠️ Recommendations", "📝 Engineer narrative"
    ])

    with tab0:
        with st.spinner("Rendering annotated drawing…"):
            annotated = render_overlay(full_img, cropped_img, crop_box, walls, openings, stacking_map)
            annotated = add_legend(annotated)
        st.image(annotated, caption=f"Annotated — {label}", use_column_width=True)
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        st.download_button(
            f"⬇ Download annotated drawing",
            data=buf.getvalue(),
            file_name=f"structural_{label.replace(' ', '_')}.png",
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
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Roof DL:**", f"{summary.get('roofDeadLoad', 15)} psf")
            st.write("**Roof LL:**", f"{summary.get('roofLiveLoad', 20)} psf")
            st.write("**Floor DL:**", f"{summary.get('floorDeadLoad', 10)} psf")
            st.write("**Floor LL:**", f"{summary.get('floorLiveLoad', 40)} psf")
        with c2:
            st.write("**Snow:**", f"{params['snow']} psf")
            st.write("**Combo:**", summary.get("governingCombo", "D + L"))
            st.write("**Seismic:**", params["seismic"])
            st.write("**Wind:**", params["wind"])

    with tab2:
        st.subheader("Wall classification & preliminary loads")
        if walls:
            import pandas as pd
            rows = []
            for w in walls:
                flag_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(w.get("flagSeverity", ""), "")
                partner = stacking_map.get(w.get("id", ""), w.get("stacksWithWall"))
                rows.append({
                    "ID": w.get("id", ""),
                    "Description": w.get("description", ""),
                    "Load bearing": "✅ Yes" if w.get("loadBearing") else "— No",
                    "Stacks with": partner or "—",
                    "Length (ft)": w.get("estimatedLength", ""),
                    "Trib. (ft)": w.get("tributaryWidth", "—") if w.get("loadBearing") else "—",
                    "Load (plf)": w.get("totalLoadPlf", "—") if w.get("loadBearing") else "—",
                    "Flag": f"{flag_icon} {w.get('flag', '')}" if w.get("flag") else "",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        critical = next((w for w in walls if w.get("id") == summary.get("criticalWall")), None)
        if critical:
            st.warning(f"**Critical — {critical.get('id')}: {critical.get('description')}** | {critical.get('totalLoadPlf')} plf | Trib: {critical.get('tributaryWidth')} ft")
        if stacking_count:
            pairs = ", ".join(f"{k}↕{v}" for k, v in stacking_map.items())
            st.info(f"**{stacking_count} stacking wall(s)** — {pairs} — shown purple. Verify continuous load path to foundation.")

    with tab3:
        st.subheader("Openings & headers")
        if openings:
            import pandas as pd
            st.dataframe(pd.DataFrame([{
                "Wall": o.get("wallId", ""), "Type": o.get("type", ""),
                "Size": o.get("size", ""), "Header": o.get("headerRequired", ""),
                "Notes": o.get("notes", "")
            } for o in openings]), use_container_width=True, hide_index=True)
        else:
            st.info("No openings flagged.")

    with tab4:
        st.subheader("Preliminary recommendations")
        priority_order = {"high": 0, "medium": 1, "low": 2}
        for rec in sorted(recs, key=lambda r: priority_order.get(r.get("priority", "low"), 2)):
            icon = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(rec.get("priority", "low"), "🔵")
            lbl = {"high": "High", "medium": "Medium", "low": "Low"}.get(rec.get("priority", "low"), "")
            with st.expander(f"{icon} {rec.get('id','')} — {rec.get('title','')} ({lbl} priority)"):
                st.write(rec.get("detail", ""))

    with tab5:
        st.subheader("Engineering narrative")
        st.write(result.get("engineerNarrative", ""))
        st.markdown("---")
        st.markdown(f"*{params['wall_material']} · {params['floor_system']} · {params['seismic']} · {params['wind']} · Snow {params['snow']} psf · ASCE 7/IBC*")
        st.markdown("> ⚠️ **Preliminary only.** Must be reviewed and stamped by a licensed structural engineer.")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏗️ Structural Wall Analyzer")
    st.caption("Preliminary analysis tool for structural engineers")
    st.markdown("""<div class="disclaimer"><strong>Engineering review required.</strong>
    Preliminary analysis only. Must be reviewed and stamped by a licensed structural engineer.</div>""",
    unsafe_allow_html=True)

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


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("Structural Wall Analyzer")
st.caption("Upload a PDF drawing set — floor plan pages are detected automatically, walls highlighted using OpenCV line detection.")

uploaded_file = st.file_uploader("Upload drawing set (PDF)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = uploaded_file.read()

    with st.spinner("Loading drawing…"):
        try:
            if uploaded_file.name.lower().endswith(".pdf"):
                all_images = pdf_to_images(file_bytes)
                st.info(f"📄 **{uploaded_file.name}** — {len(all_images)} page(s)")
            else:
                img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                all_images = [img]
                st.info(f"🖼️ **{uploaded_file.name}** loaded")
        except Exception as e:
            st.error(f"Could not load file: {e}")
            st.stop()

    if st.button("▶ Run structural analysis", type="primary", use_container_width=True):
        if not api_key_input:
            st.error("API key not configured.")
        else:
            params = {
                "building_type": building_type, "stories": stories,
                "wall_material": wall_material, "floor_system": floor_system,
                "roof_type": roof_type, "seismic": seismic,
                "wind": wind, "snow": snow
            }

            # Step 1: identify floor plan pages
            if len(all_images) > 1:
                with st.spinner(f"Scanning {len(all_images)} pages for floor plans…"):
                    try:
                        floor_pages, page_descriptions = identify_floor_plan_pages(all_images, api_key_input)
                    except Exception as e:
                        st.warning(f"Page detection failed ({e}), using page 0.")
                        floor_pages = [0]
                        page_descriptions = {"0": "Floor plan"}
                if not floor_pages:
                    st.error("No floor plan pages found.")
                    st.stop()
                st.success(f"✅ Floor plan page(s): {', '.join(str(p) for p in floor_pages)}")
                with st.expander("📋 All pages"):
                    for pg_idx, desc in page_descriptions.items():
                        icon = "🏠" if int(pg_idx) in floor_pages else "📐"
                        st.write(f"{icon} **Page {pg_idx}:** {desc}")
            else:
                floor_pages = [0]
                page_descriptions = {"0": "Floor plan"}

            # Step 2: analyze each floor plan
            floor_analyses = []  # (label, full_img, cropped_img, crop_box, result)
            for pg in floor_pages:
                label = page_descriptions.get(str(pg), f"Floor plan — page {pg}")
                with st.spinner(f"Analyzing: {label}…"):
                    try:
                        result, cropped_img, crop_box = run_analysis(all_images[pg], params, api_key_input)
                        floor_analyses.append((label, all_images[pg], cropped_img, crop_box, result))
                    except Exception as e:
                        st.error(f"Analysis failed for {label}: {e}")

            # Step 3: cross-reference stacking walls across floors
            floor_results_for_stacking = [(label, result) for label, _, _, _, result in floor_analyses]
            stacking_by_floor = find_stacking_pairs(floor_results_for_stacking)

            st.session_state["floor_analyses"] = floor_analyses
            st.session_state["stacking_by_floor"] = stacking_by_floor
            st.session_state["last_params"] = params

if "floor_analyses" in st.session_state and st.session_state["floor_analyses"]:
    st.markdown("---")
    st.success("Analysis complete — review all findings with your structural engineer.")
    analyses = st.session_state["floor_analyses"]
    stacking_by_floor = st.session_state["stacking_by_floor"]
    params = st.session_state["last_params"]

    if len(analyses) == 1:
        label, full_img, cropped_img, crop_box, result = analyses[0]
        st.subheader(f"📐 {label}")
        render_floor_analysis(label, full_img, cropped_img, crop_box, result, params, stacking_by_floor.get(0, {}))
    else:
        tabs = st.tabs([f"📐 {label}" for label, *_ in analyses])
        for i, (tab, (label, full_img, cropped_img, crop_box, result)) in enumerate(zip(tabs, analyses)):
            with tab:
                render_floor_analysis(label, full_img, cropped_img, crop_box, result, params, stacking_by_floor.get(i, {}))
else:
    if not uploaded_file:
        st.markdown("---")
        st.markdown("""
**How it works:**
1. Upload a PDF drawing set
2. Set project parameters in the sidebar
3. Click **Run structural analysis**
4. App scans all pages, finds floor plans automatically, analyzes each one
5. Walls are highlighted using OpenCV line detection for accuracy

**Color legend:**
- 🔴 Red — load bearing
- 🟣 Purple — stacking walls across floors (↕ label shows partner wall)
- ⬜ Gray — non-structural
- 🟠 Orange — openings requiring headers

*Reviewed by Claude (claude-sonnet-4-6). Engineer stamp required.*
        """)
