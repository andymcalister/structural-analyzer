import streamlit as st
import anthropic
import base64
import json
import re
import io
from PIL import Image
from wall_detector import (
    detect_wall_segments, classify_walls,
    find_stacking_walls, render_wall_overlay, add_legend
)

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
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 13px;
        color: #5d4037;
        margin-bottom: 1rem;
    }
    h1 { font-size: 1.6rem !important; }
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
        images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    doc.close()
    return images


def encode_image(img: Image.Image, max_px: int = 1500) -> tuple[str, str]:
    img = img.copy()
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), "image/png"


def parse_json(raw: str) -> dict:
    clean = raw.replace("```json","").replace("```","").strip()
    match = re.search(r'\{[\s\S]*\}', clean)
    if match:
        clean = match.group(0)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        clean = re.sub(r',\s*([}\]])', r'\1', clean)
        return json.loads(clean)


def identify_floor_pages(images: list[Image.Image], api_key: str) -> tuple[list[int], dict]:
    client = anthropic.Anthropic(api_key=api_key)
    content = []
    for i, img in enumerate(images):
        b64, mt = encode_image(img, max_px=1200)
        content.append({"type": "text", "text": f"Page {i}:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64}})
    content.append({"type": "text", "text": (
        "Which pages are floor plans (plan views of rooms from above)? "
        "Exclude elevations, sections, details, foundation plans, roof plans, title sheets.\n"
        'Return ONLY JSON: {"floorPlanPages":[0,2],"pageDescriptions":{"0":"1st floor plan","1":"elevations","2":"2nd floor plan"}}'
    )})
    response = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=400,
        messages=[{"role": "user", "content": content}]
    )
    raw = "".join(b.text for b in response.content if hasattr(b, "text"))
    result = parse_json(raw)
    return result.get("floorPlanPages", [0]), result.get("pageDescriptions", {})


def render_floor_tab(label: str, img: Image.Image, walls: list):
    """Show annotated drawing + wall table for one floor."""
    lb_count = sum(1 for w in walls if w.get("loadBearing"))
    stack_count = sum(1 for w in walls if "stacksWith" in w)

    c1, c2, c3 = st.columns(3)
    c1.metric("Walls detected", len(walls))
    c2.metric("Load bearing", lb_count)
    c3.metric("Stacking ↕", stack_count)

    tab_draw, tab_table = st.tabs(["🗺️ Annotated drawing", "🧱 Wall table"])

    with tab_draw:
        annotated = render_wall_overlay(img, walls, label)
        annotated = add_legend(annotated, label)
        st.image(annotated, use_column_width=True)
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        st.download_button(
            "⬇ Download annotated drawing",
            data=buf.getvalue(),
            file_name=f"walls_{label.replace(' ','_')}.png",
            mime="image/png",
            key=f"dl_{label}"
        )

    with tab_table:
        import pandas as pd
        rows = []
        for w in walls:
            rows.append({
                "ID": w["id"],
                "Orientation": w["orient"].title(),
                "Load bearing": "✅ Yes" if w.get("loadBearing") else "— No",
                "Stacks with": w.get("stacksWith", "—"),
                "Length (% of plan)": f"{w['length_pct']}%",
                "Position": f"{'y' if w['orient']=='horizontal' else 'x'}={w['y_pct' if w['orient']=='horizontal' else 'x_pct']}%",
                "Reason": w.get("reason", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if stack_count:
            pairs = ", ".join(f"{w['id']}↕{w['stacksWith']}" for w in walls if "stacksWith" in w)
            st.info(f"**Stacking pairs:** {pairs} — verify continuous load path to foundation.")

        st.markdown("> ⚠️ **Preliminary only.** Must be reviewed by a licensed structural engineer.")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏗️ Structural Wall Analyzer")
    st.caption("Preliminary tool — engineer review required")
    st.markdown("""<div class="disclaimer"><strong>Preliminary analysis only.</strong>
    Must be reviewed and stamped by a licensed structural engineer before use in construction.</div>""",
    unsafe_allow_html=True)

    api_key_input = get_api_key()

    st.subheader("Project parameters")
    building_type = st.selectbox("Building type", [
        "Single-family residential", "Multi-family residential",
        "Light commercial", "Mixed use"
    ])
    wall_material = st.selectbox("Wall material", [
        "Wood frame (2×6 @ 16\" OC)", "Wood frame (2×4 @ 16\" OC)",
        "CMU (8\" block)", "CMU (12\" block)",
        "Steel stud (3-5/8\")", "Steel stud (6\")",
        "ICF (6\" core)", "Poured concrete (6\")"
    ])
    floor_system = st.selectbox("Floor system above", [
        "Wood joists (2×10 @ 16\")", "Wood joists (2×12 @ 16\")",
        "TJI / engineered joists", "Concrete slab (5\")",
        "Concrete slab (8\")", "None (roof only)"
    ])
    roof_type = st.selectbox("Roof type", [
        "Gable (asphalt shingle)", "Hip roof",
        "Flat / low-slope", "Shed roof", "Complex / custom"
    ])
    seismic = st.selectbox("Seismic design category", [
        "Low (SDC A/B)", "Moderate (SDC C)",
        "High (SDC D)", "Very high (SDC E/F)"
    ])
    wind = st.selectbox("Wind exposure", [
        "Exposure B (suburban)", "Exposure C (open terrain)", "Exposure D (coastal)"
    ])
    snow = st.number_input("Ground snow load (psf)", min_value=0, max_value=150, value=0, step=5)


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Structural Wall Analyzer")
st.caption("Upload a PDF drawing set — walls are detected using computer vision and classified by Claude.")

uploaded_file = st.file_uploader("Upload drawing set (PDF or image)", type=["pdf","png","jpg","jpeg"])

if uploaded_file:
    # Reset on new file
    if st.session_state.get("last_filename") != uploaded_file.name:
        for key in ["all_images","page_descriptions","detected_floor_pages","floor_results","last_params"]:
            st.session_state.pop(key, None)
        st.session_state["last_filename"] = uploaded_file.name

    # Load file
    if "all_images" not in st.session_state:
        file_bytes = uploaded_file.read()
        with st.spinner("Loading file…"):
            try:
                if uploaded_file.name.lower().endswith(".pdf"):
                    imgs = pdf_to_images(file_bytes)
                else:
                    imgs = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]
                st.session_state["all_images"] = imgs
            except Exception as e:
                st.error(f"Could not load file: {e}")
                st.stop()

    all_images = st.session_state["all_images"]
    st.info(f"📄 **{uploaded_file.name}** — {len(all_images)} page(s)")

    # ── STEP 1: Scan pages ────────────────────────────────────────────────────
    if "detected_floor_pages" not in st.session_state:
        if st.button("🔍 Scan pages for floor plans", use_container_width=True):
            if not api_key_input:
                st.error("API key not configured.")
            else:
                if len(all_images) > 1:
                    with st.spinner("Scanning pages…"):
                        try:
                            floor_pages, page_desc = identify_floor_pages(all_images, api_key_input)
                        except Exception as e:
                            st.warning(f"Scan failed ({e}) — defaulting to page 0.")
                            floor_pages, page_desc = [0], {"0": "Floor plan"}
                else:
                    floor_pages, page_desc = [0], {"0": "Floor plan"}

                st.session_state["detected_floor_pages"] = floor_pages
                st.session_state["page_descriptions"] = page_desc
                st.rerun()

    else:
        # ── STEP 2: Engineer confirms floors ─────────────────────────────────
        floor_pages = st.session_state["detected_floor_pages"]
        page_desc = st.session_state.get("page_descriptions", {})

        st.markdown("---")
        st.subheader("📋 Detected floor plans")
        st.caption("Check the pages to include. Uncheck false positives or add missed floors.")

        n = len(all_images)
        confirmed = []
        cols_per_row = min(4, n)
        rows = [list(range(n))[i:i+cols_per_row] for i in range(0, n, cols_per_row)]

        for row in rows:
            cols = st.columns(len(row))
            for col, pg in zip(cols, row):
                with col:
                    desc = page_desc.get(str(pg), f"Page {pg}")
                    is_floor = pg in floor_pages
                    thumb = all_images[pg].copy()
                    thumb.thumbnail((280, 280))
                    st.image(thumb, use_column_width=True)
                    if st.checkbox(
                        f"{'🏠' if is_floor else '📐'} {desc}",
                        value=is_floor,
                        key=f"chk_{pg}"
                    ):
                        confirmed.append(pg)

        confirmed.sort()

        if not confirmed:
            st.warning("Select at least one floor plan page.")
        else:
            n_floors = len(confirmed)
            st.success(f"✅ **{n_floors} floor{'s' if n_floors>1 else ''} selected** — pages: {', '.join(str(p) for p in confirmed)}")

            if st.button(f"▶ Detect & classify walls — {n_floors} floor{'s' if n_floors>1 else ''}", type="primary", use_container_width=True):
                if not api_key_input:
                    st.error("API key not configured.")
                else:
                    params = {
                        "building_type": building_type,
                        "stories": str(n_floors),
                        "wall_material": wall_material,
                        "floor_system": floor_system,
                        "roof_type": roof_type,
                        "seismic": seismic,
                        "wind": wind,
                        "snow": snow,
                    }

                    floor_results = []  # (label, img, walls)
                    all_floor_walls = []

                    for pg in confirmed:
                        label = page_desc.get(str(pg), f"Floor plan — page {pg}")
                        img = all_images[pg]

                        with st.spinner(f"Detecting walls: {label}…"):
                            try:
                                walls, plan_box = detect_wall_segments(img)
                                st.write(f"  → {len(walls)} wall segments detected")
                            except Exception as e:
                                import traceback
                                err_msg = f"Wall detection failed for {label}: {e}"
                                st.session_state.setdefault("errors", []).append(err_msg)
                                st.session_state.setdefault("errors", []).append(traceback.format_exc())
                                st.error(err_msg)
                                continue

                        with st.spinner(f"Classifying walls: {label}…"):
                            try:
                                walls = classify_walls(walls, img, plan_box, params, api_key_input, label)
                            except Exception as e:
                                import traceback
                                err_msg = f"Classification failed for {label}: {e}"
                                st.session_state.setdefault("errors", []).append(err_msg)
                                st.session_state.setdefault("errors", []).append(traceback.format_exc())
                                st.error(err_msg)
                                continue

                        floor_results.append((label, img, walls))
                        all_floor_walls.append(walls)

                    # Cross-reference stacking walls across floors
                    if len(all_floor_walls) > 1:
                        all_floor_walls = find_stacking_walls(all_floor_walls)
                        floor_results = [(label, img, walls)
                                         for (label, img, _), walls
                                         in zip(floor_results, all_floor_walls)]

                    st.session_state["floor_results"] = floor_results
                    st.session_state["last_params"] = params
                    st.rerun()

# ── Error log ────────────────────────────────────────────────────────────────
if st.session_state.get("errors"):
    with st.expander(f"⚠️ Errors ({len(st.session_state['errors'])} entries) — click to expand", expanded=True):
        for err in st.session_state["errors"]:
            st.code(err)
    if st.button("Clear error log"):
        st.session_state["errors"] = []
        st.rerun()

# ── Results ───────────────────────────────────────────────────────────────────
if "floor_results" in st.session_state and st.session_state["floor_results"]:
    st.markdown("---")
    st.success("Analysis complete — review with your structural engineer.")
    results = st.session_state["floor_results"]

    if len(results) == 1:
        label, img, walls = results[0]
        st.subheader(f"📐 {label}")
        render_floor_tab(label, img, walls)
    else:
        tabs = st.tabs([f"📐 {label}" for label, _, _ in results])
        for tab, (label, img, walls) in zip(tabs, results):
            with tab:
                render_floor_tab(label, img, walls)

elif not uploaded_file:
    st.markdown("---")
    st.markdown("""
**How it works:**
1. Upload a PDF drawing set
2. Click **Scan pages** — floor plan pages are identified automatically
3. Review the thumbnail grid, adjust if needed, then click **Detect & classify walls**
4. OpenCV detects actual wall line segments — no coordinate guessing
5. Claude classifies each detected wall as load bearing or non-structural
6. Stacking walls across floors are cross-referenced and shown in purple

**Color legend:**
- 🔴 Red — load bearing
- 🟣 Purple ↕ — stacking walls (continuous load path across floors)
- ⬜ Gray — non-structural partitions

*Walls detected by OpenCV · Classified by Claude (claude-sonnet-4-6) · Engineer review required*
    """)
