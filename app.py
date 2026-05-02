import streamlit as st
import anthropic
import base64
import json
import re

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


def encode_file(uploaded_file) -> tuple[str, str]:
    file_bytes = uploaded_file.read()
    b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        media_type = "application/pdf"
    elif name.endswith(".png"):
        media_type = "image/png"
    elif name.endswith((".jpg", ".jpeg")):
        media_type = "image/jpeg"
    else:
        media_type = "image/png"
    return b64, media_type


def build_system_prompt(params: dict) -> str:
    return f"""You are a structural engineering analysis assistant. A licensed structural engineer will review all your outputs. You are performing PRELIMINARY analysis only to accelerate the engineer's workflow. Never claim your outputs are final or construction-ready.

Analyze the uploaded architectural floor plan and perform preliminary structural wall identification and load calculations.

Return ONLY valid JSON (no markdown fences, no preamble) with this exact structure:
{{
  "drawingDescription": "detailed description of what you see",
  "sheetsIdentified": ["Floor Plan", "Elevations", "Foundation Plan"],
  "walls": [
    {{
      "id": "W1",
      "description": "North exterior wall, full building width",
      "location": "North exterior",
      "loadBearing": true,
      "estimatedLength": 48,
      "estimatedHeight": 9,
      "tributaryWidth": 6,
      "deadLoad": 15,
      "liveLoad": 20,
      "totalLoadPsf": 35,
      "totalLoadPlf": 210,
      "flag": "Verify rafter tie connection",
      "flagSeverity": "info"
    }}
  ],
  "openings": [
    {{
      "wallId": "W4",
      "type": "Sliding door",
      "size": "6-0 x 6-8",
      "headerRequired": "4x12 minimum",
      "notes": "Verify jack stud count and bearing length"
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
    "houseArea": 1136,
    "garageArea": 280
  }},
  "recommendations": [
    {{
      "id": "R1",
      "priority": "high",
      "title": "Interior bearing wall foundation alignment",
      "detail": "Detailed explanation of the recommendation."
    }}
  ],
  "engineerNarrative": "2-3 paragraph overall narrative for the reviewing engineer."
}}

flagSeverity options: critical, warning, info
recommendation priority options: high, medium, low

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


def run_analysis(uploaded_file, params: dict, api_key: str):
    b64, media_type = encode_file(uploaded_file)
    client = anthropic.Anthropic(api_key=api_key)

    if media_type == "application/pdf":
        content_block = {"type": "document", "source": {"type": "base64", "media_type": media_type, "data": b64}}
    else:
        content_block = {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}}

    with st.spinner("Analyzing drawing — this may take 20–40 seconds…"):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=build_system_prompt(params),
            messages=[{"role": "user", "content": [
                content_block,
                {"type": "text", "text": "Analyze this architectural drawing set and return the structural analysis JSON."}
            ]}]
        )

    raw = "".join(b.text for b in response.content if hasattr(b, "text"))
    clean = raw.replace("```json", "").replace("```", "").strip()
    match = re.search(r'\{[\s\S]*\}', clean)
    if match:
        clean = match.group(0)
    return json.loads(clean)


def render_results(result: dict, params: dict):
    summary = result.get("summary", {})
    walls = result.get("walls", [])
    openings = result.get("openings", [])
    recs = result.get("recommendations", [])

    st.success("Analysis complete — review all findings with your structural engineer.")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Walls identified", summary.get("totalWalls", len(walls)))
    col2.metric("Load bearing", summary.get("loadBearingCount", sum(1 for w in walls if w.get("loadBearing"))))
    col3.metric("Critical wall", summary.get("criticalWall", "—"))
    col4.metric("Peak load", f"{summary.get('criticalWallLoad', '—')} plf")
    col5.metric("Governing load", summary.get("governingLoad", "Gravity"))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📋 Drawing overview", "🧱 Wall analysis", "🚪 Openings", "⚠️ Recommendations", "📝 Engineer narrative"]
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
                    "Load bearing": "✅ Load bearing" if w.get("loadBearing") else "— Non-structural",
                    "Length (ft)": w.get("estimatedLength", ""),
                    "Height (ft)": w.get("estimatedHeight", ""),
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
            label = {"high": "High priority", "medium": "Medium priority", "low": "Low priority"}.get(priority, "")
            with st.expander(f"{icon} {rec.get('id', '')} — {rec.get('title', '')} ({label})"):
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
    <strong>Engineering review required.</strong> This tool generates preliminary analysis only. 
    All outputs must be reviewed and stamped by a licensed structural engineer. 
    The engineer of record bears full professional liability.
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
st.caption("Upload an architectural floor plan to identify load-bearing walls and generate preliminary structural calculations.")

uploaded_file = st.file_uploader(
    "Upload floor plan drawing",
    type=["pdf", "png", "jpg", "jpeg"],
    help="PDF, PNG, or JPG — architectural drawings, CAD exports, or scanned plans"
)

if uploaded_file:
    if "image" in uploaded_file.type:
        st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)
    else:
        st.info(f"📄 Loaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.0f} KB)")

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
            try:
                uploaded_file.seek(0)
                result = run_analysis(uploaded_file, params, api_key_input)
                st.session_state["last_result"] = result
                st.session_state["last_params"] = params
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse analysis response. Try again. Detail: {e}")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

if "last_result" in st.session_state:
    st.markdown("---")
    render_results(st.session_state["last_result"], st.session_state["last_params"])
else:
    if not uploaded_file:
        st.markdown("---")
        st.markdown("""
        **How it works:**
        1. Upload a floor plan (PDF or image) using the uploader above
        2. Set your project parameters in the sidebar
        3. Click **Run structural analysis**
        4. Review the wall-by-wall breakdown, load calculations, and engineering recommendations

        *All analysis is performed by Claude (claude-sonnet-4-6) and must be reviewed by a licensed structural engineer.*
        """)
