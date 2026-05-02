# Structural Wall Analyzer

Preliminary load-bearing wall identification and structural calculations from architectural floor plans. Built with Streamlit + Claude API.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your Anthropic API key

**Option A — secrets file (recommended for local use):**
Edit `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

**Option B — environment variable:**
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

**Option C — enter it in the app sidebar** at runtime (not persisted).

### 3. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Usage

1. Upload a floor plan PDF or image using the file uploader
2. Set project parameters in the left sidebar (building type, wall material, seismic zone, etc.)
3. Click **Run structural analysis**
4. Review results across 5 tabs: Drawing overview, Wall analysis, Openings, Recommendations, Engineer narrative

## Output tabs

| Tab | Contents |
|-----|----------|
| Drawing overview | Interpretation of the drawing, sheets identified, load assumptions used |
| Wall analysis | Wall-by-wall table: load bearing status, dimensions, tributary width, total load in plf |
| Openings | Header requirements for doors and windows |
| Recommendations | Prioritized engineering action items (high / medium / low) |
| Engineer narrative | Summary narrative for the reviewing engineer |

## Supported file types
- PDF (multi-sheet drawing sets)
- PNG, JPG/JPEG (exported or scanned plans)

## Disclaimer

**This tool generates preliminary analysis only.** All outputs must be reviewed, verified, and stamped by a licensed structural engineer before use in construction documents. The engineer of record bears full professional liability.

---

## Deployment (Streamlit Cloud)

1. Push this folder to a GitHub repo
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Set `ANTHROPIC_API_KEY` in the Streamlit Cloud secrets panel
