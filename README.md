# SARang

**SARang** converts grayscale Synthetic Aperture Radar imagery into georeferenced, uncertainty-aware, analyst-friendly color visuals.

> SAR sees through clouds and darkness. SARang makes it understandable.

---

## Why SARang?

SAR imagery is structurally rich but visually difficult to interpret. Optical imagery is intuitive but often unavailable due to clouds, night conditions, or revisit gaps.

SARang is not just a colorization model. It is an interpretation workbench that combines:

- SAR preprocessing
- Deep learning colorization
- Geospatial export
- Uncertainty estimation
- Insight modes for geology, landcover, and disaster analysis
- Analyst-friendly comparison tools

---

## Key Features

- Upload GeoTIFF SAR image
- Generate colorized SAR output
- Preserve georeference where available
- Confidence/uncertainty overlay
- Insight modes:
  - Natural Color
  - Landcover Insight
  - Geology Insight
  - Water/Disaster Insight
- Export as GeoTIFF/PNG
- Before/after slider
- Pixel-level explanation
- Evaluation scripts for PSNR, SSIM, and optional perceptual metrics

---

## Important Ethical Note

SARang produces **interpretive visual enhancement**, not physical ground truth. The output should support analysis, not replace calibrated SAR or optical measurements.

---

## Quickstart

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

##Frontend
cd frontend
npm install
npm run dev

