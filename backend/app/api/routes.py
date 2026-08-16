from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.schemas.common import ColorizeRequest, ColorizeResponse, HealthResponse
from app.services import audit, export, insight_modes, preprocess
from app.services.inference import run_tiled_inference
from app.services.explain import explain_pixel
from app.utils.files import output_path, save_upload_file

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", app=settings.app_name)


@router.post("/colorize", response_model=ColorizeResponse)
async def colorize(
    file: UploadFile = File(...),
    mode: str = Form("natural"),
    enable_uncertainty: bool = Form(True),
    gamma: float = Form(1.0),
):
    if mode not in ["natural", "landcover", "geology", "water"]:
        raise HTTPException(status_code=400, detail="Invalid mode")

    if file.size and file.size > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    input_path = save_upload_file(file)

    try:
        img, meta = preprocess.read_geotiff(str(input_path))
        features = preprocess.build_feature_stack(img)

        rgb, confidence = run_tiled_inference(
            features=features,
            tile_size=settings.tile_size,
            overlap=settings.overlap,
            uncertainty=enable_uncertainty,
        )

        if mode == "natural":
            rgb = insight_modes.natural_mode(rgb)
        elif mode == "landcover":
            rgb = insight_modes.landcover_mode(rgb, confidence)
        elif mode == "geology":
            gray = features[0]
            rgb = insight_modes.geology_mode(rgb, gray)
        elif mode == "water":
            gray = features[0]
            rgb = insight_modes.water_mode(rgb, gray)

        rgb = insight_modes.apply_gamma(rgb, gamma)

        tif_path = output_path(prefix=f"{mode}_color", extension="tif")
        conf_path = output_path(prefix=f"{mode}_confidence", extension="tif")

        export.write_rgb_geotiff(tif_path, rgb, meta)
        export.write_confidence_geotiff(conf_path, confidence, meta)

        audit.log_event(
            "colorize_success",
            {
                "input": str(input_path),
                "mode": mode,
                "enable_uncertainty": enable_uncertainty,
                "width": meta.get("width"),
                "height": meta.get("height"),
            },
        )

        return ColorizeResponse(
            status="success",
            message="Colorization completed",
            output_image=f"/outputs/{tif_path.name}",
            confidence_image=f"/outputs/{conf_path.name}",
            mode=mode,
            width=meta.get("width"),
            height=meta.get("height"),
        )

    except Exception as exc:
        audit.log_event(
            "colorize_error",
            {
                "input": str(input_path),
                "mode": mode,
                "error": str(exc),
            },
        )
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")


@router.post("/explain")
async def explain(
    file: UploadFile = File(...),
    x: int = Form(0),
    y: int = Form(0),
):
    input_path = save_upload_file(file)

    try:
        img, _ = preprocess.read_geotiff(str(input_path))
        features = preprocess.build_feature_stack(img)
        return explain_pixel(features, x=x, y=y)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explain failed: {exc}")
