# Changelog

All notable changes to SARang will be documented here.

## [0.4.0] - Current Hackathon Build

### Added
- GeoTIFF upload endpoint.
- Feathered tiled inference to reduce seams.
- Uncertainty estimation using repeated inference passes.
- Insight modes: natural, landcover, geology, water.
- Pixel explanation endpoint.
- GeoTIFF export.
- Evaluation script for PSNR and SSIM.

### Changed
- Replaced simple min-max normalization with percentile normalization.
- Reduced over-saturation in urban areas by adjusting gamma.
- Improved speckle filtering for single-band SAR input.

### Fixed
- Crash when uploaded raster has no CRS.
- Empty output when image size is smaller than tile size.
- Frontend slider overflow on small screens.

## [0.3.0]

### Added
- Initial U-Net colorization model.
- Demo synthetic dataset generator.

### Known Issues
- Temporal consistency not implemented.
- Optical reference alignment is manual in demo dataset.
