# Subscript Project Timeline

This document provides a high-level overview of the major features, fixes, and milestones in the development of the Subscript project.

## 2025-11-18: Antigravity Launch
- First public release of Google DeepMind's Antigravity AI coding assistant.
## 2025-11-20: Early Experimentation (Subscript 1.0)
- Initial experiments using Antigravity to create a Python script.
- Initial work to use the Gemini API to transcribe images from the CLI.

## 2025-11-23 to 2025-11-29: Core Development (Subscript 2.0)
- **Initial Setup (2025-11-23):** Initial commit of the subscript project and added basic `.env.example` file.
- **Phase 1-2:** Infrastructure setup and Kraken segmentation with Gemini visual tagging. Transitioned from legacy slicing architecture to full-page processing.
- **Phase 3-4:** CLI refinement, multi-model support, and PDF combination feature added.
- **Phase 5-6:** Output, logging, and advanced configuration refinements.
- **Major Refactor:** Introduced multi-provider support, improved configuration management, and initial documentation.
- **Features:**
  - Added `--nopdf` option to skip PDF generation.
  - Added capability to combine TXT files with dividers when `--combine` is used.
- **Fixes:**
  - Resolved duplicate line writing bug in TXT output.

## 2025-12-02: Version 1.0 Release
- **Features:**
  - Added image preprocessing options (resize, contrast, binarize).
  - Renamed `image_size` argument to `resize_image` for clarity.
- **Packaging:** First iteration of packaging `subscript` for installation via `pip`.

## 2025-12-05 to 2025-12-08: Version 1.1 Release
- **Features:**
  - Improved PageXML structure with `TextLine` elements in output.
  - Added `--onlypdf` option to skip segmentation/transcription and reconstruct PDF directly from existing IMG and XML files.
- **Fixes:**
  - Used relative imports for better package compatibility.
  - Fixed issue where processed input files were not removed as expected during synchronization.
  - Resolved missing implementation files for the `--onlypdf` feature.

## 2025-12-13: Version 1.2 Release
- **Features:**
  - Added Preprocessing CLI arguments directly accessible (Resize, Contrast, Binarize, Invert).
  - Added Configuration Manifest logging for better reproducibility.
- **Fixes:**
  - Fixed Temperature control (`Temp`) override logic.

## 2025-12-23: Version 1.3 and 1.4 Releases
- **Features:**
  - Migrated to the new `google.genai` SDK (`v1.3.0`).
  - Implemented Structured Outputs and optimized prompting strategies (`v1.4.0`).
