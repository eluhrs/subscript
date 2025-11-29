# Subscript HTR pipeline: image segmentation, transcription, and searchable PDF conversion

A flexible tool for transcribing images of handwritten manuscript into searchable PDFs using a combination of **Kraken** (for layout analysis), **LLMs** (for handwriting recognition), and **ReportLab** (for PDF generation).

This tool is designed to be accessible for Digital Humanities researchers while remaining hackable for developers.

## Features
-   **Hybrid Pipeline:** Uses Kraken's segmentation features to find lines of text, then send a numbered annotation map to a Generative AI model for high-accuracy transcription. Note that Kraken only works well with full-page images, not smaller fragments of paper.  Support for other segmentation providers may be added in the future.
-   **Batch Processing:** Handle single images or glob patterns (e.g., `filename??.jpg` or `*.jpg`).
-   **Combined Output:** Optionally combine multiple input images into a single PDF output file.
-   **Searchable PDF Output:** Generates PDFs in which the image is visible, but the text is and searchable and selectable (invisible text layer).

## Installation

### Prerequisites
1.  **Python 3.10 or 3.11** (Recommended for Kraken compatibility).
    *   *Note: Newer versions of Python may have compatibility issues with some dependencies.*
2.  **API Key**: You will need an API key for any model provider you intend to use (e.g., Google Gemini, OpenAI, etc).

### Setup
1.  **Clone or Download** this repository.
2.  **Create a Virtual Environment** (Recommended):
    ```bash
    # macOS/Linux
    python3.10 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Environment**:
    Create a `.env` file in the project root using the steps below (or rename the provided example.env to `.env`). In either case, you will also need to add your API key(s) to the file.
    ```bash
    # Create .env file
    echo "GEMINI_API_KEY=your_api_key_here" > .env
    echo "OPENAI_API_KEY=your_api_key_here" >> .env
    echo "ANTHROPIC_API_KEY=your_api_key_here" >> .env
    ```

## Usage
The main script is `subscript.py`. It should be run directly from the terminal.

### Basic Syntax
```bash
./subscript.py [SEGMENTATION-NICKNAME] [MODEL-NICKNAME] INPUT [OPTIONS]
```
-   **SEGMENTATION-NICKNAME**: (Optional) The nickname of the segmentation model to use (defined in `config.yml`), e.g., `historical-manuscript`. If omitted, the segmentation model defined as default is used.
-   **MODEL-NICKNAME**: (Optional) The nickname of the transcription model to use (defined in `config.yml`), e.g., `gemini-flash`. If omitted, the transcription model defined as default is used.
-   **INPUT**: Path to an image or a wildcard pattern for multiple images

### Examples
**1. Transcribe a single image (using defaults):**
```bash
./subscript.py input/sample.jpg
```

**2. Transcribe using a specific transcription model:**
```bash
./subscript.py gemini-flash input/sample.jpg
```

**3. Transcribe using a specific segmentation model:**
```bash
./subscript.py historical-manuscript input/sample.jpg
```

**4. Transcribe using specific models for both:**
```bash
./subscript.py historical-manuscript gemini-flash input/sample.jpg
```

**5. Combine multiple images into one book:**
```bash
./subscript.py "input/*.jpg" --combine my_filename.pdf
```
*Output: `output/my_filename.pdf` and `output/my_filename.txt` (all pages)*

### Options
| Flag | Description |
| :--- | :--- |
| `--help` | Show this help message and exit. |
| `--config` | Path to alternate config file (default: `./config.yml`). |
| `--output` | Path to alternate output directory (default: `./output`). |
| `--combine` | Combine multiple input images into specified PDF filename. |
| `--prompt` | Override model prompt defined in `config.yml`. |
| `--temp` | Override temperature defined in `./config.yml`. |

## Configuration (config.yml)
The `config.yml` file defines the available models, segmentation providers, and their default settings.

```yaml
# --- Segmentation Analysis ---
segmentation:
  default_segmentation: "historical-manuscript"
  
  # Define available segmentation models (additional models to be added in the future)
  models:
    historical-manuscript:
      provider: "kraken"
      model: "default"

# --- Transcription ---
transcription:
  default_model: "gemini-pro-3"

  # Define available models here
  models:
    gemini-pro-3:
      provider: "gemini"
      model: "gemini-3-pro-preview"
      prompt: "You are a literal transcription engine..."
      cost_config:
        input_token_cost: 2.0
        output_token_cost: 12.0
      API_passthrough: # provide model-specific settings below
        temperature: 0.0
        max_output_tokens: 8192
```

## License
**GNU General Public License v3.0**

Copyright (c) 2025

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

---
*Built using:*
-   **[Kraken](https://kraken.re/)** for segmentation.
-   **[ReportLab](https://www.reportlab.com/)** for PDF generation.
-   **[Google Gemini](https://deepmind.google/technologies/gemini/)** for transcription.

*Inspiration:*
-   **[htr](https://github.com/lehigh-university-libraries/htr)**.
-   **[Coded with Google AntiGravity](https://antigravity.google/)**.
