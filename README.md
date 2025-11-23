# Manuscript OCR Pipeline

A flexible tool for transcribing images of handwritten manuscript into searchable PDFs using a combination of **Kraken** (for layout analysis), **LLMs** (for handwriting recognition), and **ReportLab** (for PDF generation).

This tool is designed to be accessible for Digital Humanities researchers while remaining hackable for developers.

## Features

-   **Hybrid Pipeline:** Uses Kraken's robust segmentation to find lines of text, then sends each line to a Generative AI model for high-accuracy transcription.
-   **Searchable PDF Output:** Generates PDFs where the image is visible, but the text is selectable and searchable (invisible text layer).
-   **Batch Processing:** Handle single images, directories, or glob patterns (e.g., `*.jpg`).
-   **Combined Output:** Option to combine multiple pages into a single PDF and text file.
-   **Context-Aware:** Sends previous lines as context to the model to improve accuracy on ambiguous handwriting.

## Installation

### Prerequisites

1.  **Python 3.10 or 3.11** (Recommended for Kraken compatibility).
    *   *Note: Newer versions of Python may have compatibility issues with some dependencies.*
2.  **API Key**: You will need an API key for the model provider you intend to use (e.g., Google Gemini).

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
    Create a `.env` file in the project root to store your API key.
    ```bash
    # Create .env file
    echo "GEMINI_API_KEY=your_api_key_here" > .env
    ```

## Usage

The main script is `subscript.py`. It can be run directly from the terminal.

### Basic Syntax

```bash
./subscript.py [MODEL] [INPUT] [OPTIONS]
```

-   **MODEL**: The nickname of the model to use (defined in `models.yml`), e.g., `gemini-pro-preview`.
-   **INPUT**: Path to an image, a directory of images, or a wildcard pattern.

### Examples

**1. Transcribe a single image:**
```bash
./subscript.py gemini-pro-preview my_page.jpg
```
*Output: `output/my_page.pdf`, `output/my_page.txt`, `output/my_page.xml`*

**2. Transcribe an entire directory:**
```bash
./subscript.py gemini-pro-preview ./scans/
```

**3. Combine multiple images into one book:**
```bash
./subscript.py gemini-pro-preview "scans/*.jpg" --combine my_manuscript
```
*Output: `output/my_manuscript.pdf` (all pages), `output/my_manuscript.txt`*

### Options

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--output-dir` | Directory to save output files. | `./output` |
| `--combine` | Combine all inputs into a single PDF/TXT with this filename. | None |
| `--context` | Number of previous lines to send to the model as context. Higher values improve accuracy but use more tokens. | `5` |
| `--prompt` | Override the system prompt defined in `models.yml`. | (See models.yml) |
| `--temp` | Override the temperature (creativity). 0.0 is best for accuracy. | `0.0` |

## Configuration (models.yml)

The `models.yml` file defines the available models and their default settings. You can add your own custom prompts here.

```yaml
models:
  gemini-pro-3:
    provider: gemini
    model: gemini-3-pro-preview
    temperature: 0.0
    timeout: 30m
    max_resolution: MEDIA_RESOLUTION_HIGH
    max_resolution_fallback: true
    prompt: "You are a literal transcription engine for 19th-century handwritten manuscripts. Extract text from the supplied image exactly as written..."
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
-   **[Google Gemini](https://deepmind.google/technologies/gemini/)** (specifically `gemini-3-pro-preview`) for transcription.
