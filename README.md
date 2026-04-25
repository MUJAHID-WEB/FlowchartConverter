# FlowchartConverter

Automatically converts SVG/image flowcharts into structured Functional Requirements (FRs) using a six-phase pipeline. Outputs a downloadable PDF report with traceability metrics.

---

## Prerequisites

Before running the project, install the following tools:

| Tool | Purpose | Download |
|------|---------|----------|
| Python 3.9+ | Runtime | https://www.python.org/downloads/ |
| Tesseract OCR | Text extraction from images | https://github.com/UB-Mannheim/tesseract/wiki |
| Ollama | Local LLM server | https://ollama.com/download |

> **Important:** Install Tesseract to the default path `C:\Program Files\Tesseract-OCR\`. If you use a different path, update line 35 in `flowchart_converter.py`.

---

## Step-by-Step Setup

### Step 1 — Clone or download the project

Place the project folder somewhere on your machine, e.g.:
```
C:\Users\YourName\Desktop\FlowchartConverter\
```

### Step 2 — Create and activate a virtual environment

Open a terminal in the project folder:

```bash
python -m venv venv
venv\Scripts\activate
```

Your prompt should now show `(venv)` at the start.

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install and verify Tesseract OCR

1. Download the Windows installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer and keep the default install path (`C:\Program Files\Tesseract-OCR\`)
3. Verify the installation:

```bash
"C:\Program Files\Tesseract-OCR\tesseract.exe" --version
```

### Step 5 — Install Ollama and pull the Mistral model

1. Download and install Ollama from https://ollama.com/download
2. Open a terminal and pull the Mistral model:

```bash
ollama pull mistral
```

3. Verify Ollama is working:

```bash
ollama list
```

You should see `mistral` in the list.

> **Memory tip:** If your machine has limited RAM and Mistral fails, try a smaller model:
> ```bash
> ollama pull llama2
> # or
> ollama pull neural-chat
> ```
> Then select that model in the app's model dropdown before processing.

---

## Running the Application

Make sure your virtual environment is active, then run:

```bash
python flowchart_converter.py
```

The GUI window will open automatically.

---

## Using the Application

### 1. Upload and process files

- Click **"Upload & Process Files"**
- Select one or more flowchart files (supported formats: `.svg`, `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.gif`, `.zip`)
- The app will extract text, build a Mermaid graph, and generate Functional Requirements using Ollama
- Progress and accuracy scores are shown in the log area

### 2. Change the Ollama model (optional)

- Use the **model dropdown** to select a different model (e.g., `llama2`, `neural-chat`)
- Click **"Refresh Models"** to load models currently available in your Ollama installation

### 3. View results

- Click **"View Results"** to see a summary of all processed files and their accuracy scores

### 4. Generate a PDF report

- Click **"Generate PDF Report"**
- Choose a save location — the PDF includes:
  - Executive summary
  - Accuracy analysis (text extraction, structure detection, FR generation)
  - File-by-file results with Mermaid code
  - Full Functional Requirements
  - Recommendations

### 5. Clear results

- Click **"Clear Results"** to reset and start a new batch

---

## Accuracy Metrics

| Metric | Weight | Description |
|--------|--------|-------------|
| `Qtxt` — Text Quality | 30% | Quality of OCR / SVG text extraction |
| `Qstr` — Structure Quality | 40% | Accuracy of Mermaid graph generation |
| `Qfr` — FR Accuracy | 30% | Relevance of generated requirements |

**Overall:** `Qoverall = 0.30·Qtxt + 0.40·Qstr + 0.30·Qfr`

---

## How It Works (Pipeline Overview)

1. **Input Validation** — Validates file format and integrity
2. **Content Extraction** — Parses SVG XML or runs Tesseract OCR on images
3. **Structure Detection** — Identifies ovals (start/end), rectangles (processes), diamonds (decisions); builds directed graph `G = (V, E)`
4. **Mermaid Code Generation** — Converts graph to Mermaid `graph TD` syntax
5. **LLM Composition** — Mistral generates FRs using the Mermaid code, extracted text, and structure description; enforces "shall" modal verbs and atomicity
6. **PDF Report** — Generates report with traceability matrix and quality scores

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Tesseract not found` | Install Tesseract or update the path on line 35 of `flowchart_converter.py` |
| `Ollama not running` | Run `ollama serve` in a separate terminal, then restart the app |
| `No models available` | Run `ollama pull mistral` in a terminal |
| Mistral crashes (out of memory) | Switch to `llama2` or `neural-chat` in the model dropdown |
| Empty FR output | Ensure the flowchart contains readable text labels in the nodes |
