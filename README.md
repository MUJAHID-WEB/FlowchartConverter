# FlowchartConverter

Automatically converts SVG flowchart-based use cases into structured Functional Requirements (FRs) through a six-phase pipeline that preserves control-flow semantics and ensures traceability.

## How It Works

### Phase 1: Input Validation
- Validates SVG format, resolution, and structural integrity
- Preprocesses the artefact for consistent processing

### Phase 2: Content Extraction
- Parses SVG XML to extract embedded text labels from nodes
- Captures geometric information (node positions, shapes, connector paths)
- Outputs token sequence: `Ti = {ti1, ti2, ..., tim}`

### Phase 3: Structure Detection & Normalization
- Recognizes flowchart shapes: ovals (start/end), rectangles (processes), diamonds (decisions)
- Constructs directed graph `Gi = (Vi, Ei)` where `Vi` = action/decision nodes, `Ei` = control-flow transitions
- Normalizes structure by standardizing element types and validating flow consistency

### Phase 4: Mermaid Code Generation
- Converts normalized graph into Mermaid syntax representation
- Serves as standardized intermediate format for requirement generation

### Phase 5: Constrained LLM Composition
Uses a local Ollama model (Mistral) with:
- Normalized graph structure
- Mermaid code as context
- Extracted text labels

Enforces constraints: atomicity, "shall" modal verbs, traceability to source nodes.

Each FR is computed as `FR = f(vk, context(vk))` where `vk` is an action node.

### Phase 6: Report Creation
- Generates PDF with traceability matrix linking FRs to source elements
- Reports quality metrics: `Qtxt` (text extraction), `Qstr` (structure recovery), `Qfr` (FR correctness)

## Key Formulas

| Metric | Formula |
|--------|---------|
| Structure Accuracy | `Si = |E_detected ∩ E_reference| / |E_reference|` |
| Overall Quality | `Qoverall = 0.24·Qtxt + 0.26·Qstr + 0.50·Qfr` |

## Example

**Input:** SVG flowchart with a "Check Book Availability" decision node

**Output FR:** _"The system shall verify book availability in the catalog before allowing a borrowing transaction."_

## Key Advantage

Separates structure detection from Mermaid generation, enabling independent validation of control-flow recovery accuracy before requirement composition.

## Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) running locally with the Mistral model pulled

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Pull the Mistral model:

```bash
ollama pull mistral
```

## Usage

```bash
python flowchart_converter.py <path-to-svg>
```
