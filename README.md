# AI-AOL: A Hybrid Symbolic–Neural Agentic Architecture for Autonomous Music Theory Analysis, Pedagogical Adaptation, and Guided Knowledge Retrieval from Optical Music Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black.svg)](https://ollama.ai/)

> **An autonomous, goal-driven AI system that functions as an intelligent music theory tutor — analyzing sheet music through optical recognition, performing structured multi-agent theory interpretation with iterative self-critique, adapting explanations to student proficiency, and generating targeted exercises — all with minimal human intervention.**

**Authors**: Weneville, Nathanael Romaloburju, and Robben Wijanathan

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Overview](#-overview)
- [What Makes This System "Agentic"?](#-what-makes-this-system-agentic)
- [System Architecture](#-system-architecture)
- [Agent Descriptions](#-agent-descriptions)
- [Core Workflow](#-core-workflow)
- [Technology Stack](#-technology-stack)
- [Design Principles](#-design-principles)
- [Research Contribution](#-research-contribution)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🔬 Problem Statement

### The Challenge of Scalable Music Theory Education

Music theory comprehension is an inherently **multi-dimensional cognitive task**. Mastery requires simultaneous understanding of:

| Domain | Examples |
|---|---|
| **Tonal Structure** | Key signatures, scales, modes |
| **Harmony** | Chord functions (I, IV, V, ii, vi…), inversions, voice leading |
| **Progression & Cadence** | Authentic, half, deceptive, and plagal cadences |
| **Modulation** | Pivot chords, direct modulation, modal interchange |
| **Rhythm & Meter** | Time signatures, syncopation, tempo relationships |
| **Style & Interpretation** | Period-specific conventions, genre idioms |

Traditional one-on-one tutoring remains the gold standard for music theory instruction, yet it introduces **systemic constraints** that limit accessibility:

- **High cost** — Private music theory tutors charge \$40–\$120+/hour, placing sustained instruction out of reach for many students.
- **Scheduling rigidity** — Synchronous sessions require fixed time commitments from both parties.
- **Geographic dependency** — Quality instruction concentrates in urban centers and university towns.
- **Scalability ceiling** — A single tutor serves one student at a time; demand vastly exceeds supply.
- **Inconsistent pedagogical adaptation** — Even expert tutors vary in their ability to diagnose and respond to individual student misconceptions in real time.

Existing AI-assisted music education tools fall short because they either:

1. **Rely on single-pass LLM inference** — producing plausible but unverified, hallucination-prone explanations with no self-critique mechanism.
2. **Lack symbolic grounding** — treating music theory as free-form text generation rather than a formal rule-governed system.
3. **Ignore pedagogical modeling** — delivering uniform explanations regardless of student proficiency.
4. **Cannot process sheet music directly** — requiring manual transcription or pre-existing digital scores.

### Our Proposal

We propose **AI-AOL** — an **autonomous agentic AI tutor** that closes these gaps through a novel hybrid architecture combining:

- **Optical Music Recognition (OMR)** for direct sheet music intake via image upload.
- **Deterministic symbolic music theory analysis** for rule-based, verifiable harmonic interpretation.
- **Multi-role agentic reasoning** over a shared LLM backbone, where specialized agents iteratively hypothesize, critique, and refine theoretical interpretations until confidence thresholds are met.
- **Guided Retrieval-Augmented Generation (RAG)** grounded in structured domain knowledge to minimize hallucination.
- **Pedagogical adaptation modeling** that dynamically adjusts explanation complexity and generates targeted exercises based on declared student level.

The result is a system that delivers **expert-level, self-correcting music theory analysis and instruction** — running entirely offline on consumer hardware (16 GB RAM, single 5–7B parameter quantized model).

---

## 🎵 Overview

**AI-AOL (Enhanced Agentic Music Notation Analyzer)** is an autonomous, goal-driven AI system designed to function as an intelligent music theory tutor.

The system:

1. **Accepts** sheet music images (PNG, JPG, PDF) uploaded by students.
2. **Recognizes** musical notation using Optical Music Recognition (oemer) and converts it to structured MusicXML.
3. **Analyzes** the score through a deterministic symbolic music theory engine (key detection, chord function labeling, cadence identification, modulation detection).
4. **Interprets** the analysis via a Hypothesis Agent that proposes structured theoretical interpretations with confidence scores.
5. **Critiques** interpretations through a Critique Agent that validates tonal coherence, flags rule violations, and considers alternative readings.
6. **Iterates** through hypothesis–critique loops until confidence thresholds are satisfied or iteration limits are reached.
7. **Retrieves** domain-specific knowledge through a Guided RAG layer backed by a curated SQLite knowledge base.
8. **Adapts** explanations and generates exercises through a Pedagogical Agent tuned to the student's declared proficiency level.
9. **Plays back** the analyzed sheet music as audio for aural reinforcement.

All agents share a **single LLM backbone** (5–7B parameter quantized model via Ollama) but operate with **distinct role prompts and structured JSON output schemas**, enabling complex multi-perspective reasoning without multi-model overhead.

---

## 🤖 What Makes This System "Agentic"?

An **agentic system** is one that is:

| Property | How AI-AOL Implements It |
|---|---|
| **Goal-directed** | Each agent pursues a specific objective (hypothesize, critique, teach) |
| **Autonomous** | After a single trigger (upload), the system runs to completion without step-by-step human control |
| **Self-evaluating** | The Critique Agent scores and challenges the Hypothesis Agent's outputs |
| **Self-refining** | Low-confidence interpretations trigger re-analysis loops |
| **Knowledge-seeking** | The RAG layer is queried dynamically based on analysis needs |
| **Adaptive** | The Pedagogical Agent modulates output complexity per student level |

> **Key distinction**: AI-AOL does **not** rely on a single-pass LLM response. It implements **iterative role-based reasoning** where multiple logical agents collaborate, critique, and refine outputs over a shared model — embodying the core principles of agentic AI.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│                     React Frontend (Web)                         │
└──────────────────────┬───────────────────────────────────────────┘
                       │  Upload sheet music image
                       │  Select difficulty level
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND (API)                        │
│              Request handling, session management                │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (LangGraph)                       │
│         Controls execution flow, enforces iteration limits,      │
│         manages confidence thresholds, coordinates agents         │
│                     "CEO Agent"                                  │
└───┬──────────┬──────────┬──────────┬──────────┬─────────────────┘
    │          │          │          │          │
    ▼          ▼          ▼          ▼          ▼
┌────────┐┌────────┐┌──────────┐┌────────┐┌──────────────┐
│  OMR   ││ Theory ││ Critique ││  RAG   ││ Pedagogical  │
│ Module ││ Agent  ││  Agent   ││ Layer  ││   Agent      │
│(oemer) ││        ││          ││(SQLite)││              │
└───┬────┘└───┬────┘└────┬─────┘└───┬────┘└──────┬───────┘
    │         │          │          │             │
    │         └────┬─────┘          │             │
    │              │ Iterative      │             │
    │              │ Refinement     │             │
    │              │ Loop           │             │
    │              ▼                │             │
    │    ┌─────────────────┐       │             │
    │    │  Shared LLM     │◄──────┘             │
    │    │  (Ollama 5-7B)  │                     │
    │    │  Single backbone │                     │
    │    └─────────────────┘                     │
    │                                            │
    ▼                                            ▼
┌────────────────┐                 ┌──────────────────────┐
│  MusicXML      │                 │  Adapted Explanation │
│  Structured    │                 │  Practice Exercises  │
│  Notation      │                 │  Audio Playback      │
└────────────────┘                 └──────────────────────┘
```

### Iterative Hypothesis–Critique Loop

```
                    ┌─────────────────────┐
                    │  Symbolic Analysis   │
                    │  (Deterministic)     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
              ┌────▶│  Hypothesis Agent   │
              │     │  Proposes interpret. │
              │     │  + confidence score  │
              │     └──────────┬──────────┘
              │                │
              │                ▼
              │     ┌─────────────────────┐
              │     │   Critique Agent    │
              │     │  Validates logic    │
              │     │  Flags violations   │
              │     │  Updates confidence │
              │     └──────────┬──────────┘
              │                │
              │                ▼
              │        Confidence ≥ θ ?
              │         /          \
              │       No            Yes
              │       /              \
              └──────┘        ┌──────────────┐
           (max iterations    │ Accepted     │
            enforced)         │ Interpretation│
                              └──────────────┘
```

---

## 🧠 Agent Descriptions

### 1. OMR Module

| Attribute | Detail |
|---|---|
| **Tool** | [oemer](https://github.com/BreezeWhite/oemer) |
| **Input** | Sheet music image (PNG, JPG, PDF) |
| **Output** | Structured notation (MusicXML or equivalent symbolic format) |
| **Role** | Converts visual notation into machine-readable symbolic representation |
| **Type** | Deterministic (non-LLM) |

### 2. Theory (Hypothesis) Agent

| Attribute | Detail |
|---|---|
| **Goal** | Propose harmonic and theoretical interpretations |
| **Identifies** | Key signature, tempo, chord functions, cadences, possible modulations |
| **Output** | Structured JSON with interpretation + confidence score |
| **Prompt Role** | `"You are an expert music theorist. Analyze the following score data..."` |
| **Type** | LLM-powered (shared backbone) |

### 3. Critique Agent

| Attribute | Detail |
|---|---|
| **Goal** | Evaluate theoretical consistency and correctness |
| **Validates** | Tonal coherence, rule violations, alternative interpretations |
| **Action** | Refines or rejects hypotheses; produces updated confidence score |
| **Prompt Role** | `"You are a rigorous music theory reviewer. Evaluate this analysis..."` |
| **Type** | LLM-powered (shared backbone) |

### 4. Pedagogical Agent

| Attribute | Detail |
|---|---|
| **Goal** | Adapt explanation to student proficiency level |
| **Evaluates** | Difficulty of material relative to student level |
| **Generates** | Level-adjusted explanation, practice exercises, reinforcement suggestions |
| **Prompt Role** | `"You are a music theory tutor. Explain this analysis to a [level] student..."` |
| **Type** | LLM-powered (shared backbone) |

### 5. Retrieval-Augmented Generation (RAG) Layer

| Attribute | Detail |
|---|---|
| **Purpose** | Reduce hallucination and inject domain-specific knowledge |
| **Storage** | SQLite database |
| **Retrieval Type** | Structured lookup (primary); vector similarity search (optional extension) |
| **Contains** | Formal music theory definitions, style-specific harmonic traits, common student misconceptions, pedagogical explanation templates |
| **Type** | Deterministic retrieval + LLM-augmented generation |

### 6. Orchestrator ("CEO Agent")

| Attribute | Detail |
|---|---|
| **Tool** | LangGraph (structured graph-based execution) |
| **Controls** | Agent call order, iteration limits, confidence thresholds |
| **Role** | Coordination controller — ensures the pipeline runs autonomously from trigger to completion |
| **Enforces** | Maximum refinement loops, minimum confidence for acceptance |

---

## 🔄 Core Workflow

```
Step 1 ─── User uploads sheet music image via React frontend
               │
Step 2 ─── OMR Module (oemer) scans and converts to MusicXML
               │
Step 3 ─── Symbolic Theory Engine performs deterministic analysis
           (key detection, chord labeling, cadence ID — no LLM)
               │
Step 4 ─── Hypothesis Agent proposes structured interpretation
           with confidence score (LLM-powered)
               │
Step 5 ─── Critique Agent validates consistency and confidence
           (LLM-powered, different role prompt)
               │
        ┌──── Confidence < threshold? ──── Yes ──→ Return to Step 4
        │                                          (max iterations enforced)
        No
        │
Step 6 ─── Guided RAG retrieves domain knowledge from SQLite
           to ground and enrich the accepted interpretation
               │
Step 7 ─── Pedagogical Agent adapts explanation to student level
           and generates targeted exercises
               │
Step 8 ─── Audio playback engine renders the analyzed music
               │
Step 9 ─── Results delivered to frontend:
           • Theory analysis (structured JSON)
           • Adapted explanation (natural language)
           • Practice exercises
           • Audio playback
```

---

## 🛠️ Technology Stack

### Backend

| Component | Technology | Purpose |
|---|---|---|
| **Web Framework** | Python 3.11+ / FastAPI | High-performance API endpoints, async support |
| **Agent Orchestration** | LangGraph | Graph-based agent execution flow, iteration control |
| **LLM Runtime** | Ollama (local) | Serves 5–7B parameter quantized model |
| **LLM Model** | Shared single backbone (e.g., Mistral 7B, Llama 3 8B) | Powers all agents via role-separated prompts |
| **OMR Engine** | oemer | Optical Music Recognition → MusicXML |
| **RAG Database** | SQLite | Structured domain knowledge storage |
| **Music Analysis** | music21 | Deterministic symbolic music theory analysis |

### Frontend

| Component | Technology | Purpose |
|---|---|---|
| **UI Framework** | React 18+ | Interactive web interface |
| **State Management** | React Context / Zustand | Application state |
| **Audio Playback** | Web Audio API / Tone.js | In-browser music rendering |

### Infrastructure

| Component | Technology | Purpose |
|---|---|---|
| **Containerization** | Docker & Docker Compose | Reproducible deployment |
| **Database** | SQLite | Lightweight, zero-config (RAG + app data) |
| **Reverse Proxy** | Nginx | Production serving |

### Architecture Notes

- **Single LLM instance** — all agents share one model, differentiated by prompt engineering and output schemas.
- **Structured JSON communication** between agents — enforced schemas prevent free-form drift.
- **Deterministic symbolic layer** (music21) — non-LLM music theory validation provides ground truth.
- **Controlled iteration** — maximum refinement loops prevent infinite agent cycling.
- **Confidence-based evaluation** — numerical thresholds gate progression through the pipeline.

---

## 🎯 Design Principles

| Principle | Implementation |
|---|---|
| **One shared LLM backbone** | Single Ollama-served model reduces resource usage |
| **Multiple role-separated agents** | Distinct system prompts + JSON schemas per agent |
| **Deterministic symbolic validation** | music21 provides non-LLM ground truth |
| **Guided retrieval over blind RAG** | Structured SQLite lookups, not unfiltered vector search |
| **Minimal hallucination risk** | Symbolic grounding + critique loop + domain RAG |
| **Offline-capable** | No external API calls; runs fully local |
| **Optimized for 16 GB RAM** | Single 5–7B quantized model, SQLite, no GPU required |

---

## 📐 Research Contribution

### Thesis

> We propose **a hybrid symbolic–neural agentic architecture** for structured music theory analysis and pedagogical adaptation, demonstrating that iterative multi-role reasoning over a single lightweight LLM backbone — grounded by deterministic symbolic validation and guided domain retrieval — can produce expert-level, self-correcting music theory instruction from raw sheet music images.

### Key Innovation Points

1. **Iterative hypothesis–critique validation loop** — Multiple logical agents collaborate to propose, evaluate, and refine theoretical interpretations, moving beyond single-pass LLM generation.

2. **Confidence-based refinement gating** — Numerical confidence thresholds control iteration depth, balancing analysis quality against computational cost.

3. **Structured RAG for formal domain knowledge** — Domain-specific retrieval from a curated knowledge base, privileging structured lookup over noisy vector similarity to minimize hallucination in a formal rule-governed domain.

4. **Role-separated reasoning over a shared model backbone** — Demonstrates that meaningful multi-perspective agentic behavior can emerge from prompt engineering alone, without requiring multiple specialized models.

5. **Hybrid symbolic–neural pipeline** — Deterministic music theory analysis (music21) provides verifiable ground truth that constrains and validates LLM-generated interpretations, creating a feedback loop between formal rules and neural reasoning.

6. **Adaptive difficulty-aware explanation generation** — Pedagogical modeling layer that dynamically adjusts explanation complexity, exercise difficulty, and terminology based on declared student proficiency.

7. **End-to-end from image to instruction** — Complete pipeline from raw sheet music photograph to structured analysis, adapted explanation, exercises, and audio playback — requiring no manual transcription or pre-existing digital scores.

---

## 📦 Prerequisites

### Minimum System Requirements

- **RAM**: 16 GB (minimum for running quantized 5–7B LLM locally)
- **Disk**: 10 GB free space (model weights + dependencies)
- **OS**: Linux, macOS, or Windows with WSL2
- **Docker** (version 20.10+) and **Docker Compose** (version 2.0+)
- **Git** for cloning the repository

### For Local Development (without Docker)

- **Python 3.11+**
- **Node.js 18+** (for React frontend)
- **Ollama** (installed and running locally)
- A downloaded quantized model (e.g., `ollama pull mistral` or `ollama pull llama3`)

---

## 🚀 Installation

### Quick Start with Docker Compose

```bash
# 1. Clone the repository
git clone https://github.com/CS-Math-Class-Group/AI-AOL.git
cd AI-AOL

# 2. Create environment configuration
cp .env.example .env
# Edit .env with your configuration (see Configuration section)

# 3. Start all services
docker-compose up -d

# 4. Verify services are running
docker-compose ps

# 5. Access the application
# Frontend UI:    http://localhost:3000
# FastAPI API:    http://localhost:8000/docs
```

### Manual Installation (Development)

#### Backend (FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

#### Frontend (React)

```bash
cd frontend
npm install
npm start
```

#### Ollama (Local LLM)

```bash
# Install Ollama (https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull mistral    # or: ollama pull llama3

# Ollama runs automatically as a service on port 11434
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env

# FastAPI Backend
# (Add any FastAPI-specific environment variables here)

# Ollama LLM Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=mistral          # or llama3, phi3, etc.

# Agent Configuration
CONFIDENCE_THRESHOLD=0.75     # Minimum confidence to accept interpretation
MAX_REFINEMENT_ITERATIONS=3   # Maximum hypothesis-critique loops
STUDENT_DIFFICULTY_DEFAULT=intermediate  # beginner | intermediate | advanced

# RAG Database
RAG_DB_PATH=./data/rag_knowledge.db

# Storage
UPLOAD_PATH=./data/uploads
OUTPUT_PATH=./data/outputs

# Frontend
REACT_APP_API_URL=http://localhost:8000/api
```

---

## 📖 Usage

### Web Interface

1. **Navigate to** http://localhost:3000
2. **Upload sheet music** — drag & drop or click to select an image (PNG, JPG, PDF)
3. **Select your proficiency level** — Beginner, Intermediate, or Advanced
4. **Click "Analyze"** — the agentic pipeline processes your score autonomously
5. **Review results**:
   - 📊 **Theory Analysis** — key, chords, cadences, modulations (structured)
   - 📝 **Adapted Explanation** — natural language at your level
   - ✏️ **Practice Exercises** — targeted to reinforce concepts found in the score
   - 🔊 **Audio Playback** — listen to the analyzed music

### API Usage

#### Upload and Analyze Sheet Music

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@path/to/sheet-music.png" \
  -F "student_level=intermediate" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

Response:

```json
{
  "job_id": "uuid-here",
  "status": "processing",
  "message": "Sheet music uploaded. Agentic analysis started."
}
```

#### Check Analysis Status

```bash
curl -X GET http://localhost:8000/api/v1/status/uuid-here \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

Response:

```json
{
  "job_id": "uuid-here",
  "status": "completed",
  "iterations": 2,
  "final_confidence": 0.87,
  "results": {
    "theory_analysis": { "key": "C major", "chords": [...], "cadences": [...] },
    "explanation": "This piece is in C major and uses a I-IV-V-I progression...",
    "exercises": [...],
    "audio_url": "/api/v1/playback/uuid-here"
  }
}
```

---

## 🔌 API Documentation

### Authentication

```
Authorization: Bearer YOUR_API_TOKEN
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/analyze` | Upload sheet music and start agentic analysis |
| `GET` | `/api/v1/status/:job_id` | Check processing status and retrieve results |
| `GET` | `/api/v1/playback/:job_id` | Stream audio playback of analyzed score |
| `GET` | `/api/v1/jobs` | List all analysis jobs for authenticated user |
| `DELETE` | `/api/v1/jobs/:job_id` | Delete a job and its associated data |

### WebSocket Support

Real-time progress updates during agentic processing:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/analysis/:job_id');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Stage:', data.stage);        // e.g., "critique_agent"
  console.log('Iteration:', data.iteration); // e.g., 2
  console.log('Confidence:', data.confidence); // e.g., 0.82
};
```

---

## 📁 Project Structure

```
AI-AOL/
├── README.md                       # This file
├── docker-compose.yml              # Docker orchestration
├── .env.example                    # Environment template
├── .gitignore
│
├── backend/                        # FastAPI Backend
│   ├── manage.py
│   ├── requirements.txt
│   ├── main.py                     # FastAPI entrypoint
│   │
│   ├── omr/                        # OMR Module (oemer integration)
│   │   ├── scanner.py              # Image → MusicXML conversion
│   │   └── preprocessor.py         # Image preprocessing
│   │
│   ├── theory/                     # Symbolic Music Theory Engine
│   │   ├── analyzer.py             # Deterministic analysis (music21)
│   │   ├── key_detector.py         # Key signature detection
│   │   ├── chord_labeler.py        # Chord function labeling
│   │   └── cadence_identifier.py   # Cadence identification
│   │
│   ├── agents/                     # Agentic AI Layer
│   │   ├── orchestrator.py         # LangGraph orchestrator (CEO agent)
│   │   ├── hypothesis_agent.py     # Theory interpretation agent
│   │   ├── critique_agent.py       # Validation & critique agent
│   │   ├── pedagogical_agent.py    # Teaching & exercise agent
│   │   ├── prompts/                # Role-specific system prompts
│   │   │   ├── hypothesis.txt
│   │   │   ├── critique.txt
│   │   │   └── pedagogical.txt
│   │   └── schemas/                # JSON output schemas per agent
│   │       ├── hypothesis_schema.json
│   │       ├── critique_schema.json
│   │       └── pedagogical_schema.json
│   │
│   ├── rag/                        # RAG Knowledge Layer
│   │   ├── knowledge_base.py       # SQLite retrieval interface
│   │   ├── seed_data.py            # Knowledge base seeder
│   │   └── data/
│   │       └── rag_knowledge.db    # Curated music theory knowledge
│   │
│   ├── audio/                      # Audio Playback Engine
│   │   ├── renderer.py             # MusicXML → audio conversion
│   │   └── player.py               # Streaming playback API
│   │
│   └── api/                        # REST API Layer
│       ├── views.py
│       ├── serializers.py
│       └── urls.py
│
├── frontend/                       # React Frontend
│   ├── package.json
│   ├── public/
│   └── src/
│       ├── App.jsx
│       ├── components/
│       │   ├── UploadPanel.jsx     # Sheet music upload
│       │   ├── AnalysisView.jsx    # Theory analysis display
│       │   ├── ExplanationView.jsx # Adapted explanation
│       │   ├── ExercisePanel.jsx   # Practice exercises
│       │   └── AudioPlayer.jsx     # Music playback
│       └── services/
│           └── api.js              # API client
│
├── data/                           # Runtime data (gitignored)
│   ├── uploads/                    # Uploaded sheet music
│   ├── outputs/                    # Generated results
│   └── rag_knowledge.db            # RAG database
│
├── nginx/                          # Nginx configuration
│   └── nginx.conf
│
└── docs/                           # Additional documentation
    ├── ARCHITECTURE.md
    ├── AGENTS.md
    ├── RAG_SCHEMA.md
    └── CONTRIBUTING.md
```

---

## 💻 Development

### Running Tests

```bash
# Backend tests
cd backend
python manage.py test

# Frontend tests
cd frontend
npm test
```

### Agent Development

Each agent is defined by three components:

1. **Role prompt** (`backend/agents/prompts/`) — system message defining the agent's persona and objective.
2. **Output schema** (`backend/agents/schemas/`) — JSON schema enforcing structured output.
3. **Agent logic** (`backend/agents/`) — Python module handling input preparation, LLM invocation, and output parsing.

To add a new agent:

```bash
# 1. Create the prompt file
echo "You are a [role]. Your goal is..." > backend/agents/prompts/new_agent.txt

# 2. Define the output schema
cat > backend/agents/schemas/new_agent_schema.json << 'EOF'
{ "type": "object", "properties": { ... }, "required": [...] }
EOF

# 3. Implement the agent logic
touch backend/agents/new_agent.py

# 4. Register in the orchestrator graph (backend/agents/orchestrator.py)
```

### Adding New Features

1. **Create a feature branch**: `git checkout -b feature/your-feature-name`
2. **Implement changes** following existing code patterns
3. **Write tests** for new functionality
4. **Run the full test suite**: `python manage.py test && cd ../frontend && npm test`
5. **Submit a pull request** with a clear description

---

## 🗺️ Roadmap

### Completed

- [x] Project architecture design
- [x] OMR module integration (oemer)
- [x] Symbolic music theory engine (music21)
- [x] Multi-agent hypothesis–critique loop
- [x] Guided RAG knowledge layer
- [x] Pedagogical adaptation agent
- [x] Audio playback engine
- [x] React frontend

### In Progress

- [ ] Confidence calibration tuning
- [ ] RAG knowledge base expansion
- [ ] End-to-end integration testing

### Future Extensions

- [ ] **Student learning memory persistence** — track progress across sessions
- [ ] **Difficulty progression tracking** — automatic level advancement
- [ ] **Multi-style harmonic comparison** — Baroque vs. Classical vs. Jazz analysis
- [ ] **Complexity scoring metrics** — quantitative difficulty assessment
- [ ] **Performance analytics dashboard** — learning outcome visualization
- [ ] **Multi-page sheet music processing** — handle full scores
- [ ] **Vector similarity search** for RAG — complement structured lookup
- [ ] **Mobile application** — on-the-go sheet music scanning

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details on our code of conduct and development process.

### Guidelines

- Follow existing code style and patterns
- Write tests for all new features
- Update documentation (including agent prompts/schemas)
- Keep commits atomic and well-described
- Ensure all tests pass before submitting PR

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Authors

- **Weneville** — Project Lead & System Architect
- **Nathanael Romaloburju** — Core Developer
- **Robben Wijanathan** — Core Developer

### Technologies & Libraries

- [FastAPI](https://fastapi.tiangolo.com/) — Python web framework
- [LangGraph](https://github.com/langchain-ai/langgraph) — Agent orchestration framework
- [Ollama](https://ollama.ai/) — Local LLM runtime
- [oemer](https://github.com/BreezeWhite/oemer) — Optical Music Recognition
- [music21](http://web.mit.edu/music21/) — Music theory analysis toolkit
- [React](https://reactjs.org/) — Frontend UI framework
- [SQLite](https://www.sqlite.org/) — Embedded database

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/CS-Math-Class-Group/AI-AOL/issues)
- **Discussions**: [GitHub Discussions](https://github.com/CS-Math-Class-Group/AI-AOL/discussions)
- **Email**: Contact the maintainers through GitHub

---

**Made with ❤️ by the AI-AOL Team**
