# AI-AOL: Optical Music Recognition with Agentic AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent Optical Music Recognition (OMR) system that uses Agentic AI to transform sheet music into live coding languages and MIDI files in real-time.

**Created by**: Weneville, Nathanael Romaloburju, Robben Wijanathan, and Evorius Valens Taruna

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Workflow](#workflow)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🎵 Overview

AI-AOL is an advanced Optical Music Recognition (OMR) system that leverages Agentic AI methodologies to convert sheet music into executable music code. The system processes uploaded sheet music images, analyzes musical notation using computer vision and AI, and generates real-time music code compatible with platforms like Sonic Pi, as well as standard MIDI files.

### What is OMR?

Optical Music Recognition (OMR) is the technology that automatically reads and interprets musical notation from images or scanned documents, converting them into machine-readable formats.

### What are Agentic AI Methods?

Agentic AI refers to artificial intelligence systems that can act autonomously, make decisions, and execute tasks with minimal human intervention. In this project, agentic AI orchestrates the entire pipeline from image recognition to code generation.

---

## ✨ Features

- **🎼 Sheet Music Recognition**: Upload and process sheet music images in various formats (PNG, JPG, PDF)
- **🤖 Agentic AI Processing**: Intelligent agents coordinate the conversion pipeline
- **🎹 Live Music Code Generation**: Generate real-time music code compatible with:
  - Sonic Pi (live coding language)
  - MIDI files for DAWs and music software
  - Other music programming formats
- **⚡ Real-time Processing**: Fast processing pipeline for near-instant results
- **🔄 Workflow Automation**: n8n integration for complex workflow orchestration
- **🐳 Containerized Architecture**: Docker Compose for easy deployment and scalability
- **🌐 RESTful API**: Go-based API for high-performance request handling
- **🧠 LLM Integration**: Large Language Models for intelligent music interpretation
- **📊 Web Interface**: Django-based admin and user interface

---

## 🏗️ Architecture

The system follows a microservices architecture with the following components:

```
┌─────────────────┐
│   User Upload   │
│  (Sheet Music)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   n8n Workflow  │
│   Orchestrator  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Go/Gin API    │
│  (API Gateway)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Python/Django  │
│   (OMR Engine)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Agentic AI     │
│  (LLM + Agents) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Code Generator │
│ (Sonic Pi/MIDI) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Files   │
│  (.rb, .mid)    │
└─────────────────┘
```

### Component Details

The system follows a **microservices architecture** where each service is independently deployable and communicates through well-defined APIs:

1. **n8n Workflow Orchestrator**: Manages the entire pipeline, handles file uploads, and coordinates between microservices through HTTP/REST APIs and webhooks
2. **Go/Gin API Gateway** (Microservice): 
   - Acts as the single entry point for all client requests (API Gateway pattern)
   - High-performance reverse proxy and request router
   - Handles authentication, rate limiting, and request validation
   - Routes requests to appropriate microservices (Python OMR, AI agents)
   - Implements service discovery and load balancing
3. **Python/Django OMR Engine** (Microservice):
   - Independent service focused solely on optical music recognition
   - Exposes REST API endpoints consumed by the API Gateway
   - Core OMR processing using computer vision and machine learning models
   - Containerized with its own database and dependencies
4. **Agentic AI Layer** (Microservice):
   - Separate service for LLM-powered agents
   - Interprets musical structure and generates code
   - Can be scaled independently based on AI processing demand
5. **Code Generator** (Microservice):
   - Dedicated service for converting interpreted music into executable formats
   - Generates Sonic Pi and MIDI files
   - Maintains its own output processing pipeline

---

## 🛠️ Technology Stack

### Backend Services (Microservices Architecture)
- **Go 1.21+** with **Gin** framework - API Gateway microservice
  - Service discovery and routing
  - Authentication and authorization
  - Request aggregation and transformation
  - Rate limiting and circuit breaking
- **Python 3.11+** with **Django 4.2+** - OMR Processing microservice
  - Independent service with REST API endpoints
  - Computer vision and ML model serving
  - Separate database instance
  - Horizontally scalable
- **n8n** - Workflow orchestration layer
  - Coordinates microservices communication
  - Implements saga pattern for distributed transactions
  - Event-driven architecture support

### AI & Machine Learning
- **Large Language Models (LLMs)** - GPT-4, Claude, or open-source alternatives
- **OpenCV** - Computer vision for image processing
- **TensorFlow/PyTorch** - Deep learning for music notation recognition
- **Music21** - Music theory and notation analysis

### Infrastructure
- **Docker** & **Docker Compose** - Containerization and orchestration
- **PostgreSQL** - Primary database
- **Redis** - Caching and message queue
- **Nginx** - Reverse proxy and load balancing

### Output Formats
- **Sonic Pi** - Live coding language for music
- **MIDI** - Standard music file format
- **MusicXML** - Sheet music exchange format

---

## 📦 Prerequisites

Before you begin, ensure you have the following installed:

- **Docker** (version 20.10+) and **Docker Compose** (version 2.0+)
- **Git** for cloning the repository
- At least **8GB RAM** recommended
- **10GB free disk space** for images and dependencies

Optional for local development:
- **Go 1.21+** for Go service development
- **Python 3.11+** for Django development
- **Node.js 18+** for n8n customization

---

## 🚀 Installation

### Quick Start with Docker Compose

1. **Clone the repository**
   ```bash
   git clone https://github.com/Wenev/AI-AOL.git
   cd AI-AOL
   ```

2. **Create environment configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start all services**
   ```bash
   docker-compose up -d
   ```

4. **Verify services are running**
   ```bash
   docker-compose ps
   ```

5. **Access the services**
   - n8n Workflow UI: http://localhost:5678
   - Go API Gateway: http://localhost:8080
   - Django Admin: http://localhost:8000/admin
   - Frontend UI: http://localhost:3000

### Manual Installation (Development)

#### Go API Service

```bash
cd services/api-gateway
go mod download
go build -o api-gateway
./api-gateway
```

#### Python/Django OMR Service

```bash
cd services/omr-engine
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

#### n8n Setup

```bash
cd services/n8n
npm install -g n8n
n8n start
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# API Gateway (Go/Gin)
API_PORT=8080
API_SECRET_KEY=your-secret-key-here
CORS_ORIGINS=http://localhost:3000

# Django OMR Engine
DJANGO_SECRET_KEY=your-django-secret-key
DJANGO_DEBUG=False
DATABASE_URL=postgresql://user:password@postgres:5432/omr_db

# n8n Configuration
N8N_BASIC_AUTH_ACTIVE=true
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=admin
N8N_HOST=localhost
N8N_PORT=5678

# LLM Configuration
OPENAI_API_KEY=your-openai-api-key
# Or use open-source alternatives
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=gpt-4

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# Storage
UPLOAD_PATH=/data/uploads
OUTPUT_PATH=/data/outputs
```

### Docker Compose Configuration

The `docker-compose.yml` orchestrates all services:

- **api-gateway**: Go/Gin service (Port 8080)
- **omr-engine**: Django/Python service (Port 8000)
- **n8n**: Workflow orchestrator (Port 5678)
- **postgres**: PostgreSQL database (Port 5432)
- **redis**: Redis cache (Port 6379)
- **nginx**: Reverse proxy (Port 80/443)

---

## 📖 Usage

### Web Interface

1. **Navigate to the frontend**: http://localhost:3000
2. **Upload sheet music**: Click "Upload" and select your music file (PNG, JPG, PDF)
3. **Select output format**: Choose Sonic Pi, MIDI, or both
4. **Process**: Click "Generate" to start the AI processing
5. **Download**: Once complete, download your generated files

### API Usage

#### Upload Sheet Music

```bash
curl -X POST http://localhost:8080/api/v1/upload \
  -F "file=@path/to/sheet-music.png" \
  -F "output_format=sonic_pi" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

Response:
```json
{
  "job_id": "uuid-here",
  "status": "processing",
  "message": "Sheet music uploaded successfully"
}
```

#### Check Processing Status

```bash
curl -X GET http://localhost:8080/api/v1/status/uuid-here \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

Response:
```json
{
  "job_id": "uuid-here",
  "status": "completed",
  "progress": 100,
  "output_files": [
    "/downloads/uuid-here/music.rb",
    "/downloads/uuid-here/music.mid"
  ]
}
```

#### Download Generated Files

```bash
curl -X GET http://localhost:8080/api/v1/download/uuid-here/music.rb \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -O
```

---

## 🔌 API Documentation

### Authentication

All API requests require authentication using Bearer tokens:

```
Authorization: Bearer YOUR_API_TOKEN
```

### Endpoints

#### `POST /api/v1/upload`
Upload sheet music for processing.

**Request Body (multipart/form-data)**:
- `file`: Music sheet image (PNG, JPG, PDF)
- `output_format`: Output format (sonic_pi, midi, both)
- `options`: JSON object with processing options

**Response**: Job ID and status

#### `GET /api/v1/status/:job_id`
Check the processing status of a job.

**Response**: Current status, progress percentage, and output files if completed

#### `GET /api/v1/download/:job_id/:filename`
Download generated music code or MIDI file.

**Response**: File download

#### `GET /api/v1/jobs`
List all jobs for the authenticated user.

**Query Parameters**:
- `status`: Filter by status (pending, processing, completed, failed)
- `page`: Page number for pagination
- `limit`: Results per page

#### `DELETE /api/v1/jobs/:job_id`
Delete a job and its associated files.

### WebSocket Support

Real-time updates via WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/jobs/:job_id');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Progress:', data.progress);
};
```

---

## 💻 Development

### Project Structure

```
AI-AOL/
├── docker-compose.yml          # Docker orchestration
├── .env.example                # Environment template
├── README.md                   # This file
│
├── services/
│   ├── api-gateway/            # Go/Gin API service
│   │   ├── main.go
│   │   ├── handlers/
│   │   ├── middleware/
│   │   ├── models/
│   │   └── Dockerfile
│   │
│   ├── omr-engine/             # Python/Django OMR service
│   │   ├── manage.py
│   │   ├── omr/                # Django app
│   │   ├── recognition/        # OMR modules
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   └── n8n/                    # n8n workflows
│       ├── workflows/
│       └── credentials/
│
├── ai-agents/                  # Agentic AI modules
│   ├── music-interpreter/
│   ├── code-generator/
│   └── llm-orchestrator/
│
├── frontend/                   # Web UI (React/Vue)
│   ├── src/
│   ├── public/
│   └── Dockerfile
│
├── nginx/                      # Nginx configuration
│   └── nginx.conf
│
└── docs/                       # Additional documentation
    ├── API.md
    ├── ARCHITECTURE.md
    └── CONTRIBUTING.md
```

### Running Tests

#### Go Tests
```bash
cd services/api-gateway
go test ./...
```

#### Python Tests
```bash
cd services/omr-engine
python manage.py test
```

### Adding New Features

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the project's coding standards

3. **Test your changes**
   ```bash
   docker-compose up --build
   ```

4. **Submit a pull request** with a clear description

---

## 🔄 Workflow

### Processing Pipeline

1. **Upload Phase**
   - User uploads sheet music via web UI or API
   - File is validated and stored in the upload directory
   - Job is created with a unique ID

2. **Recognition Phase**
   - n8n triggers the OMR engine
   - Django service processes the image using computer vision
   - Musical elements are detected (notes, clefs, time signatures, etc.)

3. **Interpretation Phase**
   - Agentic AI analyzes the recognized elements
   - LLM interprets musical structure and intent
   - Musical context and relationships are established

4. **Generation Phase**
   - Code generator creates Sonic Pi code
   - MIDI generator creates MIDI file
   - Both outputs are optimized and validated

5. **Delivery Phase**
   - Files are stored in the output directory
   - User is notified via WebSocket
   - Files are available for download

### n8n Workflow Example

The n8n workflow orchestrates the entire pipeline:

```
[Webhook] → [File Storage] → [OMR Processing] → [AI Interpretation] → [Code Generation] → [Notification]
```

---

## 🎼 Output Formats

### Sonic Pi

Sonic Pi is a live coding language for creating music. Example output:

```ruby
# Generated by AI-AOL
use_bpm 120

live_loop :melody do
  play :c4, release: 0.5
  sleep 0.5
  play :e4, release: 0.5
  sleep 0.5
  play :g4, release: 1
  sleep 1
end
```

### MIDI

Standard MIDI files compatible with any DAW (Digital Audio Workstation):
- Format: MIDI Type 1
- Tracks: Separate tracks for different instruments
- Tempo: Preserved from original sheet music

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details on our code of conduct and development process.

### Development Guidelines

- Follow the existing code style
- Write tests for new features
- Update documentation as needed
- Keep commits atomic and well-described
- Ensure all tests pass before submitting PR

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Weneville** - Project Lead
- **Nathanael Romaloburju** - Core Developer
- **Robben Wijanathan** - Core Developer

### Technologies & Libraries

- [n8n.io](https://n8n.io/) - Workflow automation
- [Gin Web Framework](https://gin-gonic.com/) - Go web framework
- [Django](https://www.djangoproject.com/) - Python web framework
- [OpenCV](https://opencv.org/) - Computer vision
- [Sonic Pi](https://sonic-pi.net/) - Live coding music platform
- [music21](http://web.mit.edu/music21/) - Music analysis toolkit

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/Wenev/AI-AOL/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Wenev/AI-AOL/discussions)
- **Email**: Contact the maintainers through GitHub

---

## 🗺️ Roadmap

- [x] Basic OMR functionality
- [x] Agentic AI integration
- [x] Sonic Pi output generation
- [x] MIDI file generation
- [ ] Support for complex musical notation
- [ ] Multi-page sheet music processing
- [ ] Real-time preview of generated music
- [ ] Mobile app for on-the-go scanning
- [ ] Collaborative editing features
- [ ] Plugin system for custom output formats

---

**Made with ❤️ by the AI-AOL Team**
