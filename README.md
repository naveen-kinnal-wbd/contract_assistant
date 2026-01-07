# Contract Assistance

A Python-based contract document processing application with a Streamlit frontend and FastAPI backend for uploading contracts and extracting metadata.

## Architecture

The application consists of two separate containerized services:

- **Frontend**: Streamlit-based web UI for document upload and workflow tracking
- **Backend**: FastAPI-based REST API for async document processing

```
contract_assistant/
├── backend/                    # FastAPI Backend
│   ├── app/
│   │   ├── main.py            # FastAPI app entry point
│   │   ├── models/            # Pydantic schemas
│   │   ├── routers/           # API endpoints
│   │   └── services/          # Business logic
│   ├── tests/                 # Unit tests
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                   # Streamlit Frontend
│   ├── app.py                 # Main Streamlit app
│   ├── components/            # UI components
│   ├── services/              # API client
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## Features

### Contracts Manager
- Upload contract documents (PDF, DOCX, DOC)
- Support for three document types:
  - **Standalone**: Single contract document
  - **Master**: Main contract with optional attachments
  - **Waiver**: Waiver documents
- Document tagging based on file type (MASTER, ATTACHMENT, STANDALONE, WAIVER)
- Toggle for blueprint refinement processing
- Real-time workflow progress tracking with status indicators

### Blueprint Manager
- Coming soon - Will manage extraction blueprints

### Workflow Status Indicators
- 🔵 **In Progress**: Document is being processed
- 🟠 **Awaiting Feedback**: Requires user input
- 🟢 **Completed**: Processing finished successfully
- 🔴 **Failed**: Processing encountered an error

## Quick Start

### Using Docker Compose (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd contract_assistant
```

2. Build and start the services:
```bash
docker-compose up --build
```

3. Access the application:
   - Frontend UI: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Local Development

#### Backend

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the FastAPI server:
```bash
uvicorn app.main:app --reload --port 8000
```

#### Frontend

1. Navigate to the frontend directory (in a new terminal):
```bash
cd frontend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## API Endpoints

### Contract Processing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/contracts/processContractInference` | POST | Process contracts without blueprint refinement |
| `/api/contracts/processBlueprintsRefinement` | POST | Process contracts with blueprint refinement |
| `/api/contracts/progress/{group_id}` | GET | Get workflow progress for a document group |
| `/api/contracts/progress/{group_id}/stream` | GET | Stream workflow progress via SSE |
| `/api/contracts/progress/{group_id}` | DELETE | Clear workflow data |

### Health Checks

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with app info |
| `/health` | GET | Global health check |
| `/api/contracts/health` | GET | Contracts service health |

## Running Tests

Navigate to the backend directory and run:

```bash
cd backend
pytest tests/ -v
```

For async tests:
```bash
pytest tests/ -v --asyncio-mode=auto
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `http://localhost:8000` | Backend API URL (for frontend) |
| `LOG_LEVEL` | `INFO` | Logging level |

### Docker Compose

When running with Docker Compose, the frontend automatically connects to the backend using the internal Docker network (`http://backend:8000`).

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **Pydantic**: Data validation and settings
- **Uvicorn**: ASGI server
- **pytest**: Testing framework

### Frontend
- **Streamlit**: Python web app framework
- **httpx**: HTTP client for API communication

### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration

## Development

### Adding New Endpoints

1. Define schemas in `backend/app/models/schemas.py`
2. Add business logic in `backend/app/services/`
3. Create endpoint in `backend/app/routers/`
4. Write tests in `backend/tests/`

### Adding New UI Components

1. Create component in `frontend/components/`
2. Export from `frontend/components/__init__.py`
3. Import and use in `frontend/app.py` or other components

## License

MIT License

