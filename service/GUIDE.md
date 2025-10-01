# Service Directory

This directory contains the FastAPI service for stock predictions.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API
python run_api.py

# Test the API
python test_api.py
```

## Files

- `app.py` - Main FastAPI application
- `run_api.py` - Startup script
- `test_api.py` - API testing suite
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container configuration
- `Stock Prediction Workflow.json` - n8n automation workflow

## Documentation

See the main project README for complete documentation:
- API endpoints and usage
- Docker deployment
- n8n integration
- Troubleshooting guide

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Single stock prediction
- `POST /predict/batch` - Multiple stock predictions
- `GET /docs` - Interactive API documentation
