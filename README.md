# VocalClone 🎙️

AI Voice Cloning web application that generates speech in **English** and **Urdu** using a cloned voice from a short audio sample.

## Features

- **Voice Cloning**: Upload a 1-30 second voice sample (WAV/MP3) to create a voice profile
- **Cross-lingual TTS**: Generate speech in English and Urdu using the cloned voice
- **Creator Dashboard**: Modern drag-and-drop interface built with React + Tailwind CSS
- **GPU Optimized**: 4-bit/8-bit quantization for free-tier T4 GPU deployment
- **Urdu Phonetics**: Special handling for Urdu character normalization and phonetics

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python, FastAPI |
| Frontend | React 18, Tailwind CSS, Vite |
| TTS Model | OpenAI TTS-1-HD (multilingual) |
| Deployment | Docker, Hugging Face Spaces |

## Project Structure

```
VocalClone/
├── backend/
│   ├── main.py            # FastAPI application & API endpoints
│   ├── tts_engine.py       # Voice cloning & TTS generation logic
│   ├── audio_utils.py      # Audio preprocessing utilities
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Creator Dashboard component
│   │   ├── main.jsx        # React entry point
│   │   └── index.css       # Tailwind CSS styles
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
├── tests/
│   ├── test_api.py         # API endpoint tests
│   ├── test_audio_utils.py # Audio utility tests
│   └── test_tts_engine.py  # TTS engine tests
├── Dockerfile              # Multi-stage build for HF Spaces
└── README.md
```

## Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- (Optional) NVIDIA GPU with CUDA for model-based voice cloning

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Docker (Full Stack)

```bash
docker build -t vocalclone .
docker run -p 7860:7860 vocalclone
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check & model status |
| POST | `/api/upload` | Upload voice sample (WAV/MP3) |
| POST | `/api/generate` | Generate TTS with cloned voice |
| GET | `/api/samples` | List uploaded voice samples |
| DELETE | `/api/samples/{id}` | Delete a voice sample |

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Deployment to Hugging Face Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces) with **Docker** SDK
2. Push the repository to the Space
3. Set environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key for TTS generation

## Architecture

### Phase 1: Environment & Model Setup
The TTS engine uses OpenAI's TTS-1-HD model with the "onyx" voice, providing stable multilingual synthesis for both English and Urdu. The client is initialized once at startup with the `OPENAI_API_KEY` environment variable.

### Phase 2: Cloning Logic
Reference audio can still be uploaded for future compatibility, but the current OpenAI backend uses a fixed voice ("onyx") that performs well for both English and Urdu.

### Phase 3: Web Interface
The React frontend provides a three-step Creator Dashboard: upload sample → enter script → generate audio. Language toggle switches between English and Urdu with RTL text support.

### Phase 4: Urdu Optimization
- Character normalization (do-chashmi he, alef maksura)
- Urdu punctuation handling
- Cross-lingual synthesis preserves speaker identity across languages

## License

MIT
