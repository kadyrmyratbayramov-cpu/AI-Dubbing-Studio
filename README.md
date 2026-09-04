# AI Dubbing Studio v1.0

An intelligent audio dubbing studio powered by artificial intelligence for automated voice synthesis and dubbing generation.

## Current Status

This repository now includes a runnable application foundation on top of the existing scaffold.
The current scope is limited to local application startup, configuration loading, backend/frontend entry points,
and development tooling. No Whisper, TTS, translation, diarization, lip-sync, or external AI services are enabled yet.

## Features

- **Runnable foundation**: CLI, backend skeleton, and frontend preview entry points
- **Voice Synthesis scaffold**: Placeholder module for future text-to-speech generation
- **Audio Processing scaffold**: Existing utilities and pipeline placeholders remain intact
- **Development setup**: Environment template, development config, and Make targets
- **Quality checks**: Existing tests preserved and new foundation tests added

## Project Structure

```
AI-Dubbing-Studio/
├── app/
│   ├── __init__.py
│   └── factory.py
├── api/
│   ├── __init__.py
│   ├── __main__.py
│   └── server.py
├── cli/
│   ├── __init__.py
│   ├── __main__.py
│   └── main.py
├── config/
│   ├── __init__.py
│   ├── config.yaml
│   ├── development.yaml
│   └── model_config.json
├── src/
│   ├── __init__.py
│   ├── config/
│   ├── core/
│   ├── models/
│   └── utils/
├── tests/
│   ├── __init__.py
│   ├── test_app_foundation.py
│   ├── test_core.py
│   ├── test_models.py
│   └── test_utils.py
├── web/
│   ├── __init__.py
│   ├── __main__.py
│   ├── static/
│   │   └── index.html
│   └── templates/
│       └── base.html
├── .env.example
├── .gitignore
├── main.py
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
git clone https://github.com/kadyrmyratbayramov-cpu/AI-Dubbing-Studio.git
cd AI-Dubbing-Studio
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

### Environment configuration

```bash
cp .env.example .env
```

Optional runtime overrides can be provided through `.env` values or `config/development.yaml`.

## Run the application

### CLI metadata

```bash
python main.py info
```

### Initialization check

```bash
python main.py check
```

### Backend API skeleton

```bash
python main.py serve-api
```

Available routes:

- `GET /health`
- `GET /config`
- `GET /`

### Frontend preview

```bash
python main.py serve-web
```

## Development

### Make targets

```bash
make install
make check
make test
make run-api
make run-web
```

### Running tests

```bash
python -m pytest
```

## Configuration

Configuration files are located in the `config/` directory:

- `config.yaml`: Base scaffold configuration
- `development.yaml`: Local development overrides for the runnable foundation
- `model_config.json`: Future model-specific parameters

## Support

For issues and questions, please open an issue on GitHub.
