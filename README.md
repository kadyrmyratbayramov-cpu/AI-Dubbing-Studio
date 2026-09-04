# AI Dubbing Studio v1.0

An intelligent audio dubbing studio powered by artificial intelligence for automated voice synthesis and dubbing generation.

## Features

- **Voice Synthesis**: AI-powered text-to-speech generation
- **Audio Processing**: Advanced audio manipulation and enhancement
- **Dubbing Pipeline**: Complete workflow for dubbing video content
- **Multi-language Support**: Support for multiple languages and accents
- **Quality Control**: Built-in validation and quality assurance

## Project Structure

```
AI-Dubbing-Studio/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── dubbing_pipeline.py
│   │   └── audio_processor.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── voice_synthesis.py
│   │   └── model_loader.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── audio_utils.py
│   │   ├── text_utils.py
│   │   └── validators.py
│   └── config/
│       ├── __init__.py
│       └── settings.py
├── config/
│   ├── __init__.py
│   ├── config.yaml
│   └── model_config.json
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   ├── test_utils.py
│   └── test_models.py
├── pyproject.toml
├── README.md
├── .gitignore
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.8+
- pip or poetry

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kadyrmyratbayramov-cpu/AI-Dubbing-Studio.git
cd AI-Dubbing-Studio
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using poetry:
```bash
poetry install
```

## Quick Start

```python
from src.core.dubbing_pipeline import DubbingPipeline
from src.config.settings import Config

# Initialize configuration
config = Config()

# Create dubbing pipeline
pipeline = DubbingPipeline(config)

# Process audio/video
result = pipeline.process(input_file="input.mp4")
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Style

This project follows PEP 8 style guidelines. Use black for formatting:

```bash
black src/ tests/
```

## Configuration

Configuration files are located in the `config/` directory:

- `config.yaml`: Main configuration settings
- `model_config.json`: Model-specific parameters

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Support

For issues and questions, please open an issue on GitHub.
