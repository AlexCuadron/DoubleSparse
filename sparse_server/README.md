# DoubleSparse API Server

An OpenAI-compatible REST API for LLM inference using DoubleSparse with sparse attention.

## Author
- **Name:** AlexCuadron
- **Email:** alex.cl.2000@gmail.com

## Features
- OpenAI-compatible chat completions API
- Sparse attention for efficient inference
- Streaming support
- Token usage tracking
- CORS support
- Configurable model parameters

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AlexCuadron/DoubleSparse.git
cd DoubleSparse/sparse_server
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
python run.py
```

The server will be available at `http://localhost:52309` with these endpoints:
- GET `/` - API information
- GET `/v1/models` - List available models
- POST `/v1/chat/completions` - Create chat completions

### Example Usage

Regular completion:
```bash
curl http://localhost:52309/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

Streaming:
```bash
curl http://localhost:52309/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 100,
    "stream": true
  }'
```

## Configuration

Server settings can be configured through environment variables or a `.env` file:

```env
API_TITLE="DoubleSparse API"
API_VERSION="1.0.0"
HOST="0.0.0.0"
PORT=52309
DEFAULT_MODEL="meta-llama/Llama-2-7b-chat-hf"
HEAVY_CONST=128
GROUP_FACTOR=4
```

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.