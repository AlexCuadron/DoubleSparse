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

### Environment Variables

Server settings can be configured through environment variables or a `.env` file:

```env
API_TITLE="DoubleSparse API"
API_VERSION="1.0.0"
HOST="0.0.0.0"
PORT=52309
DEFAULT_MODEL="meta-llama/Llama-2-7b-chat-hf"
MODEL_ARCHITECTURE="llama"  # llama, mistral, or qwen2
HEAVY_CONST=128    # Sparse attention parameter
GROUP_FACTOR=4     # Sparse attention parameter
CHANNEL="qk"       # Channel selection (q, k, qk)
```

### Sparse Attention Parameters

The server uses DoubleSparse's efficient attention mechanism with two key parameters:

1. `HEAVY_CONST` (Token Sparsity):
   - Controls how many tokens are kept for attention computation
   - Higher values = more tokens = more accuracy but slower
   - Lower values = fewer tokens = faster but potentially less accurate
   - Default: 128

2. `GROUP_FACTOR` (Channel Sparsity):
   - Controls channel grouping for attention computation
   - Higher values = more sparsity = faster but potentially less accurate
   - Lower values = less sparsity = more accurate but slower
   - Default: 4

## Client Integration

For detailed instructions on how to use the API from different programming languages and frameworks, see our [Client Guide](docs/client_guide.md). The guide includes:

- Python examples using OpenAI's client library
- Python examples using standard libraries
- JavaScript/TypeScript examples
- cURL examples
- Error handling guidelines
- Best practices

## Performance Considerations

1. **Memory Usage**:
   - The server uses DoubleSparse's efficient attention mechanism
   - Memory usage scales with `HEAVY_CONST` and `GROUP_FACTOR`
   - Monitor GPU memory usage to optimize these parameters

2. **Throughput**:
   - Higher `HEAVY_CONST` = lower throughput
   - Higher `GROUP_FACTOR` = higher throughput
   - Find the right balance for your use case

3. **Latency**:
   - Use streaming for better perceived latency
   - First token latency depends on prompt length
   - Subsequent tokens benefit from sparse attention

## Error Handling

The server implements comprehensive error handling:
- Invalid requests return 400 with details
- Server errors return 500 with stack traces
- Model loading issues return 503

Error responses follow this format:
```json
{
    "error": {
        "message": "Error description",
        "type": "error_type",
        "code": 400
    }
}
```

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.