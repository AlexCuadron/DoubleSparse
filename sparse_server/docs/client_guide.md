# DoubleSparse API Client Guide

This guide explains how to interact with the DoubleSparse API from different clients and programming languages.

## API Endpoints

The API follows OpenAI's format and provides these main endpoints:
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Create chat completions

## Python Examples

### Using OpenAI's Client Library

```python
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:52309/v1",  # Your server URL
    api_key="not-needed"  # API key isn't used but required by the client
)

# Simple completion
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "user", "content": "Hello! How are you?"}
    ],
    temperature=0.7,
    max_tokens=100
)
print(response.choices[0].message.content)

# Streaming completion
stream = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "user", "content": "Write a story about a robot."}
    ],
    temperature=0.7,
    max_tokens=100,
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### Using Standard Python Libraries

```python
import requests
import json
import sseclient

def generate_text(prompt, stream=False):
    url = "http://localhost:52309/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": stream
    }
    
    if not stream:
        response = requests.post(url, headers=headers, json=data)
        return response.json()["choices"][0]["message"]["content"]
    else:
        response = requests.post(url, headers=headers, json=data, stream=True)
        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.data != "[DONE]":
                chunk = json.loads(event.data)
                if chunk["choices"][0]["delta"].get("content"):
                    yield chunk["choices"][0]["delta"]["content"]

# Regular completion
text = generate_text("Hello! How are you?")
print(text)

# Streaming completion
for chunk in generate_text("Write a story about a robot.", stream=True):
    print(chunk, end="")
```

## cURL Examples

### List Available Models
```bash
curl http://localhost:52309/v1/models
```

### Generate Completion
```bash
curl http://localhost:52309/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [{"role": "user", "content": "Hello! How are you?"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

### Stream Completion
```bash
curl http://localhost:52309/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [{"role": "user", "content": "Write a story about a robot."}],
    "temperature": 0.7,
    "max_tokens": 100,
    "stream": true
  }'
```

## JavaScript/TypeScript Example

```typescript
async function generateText(prompt: string, stream = false) {
    const response = await fetch('http://localhost:52309/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model: 'meta-llama/Llama-2-7b-chat-hf',
            messages: [{role: 'user', content: prompt}],
            temperature: 0.7,
            max_tokens: 100,
            stream: stream
        })
    });

    if (!stream) {
        const data = await response.json();
        return data.choices[0].message.content;
    } else {
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();
        
        while (reader) {
            const {value, done} = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') break;
                    
                    try {
                        const parsed = JSON.parse(data);
                        const content = parsed.choices[0].delta.content;
                        if (content) {
                            yield content;
                        }
                    } catch (e) {
                        console.error('Error parsing chunk:', e);
                    }
                }
            }
        }
    }
}

// Regular completion
const text = await generateText("Hello! How are you?");
console.log(text);

// Streaming completion
for await (const chunk of generateText("Write a story about a robot.", true)) {
    process.stdout.write(chunk);
}
```

## Error Handling

The API returns standard HTTP status codes:
- 200: Success
- 400: Bad Request (invalid parameters)
- 500: Server Error
- 503: Service Unavailable (model not loaded)

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

## Configuration Parameters

You can configure these parameters in your requests:
- `temperature` (0-1): Controls randomness. Higher values make output more random.
- `max_tokens` (int): Maximum number of tokens to generate.
- `top_p` (0-1): Nucleus sampling parameter.
- `stream` (boolean): Whether to stream the response.

## Best Practices

1. **Error Handling**: Always implement proper error handling in your client code.
2. **Streaming**: Use streaming for long responses to get faster initial output.
3. **Connection Management**: Implement retry logic for failed requests.
4. **Resource Management**: Close connections and streams properly.
5. **Rate Limiting**: Consider implementing rate limiting in your client code.