# Welcome to your Bonsai Garden

Run large language models entirely in your browser. 
No server, no API key, completely private.

Built on top of [webml-community/bonsai-webgpu](https://huggingface.co/spaces/webml-community/bonsai-webgpu) by Xenova / HuggingFace.

**[→ Try it live](https://twinkites.github.io/bonsai-garden/)** 


---

## Models

| Model | Size | Quantization | Knowledge cutoff |
|---|---|---|---|
| Bonsai 1.7B | 290 MB | 1-bit (q1) | Late 2024 |
| Bonsai 4B | ~584 MB | 4-bit (q4) | Late 2024 |
| DeepSeek R1 1.5B | ~900 MB | q4f16 | Jul 2024 |
| Qwen3 0.6B | ~400 MB | 4-bit (q4) | Early 2025 |

Models are downloaded once and cached by your browser. Subsequent loads are instant.

---

## Features

- **100% in-browser** - inference runs on your GPU via WebGPU, nothing leaves your device
- **Multi-turn chat** - full conversation history with context window management
- **Markdown rendering** - with syntax-highlighted code blocks
- **Reasoning support** - DeepSeek R1 and Qwen3 chain-of-thought displayed in a collapsible section
- **Export conversations** - download any chat as a Markdown file
- **Customizable system prompt** - adjust model behavior before loading
- **Dark / light mode** - toggleable in the header
- **PWA installable** - add to home screen on desktop or mobile
- **Keyboard shortcuts** - `Enter` to send, `Escape` to stop generation

---

## Browser requirements

WebGPU is required. Use **Chrome** or **Edge** version 113 or later.  
Safari does not currently support WebGPU.

## License

MIT © 2026 [Twin Kites LLC](https://twinkites.com)
