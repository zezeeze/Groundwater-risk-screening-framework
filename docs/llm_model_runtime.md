# Local LLM and runtime

- Model repository: `Qwen/Qwen3-8B-GGUF`
- Quantization: `Q4_K_M`
- Model revision: `7c41481f57cb95916b40956ab2f0b139b296d974`
- Model file: `Qwen3-8B-Q4_K_M.gguf`
- Model SHA-256: `D98CDCBD03E17CE47681435B5150E34C1417F50B5C0019DD560E4882C5745785`
- Runtime: llama.cpp `b11026`, commit `b49650adb`
- Reported backend: CUDA 12.4 on an NVIDIA GeForce RTX 5070 Ti Laptop GPU

The approximately 5 GB model binary and llama.cpp runtime are not stored in this repository. Place the downloaded files at:

```text
models/llm/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf
models/llm/llama.cpp-b11026-cuda12.4/bin/llama-server.exe
```

CPU and other GPU backends can be used by editing `code/rag_llm/start_local_qwen_server.ps1`; runtime and small floating-point differences may occur.
