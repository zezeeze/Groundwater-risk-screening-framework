$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$server = Join-Path $projectRoot 'models\llm\llama.cpp-b11026-cuda12.4\bin\llama-server.exe'
$model = Join-Path $projectRoot 'models\llm\Qwen3-8B-GGUF\Qwen3-8B-Q4_K_M.gguf'
$runDir = Join-Path $projectRoot 'data\llm\local_llm_run'
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

& $server `
  -m $model `
  --host 127.0.0.1 `
  --port 8088 `
  --device CUDA0 `
  -ngl all `
  -c 24576 `
  -np 1 `
  -ctk q8_0 `
  -ctv q8_0 `
  -fa on `
  --reasoning off `
  --reasoning-format none `
  --no-webui `
  --metrics `
  --log-file (Join-Path $runDir 'llama-server-cuda.log') `
  --log-timestamps
