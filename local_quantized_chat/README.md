# Local Quantized Console Chatbot

A small, fully local console chatbot for macOS or Windows computers with 8 GB of RAM.

The program uses:

- `llama-cpp-python` as the inference engine
- GGUF quantized models
- Qwen2.5 0.5B Instruct Q4_K_M by default
- Optional Qwen2.5 1.5B Instruct Q4_K_M for better answers

No Hugging Face token is required for these public model repositories. Internet access is needed only for the initial model download.

## Model choice

| Selection | Intended use | Approximate behavior |
|---|---|---|
| `0.5b` | Fastest learning/demo option | Very quick, but limited reasoning and factual accuracy |
| `1.5b` | Better general chat on 8 GB RAM | Slower than 0.5B, but noticeably more coherent |

Start with `0.5b`. Move to `1.5b` after confirming that the installation works.

## Files

- `chat.py` — console chatbot
- `requirements.txt` — Python dependencies

## macOS installation

### 1. Install prerequisites

Install Python 3.11 or 3.12. On Apple Silicon, ensure that Python itself is ARM64, not an Intel/x86 build running through Rosetta.

I prefer to start with miniforge so that multiple python environments can exist of on my Mac. If miniforge is not installed, go ahead and install it.

```bash
brew install --cask miniforge
```

### 2. Create a virtual environment

Start off by installing llama.cpp
```bash
brew install llama.cpp
```
Next create a conda environment with the correct version of Python. Currently, gwen requires python 11 or 12 only.
```bash
conda create -n localchat python=3.11 -y
conda activate localchat
```
Use pip to install remaining libraries.
```bash
 python -m venv venv
 source ./venv/bin/activate
 python -m pip install --upgrade pip setuptools wheel
```


### 3. Install on an Apple Silicon Mac with Metal

```bash
CMAKE_ARGS="-DGGML_METAL=on" \
pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

pip install "huggingface-hub>=0.24"
```
Some current wheels/builds may enable Metal automatically. The explicit build flag makes the requested backend unambiguous.



### 4. Run

Fastest model:

```bash
python chat.py
```

Better-quality model:

```bash
python chat.py --model 1.5b
```

The first run downloads the selected model and stores it in the Hugging Face cache.

## Windows installation

### 1. Install prerequisites

Install:

- 64-bit Python 3.11 or 3.12
- Microsoft C++ Build Tools if `llama-cpp-python` cannot find a compatible wheel

Open PowerShell in the project directory.

### 2. Create a virtual environment

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. Run

```powershell
python chat.py
```

For better output:

```powershell
python chat.py --model 1.5b
```

The default Windows configuration uses the CPU. This avoids GPU-specific installation complexity and is suitable for a learning application.

## Commands inside the chatbot

```text
/help
/clear
/exit
```

## Mac optimization

### Apple Silicon: use native ARM64 Python

Check the architecture:

```bash
python -c "import platform; print(platform.machine())"
```

It should print:

```text
arm64
```

An `x86_64` result usually means that Python is running under Rosetta, which can substantially reduce performance and may prevent the preferred Metal path.

### Use Metal offload

The program automatically sets all model layers for GPU offload on Apple Silicon:

```text
n_gpu_layers=-1
```

To override it:

```bash
python chat.py --gpu-layers 0
```

That command forces CPU-only execution and is useful for troubleshooting.

### Keep the context small

KV-cache memory increases with context length. For an 8 GB Mac, start with:

```bash
python chat.py --ctx-size 1024 --history-turns 4
```

Increase context only when the application genuinely needs longer conversations.

### Use the smaller model for response speed

```bash
python chat.py --model 0.5b --ctx-size 1024 --max-tokens 160
```

The most important speed controls are:

1. Smaller model
2. Fewer generated tokens
3. Smaller context
4. Metal GPU offload on Apple Silicon

### Tune CPU threads

The default leaves approximately one logical CPU free and caps worker threads at eight. Try values around the number of performance cores rather than blindly using every logical core:

```bash
python chat.py --threads 4
python chat.py --threads 6
python chat.py --threads 8
```

The fastest setting varies by Mac model and thermal conditions.

### Avoid memory locking on an 8 GB machine

The program uses memory mapping but disables `mlock`. This lets macOS manage memory pressure instead of forcing the entire model to remain resident.

### Close memory-heavy applications

Browsers with many tabs, IDEs, containers, and video applications can force swapping. Swap activity can make local inference much slower even when the model technically fits.

## Useful launch presets

Maximum speed:

```bash
python chat.py \
  --model 0.5b \
  --ctx-size 1024 \
  --max-tokens 160 \
  --history-turns 4
```

Balanced for 8 GB:

```bash
python chat.py \
  --model 1.5b \
  --ctx-size 1536 \
  --max-tokens 256 \
  --history-turns 6
```

Lowest-memory troubleshooting mode:

```bash
python chat.py \
  --model 0.5b \
  --ctx-size 512 \
  --max-tokens 128 \
  --history-turns 2 \
  --gpu-layers 0
```

## Using a separately downloaded GGUF model

```bash
python chat.py --model-file "/path/to/model.gguf"
```

Use an instruction/chat model in GGUF format. The embedded chat template must be compatible with the installed llama.cpp version.

## Troubleshooting

### Installation compiles for a long time

`llama-cpp-python` includes native code. If no compatible wheel is found, pip compiles it locally. Install CMake and the platform compiler toolchain.

### `ModuleNotFoundError: llama_cpp`

Activate the virtual environment and install the package again.

macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### Model download is slow

The model is downloaded only on first use. Later runs use the local Hugging Face cache.

### Generation fails after a long conversation

Reduce retained history or increase the context modestly:

```bash
python chat.py --history-turns 3 --ctx-size 1536
```

For an 8 GB system, reducing history is generally preferable to allocating a very large context.
