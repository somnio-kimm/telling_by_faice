# TellingByfAIce

Emotion-reaction game. A Godot 4 client talks to a FastAPI backend that runs face detection (YOLO) and emotion classification (EfficientNet-B3), plus LLM scenario generation via LangChain + OpenAI.

## Layout

```text
.
├── server/                 # Python backend (FastAPI)
│   ├── pyproject.toml
│   ├── src/faice/          # importable package — `from faice.main import app`
│   └── tests/
├── game/                   # Godot 4 project
│   ├── project.godot
│   ├── assets/
│   ├── data/dialogues/
│   ├── globals/            # autoload singletons
│   └── scenes/{title,stage,credits,dialogue}/
├── notebooks/              # training + eval notebooks
├── models/                 # model weights (gitignored)
└── .env.example
```

## Setup

```bash
cp .env.example .env         # fill in OPENAI_API_KEY
make install                 # uv sync in server/, install pre-commit
```

Place `yolo.onnx` and `efficientnet_b3.pth` into `models/`.

## Run

```bash
make server                  # FastAPI backend on :8000
```

Open `game/` in Godot 4.4+ for the client.

## Conventions

See [REVIEW.md](REVIEW.md) for branch, commit, and PR conventions, and [CONTRIBUTING.md](CONTRIBUTING.md) for the dev loop.
