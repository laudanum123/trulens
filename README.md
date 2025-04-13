# RAG Project Documentation

## Building the RAG Index

1. Install dependencies:
```bash
uv pip install -r hf_rag_project/requirements.txt
```

2. Run the build script (automatically downloads Wikipedia dataset):
```bash
python hf_rag_project/src/build_rag.py
```

## Evaluating the RAG Pipeline

1. Start LM Studio server (required for answer generation)

2. Run evaluation with TruLens:
```bash
python hf_rag_project/src/evaluate_rag.py
```

Optional flags:
- `--reset-db` to clear previous evaluation records

## Key Components
- `build_rag.py`: Builds FAISS index from Wikipedia dataset
- `evaluate_rag.py`: Evaluates pipeline with TruLens metrics
- Uses LM Studio for local LLM inference