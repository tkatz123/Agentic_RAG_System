# Agentic RAG System

An agentic research assistant built over a corpus of 100 AI engineer job postings and 250 arXiv AI/ML abstracts. Goes beyond a basic "chat with your docs" RAG pipeline by adding an agent layer that decides when to retrieve, reformulates weak queries, and cites its sources — with a full evaluation harness measuring retrieval and answer quality.

## Demo

Ask questions like:
- *"What Python skills do companies look for in AI engineers?"*
- *"What is retrieval augmented generation?"*
- *"What cloud platforms does Booz Allen Hamilton require?"*

The agent retrieves relevant chunks from the corpus, grades their relevance, reformulates the query if needed, and returns a grounded answer with source citations.

## Architecture

```
User Question
     │
     ▼
 [retrieve]  ◄──────────────────┐
     │                          │
     ▼                          │
[grade_chunks]                  │
     │                          │
     ├── relevant ──► [generate] ──► Answer + Sources
     │
     └── not relevant ──► [reformulate] ──┘
```

- **Retrieve** — embeds the query and searches 2,299 chunks in Chroma (cosine similarity, k=10)
- **Grade** — LLM judges whether retrieved chunks are relevant to the question
- **Reformulate** — LLM rewrites the query if chunks aren't relevant (max 2 retries)
- **Generate** — stuffing prompt with retrieved context → GPT-4o-mini → answer

## Evaluation Results

| Metric | Score |
|---|---|
| Retrieval hit-rate (k=10) | 90% (18/20) |
| Retrieval hit-rate (k=5, baseline) | 85% (17/20) |
| Answer faithfulness — agent with reformulation | 2.80 / 3 |
| Answer faithfulness — simple pipeline (no reformulation) | 2.75 / 3 |

Evaluated on a hand-curated gold set of 20 questions covering 8 job postings and 4 arXiv papers. Faithfulness scored by LLM-as-judge on a 1–3 scale.

## Tech Stack

| Layer | Choice |
|---|---|
| Agent orchestration | LangGraph |
| Vector database | Chroma (local) |
| Embeddings + LLM | OpenAI (text-embedding-ada-002 + GPT-4o-mini) |
| API service | FastAPI |
| Frontend | Streamlit |
| Containerization | Docker |
| Evaluation | LLM-as-judge (faithfulness) + hit-rate (retrieval) |

## Corpus

- **100 AI engineer job postings** — scraped from LinkedIn, chunked into 1,524 chunks
- **250 arXiv abstracts** — fetched via the arXiv API across 5 queries (RAG, LLM agents, vector search, etc.), chunked into 775 chunks
- **Total: 2,299 chunks** stored in Chroma with source metadata

## Running Locally

**1. Clone and set up environment**
```bash
git clone <repo-url>
cd Agentic_RAG_System
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Set up environment variables**
```bash
cp .env.example .env
# add your OpenAI API key to .env
```

**3. Ingest the corpus**
```bash
python ingestion/ingest.py
```

**4. Run the Streamlit UI**
```bash
.venv/bin/streamlit run frontend/app.py
```

**5. Or run the FastAPI service**
```bash
uvicorn api.main:app --reload
# visit http://127.0.0.1:8000/docs
```

## Running with Docker

```bash
docker build -t agentic-rag .
docker run -p 8000:8000 \
  --env OPENAI_API_KEY=your-key-here \
  -v "$(pwd)/data:/app/data" \
  agentic-rag
```

## Project Structure

```
Agentic_RAG_System/
├── ingestion/
│   ├── ingest.py          # chunk, embed, load into Chroma
│   ├── fetch_arxiv.py     # fetch arXiv abstracts
│   └── explore.py         # corpus profiling
├── retrieval/
│   ├── retrieve.py        # basic similarity search
│   └── rag.py             # simple end-to-end RAG pipeline
├── agent/
│   └── agent.py           # LangGraph agent (retrieve→grade→reformulate→generate)
├── api/
│   └── main.py            # FastAPI service
├── frontend/
│   └── app.py             # Streamlit UI
├── eval/
│   ├── gold_set.csv        # 20 hand-curated question/source pairs
│   └── evaluate.py        # retrieval hit-rate + faithfulness scoring
└── data/
    ├── raw/               # source CSVs
    └── chroma_db/         # vector database (generated, not committed)
```

## Limitations & Next Steps

- Chroma runs locally — would swap for Pinecone for a cloud deployment
- No streaming responses yet — FastAPI endpoint returns full answer at once
- Faithfulness delta between agent and simple pipeline is small (2.80 vs 2.75) — larger corpus or harder questions would widen the gap
- Max 2 reformulation retries — could be tuned

## Author

Tyler Katz

B.S. in Applied Data Analytics, Class of 2026
Syracuse University

[GitHub Profile](https://github.com/tkatz123) • [LinkedIn](https://www.linkedin.com/in/tylerkatz1/)

## License

This projest is licensed under the MIT Licesne. See the LICESNE for details.
