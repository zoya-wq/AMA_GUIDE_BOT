# AMA Guides Ingestion and Retrieval System -- Documentation

## Overview

This system ingests the AMA Guides to the Evaluation of Permanent Impairment (5th Edition) PDF and makes its content queryable through a Retrieval-Augmented Generation (RAG) pipeline. Users upload the PDF via a Streamlit web interface, the system parses and stores the content in vector and document databases, and then answers natural language questions grounded in the extracted content.

---

## Project Structure

```
AMA_GUIDE/
  app.py                  -- Streamlit frontend (UI, tabs, chat)
  ingestion_pipeline.py   -- PDF parsing, content extraction, embedding, storage
  retrieval_engine.py     -- Query classification, semantic search, answer generation
  config.py               -- Centralised configuration (env vars, defaults)
  models.py               -- Data classes (Table, Formula, Section, Paragraph, IngestionProgress)
  utils.py                -- Shared utilities (hashing, chunking, citation validation)
  requirements.txt        -- Python dependencies
  .env                    -- Environment variables (API keys, endpoints)
```

---

## Technology Stack

| Layer               | Technology                                     |
|---------------------|------------------------------------------------|
| Frontend            | Streamlit (tabs, file upload, chat, charts)    |
| PDF Parsing         | Azure Document Intelligence OR LlamaParse      |
| Embedding Model     | Azure OpenAI `text-embedding-3-large` (3072d)  |
| Chat/LLM            | Azure OpenAI (configurable deployment name)    |
| Vector Database     | Qdrant Cloud (cosine similarity)               |
| Document Database   | Azure Cosmos DB (MongoDB API)                  |
| Retry/Resilience    | tenacity (exponential backoff, 3 attempts)     |
| Charts              | Plotly                                         |

---

## Ingestion Pipeline Flow

The ingestion runs as a single async pipeline (`run_pipeline`) with three sequential stages:

### Stage 1 -- Parse PDF

Two parser backends are supported, selected via the `USE_LLAMA_PARSE` config flag:

**Azure Document Intelligence (default)**
- Uses the `prebuilt-layout` model.
- Returns structured objects: pages, paragraphs (with bounding regions), and tables (with cell-level row/column indices).
- Tables are extracted with headers and rows by iterating `table.cells`.
- Formulas are found by regex over concatenated paragraph text.
- Sections are detected by the `heading` role on paragraphs, or heuristic checks (all-caps lines, "Chapter"/"Section" prefixes).

**LlamaParse (optional)**
- Writes PDF bytes to a temp file, calls `LlamaParse.load_data()`.
- Returns one document object per page in markdown format.
- Tables are extracted from four sub-formats: pipe-delimited markdown, grid/box-drawn (+---+), CSV/space-separated, and HTML `<table>`.
- Paragraphs are split on double-newline boundaries.
- Formulas and sections are extracted by regex from the combined raw text.

**Output of Stage 1**: A dictionary with keys `tables`, `paragraphs`, `formulas`, `sections`, and `pages`.

### Stage 2 -- Store in Cosmos DB

Each extracted item is upserted into Cosmos DB (MongoDB API) with a deterministic `_id` based on content hashes. Document schema varies by content type:

- `content_type: "table"` -- stores headers, rows, pages, natural language representation
- `content_type: "paragraph"` -- stores text, page number, section ID
- `content_type: "formula"` -- stores formula text, variables, page
- `content_type: "section"` -- stores title, page range, content

Cosmos DB is optional. If the connection fails at startup, the pipeline logs a warning and continues with Qdrant only.

### Stage 3 -- Embed and Store in Qdrant

Each content item is embedded via Azure OpenAI (`text-embedding-3-large`, 3072 dimensions) and upserted into one of four Qdrant collections:

| Collection        | Content Stored                                             |
|-------------------|------------------------------------------------------------|
| `ama_paragraphs`  | One point per paragraph (text as payload)                  |
| `ama_tables`      | Up to 3 points per table (natural language, QA pairs, JSON)|
| `ama_formulas`    | One point per formula (natural language representation)    |
| `ama_sections`    | One point per section (title + content)                    |

Tables get multiple vector representations to improve retrieval from different query angles (natural language description, question-answer pairs, raw JSON).

Points are batch-upserted in groups of 50 with retry logic (3 attempts, exponential backoff).

String IDs are converted to UUIDs for Qdrant compatibility using `uuid5` with a fixed namespace.

---

## Retrieval Engine Flow

### Step 1 -- Intent Classification

The query is classified into one of four intents using keyword matching:

| Intent                 | Trigger Keywords                                              |
|------------------------|---------------------------------------------------------------|
| `table_lookup`         | table, category, dre, impairment %, percentage, rating, etc   |
| `formula_calculation`  | calculate, formula, computation, whole person, wpi, etc       |
| `comparison`           | compare, difference, versus, vs, between                      |
| `concept_explanation`  | Default fallback -- searches all collections                  |

### Step 2 -- Retrieval Routing

Each intent uses a different retrieval strategy:

**table_lookup**
- First tries deterministic lookup by table number (regex extracts "Table X-Y" from query, maps to Cosmos `_id`).
- Falls back to semantic search on `ama_tables`.

**formula_calculation**
- Semantic search on `ama_formulas` to find the formula.
- Fetches full formula details from Cosmos DB.
- Fetches related tables referenced by the formula.
- Searches `ama_paragraphs` for calculation examples.
- Generates step-by-step calculation guide programmatically.

**comparison**
- Searches `ama_paragraphs` (limit 3) and `ama_tables` (limit 2) and merges results.

**concept_explanation**
- Searches ALL four collections: paragraphs (3), sections (2), tables (2), formulas (1).
- Merges all results for maximum coverage.

### Step 3 -- Semantic Search

Each search call:
1. Generates an embedding for the query using `text-embedding-3-large`.
2. Calls `qdrant_client.query_points()` with cosine similarity and a score threshold of 0.3.
3. Returns the payload dictionaries from matching points.

### Step 4 -- Answer Generation

The retrieved context is formatted and sent to Azure OpenAI:
- System prompt instructs the model to answer ONLY from the provided context and include page citations.
- Temperature is set to 0.2 for factual precision.
- `max_completion_tokens` is set to 1500.
- Citations are extracted from the result payloads (type, page, content snippet).

### Step 5 -- Citation Extraction

Citations are pulled from the retrieval results, extracting:
- Content type (table, paragraph, formula)
- Page number(s)
- Content preview (first 200 characters)

---

## Data Models

### Table
- `table_id` -- deterministic hash of headers + page
- `pages` -- list of page numbers (supports multi-page tables)
- `headers` -- column names
- `rows` -- list of row data
- `footnotes`, `caption`, `related_formulas`, `related_tables`
- Methods: `to_json()`, `to_natural_language()`, `generate_qa_pairs()`

### Formula
- `formula_id` -- sequential ID
- `formula_text` -- raw formula string (limited to 400-500 chars)
- `page`, `section`, `variables` (dict), `conditions` (list)
- `example_calculation`, `related_tables`
- Method: `to_natural_language()`

### Section
- `section_id` -- hash-based
- `title`, `page_start`, `page_end`, `content`
- `parent_section`, `subsections`

### Paragraph
- `paragraph_id` -- hash of text + page
- `text`, `page`, `section_id`, `section_title`, `chunk_index`

---

## Key Configuration (config.py)

All configuration is driven by environment variables with sensible defaults:

| Variable                           | Purpose                        | Default                     |
|------------------------------------|--------------------------------|-----------------------------|
| `AZURE_DOC_INTELLIGENCE_ENDPOINT`  | Document parsing endpoint      | (required)                  |
| `AZURE_DOC_INTELLIGENCE_KEY`       | Document parsing key           | (required)                  |
| `AZURE_OPENAI_ENDPOINT`            | OpenAI API endpoint            | (required)                  |
| `AZURE_OPENAI_KEY`                 | OpenAI API key                 | (required)                  |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`| Embedding model deployment     | `text-embedding-3-large`    |
| `AZURE_OPENAI_CHAT_DEPLOYMENT`     | Chat model deployment          | `gpt-4`                     |
| `AZURE_COSMOS_CONNECTION_STRING`   | Cosmos DB connection           | (optional)                  |
| `QDRANT_URL`                       | Qdrant Cloud endpoint          | (hardcoded default)         |
| `QDRANT_API_KEY`                   | Qdrant auth token              | (hardcoded default)         |
| `EMBEDDING_DIMENSIONS`             | Vector size                    | `3072`                      |
| `CHUNK_SIZE`                       | Text chunk size in tokens      | `512`                       |
| `CHUNK_OVERLAP`                    | Overlap between chunks         | `50`                        |
| `USE_LLAMA_PARSE`                  | Use LlamaParse instead of Azure| `false`                     |

---

## Startup Behaviour

On app startup, the system:
1. Instantiates `AMARetrievalEngine`, which connects to Qdrant, Cosmos DB, and Azure OpenAI.
2. Calls `is_populated()` to check if any Qdrant collection already has data.
3. If data exists, sets `ingestion_complete = True` and skips the requirement for fresh PDF upload -- the "Ask Questions" tab becomes immediately available.
4. If no data exists, the user must upload and ingest a PDF before querying.

---

## Known Issues and Gotchas

1. **Qdrant client version**: The `qdrant-client` v1.7+ removed `.search()` in favour of `.query_points()`. The code handles both APIs with a try/except fallback.

2. **OpenAI model compatibility**: Newer Azure OpenAI models (gpt-5.x, o-series) require `max_completion_tokens` instead of `max_tokens`. The code uses the newer parameter.

3. **Cosmos DB is optional**: If the connection string is missing or the server is unreachable, the pipeline continues with Qdrant only. Deterministic table lookups by ID will not work without Cosmos.

4. **Formula extraction is regex-based**: The formula detection uses heuristic regex patterns. Complex multi-line formulas or formulas embedded in unusual formatting may be missed.

5. **Section detection heuristics**: Sections are detected by doc-intelligence heading roles or by checking for all-caps lines and "Chapter/Section/Part" prefixes. This can produce false positives on documents with unconventional formatting.

6. **Embedding truncation**: Input text to the embedding model is truncated to 8000 characters. Very long paragraphs or tables lose trailing content.

7. **Final upsert routing**: The final batch upsert at the end of `store_in_qdrant` routes based on the FIRST point's type. If the batch contains mixed types (unlikely but possible), items may land in the wrong collection.

8. **Progress tracking**: The `paragraphs_chunked` counter is incremented for ALL types during Qdrant storage, not just paragraphs. The name is misleading.

---

## Improvement Opportunities

### High Impact

1. **Chunking strategy** -- The current system stores entire paragraphs as-is. Implementing a proper chunking strategy with configurable chunk sizes and overlap (the `utils.py` has `chunk_text()` but it is never called) would improve retrieval precision for long passages.

2. **Hybrid search (BM25 + vector)** -- Add keyword-based search alongside vector search. Qdrant supports payload-based filtering. Combining BM25 text matching with semantic similarity would catch exact-match queries that embeddings miss (e.g., specific table numbers, codes, DRE categories).

3. **Re-ranking** -- Add a cross-encoder re-ranker after initial retrieval. The current system returns raw vector similarity scores. A re-ranker (e.g., Cohere Rerank, BGE reranker) would significantly improve precision by scoring query-document relevance more accurately.

4. **Multi-turn conversation memory** -- The current system treats each query independently. Adding conversation context (previous Q&A pairs) to the LLM prompt would enable follow-up questions like "What about Category III?" after asking about Category II.

5. **Page-level attribution** -- Track exact page numbers through the entire pipeline (LlamaParse currently defaults most pages to 1). This would make citations more accurate and trustworthy.

### Medium Impact

6. **Query expansion** -- Before searching, use the LLM to expand the query with synonyms and related medical terms. "WPI" should also search for "Whole Person Impairment", "upper extremity" should match "arm, hand, wrist, shoulder".

7. **Table-aware chunking** -- Tables should be stored with their surrounding context (the paragraph before/after the table) to help the LLM understand what the table represents.

8. **Structured formula execution** -- Instead of just describing formulas, parse them into executable Python expressions so the system can actually compute impairment ratings given input values.

9. **Collection-aware final upsert** -- Fix the final batch upsert to group points by type and upsert to the correct collection, rather than routing by the first point's type.

10. **Caching embeddings** -- Cache generated embeddings (e.g., in-memory LRU or Redis) to avoid re-embedding identical or near-identical queries.

### Lower Priority

11. **OCR fallback** -- For scanned PDFs where Document Intelligence returns sparse results, add an OCR fallback (e.g., Tesseract) to extract text from page images.

12. **Incremental ingestion** -- Support ingesting additional documents or updated versions without re-processing the entire corpus. Use content hashes to detect changed pages.

13. **Evaluation framework** -- Build a test set of known questions and expected answers to measure retrieval accuracy (recall@k, MRR) and answer quality (faithfulness, relevance) as the system evolves.

14. **Streaming responses** -- Use OpenAI streaming to display the answer token-by-token in the Streamlit UI for better perceived performance.

15. **Access control** -- Add authentication to the Streamlit app and scope queries by user or organisation.

16. **Metadata filtering** -- Allow users to filter searches by chapter, page range, or content type (tables only, formulas only) to narrow results.

17. **Export / audit trail** -- Log all queries, retrieved context, and generated answers to a database for compliance auditing and system improvement.

---

## Dependencies

```
streamlit                       -- Web UI framework
azure-ai-documentintelligence   -- PDF parsing (Azure)
azure-identity                  -- Azure auth
pymongo                         -- Cosmos DB (MongoDB API)
qdrant-client                   -- Vector database
openai                          -- Azure OpenAI SDK
python-dotenv                   -- .env loading
pypdf                           -- PDF utilities
pandas                          -- Data manipulation
numpy                           -- Numerical operations
tiktoken                        -- Token counting
langchain                       -- (imported but usage minimal)
tenacity                        -- Retry logic
plotly                          -- Charts in statistics tab
rich                            -- Logging formatting
llama-parse                     -- Alternative PDF parser
```
