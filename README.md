# ClinicoRe — Multimodal Clinical Intelligence System

ClinicoRe is a multimodal AI system designed to support clinical workflows through the integration of:

* Retrieval-Augmented Generation (RAG) for document-based question answering
* Deep learning–based computer vision for skin lesion classification

The system combines structured knowledge retrieval with data-driven inference to provide grounded, explainable outputs.

---

## Overview

ClinicoRe addresses two core challenges in medical AI systems:

1. Efficient access to domain-specific knowledge from unstructured documents
2. Automated interpretation of medical imagery for early diagnostic support

The platform is modular, scalable, and designed for integration into API-driven environments.

---

## Key Capabilities

### Document Intelligence (RAG)

* Ingestion and indexing of PDF documents
* Semantic chunking and embedding generation
* Vector similarity search using FAISS
* Context-grounded response generation via large language models
* Source attribution and traceability

### Vision Intelligence

* Skin lesion classification using a fine-tuned EfficientNet-B5 model
* Support for ISIC multi-class classification
* Probabilistic outputs with ranked predictions

### System Characteristics

* Service-oriented architecture
* Lazy initialization of compute-heavy components
* Persistent vector storage
* External model loading via Hugging Face Hub

---

## System Architecture

### 1. Retrieval-Augmented Generation Pipeline

The RAG pipeline follows a structured sequence:

**Document Ingestion**

* PDF files are parsed into pages using a document loader

**Chunking Strategy**

* Recursive text splitting with configurable chunk size and overlap
* Optimized for retrieval quality and contextual coherence

**Embedding Generation**

* Each chunk is transformed into a dense vector representation

**Vector Indexing**

* Embeddings are stored in a FAISS index
* Index is persisted locally for reuse

**Retrieval**

* Top-K relevant chunks are retrieved based on semantic similarity

**Answer Generation**

* A large language model generates responses constrained to retrieved context

---

### 2. Vision Pipeline

The vision module processes dermoscopic or clinical skin images:

**Preprocessing**

* Image resizing and normalization aligned with training configuration

**Inference**

* Forward pass through EfficientNet-B5

**Postprocessing**

* Softmax probability distribution
* Extraction of top prediction and Top-K classes

---

## Project Structure

```
project/
│
├── api/
│   └── v1/
│       ├── rag/
│       │   ├── router.py
│       │   ├── schemas.py
│       │   └── service.py
│       └── vision/
│
├── core/
│   ├── config.py
│   ├── llm.py
│   ├── embeddings.py
│   └── vectorstore.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── indexes/
│
├── streamlit_app/
│
└── README.md
```

---

## Configuration

Environment variables are managed via a `.env` file:

```
GROQ_API_KEY=your_api_key
```

Default parameters:

* Model: `llama-3.3-70b-versatile`
* Chunk size: 500
* Chunk overlap: 50
* Vector storage: FAISS

---

## RAG Service Design

### Initialization Strategy

The system employs lazy initialization to defer loading of:

* Embedding model
* Language model
* FAISS index

This reduces startup latency and optimizes resource usage.

### Metadata and Traceability

Each processed chunk includes metadata fields such as:

* Document identifier
* Chunk identifier
* Source filename
* Page reference

This enables transparent answer attribution.

### Persistence

* FAISS index is stored locally
* Indexed document registry is maintained in JSON format

---

## API Functionality

### Document Ingestion

Processes uploaded PDFs into indexed vector representations.

### Query Interface

Retrieves relevant content and generates context-aware answers.

### Document Listing

Provides an overview of indexed documents.

---

## Vision Service Design

### Model Architecture

* Backbone: EfficientNet-B5
* Deployment: Loaded from Hugging Face Hub
* Training: Fine-tuned on ISIC dataset

### Inference Workflow

* Input image preprocessing
* Model forward pass
* Probability computation via softmax
* Ranking of predictions

### Output Schema

```json
{
  "predicted_class": "MEL",
  "confidence": 0.91,
  "top_predictions": [
    {"label": "MEL", "score": 0.91},
    {"label": "NV", "score": 0.05}
  ]
}
```

---

## Frontend Interface

The system includes a Streamlit-based interface that enables:

* Image upload and classification
* Visualization of prediction confidence
* Interpretation support via structured outputs

---

## Supported Classes (ISIC)

| Code | Description             | Risk Level |
| ---- | ----------------------- | ---------- |
| MEL  | Melanoma                | High       |
| NV   | Melanocytic Nevus       | Low        |
| BCC  | Basal Cell Carcinoma    | Medium     |
| AK   | Actinic Keratosis       | Medium     |
| BKL  | Benign Keratosis        | Low        |
| DF   | Dermatofibroma          | Low        |
| VASC | Vascular Lesion         | Low-Med    |
| SCC  | Squamous Cell Carcinoma | High       |

---

## Technical Highlights

* Retrieval-Augmented Generation (RAG)
* Dense vector search with FAISS
* Transformer-based language models
* Transfer learning with convolutional neural networks
* Modular backend design

---

## Limitations and Disclaimer

This system is intended for research and educational use only.
It is not a substitute for professional medical advice or diagnosis.

---

## Future Work

* Multimodal reasoning (joint vision + text inference)
* OCR integration for scanned medical documents
* Hybrid retrieval (BM25 + dense embeddings)
* Arabic clinical language support
* Evaluation framework (Recall@K, MRR, F1)


