# Clinicore

Clinicore is a multimodal clinical intelligence system that combines:

* Deep learning–based computer vision for skin lesion classification
* Retrieval-Augmented Generation (RAG) for document-based medical question answering

The system integrates data-driven inference with knowledge-grounded reasoning to produce explainable and clinically relevant outputs.

---

## Overview

Clinicore addresses two key challenges in medical AI:

1. Automated interpretation of dermatological images
2. Efficient retrieval of domain-specific knowledge from unstructured clinical documents

The platform is modular, scalable, and designed for API-driven deployment.

---

## Dataset

The dataset is split into training and validation subsets as follows:

```python
print(len(train_ds), len(val_ds))
# Output:
# 20264 5067
```

* **Training samples:** 20,264
* **Validation samples:** 5,067
* **Number of classes:** 8

| Code | Description             |
| ---- | ----------------------- |
| AK   | Actinic Keratoses       |
| BCC  | Basal Cell Carcinoma    |
| BKL  | Benign Keratosis        |
| DF   | Dermatofibroma          |
| MEL  | Melanoma                |
| NV   | Melanocytic Nevi        |
| SCC  | Squamous Cell Carcinoma |
| VASC | Vascular Lesions        |

---

## System Architecture

Clinicore consists of two primary subsystems:

---

### 1. Retrieval-Augmented Generation (RAG)

The RAG pipeline enables document-grounded question answering.

#### Pipeline Stages

**Document Ingestion**

* PDF documents are parsed into structured text

**Chunking**

* Recursive splitting

  * Chunk size: 500
  * Overlap: 50

**Embedding Generation**

* Text chunks are converted into dense vector representations

**Vector Indexing**

* Stored using FAISS with local persistence

**Retrieval**

* Top-K semantically relevant chunks are retrieved

**Generation**

* A large language model generates responses constrained to retrieved context

  * Model: `llama-3.3-70b-versatile`

#### Key Features

* Semantic search with FAISS
* Source attribution and traceability
* Persistent vector storage
* Lazy initialization of models (LLM + embeddings)

---

### 2. Vision Pipeline

The vision module performs multi-class skin lesion classification.

#### Workflow

**Preprocessing**

* Image resizing and normalization

**Inference**

* Forward pass through CNN architectures

**Postprocessing**

* Softmax probability distribution
* Top-K prediction ranking

#### Output Example

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

## Model Benchmarking

Five architectures were fine-tuned and evaluated:

---

### 1. EfficientNet-B5 (Fine-Tuned)

* **Macro F1:** 0.8409
* **Macro Recall:** 0.8398
* **Accuracy:** 0.8666
* **Inference Time:** 121.40 sec

#### Per-Class F1

| Class | F1     |
| ----- | ------ |
| AK    | 0.7394 |
| BCC   | 0.9151 |
| BKL   | 0.8101 |
| DF    | 0.8372 |
| MEL   | 0.7765 |
| NV    | 0.9116 |
| SCC   | 0.8066 |
| VASC  | 0.9307 |

---

### 2. ConvNeXt-Base (Fine-Tuned) — **Best Deployment Model**

* **Macro F1:** 0.7192
* **Accuracy:** 0.7857
* **Inference Time:** 72.45 sec

**Checkpoint:**

```
checkpoints/convnext_base_ft_final.pth
```

---

### 3. EfficientNet-B4 (Fine-Tuned)

* **Macro F1:** 0.6682
* **Accuracy:** 0.7557
* **Inference Time:** 79.90 sec

---

### 4. DenseNet121 (Fine-Tuned)

* **Epoch:** 10
* **Loss:** 0.6368
* **Macro F1:** 0.5830
* **Accuracy:** 0.6838

#### Per-Class F1

| Class | F1     |
| ----- | ------ |
| AK    | 0.4173 |
| BCC   | 0.6972 |
| BKL   | 0.5095 |
| DF    | 0.5769 |
| MEL   | 0.5928 |
| NV    | 0.8119 |
| SCC   | 0.4776 |
| VASC  | 0.5811 |

#### Key Observations

* Strong performance on **NV (0.8119)** and **BCC (0.6972)**
* Weak performance on minority classes (AK, SCC)
* High recall bias visible in confusion matrix (over-predicting dominant classes)

---

### 5. ResNet-50 (Fine-Tuned)

* **Macro F1:** 0.4557
* **Accuracy:** 0.5958
* **Inference Time:** 62.49 sec

---

## Comparative Summary

| Model           | Accuracy | Macro F1 | Inference Time | Notes                          |
| --------------- | -------- | -------- | -------------- | ------------------------------ |
| EfficientNet-B5 | 0.8666   | 0.8409   | 121.40 sec     | Best overall performance       |
| ConvNeXt-Base   | 0.7857   | 0.7192   | 72.45 sec      | Best deployment trade-off      |
| EfficientNet-B4 | 0.7557   | 0.6682   | 79.90 sec      | Moderate performance           |
| DenseNet121     | 0.6838   | 0.5830   | —              | Balanced but weaker on classes |
| ResNet-50       | 0.5958   | 0.4557   | 62.49 sec      | Lowest performance             |

---

## API Capabilities

* **Document Ingestion:** Index PDF files into vector store
* **Query Interface:** Retrieve and generate grounded responses
* **Image Classification:** Predict lesion class from input image

---

## Technical Highlights

* Retrieval-Augmented Generation (RAG)
* FAISS-based vector search
* Transformer-based LLM integration
* Transfer learning with CNN architectures
* Modular service-oriented backend design

---

## Key Takeaways

* EfficientNet-B5 achieves the highest accuracy and robustness
* ConvNeXt-Base offers the best balance between performance and efficiency
* DenseNet121 shows reasonable generalization but struggles with class imbalance
* RAG enhances explainability and clinical grounding
* The system unifies perception (vision) and reasoning (language)

---

## Future Improvements

* Multimodal reasoning (vision + RAG fusion)
* Hybrid retrieval (BM25 + dense embeddings)
* Class imbalance mitigation (re-weighting, augmentation)
* Model optimization (quantization, pruning)
* Arabic clinical language support

---

## Authors

Mostafa
Yousef
Menna
Malak

---
