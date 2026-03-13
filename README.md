# Human-in-the-Loop Legal Named Entity Annotation with LLM Assistance

**Submitted to AIAI 2026**

This is the official repository containing the source code, human-in-the-loop (HITL) annotation interface, dataset splits, and experimental results for the 22nd International Conference on Artificial Intelligence Applications and Innovations (AIAI) paper: **"Human-in-the-Loop Legal Named Entity Annotation with LLM Assistance"**.



## 📊 Dataset Statistics (Enhanced GLN v2)

Overview of the entity distribution across Train, Development, and Test splits in the enhanced Greek Legal NER dataset, generated iteratively through the HITL framework:

| Metric / Type | Train Set | Dev Set | Test Set | Automation (Memory) | Automation (Fuzzy) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Sentences** | **17,371** | **4,751** | **3,879** | - | - |
| \ORG\ | 7,712 | 1,300 | 1,774 | 55.0% | 32.3% |
| \LEG-REFS\ | 4,342 | 1,052 | 1,311 | 62.8% | 35.0% |
| \PUBLIC-DOCS\ | 3,389 | 829 | 796 | 15.5% | 84.0% |
| \GPE\ | 4,198 | 1,245 | 828 | 20.0% | 28.7% |
| \LOCATION\ | 4,685 | 174 | 707 | 35.0% | 25.0% |
| \PERSON\ | 2,017 | 314 | 516 | 85.0% | 2.7% |
| \DATE\ | 2,509 | 502 | 553 | 65.0% | 1.8% |
| \FACILITY\ | 402 | 115 | 84 | 35.0% | 6.7% |
| **Total Entities** | **29,254** | **5,331** | **6,569** | **46.1%** | **31.8%** |

---

## 🔬 Experimental Setup

To evaluate the structural integrity and complexity of the Enhanced GLN v2 dataset, we benchmarked it across diverse Named Entity Recognition paradigms using the **[LEXTREME](https://arxiv.org/abs/2301.13126)** evaluation framework. We compared traditional token-classification Transformers with modern span-based models under fully supervised, zero-shot, and few-shot scenarios.

- **Sequence Labeling Architectures**: DeBERTa, XLM-R Base, GreekBERT, RoBERTa, MiniLM, DistilBERT.
- **Span-based Architectures**: **[GLiNER](https://arxiv.org/abs/2311.08526)**, **[DiffusionNER](https://arxiv.org/abs/2305.13298)**, **[SeNER](https://arxiv.org/pdf/2502.07286)**.

### 🏆 Model Performance Comparison (Micro-F1 & Macro-F1)

| Model Architecture | Category | Micro-F1 (%) | Macro-F1 (%) |
| :--- | :--- | :---: | :---: |
| **DeBERTa** | Sequence Labeling | **71.2** | **67.3** |
| XLM-R Base | Sequence Labeling | 69.6 | 66.7 |
| GreekBERT | Sequence Labeling | 69.4 | 65.9 |
| RoBERTa | Sequence Labeling | 69.1 | 65.8 |
| MiniLM | Sequence Labeling | 68.4 | 63.5 |
| DistilBERT | Sequence Labeling | 65.5 | 62.1 |
| **GLiNER** | Span-based (Fully Supervised) | 64.3 | 61.3 |
| DiffusionNER | Span-based (Fully Supervised) | 55.9 | 53.5 |
| SeNER | Span-based (Fully Supervised) | 54.9 | 49.8 |
| GLiNER (500 samples) | Span-based (Few-Shot) | 60.5 | 57.6 |
| GLiNER (Zero-shot) | Span-based (Zero-Shot) | 32.0 | 28.0 |

---

## 📁 Repository Structure

```text
NER_GLN_v2_curation_architecture/
├── app/                  # Human-in-the-Loop (HITL) Streamlit Interface
├── config/               # Configuration files (API keys, settings)
├── dataset/              # Enhanced Greek Legal NER v2 Datasets
│   ├── conll/            # Tokens and BIO labels (spacy tokenized)
│   ├── jsonl/            # Span-level absolute character offsets
│   └── statistics/       # Annotation counts and entity distribution
├── models/               # (Not tracked) Place custom/fine-tuned models here
├── src/                  # Core algorithms and pipelines
│   ├── agents/           # LLM interaction layers and context builders
│   ├── automations/      # Background sync and mapping routines
│   ├── core/             # Base logic including vector retrieval & schema
│   ├── database/         # SQLite interfaces (auto-generated locally)
│   ├── judges/           # Logic for neural adjudication heuristics
│   └── training/         # Utility scripts (not deployed to production app)
├── README.md             # This document
└── requirements.txt      # Python dependencies
```

## 🚀 Setup & Usage

The application requires **Python 3.9+**.

### 1. Installation

Clone the repository and install the dependencies:
```bash
git clone https://github.com/syn-nomos/NER_GLN_v2_curation_architecture.git
cd NER_GLN_v2_curation_architecture
pip install -r requirements.txt
```

### 2. Loading the Transformers (Model Weights)

Due to repository size constraints, the fine-tuned Transformer models (used for the Phase 1 heuristic engine) are not included. 
To add the Transformers needed for the application to generate baseline predictions:
1. Create a `models/` directory at the root of the project: `mkdir models`
2. Place your fine-tuned Hugging Face weights inside the folder (e.g., `models/roberta_finetuned`).
3. Update the model path within the configuration files to point to this directory.

### 3. Configure External Services

Prior to launching the application, ensure that the appropriate LLM endpoints (e.g., OpenAI or Local/Ollama APIs) are strictly defined in the `config/` files (or via environment variables). These endpoints are required for the Neural Adjudication assistance to function correctly. The local SQLite database maps and indexes will be initialized automatically.

### 4. Run the Human-in-the-Loop Interface

Launch the interactive annotation UI via Streamlit:
```bash
streamlit run app/Data_Loader.py
```

### 5. Navigate the UI
- Load your unannotated JSON/CoNLL data through the **Data Loader** tab.
- Utilize the background active learning processes.
- Adjudicate and approve predictions visually via the **Annotator** page.

