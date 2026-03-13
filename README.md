# Human-in-the-Loop Legal Named Entity Annotation with LLM Assistance

**Submitted to AIAI 2026**

This repository contains the source code, human-in-the-loop (HITL) annotation interface, datasets, and experimental results for the paper **"Human-in-the-Loop Legal Named Entity Annotation with LLM Assistance"**.

## 📑 Abstract

High-quality Named Entity Recognition (NER) in the legal domain is often hindered by the scarcity of reliable datasets and the structural noise inherent in legal corpora. To address this, we present a semi-automated Human-in-the-Loop (HITL) architecture designed for the rapid development and enhancement of legal NER datasets. 

Our pipeline integrates a hybrid named-entity extraction engine, combining heuristic-based approaches with a fine-tuned Transformer, and a semantic similarity module within an active learning loop to facilitate named-entity annotations. Crucially, a Large Language Model (LLM) framework acts as a neural adjudicator assistant to resolve conflicts and enforce strict entity constraints. We demonstrate the usefulness of the proposed method by building an improved version of the Greek Legal NER dataset, expanding the named entity volume by 209% while reducing manual annotation to merely 1.4%. Finally, we report evaluation results using the new dataset and a diverse set of NER methods under fully supervised and zero/few-shot scenarios.

---

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

> **Note**: DeBERTa achieved the highest global score due to robust contextual representations, while span-based and generative models showed localized superiority in defining strict boundaries for extensive entities.

---

## 📁 Repository Structure

- \dataset/\: Contains the Enhanced Greek Legal NER datasets in multiple formats.
    - \conll/\: Standard BIO format for token classification fine-tuning.
    - \jsonl/\: Span-based format with absolute character offsets (Tokenizer Agnostic).
    - \statistics/\: Detailed statistics, counts, and B/I split reports per dataset split.
- \src/\: Source code of the extraction and evaluation system.
    - \gents/\: LLM prompts and agent behaviors for neural adjudication.
    - \core/\ & \utils/\: Helper functionalities used globally.
- \pp/\: The Human-in-the-Loop Annotation Interface (built with Streamlit).
- \config/\: System environment variables and API routing configurations.

## 🚀 Setup & Usage

The application requires Python 3.9+ and Streamlit.

1. **Install dependencies:**
   \\\ash
   pip install -r requirements.txt
   \\\

2. **Configure External Services:**
   Make sure to configure the LLM endpoints (e.g., OpenAI/Local APIs) in \config/\ as needed. The database is initialized automatically.

3. **Run the Human-in-the-Loop Interface:**
   \\\ash
   streamlit run app/Data_Loader.py
   \\\

4. **Navigate the UI:**
   Load your unannotated JSON/CoNLL data through the **Data Loader**, utilize the active learning background processes, and adjudicate predictions via the **Annotator** page.

