# Introduction to Data Science - Milestone 2

This repository contains the implementation for Milestone 2: **Hierarchical Parsing** and **Reference Matching Pipeline**.

## 1. Environment Setup

The project requires Python 3.x and several data science libraries.
To set up the environment, navigate to the root directory and run:

```bash
pip install -r requirements.txt
```

**Dependencies:**
*   `numpy`, `pandas`: Data manipulation.
*   `scikit-learn`: Machine learning models and metrics.
*   `matplotlib`, `seaborn`: Visualization and EDA.
*   `pylatexenc`: LaTeX parsing.
*   `tqdm`: Progress bars.

---

## 2. Project Structure & Configuration

Ensure your data directories are organized as follows before running the notebooks. The code relies on relative paths (e.g., `../data/raw/`).

```text
├── data/
│   ├── raw/
│   │   ├── auto/           # Contains raw paper folders (LaTeX source)
│   │   └── manual/         # Contains the 5 manually labeled papers
│   └── processed/          # Output directory (created automatically)
│       ├── parsed/         # Stores hierarchy.json and refs.bib
│       ├── ground_truth/   # Stores auto.json and manual.json
│       ├── pairs/          # Stores pairwise datasets
│       └── modelling/      # Stores feature matrices (X, y)
├── notebooks/
│   ├── 2_1/                # Parser Notebooks
│   └── 2_2/                # Machine Learning Pipeline Notebooks
└── requirements.txt
```

**Note:** All input/output paths are defined in the **Configuration block** at the top of each notebook. If your data location differs, please update `BASE_RAW_PATH` or `DATA_PATH` in the first code cell of the notebooks.

---

## 3. Execution Guide

Please run the notebooks in the following order to ensure data dependencies are met.

### Part 1: Hierarchical Parsing & Standardization
**Location:** `notebooks/2_1/`

1.  **`01_parser.ipynb`**
    *   **Goal:** Transformation of raw LaTeX into structured JSON.
    *   **Actions:**
        *   Parses LaTeX structure (Sections, Paragraphs).
        *   Normalizes Math ($...$) and Text.
        *   Extracts and dedupes references to create `refs.bib`.
    *   **Output:** Generates `hierarchy.json`, `refs.bib`, and copies metadata to `data/processed/parsed/`.

---

### Part 2: Reference Matching Pipeline
**Location:** `notebooks/2_2/`

2.  **`01_data_exploration.ipynb`**
    *   **Goal:** Feasibility analysis.
    *   **Actions:** checks data quality (missing values) and statistical distinctness of candidates.
    *   **Outcome:** Justifies the feature selection (Title/Author/Year) and labeling strategy.

3.  **`02_labelling.ipynb`**
    *   **Goal:** Ground Truth Generation.
    *   **Actions:**
        *   Loads references from `refs.bib` (Source) and `references.json` (Target).
        *   Performs cleaning and tokenization.
        *   Generates "Silver Standard" labels using strict matching (Title $\ge$ 0.8, Author $\ge$ 0.3).
    *   **Output:** `auto.json` (Ground Truth mapping).

4.  **`03_pairs_generation.ipynb`**
    *   **Goal:** Dataset Construction.
    *   **Actions:** Converts ground truth maps into Positive/Negative pairs. Uses Random Negative Sampling (Ratio 1:5).
    *   **Output:** `manual_pairs.json` and `auto_pairs.json`.

5.  **`04_feature_engineering.ipynb`**
    *   **Goal:** Vectorization.
    *   **Actions:**
        *   Splits data into Train/Val/Test by Publication ID.
        *   **Augmentation:** Applies aggressive noise (deletions, typos) to Training data to prevent overfitting.
        *   Calculates similarity features (Levenshtein, Jaccard, Year Diff).
    *   **Output:** `X_train.csv`, `y_train.csv`, etc.

6.  **`05_modelling.ipynb`**
    *   **Goal:** Training and Inference.
    *   **Actions:**
        *   Trains Logistic Regression (Baseline) and Random Forest.
        *   Dynamically selects the best model based on ROC-AUC.
        *   Evaluates MRR on the Manual Test Set.
    *   **Output:** Generates `pred.json` files for the final submission.

---

## 4. Final Submission
After running **Notebook 05**, the final `pred.json` files will be located in:
`data/processed/parsed/<partition>/<paper_id>/pred.json`

These files, along with the parsed hierarchy and bib files, constitute the final deliverable.