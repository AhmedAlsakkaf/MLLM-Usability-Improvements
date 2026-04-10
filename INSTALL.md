# Installation and Usage Instructions

This document provides instructions for setting up the artifact environment and verifying its functionality.

## 1. Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd recommending-usability-improvements-mllm
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 2. Basic Usage

The artifact's functionality can be verified by reviewing the pre-generated reports. To rerun the evaluation steps and generate new results, a Google AI API key is required.

### Option A: Reviewing Results (Quickest)
1.  **Start Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
2.  **Open the review notebook:** Navigate to `evaluation/Results_Review.ipynb`.
3.  **Run the cells** to load and browse the pre-generated usability reports. This does **not** require an API key.

### Option B: Full Replication (Functional Test)
1.  **Provide API Key:** To perform any reproduction of the MLLM evaluation (Steps 2, 3 and 4), you must provide a **Google AI API Key** in the `GENAI_API_KEY` variable within the notebook. Note that running these steps will incur costs according to your Google AI API usage plan.
2.  **Open the replication notebook:** Navigate to `evaluation/LLM_Usability_Evaluation.ipynb`.
3.  **Run the following cells:**
    -   **Step 1: Setup and Configuration**: Run the cells to load the data and heuristics.
    -   **Step 2: Usability Evaluation**: Run the cells to identify usability issues.
    -   **Step 3: Issue Aggregation**: Run the cells to process the reports and generate a summary of aggregated usability issues.
    -   **Step 4: Severity Ranking**: Run the cells to re-rank the aggregated issues based on their overall impact.

### Expected Output
You should see a list of prioritized usability problems with justifications and recommendations being generated and displayed in the notebook. This confirms that the entire evaluation and ranking pipeline is working correctly.
