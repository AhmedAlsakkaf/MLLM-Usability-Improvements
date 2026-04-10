# Artifact Requirements

This document covers the hardware and software requirements needed to install and run the artifact.

## Software Requirements

-   **Operating System**: macOS, Linux, or Windows.
-   **Python**: Version **3.9** or higher is required.
-   **Package Manager**: `pip` (included with Python).
-   **Main Dependencies**: 
    -   `jupyter`: To run the replication and review notebooks.
    -   `google-generativeai`: To interact with the Gemini MLLM (if re-running the full evaluation).
    -   `opencv-python`: For video processing.
    -   `pandas`, `numpy`: For data processing and review.
    -   `sentence-transformers`: For issue aggregation using embeddings.

A complete list of dependencies is provided in `requirements.txt`.

### API Requirements (Optional for Basic Replication)

To perform a **full** reproduction of the LLM-based steps (Heuristic Evaluation and Severity Ranking), you will need:
-   A **Google Gemini API Key** (e.g., for `gemini-2.0-flash-001`).
-   An active internet connection.

However, the artifact is designed to be **Functional** even without an API key, as pre-generated reports are included in the `data/` directory.

## Hardware Requirements

-   **Processor**: A modern multi-core CPU (e.g., Intel Core i5 or better, or Apple Silicon).
-   **Memory (RAM)**: At least **8 GB** is recommended for smooth execution of the Jupyter notebook and video loading.
-   **Disk Space**: Approximately **200 MB** of free disk space is required for the repository, including the video data and dependencies.
-   **Network**: Required for initial setup (cloning and pip install) and for LLM interaction (optional).
