# Status of Artifact Badges

We are applying for the following artifact badges for our paper: *Recommending Usability Improvements with Multimodal Large Language Models*.

## Applied Badges

### 1. Available
The artifact is publicly available in this repository, and a persistent copy is archived on Zenodo to have a permanent DOI. The license used is the MIT License (Open Source).

### 2. Functional
The artifact is documented, consistent, and complete. It includes:
-   **Raw Data**: All screen recordings (videos) and the task descriptions used in the study.
-   **Reports**: The initial MLLM evaluation reports, aggregation results, and final rankings.
-   **User Study**: All anonymized participant responses and the survey structure.
-   **Replication Code**: A Jupyter notebook that implements the evaluation pipeline (`LLM_Usability_Evaluation.ipynb`). Additionally, a second notebook to review the results of the usability evaluation documented in the paper is included (`Results_Review.ipynb`).
-   **Verification**: The artifact has been tested for clean installation and basic functionality.

### 3. Reusable
The artifact's quality exceeds minimal functionality:
-   **Structure**: The repository is clearly organized into `data/`, `evaluation/`, and `user_study/` directories, following best practices for replication packages.
-   **Documentation**: Detailed `INSTALL`, `REQUIREMENTS`, and `README` files are provided.
-   **Extensibility**: The task description JSON format and the modular notebook allow other researchers to apply the same methodology to new applications and tasks by simply providing new video recordings and descriptions.
-   **Interoperability**: Data is provided in standard formats (JSON, CSV, MP4) that are widely supported.
