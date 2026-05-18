# Recommending Usability Improvements with Multimodal Large Language Models

This repository contains the replication package for the paper: *Recommending Usability Improvements with Multimodal Large Language Models*, accepted to FSE 2026.

[![arXiv](https://img.shields.io/badge/arXiv-2604.25420-b31b1b.svg)](https://doi.org/10.48550/arXiv.2604.25420)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19498008.svg)](https://doi.org/10.5281/zenodo.19498008)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AIG-ist-tugraz/MLLM-Usability-Improvements/artifacts_FSE2026?urlpath=%2Fdoc%2Ftree%2Fevaluation%2FResults_Review.ipynb)
## Abstract
*Usability* describes quality attributes of application *user interfaces* that determine how effectively users can interact with them. Traditional usability evaluation methods require considerable expertise and resources, which can be challenging, especially for small teams and organizations. Automating usability evaluation could make it more accessible and help to improve the user experience.
The recent emergence of powerful *multimodal large language models (MLLMs)* has opened new opportunities for automating usability evaluation and recommendation of improvements. These models can process visual inputs such as images and videos alongside textual context, which enables the identification of usability issues and the generation of actionable suggestions to resolve these issues. 

In this paper, we present a novel automated approach that uses limited application context and screen recordings of user interactions as input to an MLLM. The model automatically identifies and describes usability issues based on *Nielsen’s usability heuristics*, and provides corresponding explanations and improvement recommendations. To reduce the developer effort of manual prioritization, the recommendations are ranked by severity.
The quality and practical usefulness of the generated recommendations were evaluated based on a user study that involved software engineers as participants. The evaluation focused on the highest-ranked suggestions provided by the model. The results demonstrate the potential of our approach to provide low-effort usability improvement recommendations. This makes it a promising complement to traditional evaluation methods, especially in settings with limited access to usability experts. In this sense, the approach serves as a basis for future integration into development tools to enable automated usability evaluation within software engineering workflows.
## Repository Structure

The repository is organized as follows:

- `data/`: Contains the raw data and LLM-based usability analysis results used in our study.
  - `EventHelpR/`: Data related to the EventHelpR application.
    - `organizer/`: Screen recordings of an organizer persona performing tasks.
    - `participant/`: Screen recordings of a participant persona performing tasks.
    - `reports/`: JSON files with the MLLM's analysis, including identified usability issues, rankings, and evaluations.
    - `task_description.json`: Descriptions of the user tasks for EventHelpR.
  - `KnowledgeCheckR/`: Data related to the KnowledgeCheckR application.
    - `student/`: Screen recordings of a student persona performing tasks.
    - `teacher/`: Screen recordings of a teacher persona performing tasks.
    - `reports/`: JSON files with the MLLM's analysis for KnowledgeCheckR.
    - `task_description.json`: Descriptions of the user tasks for KnowledgeCheckR.
- `evaluation/`: Contains the Jupyter notebooks for usability evaluation.
  - `Results_Review.ipynb`: A lightweight notebook to review the pre-generated usability reports of the study without an API key.
  - `LLM_Usability_Evaluation.ipynb`: The main notebook for full replication of the evaluation pipeline (requires API key).
  - `nielsen_heuristics.json`: Nielsen's heuristics used in the evaluation.
- `user_study/`: Contains materials and data from our user study.
  - `responses/`: Anonymized participant responses.
  - `survey/`: Contains the `survey_structure.md` file, which outlines the structure and questions for the user study. The file shows the template for one specific survey variant (evaluating Task 1 of the EventHelpR application). Other survey variants were created by replacing the application context, user task, video link, and the three usability improvement recommendations, corresponding to the different tasks outlined in the `task_description.json` files.
- `requirements.txt`: A list of Python dependencies required to run the code.

## Setup and Installation

To set up the environment and run the code in this repository, please follow these steps.

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd MLLM-Usability-Improvements
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## How to run the Usability Evaluation 

Please refer to [INSTALL.md](INSTALL.md) for detailed setup and usage instructions.

The main usability analysis and possibility to review pre-generated reports can be found in the Jupyter Notebook located in the `evaluation/` directory.

### Summary of Replication Paths
1.  **Results Review**: Use `Results_Review.ipynb` to browse the pre-generated reports without an API key.
2.  **Full Replication**: Use `LLM_Usability_Evaluation.ipynb` for the full pipeline (requires API key for all evaluation steps).

## Reuse

This section provides guidance on how to reuse the automated usability evaluation pipeline for new applications and data.

### Running on New Inputs

To execute the usability evaluation on new data, follow these steps:

1.  **Prepare the Data Directory**: Create a new directory within the `data/` folder for your application (e.g., `data/MyApplication/`).
2.  **Add Task Descriptions**: Create a `task_description.json` file in your new directory. This file must follow the structure described in the [Task Description File Structure](#task-description-file-structure) section.
3.  **Add Screen Recordings**: Place the screen recordings (video files) of the user interactions in the directory. Ensure that the `video_file` paths in `task_description.json` correctly point to these files.
4.  **Create a Reports Directory**: Create a subdirectory named `reports/` within your application data directory to store the generated analysis files.
5.  **Configure the Pipeline**: Open the `evaluation/LLM_Usability_Evaluation.ipynb` notebook and update the `DATA_PATH` variable to point to your new directory:
    ```python
    DATA_PATH = "../data/MyApplication/"
    ```
6.  **Execute the Evaluation**: Provide your Google AI API key in the `GENAI_API_KEY` variable and run the notebook cells sequentially.

### Configurable Parameters

Users modifying the pipeline should be aware of the following configurable parameters:

*   **`GENAI_MODEL`**: (In `evaluation/LLM_Usability_Evaluation.ipynb`) The multimodal large language model used for identifying and ranking issues. By default, it uses `gemini-2.0-flash-001`.
    > **Note**: If you encounter a `404 NOT_FOUND` error stating that `gemini-2.0-flash-001` is not available to new users, please update this variable in the notebook to a newer model version (e.g., `gemini-2.5-flash`).
*   **Similarity Threshold**: (In the `aggregate_issues_by_task_and_similarity` function) The cosine similarity threshold used to cluster redundant findings. The default value is `0.7`.
*   **Heuristics**: The `evaluation/nielsen_heuristics.json` file contains the usability principles used to guide the model. You can modify this to use different heuristics.
*   **Prompts**: The system and user prompts within the notebook can be adjusted to refine how the model identifies and describes issues.
*   **Embedding Model**: The `SentenceTransformer` model (default: `all-MiniLM-L6-v2`) used for semantic clustering can be changed to other models if needed.

## Data

The `data/` and `user_study/` directories contain all the data used in this research, including screen recordings, survey data, and application-specific materials. Please refer to the paper for a detailed description of the data collection process.

### Task Description File Structure

The `task_description.json` files in the `data/EventHelpR/` and `data/KnowledgeCheckR/` directories define the structure for the user tasks. Each file contains the following fields:

- `app_id`: A unique identifier for the application (e.g., "EHR").
- `name`: The name of the application (e.g., "EventHelpr").
- `description`: A brief description of the application.
- `personas`: An array of user personas, each with:
  - `persona_id`: A unique identifier for the persona (e.g., "P1").
  - `name`: The name of the persona (e.g., "Organizer").
  - `description`: A description of the persona's background and goals.
  - `tasks`: An array of tasks for the persona to complete, each with:
    - `task_id`: A unique identifier for the task (e.g., "T1").
    - `name`: The name of the task (e.g., "Account Registration and Login").
    - `description`: A step-by-step description of the task.
    - `video_file`: The path to the screen recording of the task being performed.

## User Study 

### Response Structure

The `user_study/responses/` directory contains the anonymized results from our user study in CSV format. Each file corresponds to a specific task performed by a user persona in one of the applications.

**File Naming Convention:**

The files are named using the format: `{app_id}_{persona_id}_{task_id}.csv`.

-   `app_id`: The application identifier (`EHR` for EventHelpR, `KCR` for KnowledgeCheckR).
-   `persona_id`: The user persona identifier (e.g., `P1`, `P2`).
-   `task_id`: The task identifier (e.g., `T1`, `T2`).

For example, `EHR_P1_T1.csv` contains the responses for Task 1 performed by Persona 1 in the EventHelpR application.

**Response ID Structure:**

Inside each CSV file, the `ID` column uniquely identifies each participant's response. The format is: `{app_id}_{persona_id}_{task_id}_{participant_number}`.

-   `participant_number`: A unique sequential number for each participant who evaluated that specific task.

This structure allows for easy traceability between the responses, the tasks, and the applications under evaluation.

## How to Cite

If you want to cite the research paper, we suggest using the following reference:
```
@inproceedings{10.1145/3797121,
    author = {Lubos, Sebastian and Felfernig, Alexander and Garber, Damian and Le, Viet-Man and Henrich, Manuel},
    title = {Recommending Usability Improvements with Multimodal Large Language Models},
    year = {2026},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3797121},
    doi = {10.1145/3797121},
    booktitle = {Proceedings of the 34th ACM International Conference on the Foundations of Software Engineering},
    numpages = {22},
    keywords = {Usability Evaluation, Multimodal Large Language Models, Recommender Systems, Human-Computer Interaction, Usability Heuristics, Software Quality},
    location = {Montreal, Canada},
}
```

To cite the replication package of the paper, we suggest using the following reference:
```
@software{Lubos_Recommending_Usability_Improvements,
  author = {Lubos, Sebastian and Felfernig, Alexander and Garber, Damian and Le, Viet-Man and Henrich, Manuel},
  title = {{Recommending Usability Improvements with Multimodal Large Language Models}},
  url = {https://doi.org/10.5281/zenodo.19498008}
}
```