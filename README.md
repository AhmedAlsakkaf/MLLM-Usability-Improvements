# Recommending Usability Improvements with Multimodal Large Language Models

This repository contains the replication package for the paper: *Recommending Usability Improvements with Multimodal Large Language Models*, submitted to FSE 2026.

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
- `evaluation/`: Contains the Jupyter notebook for usability evaluation.
  - `LLM_Usability_Evaluation.ipynb`: The main notebook for reproducing our usability evaluation.
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
    cd recommending-usability-improvements-mllm
    ```

2.  **Create and activate a Conda environment:**

    ```bash
    conda create --name usability-mllm python=3.9
    conda activate usability-mllm
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## How to run the Usability Evaluation 

The main analysis is contained in the `LLM_Usability_Evaluation.ipynb` Jupyter Notebook located in the `evaluation/` directory.

1.  **Start Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```

2.  **Open and run the notebook:**

    Navigate to `evaluation/LLM_Usability_Evaluation.ipynb` in the Jupyter interface and run the cells in the notebook to reproduce the analysis and results.

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
