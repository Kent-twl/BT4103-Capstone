# BT4103 Capstone Project

The project aims to generate key insights from trading data to improve efficiency of and add value to end-of-day reporting. 

## Usage

## Project Structure

The project structure distinguishes three kinds of folders:
- read-only (RO): not edited by either code or researcher
- human-writeable (HW): edited by the researcher only.
- project-generated (PG): folders generated when running the code; these folders can be deleted or emptied and will be completely reconstituted as the project is run.


```
.
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── data               <- All project data, ignored by git
│   ├── processed      <- The final, canonical data sets for modeling. (PG)
│   ├── raw            <- The original, immutable data dump. (RO)
│   └── temp           <- Intermediate data that has been transformed. (PG)
├── docs               <- Documentation notebook for users (HW)
│   ├── manuscript     <- Manuscript source, e.g., LaTeX, Markdown, etc. (HW)
│   └── reports        <- Other project reports and notebooks (e.g. Jupyter, .Rmd) (HW)
├── results
│   ├── figures        <- Figures for the manuscript or reports (PG)
│   └── output         <- Other output for the manuscript or reports (PG)
└── src                <- Source code for this project (HW)

```

## License

## Dependencies and environment variables
- Python version: 3.11.7 
- Docker Desktop: https://docs.docker.com/desktop/install/windows-install/

## To Run Streamlit Application
1. Clone Main Branch
2. Download Docker Desktop
3. After Docker is installed, run in command line at cloned repo directory: docker build -t my-streamlit-app .
4. To run built container, run in command line: docker run -p 8501:8501 --env-file streamlit.env -v ${PWD}:/app my-streamlit-app 
