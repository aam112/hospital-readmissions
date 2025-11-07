## Dataset Source and Citation

This project uses the **Diabetes 130-US Hospitals for Years 1999–2008 Dataset**, originally published by the Center for Clinical and Translational Research, Virginia Commonwealth University.

- Source: Kaggle — [Diabetes 130-US hospitals for years 1999–2008](https://www.kaggle.com/datasets/andrewmvd/diabetes-130-us-hospitals-for-years-19992008)
- Citation:
  > Strack B, DeShazo JP, Gennings C, et al. *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International. 2014;2014:781670.  
  > doi:[10.1155/2014/781670](https://doi.org/10.1155/2014/781670)


# Hospital Readmissions Prediction

This project predicts **30-day hospital readmissions** using the UCI Diabetes 130-US Hospitals Dataset.  
It includes scripts for data exploration, model training, and predictions.

---

## Project Structure
- `data/raw/` – original dataset  
- `data/processed/` – cleaned dataset (`readmission_dataset.csv`)  
- `models/` – saved model files  
- `src/train.py` – trains the model  
- `src/predict.py` – predicts on new data  
- `src/explore.py` – exploratory data analysis  
- `environment.yml` – reproducible conda environment  

---

## Setup
```bash
conda env create -f environment.yml
conda activate hospital-readmissions
conda install seaborn jupyterlab -y
