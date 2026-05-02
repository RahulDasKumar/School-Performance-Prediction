# Video Submission

https://drive.google.com/file/d/1dNzrGCqrasgCXYULYTrz12GscPmsrouz/view?usp=sharing 

# Steps to run the software
  - Download the data and upload to your repo
    -  https://drive.google.com/drive/folders/1W2e3wtS2WUYNf54QMVUSpf5QA93Fw-hE?usp=sharing - training-data
    -  https://drive.google.com/drive/folders/1h35MVq38TGZH9BZXm9uDuJvLzOENkj8e?usp=sharing - school-data
    -  https://drive.google.com/drive/folders/1l1S0fTsJn2APtnFMO-4i8XlcFx4ZRjoP?usp=sharing - data
  - `pip install -r requirements.txt`
  - Data Cleaning Process - `preprocess_data.py`
  - Transform data to parquet `spark-flow.py`
  - Machine Learning Process - `training.py`
  - Streaming Process / Inference Process - `streaming.py`
  - Dashboard - `dashboard.py` command to run the dashboard(streamlit run dashboard.py)
