# Project Overview
In this project, you will apply the skills acquired in this course to develop a classification model on publicly available Census Bureau data. You will create unit tests to monitor the model performance on various data slices. Then, you will deploy your model using the FastAPI package and create API tests. The slice validation and the API tests will be incorporated into a CI/CD framework using GitHub Actions.
Two datasets will be provided in the starter code on the following page to experience updating the dataset and model in git.

# Environment
- Install conda or miniconda
- Create a new environment
```
conda create -n [environment_name] python=3.8
```
- Activate the environment
```
conda activate [environment_name]
```
- Install libraries
```
pip install -r requirements.txt 
pip install dvc dvc-s3
```

# Train model
- I use DVC remote pointing to my S3 bucket and commit the data. So need pull data from S3 to get data
```
dvc pull
```
However, due to the need for information about access key and secret access key of AWS, we can simplify by copy file census.csv in root folder to folder data.
- Train model:
```
python starter/ml/train_model.py
```
- Run API on local:
```
uvicorn main:app --reload
```
- Unit Test:
```
pytest
```
- Test POSTS to the API:
```
python live_post.py
```
- Test GET to the API: open https://census-income-predict.onrender.com/
