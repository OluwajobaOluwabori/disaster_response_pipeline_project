# Disaster Response Pipeline Projects

## Description 
The data set contains real messages that were sent during disaster events. 
```

├── data 
    ├── disaster_categories.csv  # data to process 
    ├── disaster_messages.csv  # data to process
    ├── process_data.py
    ├── disaster_response.db   # database to save clean data to

├── models
    ├── train_classifier.py
    ├── model.pkl  # saved model 

├── app
    ├── template
        ├── master.html  # main page of web app
        ├── go.html  # classification result page of web app
        ├── run.py  # Flask file that runs app

```

## Installations


-   [NumPy](http://www.numpy.org/)
-   [Pandas](http://pandas.pydata.org/)
-   [nltk](https://www.nltk.org/)
-   [scikit-learn](http://scikit-learn.org/stable/)
-   [sqlalchemy](https://www.sqlalchemy.org/)
-   [plotly](https://plotly.com/)

## Project Motivation
The goal of this project is to create a machine learning pipeline to categorize these events so that messages can be sent to an appropriate disaster relief agency.

### Objective
To develop a machine learning pipeline that can categorize disaster relief messages into relevant categories so that they can be directed to the appropriate disaster relief agency.

### Description
The project was designed to create a Natural Language Processing (NLP) tool that could categorize real-life disaster messages and tweets. The data used for the project was pre-labelled and contained messages sent during disaster events. The project was divided into three sections: ETL Pipeline, Machine Learning Pipeline, and Web Development.

The ETL (Extract, Transform, Load) Pipeline involved cleaning and pre-processing the data to get it ready for the machine learning model. The Machine Learning Pipeline used NLP techniques and the scikit-learn library to train a model on the pre-processed data. The final stage involved web development, where a Flask-based web application was developed to showcase the model's categorization results.

## Project Components
There are three components used to complete this project.

- ETL Pipeline
    process_data.py consists of a data cleaning pipeline that:
    - Loads the messages and categories datasets
    - Merges the two datasets
    - Cleans the data
    - Stores it in a SQLite database
- ML Pipeline
   train_classifier.py consists of a machine learning pipeline that:
   - Loads data from the SQLite database
   - Splits the dataset into training and test sets
   - Builds a text processing and machine learning pipeline
   - Trains and tunes a model using GridSearchCV
   - Outputs results on the test set
   - Exports the final model as a pickle file
- Flask Web App
   - Uploads the database file and pkl file with your model
   - Data visualizations created using Plotly in the web app
   - Display results in a Flask web app
   
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster_response.db models/model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## References
Datasets was gotten from [Appen](https://appen.com/datasets/combined-disaster-response-data/) (formally Figure Eight):
