# import libraries
from sklearn.utils import parallel_backend
import sys
import pickle
import nltk
import pandas as pd
from sqlalchemy import create_engine
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import make_pipeline, FeatureUnion,Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
nltk.download(['stopwords','punkt','wordnet','omw-1.4'])

# load data from database
def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM master',con=engine)
    df.loc[df['related'] == '2','related'] = '1'
    X = df['message']
    Y = df.iloc[:,4:]
    Y=Y.astype('int')

    return (X,Y,Y.columns)

# this is tokenization function to process the text data
def tokenize(text):
    tokens = word_tokenize(text)
    
    tok=[WordNetLemmatizer().lemmatize(tok, pos='v') for tok in tokens]
    tok=[tok for tok in tokens if tok not in stopwords.words("english")]
         
        
    return tokens


def build_model():
    '''
    This creates creates a pipeline and uses GridSearch to find better parameters.
    '''
    pipeline = Pipeline([ 
    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
   ('clf', MultiOutputClassifier(RandomForestClassifier(max_depth=3)))
    ])
    
    parameters =  {'clf__estimator__n_estimators': [15]}

    cv = GridSearchCV(pipeline, parameters,verbose=3, n_jobs=-1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''This function tests the model and outputs the f1 score,
     precision, recall and accuracy for each output category of the dataset'''
    y_pred=model.predict(X_test)
    print(classification_report(pd.DataFrame(Y_test), pd.DataFrame(y_pred,columns=category_names),target_names=category_names))
    print((y_pred == Y_test).mean())


def save_model(model, model_filepath):
    ''' 
    This function takes in the model and filepath
    It export the model as a pickle file'''
    filename = 'model.pkl'
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        #Split data into train and test sets and train pipeline
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        with parallel_backend('multiprocessing'): 
            print('Building model...')
            model = build_model()
            
            print('Training model...')
            model.fit(X_train, Y_train)
            print(model.best_params_)
            print('Evaluating model...')
            evaluate_model(model, X_test, Y_test, category_names)

            print('Saving model...\n    MODEL: {}'.format(model_filepath))
            save_model(model, model_filepath)

            print('Trained model saved!')

    else:
            print('Please provide the filepath of the disaster messages database '\
                'as the first argument and the filepath of the pickle file to '\
                'save the model to as the second argument. \n\nExample: python '\
                'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()