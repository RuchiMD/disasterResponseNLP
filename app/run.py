import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib#from sklearn.externals import joblib
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
#engine = create_engine(r'C:\Users\ruchi\OneDrive\projects\document\datascience_nanodegree\Disaster-Response-Pipeline-master\Disaster-Response-Pipeline-master\data\disaster_response_data.db')
engine = create_engine('sqlite:///../data\DisasterResponse.db')
df = pd.read_sql_table('FigureEight', engine)

# load model
model = joblib.load(r"C:\Users\ruchi\OneDrive\projects\document\datascience_nanodegree\Disaster-Response-Pipeline-master\Disaster-Response-Pipeline-master\models\model.pkl")
    #("..\models\model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    #first graph
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #second graph
    category = list(df.columns[4:])
    category_counts = []
    genre_category_counts = []
    for column_name in category:
        category_counts.append(np.sum(df[column_name]))

    categories = df.iloc[:,4:]
    categories_mean = categories.mean().sort_values(ascending=False)#[1:11]
    categories_names = list(categories_mean.index)
    
    #third graph
    print(df.groupby(['aid_related']).sum().reset_index())
    df_grouped = df.groupby(['aid_related']).sum().reset_index()
    df_grouped = df_grouped.drop(columns=['id', 'aid_related'])
    df_aid = df_grouped.iloc[1].sort_values(ascending=False)
    df_aid_col = list(df_aid.index)

    print(df.groupby(['request']).sum().reset_index())
    df_grouped = df.groupby(['request']).sum().reset_index()
    df_grouped = df_grouped.drop(columns=['id', 'request'])
    df_request = df_grouped.iloc[1].sort_values(ascending=False)
    df_request_col = list(df_request.index)

    print(df.groupby(['weather_related']).sum().reset_index())
    df_grouped = df.groupby(['weather_related']).sum().reset_index()
    df_grouped = df_grouped.drop(columns=['id', 'weather_related'])
    df_weather = df_grouped.iloc[1].sort_values(ascending=False)
    df_weather_col = list(df_weather.index)


    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_mean         
                )
            ],

            'layout': {
                'title': 'Top 10 Message Categories',
                'yaxis': {
                    'title': "Percentage", 
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
		        'data': [
                Bar(
                    x=df_aid_col,#aid_names,
                    y=df_aid,#aid_counts
                )
            ],

            'layout': {
                'title': 'Relationship of aid-related messages with other catergories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Catergories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=df_request_col,  # aid_names,
                    y=df_request,  # aid_counts
                )
            ],

            'layout': {
                'title': 'Relationship of request messages with other catergories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=df_weather_col,  # aid_names,
                    y=df_weather,  # aid_counts
                )
            ],

            'layout': {
                'title': 'Relationship of weather related messages with other catergories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=8000, debug=True)


if __name__ == '__main__':
    main()