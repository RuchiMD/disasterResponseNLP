# disasterResponseNLP
##summary of the project

In this project, I am trying to predict the disaster category of a message. The dataset includes messages and their broad genres. It also includes 36 category lables for each message. 

##How to run the Python scripts and web app
FOr data preparation:
python ./data/process_data.py disaster_messages.csv disaster_categories.csv response_data.db

FOr generating model classifier:
python ./model/train_classifier.py ../data/response_data.db model.pkl')

For running the web app:
python ./app/run.py

##Explanation of the files in the repository
1. process_data.py
load_data(messages_filepath, categories_filepath)
>>>>>>Loads the messages and corresponding categories and merges them
visualise_data(df)
>>>>>>Helps visualise the distribution of categories
clean_data(df)
>>>>>>Generate individual category columns with binary values. Also drop duplicate entries
save_data(df, database_filename)
>>>>>>create sqlite database

2. train_classfier.py
load_data(database_filepath)
>>>>>>load the sqlite database
tokenize(text)
>>>>>>normalise the words by making the strings case insensitive
build_model()
>>>>>>build a grid search pipeline to train model using classifiers CountVectorizer, TfidfTransformer, MultiOutputClassifier.
evaluate_model(model, X_test, Y_test, category_names)
>>>>>>evaluate the model selected by grid search

3. run.py
Visualise the dataset to get a preview of distribution and relationship between categories in dataset.
We have various graphs to visualise the dataset:
1. count of messages acroos different genres of messages
2. Sort message categories across all genres from most popular to least. We can see that the dataset is biased. Some categories are not represented at all.
3. 'Aid-related', 'weather-related', 'request' are the major categories. I have plotted count of other categories for each of these categories. 

