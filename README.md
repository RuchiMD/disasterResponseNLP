# DisasterResponseNLP
## Summary of the project

In this project, I am trying to predict the disaster category of a message. The dataset includes messages and their broad genres. It also includes 36 category lables for each message. 

## How to run the Python scripts and web app
## Dependencies:
pandas
sqlalchemy
matplotlib
nltk
sklearn

## For data preparation:
python ./dataPrep/process_data.py disaster_messages.csv disaster_categories.csv data.db

## For generating model classifier:
python ./trainClassifier/train_classifier.py ../data/data.db model_recent.pkl')

## For running the web app:
python ./app/run.py

Then open 'http://192.168.1.5:8000/' on your browser and you will see the following web page prompting you to type in a message.
![image](https://user-images.githubusercontent.com/33075751/136218736-8c3b1e94-b4d5-41c2-b3ee-73c36f821680.png)

If you type in a message eg: Need a place to stay for 40 homeless people. We get following output. 
![image](https://user-images.githubusercontent.com/33075751/136222182-211755f8-dc20-4f60-87b3-b23976e86e3e.png)

The web page also provides an overview of the dataset. Further explanation about the visuals is given below 

## Explanation of the files in the repository
1. process_data.py
load_data(messages_filepath, categories_filepath)
>Loads the messages and corresponding categories and merges them

visualise_data(df)
>Helps visualise the distribution of categories

clean_data(df)
>Generate individual category columns with binary values. Also drop duplicate entries

save_data(df, database_filename)
>create sqlite database

2. train_classfier.py
load_data(database_filepath)
>load the sqlite database

tokenize(text)
>normalise the words by making the strings case insensitive

build_model()
>build a grid search pipeline to train model using classifiers CountVectorizer, TfidfTransformer, MultiOutputClassifier.
I have used 80% of dataset for training and 20% for testing.
We can see warning messages informing that some classes are under represented.
>> UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in samples with no predicted labels. 
>> UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in samples with no true labels.

evaluate_model(model, X_test, Y_test, category_names)
>evaluate the model selected by grid search
The average precision offered by the model is 81%.
![result](https://user-images.githubusercontent.com/33075751/136200953-c4629013-3868-4b9b-9e3a-1cbd9f1df61b.PNG)

3. run.py
Visualise the dataset to get a preview of distribution and relationship between categories in dataset. 
We have various graphs to visualise the dataset:
>1. count of messages acroos different genres of messages
?2. Sort message categories across all genres from most popular to least. We can see that the dataset is biased. Some categories are not represented at all. 
![topmessage](https://user-images.githubusercontent.com/33075751/136167877-f690bbcd-3320-4760-a387-49dcd9b20294.png)
>3. 'Aid-related', 'weather-related', 'request' are the major categories. I have plotted count of other categories for each of these categories. 
![aid](https://user-images.githubusercontent.com/33075751/136167824-fef56107-8742-4590-866e-db86b5a948af.png)
![ewquest](https://user-images.githubusercontent.com/33075751/136168021-5e903149-9fe1-4bd3-bd64-879c6779e0ec.png)
![weather](https://user-images.githubusercontent.com/33075751/136168043-a5741a8a-594e-4c50-9c7f-6a68fd4b782e.png)

