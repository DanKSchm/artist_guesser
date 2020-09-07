## Artist guesser

#### Project description:

Scrapes all song texts by four different artists from the website lyrics.com and builds a machine learning model.
It then takes any string as input and returns the name of the artist to whom the lyrics/string fits best.

I chose very few and distinct artists so the model wouldn’t fail.


#### Used methods:
    
Web Sraping, Regular Expressions, Bag of Words, Naive Bayes


#### The workflow can be described as follows:

1) Build a function that extracts all lyrics from an artist and store it in a list
2) Convert lyrics to PandaDataFrame, save as CSV and add Artist name
3) Define X and Y and make train_test_split
4) Import and instantiate CountVectorizer (with the default parameters)
5) learn the 'vocabulary' of the training data
6) Transform training and test data into a 'document-term matrix'
7) Convert sparse matrix to a dense matrix
8) Chose and define model
9) Save models on harddisk with pickle
10) Build user input function so that the user can interact with the model