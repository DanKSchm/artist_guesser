#!/usr/bin/env python
# coding: utf-8

# ## Import relevant libraries




import pickle


with open('vectorizer.pickle', 'rb') as f:
    vect = pickle.load(f)

with open('model.pickle', 'rb') as f:
    nb = pickle.load(f)

testtext = input("Please type in any text: ")

testtext_dtm = vect.transform([testtext])
which_artist = nb.predict(testtext_dtm)
prob_artist = nb.predict_proba(testtext_dtm)

print(which_artist)
print(prob_artist)

print(" This lyric snatch fits best to the artist " + str(which_artist) + " with a probability of " +  str(max(prob_artist[0])))

