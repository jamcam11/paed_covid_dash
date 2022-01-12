# this script aims to classify unseen articles into binary classes
# for this example the classes are severe and non-severe covid cases
# the set is drawn from PubMed using paediatrics and COVID-19 synonyms

# libraries
from classifier_functs import *
import pandas as pd
import numpy as np
import re
import string
import nltk
import scipy.stats as stats
import pickle
from collections import Counter
from sklearn import metrics
import math
from sklearn.feature_extraction.text import CountVectorizer
import scipy
import statsmodels.stats.multitest as smm
from itertools import compress
from pathlib import Path


# read in the retrieved df from our search protocol
ret_df = pickle.load(open('/home/jcampbell/paed_covid_case_reports/scispacy_analysis/hpo_ret_df2.p', 'rb'))
# the classifier is a supervised approach requiring manually classified documents
# read in the classified df 
# the indexes in this df correspond to the indexes in the retrieved df so we can transpose the classifications
class_df = pickle.load(open('/home/jcampbell/paed_covid_case_reports/classification/article_screening/class_df.p', 'rb'))

# loop through the class df and transpose the manual classification to the retireved df using the shared indexes
for index, row in class_df.iterrows():
    # class 2 is severe paediatric covid in the class df, the classes are 2= severe paeds covid, 1 = mild/mod/asymptomatic paeds covid, 0= non relevant artilce.
    if row['test'] == 2:
        # for binary classification, we will make this class 1 in our retrieved df
        ret_df.loc[index, 'class'] = 1
    else:
        # anything that is not severe, will be classified as a 0
        ret_df.loc[index, 'class'] = 0
# the training df is made up of all the classified documents.
train_df = pd.DataFrame(ret_df.loc[class_df.index, :])
print(f'\nWe\'ve got {len(train_df)} articles to train the model with')

# The test df will be the retrieved df, for simplicity we will apply the classifier to the whole corpus
test_df = ret_df

print('vectorising the training set')
# first we must vectorise the trainig data
train_df, cv = training_text_vectoriser(train_df, col='umls')

print('Building the token count dataframe from the training set')
# next we create a token dataframe to count the presence or absence of a given term in the included (severe) and excluded (non severe covid) set
token_df = term_table_maker(train_df, class_col='class', cv=cv)

# free up some memory by deleting the training df and ret df (now called test df)
del train_df
del ret_df
                         

print("Now performing Fishers exact test for token df")
# now we calculate the enrichment of each term in the include set/exclude set
token_df = fishers_exact(token_df)
print("correcting for multiple tests (bonferoni)")
# now correct for multiple testing to adjust the pvalues
token_df = bonf_correction(token_df)
pickle.dump(token_df, open(f'./token_df.p', 'wb'))


######## Classifying the test dfs
print('classifying the test df')
# we want to only keep the full text artilces 
# lets generate a drop list 
drop_list = []
# loop through the df and look for the conditions
print('Dropping the Empty fields and abstracts')
for index, row in test_df.iterrows():
    # we dont want any None values or those marked as ABS
    if any([row['umls'] == [], type(row['umls']) != list]):
        drop_list.append(index)

# drop the artilces we failed to retrieve (no UMLS tags idenified at all)
test_df.drop(labels = drop_list, inplace = True)
print(f'We\'ve got {len(test_df)} articles to classify')
# we now have a token df to use for our classifier
# lets test it against the test df
# first we process and vectorise the test df content text
print('Preparing test set for classification')
test_df = test_count_vectoriser(test_df, token_df, col = 'umls')

print("classifying Test set")
# this function takes the test df and adds columns for posterior probabiliy, log_odds of inclusion and binary class prediciton based on the threshold (threshold calculated by crossvalidation and threshold increamentation of the training set.)
test_df = bayes_classifier(test_df, token_df, threshold = 0.99)


pickle.dump(test_df, open('./classified_df.p', 'wb'))

print('complete')
