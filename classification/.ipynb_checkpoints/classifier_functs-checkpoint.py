# Script to hold all the functions for my Bayesian Classification Model

# libraries
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
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report as report

# import the nltk stopwords
from nltk.corpus import stopwords



# define our preprocessing to use in the Count vectoriser
def preprocessing(text):
    
    stop_words = stopwords.words('english')

    # i've found some additional stop words i want removed from our working text prior to tokenization, feel free to add to this
    new_stops =['http','doi','org','medrxiv','manuscript','preprint','license','creativecommons','et','al',
               'https','ti','kw','ab','nc','nd','cc','yes','no','www']

    stop_words = stop_words + new_stops
    
    """Basic cleaning of texts."""
    
    # remove html markup
    text=re.sub("(<.*?>)","",text)
    
    #remove non-ascii and digits
    text=re.sub("(\\W|\\d)"," ",text)
    
    #remove whitespace
    text=text.strip().lower()
    
    
    
    return text


def tokenizer(text):
    
    stop_words = stopwords.words('english')

    # i've found some additional stop words i want removed from our working text prior to tokenization, feel free to add to this
    new_stops =['http','doi','org','medrxiv','manuscript','preprint','license','creativecommons','et','al',
               'https','ti','kw','ab','nc','nd','cc','yes','no','www']

    stop_words = stop_words + new_stops
    
    """Tokenizing and stemming words"""
    
    # split on whitespace
    tokens = re.split("\\s+",text)
    #     # porter stemmer on each token
    #     stemmed_words=[porter_stemmer.stem(word = token) for token in tokens if (len(token) > 1) and (token not in stopwords)]
    # lemmatize each token
    
    # Init the Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    lemmas = [lemmatizer.lemmatize(token) for token in tokens if (len(token) > 1) and (token not in stop_words)]
    
    return lemmas

def dummy(text):
    return text


def training_text_vectoriser(df, col):

    # lets set up the Count Vectoriser
    # NB these parameters could be tweaked using grid search to see if we can do better.
    cv = CountVectorizer(preprocessor=dummy,
                     tokenizer=dummy,
                     lowercase=True,
                     binary=True,
                     ngram_range=(1,1),
                     min_df=1,
                     max_df=1.0)

    # fit the count vector to our training set. 
    count_vec = cv.fit_transform(df[col]).toarray()

    print('saving the count vector array for the training set')
    # store the count vecor array to the dataframe
    df['cv'] = [vec for vec in count_vec]
    
    return df, cv

# the count vector has a value for each term in the vocabulary
# we want a to count the number of included documents with each term present
# then we'll want to count the number of excluded documents with the term present
# this is probably easiest by splitting the training df into an included and excluded df
# then we can apply a binary Count vecotriser to each document in each dataframe.

def term_table_maker(df, class_col, cv):

    # note I have set the included class as 1 and excluded as 0 
    # we split the training set to do this quickly and simply, grouped by inclusion and exclusion class column
    inc_df = df[df[class_col] == 1]
    ex_df = df[df[class_col] != 1]

    print('counting up each term for each class')
    # for every term in the vector we can sum the counts for each class 
    # this list comprehension takes every position in the list of vectors 
    # (each list corresponds to a class and each document has its own vector)
    inc_term_counts = [sum(t) for t in zip(*inc_df['cv'])]
    inc_absent_counts = [(len(inc_df) - count) for count in inc_term_counts]
    ex_term_counts = [sum(t) for t in zip(*ex_df['cv'])]
    ex_absent_counts = [(len(ex_df) - count) for count in ex_term_counts]

    print('making the token df')
    # now we can map the term counts onto a dataframe to visualise the count for each term for each class.
    # can set the index as the term from the count vec dictionary
    token_df = pd.DataFrame(data={'pres_inc':inc_term_counts,
                                  'abs_inc':inc_absent_counts,
                                  'pres_ex':ex_term_counts,
                                  'abs_ex':ex_absent_counts},
                                  index=cv.get_feature_names())

    return token_df


# now we move on to calculating the fishers exact sores and likelihood ratios.
# now lets make a fishers exact pval

def fishers_exact(token_df):
    # we'll store the pvals and ORs for use later
    pvals = []
    ORs = []
    ppvs = []
    npvs = []
    LRs = []
    
    # we are using a laplacean correction to avoid infinity and 0 in our calculations
    # this is the equivalent of adding a documnet with all terms and an empty document to each class
    for term in token_df.index: 
        pres_inc = token_df.loc[term, 'pres_inc'] +1
        abs_inc = token_df.loc[term, 'abs_inc'] +1
        pres_ex = token_df.loc[term, 'pres_ex'] +1
        abs_ex = token_df.loc[term, 'abs_ex'] +1
        
        ppv = np.round(pres_inc/(pres_inc + pres_ex), 3)
        npv = np.round(abs_ex/(abs_ex + abs_inc), 3)

        table = np.array([[pres_inc, abs_inc],[pres_ex, abs_ex]])
        OR, pval = scipy.stats.fisher_exact(table, alternative='two-sided')
        
        # now get the likelihood ratio
        LR = (pres_inc/(pres_inc+abs_inc))/(pres_ex/(pres_ex+abs_ex))
        
        pvals.append(pval)
        ORs.append(OR)
        ppvs.append(ppv)
        npvs.append(npv)
        LRs.append(LR)
        
       

    token_df['LR'] = LRs 
    token_df['fe_pval'] = pvals
    token_df['OR'] = ORs
    token_df['ppv'] = ppvs
    token_df['npv'] = npvs

    return token_df

def bonf_correction(token_df):
    
    print('Now setting up Bonferroni Correction')
    # correct the pvalues using bonferoni, one step correction.
    # create an original pvalue vector for use in the bonf correction - set as a np array of float values
    pval_vec = np.asarray(token_df['fe_pval'].astype(float))
    # set the original p-value we consider sig.
    alpha = 0.05
    # assign output variables from correction test, apply test on pval vec
    reject, pval_corr, sidak, bonf = smm.multipletests(pval_vec, alpha=alpha, method='bonferroni')

    # show the new significance threshold based on the number of tests
    print(f'New alpha = {bonf}')

    # assign the corrected pval vector to the df
    pval_corr = [np.format_float_scientific(pval, precision=3, exp_digits=2) for pval in pval_corr]
    token_df['corr_pval'] = pval_corr
    # add whether or not the null hypothesis can be rejected based on the corrected pval 
    token_df['reject_null'] = reject
    
    # we are only going to return the significant tokens
    sig_df = token_df[token_df['reject_null'] == True]
    
    print(f'This training set yields {len(sig_df)} terms that reach significance')
    return sig_df

def test_set_split(df):
    # this function evaluates the size of a retrieved df and splits it in to chunks of 50,000 for more reasonable chunks for memory.
    if len(df) >50000:
        print('This is a large dataset')
        indexes = df.index
        
    else:
        print('THis dataset is less than 50,000\nNo need to split it up.')
        


def test_count_vectoriser(test_df, token_df, col):
# now we move on to classifying the unseen or test set

    # lets set up the Count Vectoriser using just the significant vocab
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(preprocessor=dummy,
                         tokenizer=dummy,
                         lowercase=True,
                         binary=True,
                         ngram_range=(1,1),
                         vocabulary=token_df.index)

    # we now need to create count vectors for each article in the test set
    count_vectors = cv.fit_transform(test_df[col]).toarray()
    # save the Cvs to the dataframe
    test_df['cvs'] = [list(vec) for vec in count_vectors]

    return test_df



# we'll use the presence or absence to create a boolean vector for each sig term
def vec_to_bool(count_vec):
    result = [True if val ==1 else False for val in count_vec]
    return result


def bayes_classifier(test_df, token_df, threshold):
    inc_count = token_df['pres_inc'][0] + token_df['abs_inc'][0]
    total_count = token_df['pres_inc'][0] + token_df['abs_inc'][0] + token_df['pres_ex'][0] + token_df['abs_ex'][0]
    prior_prob = inc_count/total_count
    prior_inc_odds = prior_prob /(1-prior_prob)

    post_probs = []
    inc_ln_odds = []
    
    for vec in test_df['cvs']:
        boolean = vec_to_bool(vec)
        prior = prior_inc_odds
        cv_LR_vec = list(compress(token_df['LR'], boolean))
        inc_odds = prior * np.prod(cv_LR_vec)
        inc_ln_odds.append(np.log(inc_odds))
        
        inc_prob = inc_odds/(1 + inc_odds)
        post_probs.append(inc_prob)
    
    test_df['post_prob'] = post_probs
    test_df['inc_ln_odds'] = inc_ln_odds
    test_df['pred'] = [1 if val >=threshold else 0 for val in post_probs]
    
    test_df.drop(columns = 'cvs', inplace = True)
    
    return test_df
        

def evaluate_classifier(test_df, col='post_prob', threshold=1):
    # nb this will only work on preclassified dataframes
    # set the scores we are using ('inc_ln_odds' or 'inc_prob')

    inc_df = test_df[test_df['class'] == 1]
    ex_df = test_df[test_df['class'] == 0]

    # for each threshold create a confusion matrix
    true_pos = sum(inc_df[col] >= threshold)
    false_neg = sum(inc_df[col] < threshold)
    true_neg = sum(ex_df[col] < threshold)
    false_pos = sum(ex_df[col] >= threshold)

    # calculate the recall for this threshold value
    recall  = (true_pos)/(true_pos+false_neg)

    precision = (true_pos)/(true_pos+false_pos)

    accuracy = (true_pos + true_neg)/(false_pos + false_neg + true_pos + true_neg)

    # calculate the matthews correlation coefficient (aparently better than f1 for recall and precision tasks though they seem the same most of the time)
    mcc = (true_pos * true_neg - false_pos * false_neg)/np.sqrt((true_pos+false_pos)*(true_pos+false_neg)*(true_neg+false_pos)*(true_neg+false_neg))  

    f1 = 2*((precision*recall)/(precision+recall))

    confusion_matrix = np.array([[true_pos, false_neg], [ false_pos, true_neg]])
    
    summary_text = f'Posterior Probability of {threshold} - Summary\n\
    F1 score = {f1}\n\
    MCC score = {mcc}\n\
    Confusion Matrix = \n{confusion_matrix}\n\
    Precision = {precision}\n\
    Recall = {recall}\n\
    Accuracy = {accuracy}\n'
    
    summary_d = {"summary_text": summary_text,
                "recall":recall,
                "precision":precision,
                "accuracy":accuracy,
                "mcc":mcc,
                "f1":f1,
                "confusion_matrix":confusion_matrix}
    
    return summary_d