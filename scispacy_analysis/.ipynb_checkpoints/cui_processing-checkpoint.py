import pickle
import pandas as pd
import numpy as np
from collections import Counter
import random
import itertools


# read in the cui_df
cui_df = pickle.load(open('./cui_df.p', 'rb'))

# read in the retrieved df with the hpo ents column present
df = pickle.load(open('hpo_ret_df2.p', 'rb'))

# get the total count of each cui for the corpus
lens = [len(set(indexes)) for indexes in cui_df['indexes']]
cui_df['doc_count'] = lens

# get the document count of each cui
lens = [len(indexes) for indexes in cui_df['indexes']]
cui_df['cui_count'] = lens

# get the keywords for the hpo_df based on hpo entities
def tfidf(df, cui_df):
    tfidf_d = {}
    total_docs = len(df)
    
    # get rid of none values and replace with empty lists.
    doc_ents = []
    for ents in df['hpo_ents']:
        if type(ents) == list:
            doc_ents.append(ents)
        else:
            doc_ents.append([])  
    df['hpo_ents'] = doc_ents
    
    count = 0
    for doc_ents in df['hpo_ents']:
        holding_d = {}
        counter_d = Counter(doc_ents)
        doc_ent_counts = len(doc_ents)
        for ent in doc_ents:
            cui_i = cui_df.index[list(cui_df['hpo_str']).index(ent)]
            # tf = number of terms in the doc / total num terms in the doc
            term_freq = counter_d[ent] / doc_ent_counts
            # DF(t) = log_e(Total number of documents / Number of documents with term t in it)
            inv_doc_freq = np.log(total_docs / cui_df.loc[cui_i,'doc_count'])
            tfidf = term_freq * inv_doc_freq
            holding_d.update({ent:tfidf})
        tfidf_d.update({df.index[count]:holding_d})
        count+=1
        
    return tfidf_d, df

# apply fcuntion to our dataframes
tfidf_d, df = tfidf(df, cui_df)
# convert dict to dataframe
tfidf_df = pd.DataFrame.from_dict(tfidf_d, orient = 'index')


# now we'll use the tfidf_df to generate the top ten, most important entites per article.
top_terms = {}
for index, row in tfidf_df.iterrows():
    top_ten = row.sort_values(ascending=False)[:10]
    # use the keys to store the top key words
    kws = list(top_ten.keys())
    top_terms.update({index:kws})
# lastly we'll add the kws to our df and save it to file
df['top_hpo'] = df.index.to_series().map(top_terms)


# now we'll get the most important 10 sentences for each cui (hpo entity) using tfidf
import spacy
import scispacy
nlp = spacy.load('en_core_sci_lg')

# function to process a cui_df to rank the entity sentences by most interesting.
# we are going to use the tfidf vectorizer and then normalize for number of entities in the sentence
def tfidf_best_sents(cui_df):
    # holding list for the top sentences for each term
    top_sents = []
    
    # dummy funct for vectorizer
    def dummy(ents):
        return(ents)
    
    # import the tfidf vectorizer from sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    # instantiate with dummy funct for preprocessing and tokenization (we will already have the entities as a list)
    tfidf_vectorizer = TfidfVectorizer(preprocessor = dummy, tokenizer = dummy, ngram_range = (1,1))

    # iterate through the data frame, processing each term and term sentences
    count = 0
    for index, row in cui_df.iterrows():
        # holding list for the spacy ents
        ents = []
        # set the term explicitly
        term = row['hpo_str']
        # set the sentences for that term
        sents = row['sents']
        # process each sentence, extracting the ents
        # we can batch these to make it run a bit quicker
        for doc in nlp.pipe(sents, batch_size = 500):
            ents.append([str(ent) for ent in doc.ents])


        # now we have the ents, lets make a tfidf_vector, one vector for each sentence
        tfidf_vec = tfidf_vectorizer.fit_transform(ents)
        # sum the values for the vecor
        sums = [sum(vec) for vec in tfidf_vec.toarray()]
        # normalise the tfidf sum using the number of ents in that sentence
        norm_sum = [a / b for a, b in zip(sums, [len(ent_list) for ent_list in ents])]
        
        # build a dataframe to sort by norm sum
        df = pd.DataFrame(data={'sents':sents, 'ents':ents, 'sums':sums, 'norm_sum':norm_sum})
        
        # do the subset for sents with at least 4 entites present 
        condition = [len(ents) >= 4 for ents in df['ents']]
        df = df.loc[condition, :]
        
        # sort by largest normalised tfidf sum
        df.sort_values(by = 'norm_sum', ascending = False, inplace = True)
        # pick the top ten sentences
        df = df.head(10)
        # add these sentences to the cui dataframe
        top_sents.append(list(df['sents']))
        count +=1
        
        if count%100 ==0:
            print(f'{count} of {len(cui_df)}')
    cui_df['top_sents'] = top_sents
        
    return cui_df

# apply the function to our cui_df
cui_df = tfidf_best_sents(cui_df)
# save the cui_df with the best sentences saved in a new column
pickle.dump(cui_df, open('./cui_df.p', 'wb'))