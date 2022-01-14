import pandas as pd
import numpy as np
import pickle
import spacy
import scispacy
from collections import Counter
import string
import re
from scispacy.abbreviation import AbbreviationDetector
from spacytextblob.spacytextblob import SpacyTextBlob
from scispacy.linking import EntityLinker
from negspacy.negation import Negex
from negspacy.termsets import termset
ts = termset("en_clinical")

# Intantiate the spacy program specifying the model you want to use
# we are using the scispaCy large vocabular model 750K term with vectors
print('Loading spaCy Model')
nlp = spacy.load("en_core_sci_lg")
print('Model Loaded')
nlp.max_size = 2000000

# set the ontology we are looking to match entities to.
ontology = 'umls' 

# Add the abbreviation pipe to the spacy pipeline.
nlp.add_pipe("abbreviation_detector")
# Add the hpo entity linker pipe to the spacy pipeline.
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True,
                                        "linker_name": ontology,
                                        'threshold':0.95,
                                        'max_entities_per_mention':1
                                       })
# add the sentiment analysis step
nlp.add_pipe('spacytextblob')
# add the negation detection
nlp.add_pipe("negex")
ts.add_patterns({
            "pseudo_negations": [],
            "termination": [],
            "preceding_negations": ["was no", "were no", "were never", "showed no", "a negative"],
            "following_negations": ["was negative", "remained negative", "was undetectable"],
        })




# set the path to the directory that holds our df of intereset
ctrl_path ='../picu_control_set/output/retrieved_df/retrieved_df2.p'

# read in the dataframe 
df = pickle.load(open(ctrl_path, 'rb'))
    

# this dictionary will hold the index as key and the doc object as the value
texts = {}

# when the text file is very big, we'll read in the first 1000000 cahracters (400000 words),  
print('cleaning up the text for processing')
for index, row in df.iterrows():
    # set the current text
    text = row['content_text']
    # check if not string or if empty string
    if type(text) != str or text == '':
        # we want to ignore these texts
        pass
    else:
        # if the text is valid, we check if it is very large
        # spacy by default happily copes with docs of 1M chrs.
        
        if len(text) > 1000000:
            text = text[:1000000]
        # get rid of Severe from the acronym sars cov2
        sars_sub = re.compile(re.escape('severe acute respiratory syndrome'), re.IGNORECASE)
        text = sars_sub.sub('SARS', text)
        texts.update({index:text})
        
print(f'{len(texts)} articles ready to process')

# this is a dictionary of CUIs with all the sentences attributed to that cui.
cui_d = {}

def entity_processing(index, doc, cui_d):
    onto_terms = []
    neg_onto_terms = []
    triggers = []
    neg_triggers =[]
    ents = [ent for ent in doc.ents]
    ents.extend([abrv for abrv in doc._.abbreviations])
    for ent in ents:
        # try mapping to HPO
        linker = nlp.get_pipe("scispacy_linker")
        if ent._.kb_ents:
            # if there is a "match/mention" you get a cui
            # we have set a high threshold and only keep one so there is only an index 0
            mention = ent._.kb_ents[0]
            if mention:
                # then we get the cui details (mainly the name)
                trigger = doc[ent.start:ent.end].text
                onto_term = linker.kb.cui_to_entity[mention[0]]
                if onto_term is not None:
                    # the result is a sring that needs a bit of parsing
                    onto_term = str(onto_term).split('\n')[0].split(', ')
                    cui = onto_term[0].replace('CUI: ','')
                    onto_term_str = onto_term[1].replace('Name: ','')
                    
                    if ent._.negex == True: 
                        # we then store attributes for each entity starting with HPO CUI
                        onto_terms.append(onto_term_str)
                        triggers.append(trigger)
                    else:
                        neg_onto_terms.append(onto_term_str)
                        neg_triggers.append(trigger)
                    # now we want to create a file for the cui or add to an existing one.
                    if cui_d.get(cui) == None:
                        cui_d.update({cui:{'indexes':[index],
                                           f'{ontology}_str':onto_term_str,
                                          'sents':[ent.sent.text],
                                          'negation':[ent._.negex],
                                          'polarity':[ent.sent._.polarity],
                                          'subj':[ent.sent._.subjectivity],
                                          'threshold':[mention[1]],
                                          'triggers':[trigger]}})
                    else:
                        cui_d[cui]['indexes'].append(index)
                        cui_d[cui]['sents'].append(ent.sent.text)
                        cui_d[cui]['negation'].append(ent._.negex)
                        cui_d[cui]['polarity'].append(ent.sent._.polarity)
                        cui_d[cui]['subj'].append(ent.sent._.subjectivity)
                        cui_d[cui]['threshold'].append(mention[1])
                        cui_d[cui]['triggers'].append(trigger)
    return onto_terms, triggers, neg_onto_terms, neg_triggers, cui_d


# set a counter for the texts within a df
doc_count=0

print('creating spacy doc objects and parsing out the entities')
# iterate through the texts in the current df
onto_d = {}
trigger_d = {}
neg_onto_d = {}
neg_trigger_d = {}
cui_d = {}
for index, text in texts.items():

    # increment count token to keep track of progress.
    doc_count+=1

    # create the spacy Doc Object
    doc = nlp(text)
    # apply our entity processing function to each entity (including abbreviations) in the doc
    # the output is a list of hpo mathced entities, one for each doc (which we will add to the retrieved df)
    # and a new dataframe indexed by each unique entity.
    onto_terms, trigger_terms, neg_onto_terms, neg_trigger_terms, cui_d = entity_processing(index, doc, cui_d)
    # we then add the hpos to the hpo_dictionary using the df index as our keys - to map back at the end.
    onto_d.update({index:onto_terms})
    trigger_d.update({index:trigger_terms})
    neg_onto_d.update({index:neg_onto_terms})
    neg_trigger_d.update({index:neg_trigger_terms})
    

    # print progress in df every 100 docs
    if doc_count % 100 == 0:
        print(f'Finished Doc {doc_count} of {len(df)}')
    else:
        pass



# lastly we'll add the df_ents to our paed_df and save it to file
df[f'{ontology}_ents'] = df.index.to_series().map(onto_d)
df[f'{ontology}_triggers'] = df.index.to_series().map(trigger_d)
# lastly we'll add the df_ents to our paed_df and save it to file
df[f'neg_{ontology}_ents'] = df.index.to_series().map(neg_onto_d)
df[f'neg_{ontology}_triggers'] = df.index.to_series().map(neg_trigger_d)
# save the dataframe
pickle.dump(df, open(f'./ctrl_spacy_{ontology}_df.p', 'wb'))
# # save the cui_df for entity exploration
cui_df = pd.DataFrame.from_dict(cui_d, orient = 'index')
pickle.dump(cui_df, open(f'./ctrl_{ontology}_cui_df.p', 'wb'))

print('Process Complete!')
