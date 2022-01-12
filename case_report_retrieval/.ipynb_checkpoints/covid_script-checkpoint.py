#!/home/jcampbell/miniconda3/envs/gopher/bin/python3.8
import os 

os.chdir('/home/jcampbell/paed_covid_case_reports/case_report_retrieval/')


from cadmus import bioscraping


email = 's0565787@ed.ac.uk'
NCBI_API_KEY = 'eeb5b003283fba487503900053450dd7c507'
CROSSREF_API_KEY = '8f5ec160-9bea2c4f-55546a8b-787ac730'



print('starting bioscraping')
val = bioscraping('(\
            (COVID-19[Title] OR Sars-CoV-2[Title] OR Coronavirus[Title] OR "severe acute respiratory syndrome coronavirus 2"[Title])\
            AND \
            (child[TIAB] OR children[TIAB] OR pediatric*[TIAB] OR paediatric*[TIAB] OR adolescen*[TIAB] OR infan*[TIAB] ))\
            AND \
            (cohort[Title] OR retrospective[Title] OR prospective[Title] OR "case series"[Title] OR "Multicenter Study"[pt] OR "Observational Study"[pt] OR "Comparative Study"[pt] OR "Case Reports"[pt])\
            NOT \
            (pregnan*[Title] OR breast[Title] or milk[Title] OR postpartum[Title] or adult[Title])\
            AND (English[LANG])\
            AND (2019/11/01:3000/01/01[Publication Date])',
            email, NCBI_API_KEY, CROSSREF_API_KEY)
print('bioscraping complete')

print('Changing directory to classification')
# once the retrieval is complete we need to classify the retrieved documents
os.chdir('/home/jcampbell/paed_covid_case_reports/classification')
print('running Niaive Bayes classifer for retrieved docs')
import clin_bayes_classifier.py
print('classfication complete')
