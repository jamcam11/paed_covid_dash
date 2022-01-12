from cadmus import bioscraping


email = 's0565787@ed.ac.uk'
NCBI_API_KEY = 'eeb5b003283fba487503900053450dd7c507'
CROSSREF_API_KEY = '8f5ec160-9bea2c4f-55546a8b-787ac730'




bioscraping('(("intensive care"[TIAB] or "Critical Care"[TIAB]) AND (child* [TIAB] OR pediatric* [TIAB] OR paediatric* [TIAB])) AND (Cohort [Title] OR retrospective [Title] OR Prospective [Title] OR "Case Series" [Title])',
            email, NCBI_API_KEY, CROSSREF_API_KEY)