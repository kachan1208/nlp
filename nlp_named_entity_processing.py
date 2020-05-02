import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import json
import random
import pprint

proposals = open('data/parsed_proposal_data.json', 'r')
data = json.loads(proposals.read())
proposals.close()
# random.shuffle(data)

nlp = en_core_web_sm.load()
for row in data[:10]:
    doc = nlp(row['text'])
    print(doc.ents)
    for geo in filter(lambda w: w.ent_type_ == "GPE", doc):
        print(geo)