import pandas as pd
import spacy

from sklearn import metrics
import jsonlines

import matplotlib.pyplot as plt

from sklearn.metrics import DetCurveDisplay, RocCurveDisplay
from sklearn.metrics import det_curve

import numpy as np

nlp = spacy.load("/home/vs428/Documents/Incarceration/incarceration_model_binary/model-best")

data = pd.read_csv("/home/vs428/project/Incarceration_Data/ed_notes_19_20.tsv", 
                   sep="\t",
                   on_bad_lines="skip",
                   header=0,
                   engine="python"
                  # quoting=2
                  )

# test = data.sample(20)


hx_incarceration = []
with nlp.select_pipes(enable="textcat_multilabel"):
    for idx, doc in enumerate(nlp.pipe(data['TEXT'].astype(str).tolist(), 
                        # disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"],
                        n_process=5
                       )):
        if idx % 1000 == 0:
            print(idx, flush=True)
        
        # Do something with the doc here
        hx_incarceration.append(doc.cats['Prior_History_Incarceration'])

# FNR, FPR: (0.3333333333333333, 0.14814814814814814)
# THRESHOLD = 0.627554178237915
# FNR, FPR:  (0.25, 0.07407407407407407)
THRESHOLD = 0.856224000453949


data['hx_incarceration_nlp_score'] = hx_incarceration
data[f'hx_incarceration_nlp_pred_{THRESHOLD:.2f}'] = data['hx_incarceration_nlp_score'] >= THRESHOLD
print(data[f'hx_incarceration_nlp_pred_{THRESHOLD:.2f}'].value_counts(normalize=True))

data.to_csv("/home/vs428/project/Incarceration_Data/ed_notes_19_20_hx_incarceration_preds.csv", index=False)