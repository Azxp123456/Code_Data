import pandas as pd
import numpy as np
from utils import EXP_CODES, get_gene_ontology
import os
import requests
from aaindex import is_ok
import gzip

DATA_ROOT = 'Wheat/sequence_Labels_Data/'


def to_pickle():
    Entry_Accessions = list()
    Entry_Name = list()
    annotations = list()
    Sequences = list()
    with open(DATA_ROOT + 'wheat.tab') as f:
        for line in f:
            items = line.strip().split('\t')
            Entry_Accessions.append(items[0])
            Entry_Name.append(items[1])
            annotations.append(items[2])
            Sequences.append(items[3])
        print("len(Entry_Accessions): {}!".format(len(Entry_Accessions)))
        print("len(Entry_Name): {}!".format(len(Entry_Name)))
        print("len(annotations): {}!".format(len(annotations)))
        print("len(Sequences): {}!".format(len(Sequences)))
    seq_df = pd.DataFrame({
        'Entry_Accessions': Entry_Accessions,
        'Entry_Name': Entry_Name,
        'annotations': annotations,
        'Sequences': Sequences
    })

    seq_df.to_pickle(DATA_ROOT + 'wheat_swissprot.pkl')

def filter_sample():
    df = pd.read_pickle(DATA_ROOT + 'wheat_swissprot.pkl')

    index = list()
    for i, row in df.iterrows():
        if is_ok(row['Sequences']):
            index.append(i)
    df = df.loc[index]

    df.to_pickle(DATA_ROOT + 'wheat_swissprot_exp2.pkl')


if __name__ == '__main__':
    # to_pickle()
    filter_sample()