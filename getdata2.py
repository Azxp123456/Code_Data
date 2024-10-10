
import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from utils import (
    get_gene_ontology,
    get_go_set,
    get_anchestors,
    FUNC_DICT,
    EXP_CODES)
from aaindex import is_ok
import tensorflow as tf
import click as ck
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from sklearn.preprocessing import MinMaxScaler

target_shape = (224, 224)

# DATA_ROOT = '/root/autodl-tmp/CNNAT/Japonica/sequence_Labels_Data'
DATA_ROOT = 'Wheat/sequence_Labels_Data/'

@ck.command()
@ck.option(
    '--function',
    default='cc',
    help='Function (mf, bp, cc)')
@ck.option(
    '--split',
    default=0.8,
    help='Train test split')

# 1
def main(function, split):
    global SPLIT
    SPLIT = split
    global GO_ID
    GO_ID = FUNC_DICT[function]
    global go
    go = get_gene_ontology('go.obo')
    global FUNCTION
    FUNCTION = function
    func_df = pd.read_pickle(DATA_ROOT + FUNCTION + '.pkl')
    global functions
    functions = func_df['functions'].values
    global func_set
    func_set = get_go_set(go, GO_ID)
    func_set.remove(GO_ID)
    print('len(functions): {}!'.format(len(functions)))
    print('len(func_set): {}!'.format(len(func_set)))
    global go_indexes
    go_indexes = dict()
    for ind, go_id in enumerate(functions):
        go_indexes[go_id] = ind
    print('len(go_indexes): {}!'.format(len(go_indexes)))
    run()

def load_data():
    ngram_df = pd.read_pickle(DATA_ROOT + '1-grams.pkl')

    vocab = {}
    for key, gram in enumerate(ngram_df['1-grams']):
        vocab[gram] = key + 1

    gram_len = len(ngram_df['1-grams'][0])
    print(('Gram length:', gram_len))  # 1
    print(('Vocabulary size:', len(vocab)))     # 20

    Entry_Accessions = list()
    Entry_Name = list()
    gos = list()
    labels = list()
    ngrams = list()
    Sequences = list()
    ss_onehot = list()
    df = pd.read_pickle(DATA_ROOT + 'wheat_swissprot_exp3.pkl')

    index = list()
    for i, row in df.iterrows():
        if is_ok(row['Sequences']):
            index.append(i)
    df = df.loc[index]

    for i, row in df.iterrows():
        go_list = row['annotations'].split(';')
        go_set = set()
        for go_id in go_list:
            if go_id in func_set:
                go_set |= get_anchestors(go, go_id)
        if not go_set or GO_ID not in go_set:
            continue
        go_set.remove(GO_ID)
        gos.append(go_list)
        Entry_Accessions.append(row['Entry_Accessions'])
        Entry_Name.append(row['Entry_Name'])
        ss_onehot.append(row['SS_onehot'])
        seq = row['Sequences']
        Sequences.append(seq)
        grams = np.zeros((len(seq) - gram_len + 1, ), dtype='int32')
        for i in range(len(seq) - gram_len + 1):
            grams[i] = vocab[seq[i: (i + gram_len)]]
        ngrams.append(grams)
        label = np.zeros((len(functions),), dtype='int32')
        for go_id in go_set:
            # if go_id in go_list and go_id in go_indexes:
            if go_id in go_indexes:
                label[go_indexes[go_id]] = 1
        labels.append(label)
    res_df = pd.DataFrame({
        'Entry_Accessions': Entry_Accessions,
        'Entry_Name': Entry_Name,
        'Sequences': Sequences,
        'ngrams': ngrams,
        'SS_onehot': ss_onehot,
        'labels': labels,
        'gos': gos})

    cMap_path = "\contact_map_path"
    record_None = 0
    cMap_data = []
    entryAccessions = []
    filelist = os.listdir(cMap_path)
    for cMap_fn in filelist:
        entry_accession = cMap_fn.split("-")[0]     # ID
        entryAccessions.append(entry_accession)

        file_path = os.path.join(cMap_path, cMap_fn)
        contactMapArray = np.load(file_path)
        dataArray = np.array(contactMapArray)[:, :, np.newaxis]
        cMap_data.append(dataArray)

    cMap_df = pd.DataFrame({
        'Entry_Accessions': entryAccessions,
        'cMap_data': cMap_data
    })

    merge_df = pd.merge(res_df, cMap_df, on='Entry_Accessions')
    print('record_None', record_None)
    return merge_df

# 2
def run(*args, **kwargs):

    df = load_data()

    index = df.index.values

    np.random.seed(seed=0)
    np.random.shuffle(index)

    tv_n = int(len(df) * SPLIT)
    test_df = df.loc[index[tv_n:]]

    train_n = int(tv_n * SPLIT)
    valid_df = df.loc[index[train_n:tv_n]]
    train_df = df.loc[index[:train_n]]

    train_df.to_pickle(DATA_ROOT + 'cc-train_data' + '.pkl')
    valid_df.to_pickle(DATA_ROOT + 'cc-valid_data' + '.pkl')
    test_df.to_pickle(DATA_ROOT + 'cc-test_data' + '.pkl')

    print("over!")

if __name__ == '__main__':
    # run()
    main()

