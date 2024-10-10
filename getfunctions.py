#!/usr/bin/env python

import pandas as pd
import click as ck
import csv
from utils import (
    get_gene_ontology,
    get_anchestors,
    FUNC_DICT,
    EXP_CODES)
import os
from collections import deque
from aaindex import is_ok
from utils import (
    get_gene_ontology,
    get_go_set,
    get_anchestors,
    get_parents,
    DataGenerator,
    FUNC_DICT,
    MyCheckpoint,
    save_model_weights,
    load_model_weights,
    get_ipro)

DATA_ROOT = 'Wheat/sequence_Labels_Data/'

@ck.command()
@ck.option(
    '--function',
    default='mf',
    help='Function (mf, bp, cc)')
@ck.option(
    '--annot-num',
    # default=50,
    default=5,
    help='Limit of annotations number for selecting function')
def main(function, annot_num):
    global FUNCTION
    FUNCTION = function
    global GO_ID
    GO_ID = FUNC_DICT[FUNCTION]
    global go
    go = get_gene_ontology('go.obo')
    global functions
    functions = deque()
    dfs(GO_ID)
    functions.remove(GO_ID)
    functions = list(functions)
    print('len(functions):  {}!'.format(len(functions)))
    global func_set
    func_set = set(functions)
    print('len(func_set):  {}!'.format(len(func_set)))
    get_functions2(annot_num)

# Add functions to deque in topological order
def dfs(go_id):
    if go_id not in functions:
        for ch_id in go[go_id]['children']:
            dfs(ch_id)
        functions.append(go_id)


def get_functions2(annot_num):
    df = pd.read_pickle(DATA_ROOT + 'wheat_swissprot_exp2.pkl')

    folder_path = '\DSSP_folder_path'
    dssp_set = set()

    for filename in os.listdir(folder_path):
        prot_ID = filename.split('-')[1]
        dssp_set.add(prot_ID)
    print('dssp_set  :{}!'.format(len(dssp_set)))

    annots = dict()
    counts = 0
    for i, row in df.iterrows():
        go_set = set()
        if not is_ok(row['Sequences']) or row['Entry_Accessions'] not in dssp_set:
            continue
        annotations_list = row['annotations'].split(';')
        for go_id in annotations_list:
            if go_id in func_set:
                go_set |= get_anchestors(go, go_id)

        for go_id in go_set:
            if go_id not in annots:
                annots[go_id] = 0
            annots[go_id] += 1
        counts += 1

    filtered = list()
    for go_id in functions:
        if go_id in annots and annots[go_id] >= annot_num:
            filtered.append(go_id)

    print("counts:", counts)
    print(filtered)
    df = pd.DataFrame({'functions': filtered})
    df.to_pickle(DATA_ROOT + FUNCTION + '.pkl')
    print('Saved ' + DATA_ROOT + FUNCTION + '.pkl')


if __name__ == '__main__':
    main()
    # get_functions2()
