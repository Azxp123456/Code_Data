
from get_StructureData import preprocess
import click as ck
import pandas as pd
import pickle
from keras.models import load_model
from aaindex import INVALID_ACIDS
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import sequence
import tensorflow as tf
from Japonica.MFRA import Inception, Multi_scale_1D_conv


MAXLEN = 1002
funcs = ['mf', 'bp', 'cc']

@ck.command()
@ck.option('--data_file', '-f', default='Japonica/sample_test/cc-select_samples.pkl', help='Pickle file with Protein prediction data')
@ck.option('--model_path', '-pmf', default='Japonica/sample_test/model_Multi-Fusion-ssscc.h5', help='Pretrained model weight file')
@ck.option('--functions', '-funcs', default='Japonica/sample_test/cc.pkl', help='label space')
@ck.option('--threshold', '-t', default=0.35, help='Prediction threshold')
@ck.option('--out_file', '-o', default='Japonica/sample_test/cc-results.tsv', help='Output result file')

def main(data_file, model_path, functions, out_file, threshold):
    data_df = pd.read_pickle(data_file)
    model = load_model(model_path, custom_objects={'Multi_scale_1D_conv': Multi_scale_1D_conv, 'Inception': Inception})

    functions_df = pd.read_pickle(functions)
    functions = functions_df['functions']
    print('len(functions):', (len(functions)))
    print('Predictions started: ')

    w = open(out_file, 'w')
    if len(data_df) == 1:
        row = data_df.iloc[0]
        result_pred, result_true = predict_single_sample(model, row, functions, threshold)

        w.write(row['Entry_Accessions'])
        w.write('\n')
        w.write('result_pred: ' + str(result_pred))
        w.write('\n')
        w.write('result_true: ' + str(result_true))
        w.write('\n')
    else:
        entries, result_preds, result_trues = predict_batch_samples(model, data_df, functions, threshold)

        for entry, result_pred, result_true in zip(entries, result_preds, result_trues):
            w.write(entry)
            w.write('\n')
            w.write('result_pred: ' + str(result_pred))
            w.write('\n')
            w.write('result_true: ' + str(result_true))
            w.write('\n')
    w.close()


def predict_single_sample(model, row, functions, threshold):
    result_pred = list()
    result_true = list()

    ngrams = sequence.pad_sequences(row['ngrams'], maxlen=MAXLEN)

    ss_onehot = row['SS_onehot']
    padded_ss_onehot = pad_sequences([ss_onehot], maxlen=1002, padding='post', truncating='post', value=0.0)[0]

    cMap_data = row['cMap_data']
    cMap_data = [tf.convert_to_tensor(cMap_data, dtype=tf.int32)]
    padded_images_tensor = preprocess(cMap_data)

    labels = row['labels']

    data = [ngrams, padded_ss_onehot, padded_images_tensor]
    predictions = model.predict([data])[0]
    pred = (predictions >= threshold).astype('int32')
    for j in range(len(functions)):
        if pred[j] == 1:
            result_pred.append('_' + functions[j] + '|' + '%.2f' % predictions[j])
    for j in range(len(functions)):
        if labels[j] == 1:
            result_true.append('_' + functions[j] + '|')
    return result_pred, result_true


def predict_batch_samples(model, data_df, functions, threshold):
    def reshape(values):
        values = np.hstack(values).reshape(len(values), len(values[0]))
        return values

    result_preds = []
    result_trues = []
    entries = []

    ngrams = sequence.pad_sequences(data_df['ngrams'].values, maxlen=MAXLEN)
    ngrams = reshape(ngrams)
    print("ngrams.shape:    {}!".format(ngrams.shape))

    ss_onehots = data_df['SS_onehot'].apply(lambda x: pad_sequences([x], maxlen=1002, padding='post', truncating='post', value=0.0)[0])
    ss_stacked_data = np.stack(ss_onehots)
    print("ss_stacked_data.shape:    {}!".format(ss_stacked_data.shape))

    cMap_datas = data_df['cMap_data']
    cMap_datas = [tf.convert_to_tensor(x, dtype=tf.int32) for x in cMap_datas]
    padded_images_tensor = preprocess(cMap_datas)
    print("Padded Image Data List Shape: {}!".format(padded_images_tensor.shape))

    labels_list = reshape(data_df['labels'].values)
    print("labels_list.shape:    {}!".format(labels_list.shape))

    predictions_batch = model.predict([ngrams, ss_stacked_data, padded_images_tensor])
    for i, predictions in enumerate(predictions_batch):
        result_pred = list()
        result_true = list()
        labels = labels_list[i]

        pred = (predictions >= threshold).astype('int32')
        for j in range(len(functions)):
            if pred[j] == 1:
                result_pred.append('_' + functions[j] + '|' + '%.2f' % predictions[j])
        for j in range(len(functions)):
            if labels[j] == 1:
                result_true.append('_' + functions[j] + '|')

        result_preds.append(result_pred)
        result_trues.append(result_true)
        entries.append(data_df.iloc[i]['Entry_Accessions'])

    return entries, result_preds, result_trues

def is_ok(seq):
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True


if __name__ == '__main__':
    main()