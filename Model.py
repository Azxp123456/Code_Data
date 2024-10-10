#!/usr/bin/env python

"""
  python nn_hierarchical_seq.py
"""
import sys
sys.path.append('/root/autodl-tmp/CNNAT')


from get_StructureData import preprocess
from tensorflow.keras.layers import Layer, Dense, Activation, Multiply
from tensorflow.keras.layers import Input, LSTM, Bidirectional
from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_recall_curve, roc_curve, auc, matthews_corrcoef
import numpy as np
import pandas as pd
import click as ck
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from tensorflow.keras.models import Sequential, Model, load_model  # 更改
#from keras.layers import (Dense, Dropout, Activation, Input, Flatten, Highway, merge, BatchNormalization)
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D, Flatten, InputLayer, \
    Dense, GlobalMaxPooling1D, concatenate, Maximum, Concatenate, add, Reshape
from tensorflow.keras.layers import Dropout, Activation, BatchNormalization  # 更改
from tensorflow.keras.layers import Embedding, MaxPooling1D  # 更改
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta      # 更改
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Lambda
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import sequence
import sys
from collections import deque
import time
import logging
from scipy.spatial import distance
from multiprocessing import Pool

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# K.set_session(sess)

# Allow GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
sys.setrecursionlimit(100000)

DATA_ROOT = '/root/autodl-tmp/CNNAT/Japonica/sequence_Labels_Data/'

NUM_heads = 8
MAXLEN = 1002
ind = 0
max_size = (1002, 1002)

@ck.command()
@ck.option(
    '--function',
    default='cc',
    help='Ontology id (mf, bp, cc)')
@ck.option(
    '--device',
    # default='gpu:0',
    default='GPU:0',
    help='GPU or CPU device id')
@ck.option('--train', is_flag=True)

# 1
def main(function, device, org, train):   # org
    global FUNCTION
    FUNCTION = function
    global GO_ID
    GO_ID = FUNC_DICT[FUNCTION]
    global go
    go = get_gene_ontology('go.obo')
    global ORG
    ORG = org
    func_df = pd.read_pickle(DATA_ROOT + FUNCTION + '.pkl')
    global functions
    functions = func_df['functions'].values
    print('len(functions):  {}!'.format(len(functions)))
    global func_set
    func_set = set(functions)
    print('len(func_set):  {}!'.format(len(func_set)))
    global all_functions
    all_functions = get_go_set(go, GO_ID)
    logging.info('Functions start: %s %d' % (FUNCTION, len(functions)))
    global go_indexes
    go_indexes = dict()
    for ind, go_id in enumerate(functions):
        go_indexes[go_id] = ind
    global node_names
    node_names = set()
    with tf.device('/' + device):
        model(is_train=train)


def generate_sequence(start):
    sequence = [start]
    current_value = start

    while current_value < 513:
        current_value *= 2
        if current_value <= 512:
            sequence.append(current_value)
    return sequence

# 3
def load_data():

    train_df = pd.read_pickle(DATA_ROOT + 'cc-train_data.pkl')
    valid_df = pd.read_pickle(DATA_ROOT + 'cc-valid_data.pkl')
    test_df = pd.read_pickle(DATA_ROOT + 'cc-test_data.pkl')

    # 5
    def reshape(values):
        values = np.hstack(values).reshape(len(values), len(values[0]))
        return values

    # 4
    def get_values(data_frame):
        #
        labels = reshape(data_frame['labels'].values)
        print("labels.shape:    {}!".format(labels.shape))

        # Dense mapping of amino acid ngrams             MAXLEN = 1002  Filling Data
        ngrams = sequence.pad_sequences(data_frame['ngrams'].values, maxlen=MAXLEN)
        ngrams = reshape(ngrams)
        print("ngrams.shape:    {}!".format(ngrams.shape))

        # Onehot coding of the secondary structure of the amino acid
        ss_onehot_data = data_frame['SS_onehot'].apply(lambda x: pad_sequences([x], maxlen=1002, padding='post', truncating='post', value=0.0)[0])
        ss_stacked_data = np.stack(ss_onehot_data)
        print("ss_stacked_data.shape:    {}!".format(ss_stacked_data.shape))

        # Structural contact map data
        Map_data = data_frame['cMap_data']
        Map_data = [tf.convert_to_tensor(x, dtype=tf.int32) for x in Map_data]
        print("len(Map_data):    {}!".format(len(Map_data)))
        padded_images_tensor = preprocess(Map_data)
        print("Padded Image Data List Shape: {}!".format(padded_images_tensor.shape))


        data = (ngrams, ss_stacked_data, padded_images_tensor)
        # data = (ngrams, ss_stacked_data)
        # data = ngrams
        return data, labels

    train = get_values(train_df)
    valid = get_values(valid_df)
    test = get_values(test_df)

    return train, valid, test, train_df, valid_df, test_df

#  7 Feature extraction sub-model —— Protein sequence branch —— Extracting protein sequence features
class Multi_scale_1D_conv(tf.keras.layers.Layer):
    def __init__(self, kernel_size, **kwargs):
        super(Multi_scale_1D_conv, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = Conv1D(filters=512, kernel_size=self.kernel_size, padding='valid', kernel_initializer='glorot_normal')
        self.pool = MaxPooling1D(pool_size=1002 - self.kernel_size + 1)
        self.flat = Flatten()

    def call(self, inputs):
        conv = self.conv(inputs)
        pool = self.pool(conv)
        flat = self.flat(pool)
        return flat

    def get_config(self):
        # Serialize layer configuration
        config = {
            'kernel_size': self.kernel_size
        }
        base_config = super(Multi_scale_1D_conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ResidualAttention(tf.keras.Model):

    def __init__(self, num_class=500, la=0.5):
        super(ResidualAttention, self).__init__()
        self.la = la
        self.fc = tf.keras.layers.Conv2D(filters=num_class, kernel_size=1, strides=1, padding='valid', use_bias=False)

    def call(self, x):
        b, h, w, c = x.shape
        y_raw = self.fc(x)
        y_avg = tf.reduce_mean(y_raw, axis=[1, 2])
        y_max = tf.reduce_max(y_raw, axis=[1, 2])
        score = y_avg + self.la * y_max
        return score

#  Feature extraction sub-model —— Protein structure branch —— Extracting protein structural features
class Multi_Conv2D(tf.keras.layers.Layer):
    def __init__(self, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, ch9x9r1, ch9x9r2, ch9x9r3, ch9x9r4, ch9x9r5, pool_proj, **kwargs):
        super(Multi_Conv2D, self).__init__()

        self.ch1x1 = ch1x1
        self.ch3x3red = ch3x3red
        self.ch3x3 = ch3x3
        self.ch5x5red = ch5x5red
        self.ch5x5 = ch5x5
        self.ch9x9r1 = ch9x9r1
        self.ch9x9r2 = ch9x9r2
        self.ch9x9r3 = ch9x9r3
        self.ch9x9r4 = ch9x9r4
        self.ch9x9r5 = ch9x9r5
        self.pool_proj = pool_proj

        '''branch1'''
        self.branch1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.ch1x1, kernel_size=(1, 1)),
            ResidualAttention(num_class=500, la=0.5)
        ])

        '''branch2'''
        self.branch2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.ch3x3red, kernel_size=(3, 3), padding='same'),
            tf.keras.layers.Conv2D(self.ch3x3, kernel_size=(1, 1)),
            ResidualAttention(num_class=500, la=0.5)
        ])

        '''branch3'''
        self.branch3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.ch5x5red, kernel_size=(5, 5), padding='same'),
            tf.keras.layers.Conv2D(self.ch5x5, kernel_size=(1, 1), padding='same'),
            ResidualAttention(num_class=500, la=0.5)
        ])

        '''branch6'''
        self.branch6 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.ch9x9r1, kernel_size=(1, 9), padding='same'),
            tf.keras.layers.Conv2D(self.ch9x9r2, kernel_size=(9, 1), padding='same'),
            tf.keras.layers.Conv2D(self.ch9x9r3, kernel_size=(1, 3), padding='same'),
            tf.keras.layers.Conv2D(self.ch9x9r4, kernel_size=(3, 1), padding='same'),
            tf.keras.layers.Conv2D(self.ch9x9r5, kernel_size=(1, 1), padding='same'),
            ResidualAttention(num_class=500, la=0.5)
        ])

        '''branch4'''
        self.branch4 = tf.keras.Sequential([
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=1, padding='same'),
            tf.keras.layers.Conv2D(self.pool_proj, kernel_size=(1, 1)),
            ResidualAttention(num_class=500, la=0.5)
        ])

    def call(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        branch6 = self.branch6(x)

        outputs = [branch1, branch2, branch3, branch4, branch6]
        return tf.concat(outputs, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'ch1x1': self.ch1x1,
            'ch3x3red': self.ch3x3red,
            'ch3x3': self.ch3x3,
            'ch5x5red': self.ch5x5red,
            'ch5x5': self.ch5x5,
            'ch9x9r1': self.ch9x9r1,
            'ch9x9r2': self.ch9x9r2,
            'ch9x9r3': self.ch9x9r3,
            'ch9x9r4': self.ch9x9r4,
            'ch9x9r5': self.ch9x9r5,
            'pool_proj': self.pool_proj
        })
        return config


def merge_outputs(outputs, name):
    if len(outputs) == 1:
        return outputs[0]
    return Concatenate(axis=1, name=name)(outputs)


def merge_nets(nets, name):
    if len(nets) == 1:
        return nets[0]
    return add(nets, name=name)

def get_node_name(go_id, unique=False):
    name = go_id.split(':')[1]
    if not unique:
        return name
    if name not in node_names:
        node_names.add(name)
        return name
    i = 1
    while (name + '_' + str(i)) in node_names:
        i += 1
    name = name + '_' + str(i)
    node_names.add(name)
    return name

# √
def get_function_node(name, inputs):
    output_name = name + '_out'
    output = Dense(1, name=output_name, activation='sigmoid')(inputs)
    return output, output


def get_layers(inputs):
    q = deque()
    layers = {}
    name = get_node_name(GO_ID)

    layers[GO_ID] = {'net': inputs}
    for node_id in go[GO_ID]['children']:
        if node_id in func_set:
            q.append((node_id, inputs))

    while len(q) > 0:
        node_id, net = q.popleft()
        name = get_node_name(node_id)
        net, output = get_function_node(name, inputs)
        if node_id not in layers:

            layers[node_id] = {'net': net, 'output': output}

            for n_id in go[node_id]['children']:
                if n_id in func_set and n_id not in layers:
                    ok = True
                    for p_id in get_parents(go, n_id):
                        if p_id in func_set and p_id not in layers:
                            ok = False
                    if ok:
                        q.append((n_id, net))

    for node_id in functions:
        childs = set(go[node_id]['children']).intersection(func_set)
        if len(childs) > 0:
            outputs = [layers[node_id]['output']]
            for ch_id in childs:
                outputs.append(layers[ch_id]['output'])
            name = get_node_name(node_id) + '_max'
            layers[node_id]['output'] = Maximum(name=name)(outputs)
    return layers


def get_model():
    logging.info("Start building the model!")
    embedding_dims = 10
    max_features = 21
    kernels = generate_sequence(8)
    print(kernels)

    # Protein sequence branch
    seq_input = Input(shape=(MAXLEN,), dtype='int16', name='input1')   # Sequence Input
    grams_embed = Embedding(max_features, embedding_dims, input_length=MAXLEN, embeddings_initializer='uniform')(seq_input)
    print("grams_embed.shape:    {}!".format(grams_embed.shape))

    ss_onehot_input = Input(shape=(1002, 9), dtype='float16', name='ss_onehot_input')   # Secondary structure input
    print("ss_onehot_input.shape:    {}!".format(ss_onehot_input.shape))
    concat_vector = Concatenate()([grams_embed, ss_onehot_input])
    print("concatenated_vector.shape:    {}!".format(concat_vector.shape))

    stru_input = Input(shape=(224, 224, 1), dtype='float16', name='input3')   # Distance structure input


    cnn_branch = []
    for i, kernel in enumerate(kernels):
        flat = Multi_scale_1D_conv(kernel)(concat_vector)
        cnn_branch.append(flat)
    net = Concatenate(axis=1)(cnn_branch)
    print("net.shape:    {}!".format(net.shape))

    # Structural branches
    Multi_Conv2D_a = Multi_Conv2D(16, 128, 32, 128, 32, 128, 128, 64, 64, 32, 32)
    x = Multi_Conv2D_a(stru_input)
    flat_layer = tf.keras.layers.Flatten()(x)
    print("flat_layer.shape:    {}!".format(flat_layer.shape))

    merge_features = Concatenate()([net, flat_layer])
    print("merge_features.shape:    {}!".format(merge_features.shape))

    net = Dense(1024, activation='relu', name='dense')(merge_features)

    layers = get_layers(net)
    output_models = []
    for i in range(len(functions)):
        output_models.append(layers[functions[i]]['output'])
    net = concatenate(output_models, axis=1)

    model = Model(inputs=[seq_input, ss_onehot_input, stru_input], outputs=net)
    # model.summary()
    logging.info('Compiling the model')
    optimizer = Adam(learning_rate=0.0003)

    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    logging.info('Compilation finished')
    return model

# 2
def model(batch_size=32, nb_epoch=200, is_train=True):
    start_time = time.time()
    logging.info("Start Loading Data!")
    train, valid, test, train_df, valid_df, test_df = load_data()
    logging.info("Finished data load")

    train_df = pd.concat([train_df, valid_df])
    test_gos = test_df['gos'].values
    train_data, train_labels = train
    valid_data, valid_labels = valid
    test_data, test_labels = test

    logging.info("Data loaded in %d sec" % (time.time() - start_time))
    logging.info("Training data size: %d" % len(train_data))
    logging.info("Validation data size: %d" % len(valid_data))
    logging.info("Test data size: %d" % len(test_data))

    model_path = DATA_ROOT + 'model_Multi-Fusion-ccsss' + FUNCTION + '.h5'

    checkpointer = ModelCheckpoint(filepath=model_path, verbose=2, save_best_only=True)

    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    logging.info('Starting training the model')

    logging.info('### Start dividing data ###')
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_data, valid_labels))
    valid_dataset = valid_dataset.batch(batch_size)
    valid_dataset = valid_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    logging.info('### data dividing ends!!! ###')

    if is_train:
        model = get_model()
        model.fit(
            train_dataset,
            epochs=nb_epoch,
            validation_data=valid_dataset,
            callbacks=[checkpointer, earlystopper])

    logging.info('Loading best model')
    model = load_model(model_path, custom_objects={'Multi_scale_1D_conv': Multi_scale_1D_conv, 'Multi_Conv2D': Multi_Conv2D})

    logging.info('Start making predictions using the test set')

    preds = model.predict(test_dataset)
    print(preds.shape, test_labels.shape)

    logging.info('Computing performance')

    f, p, r, t, preds_max = compute_performance(preds, test_labels)
    print(len(preds))
    print(len(test_labels))
    aupr = compute_aupr(test_labels, preds)
    mcc = compute_mcc(preds_max, test_labels)

    print("Fmax: {:.2f}".format(f))
    print("Precision: {:.2f}".format(p))
    print("Recall: {:.2f}".format(r))
    print("Threshold: {:.2f}".format(t))
    print("aupr:", aupr)
    print("mcc:", mcc)

    end_time = time.time()

    total_time = end_time - start_time
    print(f"Total model running time: {total_time} s")

# √
def compute_roc(preds, labels):
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


def compute_mcc(preds, labels):
    preds = preds.astype(np.float64)
    labels = labels.astype(np.float64)
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc


def compute_aupr(labels, preds):
    p, r, _ = precision_recall_curve(labels.flatten(), preds.flatten())
    aupr = auc(r, p)
    return aupr


def average_auprc(y_true_list, y_scores_list):

    total_auprc = 0.0

    for y_true, y_scores in zip(y_true_list, y_scores_list):
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        area = auc(recall, precision)
        total_auprc += area

    average_auprc = total_auprc / len(y_true_list)

    return average_auprc

# Evaluation Metrics
def compute_performance(preds, labels):
    print("Preds shape:", preds.shape)
    print("Labels shape:", labels.shape)

    preds = np.round(preds, 2)
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        total = 0
        f = 0.0
        p = 0.0
        r = 0.0
        p_total = 0
        for i in range(labels.shape[0]):
            tp = np.sum(predictions[i, :] * labels[i, :])
            fp = np.sum(predictions[i, :]) - tp
            fn = np.sum(labels[i, :]) - tp

            if tp == 0 and fp == 0 and fn == 0:
                continue
            total += 1
            if tp != 0:
                p_total += 1
                precision = tp / (1.0 * (tp + fp))
                recall = tp / (1.0 * (tp + fn))
                p += precision
                r += recall
        if p_total == 0:
            continue
        r /= total
        p /= p_total
        if p + r > 0:
            f = 2.0 * p * r / (p + r)
            if f_max < f:
                f_max = f
                p_max = p
                r_max = r
                t_max = threshold
                predictions_max = predictions
    return f_max, p_max, r_max, t_max, predictions_max


if __name__ == '__main__':
    main()