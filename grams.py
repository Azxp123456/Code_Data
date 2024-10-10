
import click as ck
import pandas as pd
from aaindex import is_ok

@ck.command()
@ck.option(
    '--length',
    default=1,
    help='Ngram length')

def main(length):
    seqs = get_sequences()
    ngrams = set()
    for seq in seqs:
        for i in range(len(seq) - length + 1):
            ngrams.add(seq[i: (i + length)])
    ngrams = list(sorted(ngrams))
    print('look↓')
    print(ngrams[:20])
    print('look↓')
    print(len(ngrams))
    df = pd.DataFrame({'1-grams': ngrams})
    df.to_pickle('Wheat/sequence_Labels_Data/1-grams.pkl')


def get_sequences():
    data = list()
    df = pd.read_pickle('Wheat/sequence_Labels_Data/wheat_swissprot_exp3.pkl')
    for i, row in df.iterrows():
        if is_ok(row['Sequences']):
            data.append(row['Sequences'])
    print(len(data))
    return data

if __name__ == '__main__':
    main()