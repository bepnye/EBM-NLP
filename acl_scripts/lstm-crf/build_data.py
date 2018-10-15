from model.config import Config
from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word
import json, re, os, sys
import nltk
from nltk import pos_tag, sent_tokenize
from nltk import ngrams
from glob import glob
from collections import defaultdict

def main():
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """

    # get config and processing of words
    # loads PubMeda articles
    config = Config(load=False)
    print('Config')
    processing_word = get_processing_word(lowercase=True)
    print('Processing_word')

    # Generators
    dev   = CoNLLDataset(config.filename_dev, processing_word)
    test  = CoNLLDataset(config.filename_test, processing_word)
    train = CoNLLDataset(config.filename_train, processing_word)
    print('Loaded dev, test, train')

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])
    print('Loading vocab_words')
    vocab_glove = get_glove_vocab(config.filename_glove)

    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)

    # Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # Trim GloVe Vectors
    vocab = load_vocab(config.filename_words)
    export_trimmed_glove_vectors(vocab, config.filename_glove,
                                config.filename_trimmed, config.dim_word)

    # Build and save char vocab
    train = CoNLLDataset(config.filename_train)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.filename_chars)

def index_map(span, indices):
    for i in range(len(indices)):
        if indices[i][0] <= span[0]:
            begin = i
        elif span[0] >= indices[i][0] and span[0] <= indices[i][1]:
            begin = i
        if indices[i][1] >= span[1]:
            end = i
            break
        elif span[1] >= indices[i][0] and span[1] <= indices[i][1]:
            end = i
            break
        if i == len(indices)-1: end = i
    return begin, end+1

def tokenize(s):
    """
    :param s: string of the abstract
    :return: list of word with original positions
    """
    def white_char(c):
        return c.isspace() or c in [',', '?']
    res = []
    i = 0
    while i < len(s):
        while i < len(s) and white_char(s[i]): i += 1
        l = i
        while i < len(s) and (not white_char(s[i])): i += 1
        r = i
        if s[r-1] == '.':       # consider . a token
            res.append( (s[l:r-1], l, r-1) )
            res.append( (s[r-1:r], r-1, r) )
        else:
            res.append((s[l:r], l, r))
    return res

def fname_to_pmid(fname):
    pmid = os.path.splitext(os.path.basename(fname))[0].split('_')[0]
    return pmid

def pre_main():
    batch_to_labels = {}
    id_to_tokens = {}
    id_to_pos = {}
    PIO = ['participants', 'interventions', 'outcomes']

    for pio in PIO:
      print('Reading files for %s' %pio)
      for fdir in ['train', 'test/gold', 'test/student']:
        batch = fdir.split('/')[-1]
        ann_fnames = glob('../../ebm_nlp_1_00/annotations/aggregated/starting_spans/%s/%s/*.ann' %(pio, fdir))
        for fname in ann_fnames:
          pmid = fname_to_pmid(fname)
          if pmid not in id_to_tokens:
            tokens, tags = zip(*nltk.pos_tag(open('../../ebm_nlp_1_00/documents/%s.tokens' %pmid).read().split()))
            id_to_tokens[pmid] = tokens
            id_to_pos[pmid] = tags
          if batch not in batch_to_labels:
            batch_to_labels[batch] = defaultdict(dict)
          batch_to_labels[batch][pmid][pio] = open(fname).read().split(',')

    for batch, batch_labels in batch_to_labels.items():
      for pmid, labels in batch_labels.items():
        if len(labels) != len(PIO):
          print 'Bad annotations for %s %s, only found %s' %(batch, pmid, ''.join(labels.keys()))

    batch_to_ids = { batch: set(batch_labels.keys()) for batch, batch_labels in batch_to_labels.items() }
    for batch, ids in batch_to_ids.items():
      print 'Found %d ids for %s' %(len(ids), batch)

    train_ids = list(batch_to_ids['train'] - batch_to_ids['student'] - batch_to_ids['gold'])
    print 'Using %d ids for train' %len(train_ids)

    dev_idx = int(len(train_ids) * 0.2)
    dev_ids, train_ids = set(train_ids[:dev_idx]), set(train_ids[dev_idx:])
    print 'Split training set in to %d train, %d dev' %(len(train_ids), len(dev_ids))

    batch_to_labels['dev'] = defaultdict(dict)
    for pmid in dev_ids:
      batch_to_labels['dev'][pmid] = batch_to_labels['train'][pmid]
    for pmid in batch_to_labels['train'].keys():
      if pmid not in train_ids:
        del batch_to_labels['train'][pmid]

    print 'Final batch sizes:'
    for batch, batch_labels in batch_to_labels.items():
      print '%s: %d' %(batch, len(batch_labels))

    for batch, id_to_labels in batch_to_labels.items():
      fout = open('data/%s.txt' %batch, 'w')
      for pmid in id_to_labels:
        fout.write('-DOCSTART- -X- O O\n\n')
        tokens = id_to_tokens[pmid]
        poss = id_to_pos[pmid]
        for i, (token, pos) in enumerate(zip(tokens, poss)):
          labels = [int(id_to_labels[pmid][pio][i]) for pio in PIO]
          label = 'N'
          for pio, is_applied in zip(PIO, labels):
            if is_applied and label == 'N':
              label = pio[0]
          fout.write('%s %s %s\n' %(token, pos, label))
          if token == '.': fout.write('\n')

if __name__ == "__main__":
    pre_main()
    main()
