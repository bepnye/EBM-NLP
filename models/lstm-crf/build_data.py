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

def generate_model_data(data_prefix = None):
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
    if data_prefix:
      cwd = os.getcwd()
      config.filename_dev   = os.path.join(cwd, 'data', data_prefix + '_' + os.path.basename(config.filename_dev))
      config.filename_test  = os.path.join(cwd, 'data', data_prefix + '_' + os.path.basename(config.filename_test))
      config.filename_train = os.path.join(cwd, 'data', data_prefix + '_' + os.path.basename(config.filename_train))

    if not os.path.isfile(config.filename_dev):
      print('Preprocessing tokens and labels to generate input data files')
      preprocess_data()

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
    pmid = os.path.splitext(os.path.basename(fname))[0].split('.')[0]
    return pmid

def preprocess_data():
    ebm_nlp = '../../ebm_nlp_2_00/'

    id_to_tokens = {}
    id_to_pos = {}
    PIO = ['participants', 'interventions', 'outcomes']
    PHASES = ['starting_spans', 'hierarchical_labels']

    batch_to_labels = {}
    for phase in PHASES:
      batch_to_labels[phase] = {}
      for pio in PIO:
        batch_to_labels[phase][pio] = {}
        print('Reading files for %s %s' %(phase, pio))
        for fdir in ['train', 'test/gold']:
          batch = fdir.split('/')[-1]
          batch_to_labels[phase][pio][batch] = dict()
          ann_fnames = glob('%s/annotations/aggregated/%s/%s/%s/*.ann' %(ebm_nlp, phase, pio, fdir))
          for fname in ann_fnames:
            pmid = fname_to_pmid(fname)
            if pmid not in id_to_tokens:
              tokens, tags = zip(*nltk.pos_tag(open('%s/documents/%s.tokens' %(ebm_nlp, pmid)).read().split()))
              id_to_tokens[pmid] = tokens
              id_to_pos[pmid] = tags
            batch_to_labels[phase][pio][batch][pmid] = open(fname).read().split()

    batch_groups = [('p1_all', ['starting_spans'], ['participants', 'interventions', 'outcomes']),
                    ('p2_p', ['hierarchical_labels'], ['participants']),
                    ('p2_i', ['hierarchical_labels'], ['interventions']),
                    ('p2_o', ['hierarchical_labels'], ['outcomes'])]
    for group_name, phases, pio in batch_groups:

      id_to_labels_list = defaultdict(list)
      batch_to_ids = defaultdict(set)
      for phase in phases:
        for e in pio:
          print('Collecting anns from %s %s' %(phase, e))
          for batch, batch_labels in batch_to_labels[phase][e].items():
            print('\t%d ids for %s' %(len(batch_labels), batch))
            batch_to_ids[batch].update(batch_labels.keys())
            for pmid, labels in batch_labels.items():
              labels = ['%s_%s' %(l, e[0]) for l in labels]
              id_to_labels_list[pmid].append(labels)

      for batch, ids in batch_to_ids.items():
        print('Found %d ids for %s' %(len(ids), batch))

      train_ids = list(batch_to_ids['train'] - batch_to_ids['gold'])
      print('Using %d ids for train' %len(train_ids))

      dev_idx = int(len(train_ids) * 0.2)
      dev_ids, train_ids = set(train_ids[:dev_idx]), set(train_ids[dev_idx:])
      print('Split training set in to %d train, %d dev' %(len(train_ids), len(dev_ids)))
      batch_to_ids['train'] = train_ids
      batch_to_ids['dev'] = dev_ids

      for batch, ids in batch_to_ids.items():
        fout = open('data/%s_%s.txt' %(group_name, batch), 'w')
        for pmid in ids:
          fout.write('-DOCSTART- -X- O O\n\n')
          tokens = id_to_tokens[pmid]
          poss = id_to_pos[pmid]
          per_token_labels = zip(*id_to_labels_list[pmid])
          for i, (token, pos, labels) in enumerate(zip(tokens, poss, per_token_labels)):
            final_label = 'N'
            for l in labels:
              if l[0] != '0':
                final_label = l
            fout.write('%s %s %s\n' %(token, pos, final_label))
            if token == '.': fout.write('\n')

if __name__ == "__main__":
  try:
    data_prefix = sys.argv[1]
  except IndexError:
    data_prefix = None

  generate_model_data(data_prefix)
