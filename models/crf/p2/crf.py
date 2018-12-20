import os, sys, re
import nltk
from collections import Counter
from sklearn import linear_model
from scipy.sparse import csr_matrix
from glob import glob
from collections import defaultdict
from operator import itemgetter
import pickle

from sklearn.metrics import precision_score, recall_score, f1_score
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

LEMMATIZER = nltk.stem.WordNetLemmatizer()
DOC_PKL = 'docs.pkl'
#EMBEDDINGS = 'embeddings.pkl'
#EMBEDDING_SIZE = 200

TOP = '../../ebm_nlp_2_00/'
UNK = '<UNK>'
BOS = '<BOS>'
EOS = '<EOS>'

def get_wordnet_pos(pos):
  if pos.startswith('J'): return nltk.corpus.wordnet.ADJ
  if pos.startswith('V'): return nltk.corpus.wordnet.VERB
  if pos.startswith('N'): return nltk.corpus.wordnet.NOUN
  if pos.startswith('R'): return nltk.corpus.wordnet.ADV
  return pos

def lemmatize(token_pos):
  token, pos = token_pos
  try:
    return LEMMATIZER.lemmatize(token, pos = get_wordnet_pos(pos)).lower()
  except KeyError:
    return LEMMATIZER.lemmatize(token).lower()

def build_data():

  print('Reading, lemmatizing, and pos taggings documents...')

  docs = {}
  doc_fnames = glob('%s/documents/*.tokens' %(TOP))
  all_tokens = set()

  for i, fname in enumerate(doc_fnames):
    pmid = os.path.basename(fname).split('.')[0]

    tokens = open(fname).read().split()
    tagged_tokens = nltk.pos_tag(tokens)
    lemmas = map(lemmatize, tagged_tokens)
    _, pos = zip(*tagged_tokens)
    docs[pmid] = {}
    docs[pmid]['tokens'] = tokens
    docs[pmid]['lemmas'] = lemmas
    docs[pmid]['pos'] = [p[:2] for p in pos]

    all_tokens.update(tokens)

    if (i // 100 != (i-1) // 100):
      sys.stdout.write('\r\tprocessed %04d / %04d docs' %(i, len(doc_fnames)))
      sys.stdout.flush()

  with open(DOC_PKL, 'wb') as fout:
    print('\nWriting doc data to %s' %DOC_PKL)
    pickle.dump(docs, fout)

  #if not os.path.isfile(EMBEDDINGS):
  #  print 'Formatting word vectors'
  #  embeddings = {}
  #  source_embeddings = open('PubMed-w2v.txt')
  #  for l in source_embeddings:
  #    e = l.strip().split()
  #    if len(e) < EMBEDDING_SIZE:
  #      continue
  #    if e[0] in all_tokens:
  #      embeddings[e[0]] = map(float, e[1:])
  #  with open(EMBEDDINGS, 'w') as fout:
  #    pickle.dump(embeddings, fout)

  return docs


def predict_labels_crf(docs = None):
  
  print('Initializing docs...')
  docs = docs or pickle.load(open(DOC_PKL))

  for pio in ['interventions']:#['participants', 'interventions', 'outcomes']:
    print('Running crf for %s' %pio)
    test_fnames  = glob('%s/annotations/aggregated/hierarchical_labels/%s/test/gold/*.ann' %(TOP, pio))
    train_fnames = glob('%s/annotations/aggregated/hierarchical_labels/%s/train/*.ann' %(TOP, pio))

    print('Reading labels for %d train and %d test docs' %(len(train_fnames), len(test_fnames)))
    test_labels = { os.path.basename(f).split('_')[0]: open(f).read().split(',') for f in test_fnames }
    train_labels = { os.path.basename(f).split('_')[0]: open(f).read().split(',') for f in train_fnames }

    train_pmids = sorted(train_labels.keys())
    test_pmids = test_labels.keys()

    assert len(set(test_pmids) & set(train_pmids)) == 0

    print('Computing test/train features')
    train_X = [doc2features(docs[p]) for p in train_pmids]
    test_X  = [doc2features(docs[p]) for p in test_pmids]

    train_Y = [train_labels[p] for p in train_pmids]
    test_Y  = [test_labels[p]  for p in test_pmids]

    crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.4, c2=0.005, max_iterations=100, all_possible_transitions=True)
    print('Fitting model for %d pmids' %len(train_pmids))
    crf.fit(train_X, train_Y)
    labels = list(crf.classes_)
    pred_Y = crf.predict(test_X)

    labels.remove('0')
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        test_Y, pred_Y, labels=sorted_labels, digits=3
    ))

def any_are(f, l):
  return any(map(f, l))

def is_float(s):
  try:
    float(s)
    return True
  except ValueError:
    return False

def word2features(i, tokens, tags, lemmas):

    w = tokens[i]
    features = {
        'bias': 1.0,
        'init_upper': w[0].isupper(),
        'internal_upper': any_are(str.isupper, w[1:]),
        'all_upper': w.isupper(),
        'all_lower': w.islower(),
        'is_float': is_float(w),
        'is_alpha': w.isalpha(),
        'has_nums_letters': any_are(str.isdigit, w) * any_are(str.isalpha, w),
        'has_oper': any([c in w for c in '+/=%']),
        'pos[:2]': tags[i][:2],
        'only_letters': ''.join(filter(str.isalpha, w)),
        'no_letters': ''.join(filter(lambda c: not c.isalpha(), w)),
        'char_pattern': re.sub(r'[a-z]', 'a', re.sub(r'[A-Z]', 'A', re.sub(r'[0-9]', '0', w)))
    }
    for i in range(3):
      features['prefix_%d' %i] = w[:i]
      features['suffix_%d' %i] = w[-i:]
    #for j in [-1, 1]:
    #  new_i = i+j
    #  if 0 <= new_i < len(tokens):
    #    features.update({
    #      str(j)+'_isdigit': tokens[new_i].isalnum(),
    #      str(j)+'_isalnum': tokens[new_i].isalnum(),
    #      str(j)+'_isparen': tokens[new_i] in ['(', ')'],
    #      str(j)+'_hasoper': any([c in tokens[new_i] for c in '+/=%']),
    #      str(j)+'_token': tokens[new_i],
    #      str(j)+'_pos': tags[new_i],
    #      str(j)+'_pos[:2]': tags[new_i][:2],
    #    })
    #  else:
    #    features.update({
    #      str(j)+'_EOD': True
    #    })

    return features

def doc2features(doc):
  tokens = doc['tokens']
  tags = doc['pos']
  lemmas = doc['lemmas']
  try:
    assert len(tokens) == len(tags) == len(lemmas)
  except AssertionError:
    for l in [tokens, tags, lemmas]:
      print(pmid, len(l), l)
  return [word2features(i, tokens, tags, lemmas) for i in range(len(tokens))]

if __name__ == '__main__':
  docs = None
  if not os.path.isfile(DOC_PKL):
    print('Building data file...')
    docs = build_data()
  predict_labels_crf(docs = docs)
