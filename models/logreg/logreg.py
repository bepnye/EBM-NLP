import os, sys
import nltk
from collections import Counter
from sklearn import linear_model
from scipy.sparse import csr_matrix
from glob import glob
from collections import defaultdict
from operator import itemgetter
import pickle

from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append('../')
import eval

DOC_PKL = 'docs.pkl'
#EMBEDDINGS = 'embeddings.pkl'
#EMBEDDING_SIZE = 200

TOP = '../../ebm_nlp_2_00/'
UNK = '<UNK>'
BOS = '<BOS>'
EOS = '<EOS>'

def build_data():

  print('Reading documents...')

  docs = {}
  doc_fnames = glob('%s/documents/*.tokens' %(TOP))

  for i, fname in enumerate(doc_fnames):
    pmid = os.path.basename(fname).split('.')[0]

    tokens = open(fname).read().split()
    tags = open(fname.replace('tokens', 'pos')).read().split()
    docs[pmid] = {}
    docs[pmid]['tokens'] = tokens
    docs[pmid]['pos'] = tags

    if (i//100 != (i-1)//100):
      sys.stdout.write('\r\tprocessed %04d / %04d docs' %(i, len(doc_fnames)))
      sys.stdout.flush()

  with open(DOC_PKL, 'wb') as fout:
    print('\nWriting doc data to %s' %DOC_PKL)
    pickle.dump(docs, fout)

  return docs

def get_idx(e, vocab):
  return vocab.get(e, vocab[UNK])

def get_X(pmids, vocabs, docs):
  indptr = [0]
  indices = []
  data = []
  window_size = 1
  for pmid in pmids:
    n = len(docs[pmid]['tokens'])
    pos    = [BOS]*window_size + docs[pmid]['pos']    + [EOS]*window_size
    tokens = [BOS]*window_size + docs[pmid]['tokens'] + [EOS]*window_size
    for i in range(window_size, n+window_size):
      col_offset = 0

      for token_offset in range(-window_size, window_size+1):
        data.append(1)
        indices.append(col_offset + get_idx(tokens[i + token_offset], vocabs['tokens']))
        col_offset += len(vocabs['tokens'])

        data.append(1)
        indices.append(col_offset + get_idx(pos[i + token_offset], vocabs['pos']))
        col_offset += len(vocabs['pos'])

      data.append(int(tokens[i].isdigit()))
      indices.append(col_offset + 0); col_offset += 1

      data.append(int(tokens[i].isalpha()))
      indices.append(col_offset + 0); col_offset += 1

      data.append(int(tokens[i].isupper()))
      indices.append(col_offset + 0); col_offset += 1

      data.append(int(any([c.isupper() for c in tokens[i]])))
      indices.append(col_offset + 0); col_offset += 1

      indptr.append(len(indices))

  X = csr_matrix((data, indices, indptr), dtype = int)
  return X

def get_Y(pmids, labels):
  Y = []
  for pmid in pmids:
    Y += labels[pmid]
  return Y

def logreg(phase = 'hierarchical_labels', pio = 'participants', docs = None):

  print('Reading document data from %s' %DOC_PKL)
  docs = docs or pickle.load(open(DOC_PKL, 'rb'))

  print('Running logreg for %s' %pio)
  test_fnames  = glob('%s/annotations/aggregated/%s/%s/test/gold/*.ann' %(TOP, phase, pio))
  train_fnames = glob('%s/annotations/aggregated/%s/%s/train/*.ann' %(TOP, phase, pio))

  print('Reading labels for %d train and %d test docs' %(len(train_fnames), len(test_fnames)))
  test_labels = { os.path.basename(f).split('.')[0]: open(f).read().split() for f in test_fnames }
  train_labels = { os.path.basename(f).split('.')[0]: open(f).read().split() for f in train_fnames }

  train_pmids = sorted(train_labels.keys())
  test_pmids = list(test_labels.keys())

  assert len(set(test_pmids) & set(train_pmids)) == 0

  for pmid in train_pmids:
    try:
      assert len(docs[pmid]['tokens']) == len(train_labels[pmid])
    except AssertionError:
      print('train_X: token / label mismatch for %s, %d != %d' %(pmid, len(docs[pmid]['tokens']), len(train_labels[pmid])))
  for pmid in test_pmids:
    try:
      assert len(docs[pmid]['tokens']) == len(test_labels[pmid])
    except AssertionError:
      print('test_X: token / label mismatch for %s, %d != %d' %(pmid, len(docs[pmid]['tokens']), len(test_labels[pmid])))
      raise

  vocabs = { 'tokens': [UNK, BOS, EOS], 'pos': [UNK, BOS, EOS] }
  for etype, cutoff in [('tokens', 20), ('pos', 0)]:
    counts = defaultdict(lambda: 0)
    for pmid in train_pmids:
      for e in docs[pmid][etype]:
        counts[e] += 1
    vocabs[etype] += [e for e, count in sorted(list(counts.items()), key = itemgetter(1), reverse = True) \
                          if count >= cutoff]
    vocabs[etype] = { e: i for i, e in enumerate(vocabs[etype]) }
    print('Constructed vocab for %s: %d elements' %(etype, len(vocabs[etype])))

  # Balanced class weights significantly improve recall for participants and outcomes
  # No class weights are better for interventions
  model = linear_model.LogisticRegression(class_weight = 'balanced')
  print('Buidling matrices')
  X = get_X(train_pmids, vocabs, docs)
  print('Built X, shape = %s' %(str(X.shape)))
  Y = get_Y(train_pmids, train_labels)
  print('Built Y, shape = (%d, 1)' %(len(Y)))
  labels = sorted(list(set(Y) - set('0')))
  
  print('Fitting model')
  model.fit(X,Y)

  print('Evaluating on test set')
  pred_labels = {}
  for pmid in test_pmids:
    Xt = get_X([pmid], vocabs, docs)
    Yt = get_Y([pmid], test_labels)
    Yp = list(model.predict(Xt))
    pred_labels[pmid] = Yp

  Yt = get_Y(test_pmids, test_labels)
  Yp = []
  for pmid in test_pmids:
    Yp += pred_labels[pmid]

  eval.eval_labels(TOP, pred_labels, phase, pio)

if __name__ == '__main__':
  docs = None
  phase, pio = sys.argv[1:3]
  if not os.path.isfile(DOC_PKL):
    print('Building data file...')
    docs = build_data()
  logreg(phase, pio, docs = docs)
