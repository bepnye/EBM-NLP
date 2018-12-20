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

LEMMATIZER = nltk.stem.WordNetLemmatizer()
DOC_PKL = 'docs.pkl'
#EMBEDDINGS = 'embeddings.pkl'
#EMBEDDING_SIZE = 200

TOP = '../../ebm_nlp_1_00/'
UNK = '<UNK>'
BOS = '<BOS>'
EOS = '<EOS>'

def get_wordnet_pos(pos):
  if pos.startswith('J'): return nltk.corpus.wordnet.ADJ
  if pos.startswith('V'): return nltk.corpus.wordnet.VERB
  if pos.startswith('N'): return nltk.corpus.wordnet.NOUN
  if pos.startswith('R'): return nltk.corpus.wordnet.ADV
  return pos

def lemmatize((token, pos)):
  try:
    return LEMMATIZER.lemmatize(token, pos = get_wordnet_pos(pos)).lower()
  except KeyError:
    return LEMMATIZER.lemmatize(token).lower()

def build_data():

  print 'Reading, lemmatizing, and pos taggings documents...'

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

    if (i/100 != (i-1)/100):
      sys.stdout.write('\r\tprocessed %04d / %04d docs' %(i, len(doc_fnames)))
      sys.stdout.flush()

  with open(DOC_PKL, 'w') as fout:
    print '\nWriting doc data to %s' %DOC_PKL
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

def get_idx(e, vocab):
  return vocab.get(e, vocab[UNK])

def get_emb(e, embeddings):
  if e in embeddings:
    return embeddings[e]
  else:
    return [0.]*EMBEDDING_SIZE
  

def get_X(pmids, vocabs, docs):
  indptr = [0]
  indices = []
  data = []
  window_size = 2
  for pmid in pmids:
    n = len(docs[pmid]['lemmas'])
    lemmas = [BOS]*window_size + docs[pmid]['lemmas'] + [EOS]*window_size
    pos    = [BOS]*window_size + docs[pmid]['pos']    + [EOS]*window_size
    tokens = [BOS]*window_size + docs[pmid]['tokens'] + [EOS]*window_size
    for i in range(window_size, n+window_size):
      col_offset = 0

      for token_offset in range(-window_size, window_size+1):
        data.append(1)
        indices.append(col_offset + get_idx(lemmas[i + token_offset], vocabs['lemmas']))
        col_offset += len(vocabs['lemmas'])

        data.append(1)
        indices.append(col_offset + get_idx(pos[i + token_offset], vocabs['pos']))
        col_offset += len(vocabs['pos'])

      #if embeddings:
      #  data += get_emb(tokens[i], embeddings)
      #  indices += range(col_offset, col_offset+EMBEDDING_SIZE)
      #  col_offset += EMBEDDING_SIZE

      data.append(lemmas[i].isdigit())
      indices.append(col_offset + 0); col_offset += 1

      data.append(lemmas[i].isalpha())
      indices.append(col_offset + 0); col_offset += 1

      data.append(tokens[i].isupper())
      indices.append(col_offset + 0); col_offset += 1

      data.append(any([c.isupper() for c in tokens[i]]))
      indices.append(col_offset + 0); col_offset += 1

      indptr.append(len(indices))

  X = csr_matrix((data, indices, indptr), dtype = int)
  return X

def get_Y(pmids, labels):
  Y = []
  for pmid in pmids:
    Y += labels[pmid]
  return Y

def logreg(docs = None):

  print 'Reading document data from %s' %DOC_PKL
  docs = docs or pickle.load(open(DOC_PKL))

  for pio in ['participants', 'interventions', 'outcomes']:
    print 'Running logreg for %s' %pio
    test_fnames  = glob('%s/annotations/aggregated/hierarchical_labels/%s/test/gold/*.ann' %(TOP, pio))
    train_fnames = glob('%s/annotations/aggregated/hierarchical_labels/%s/train/*.ann' %(TOP, pio))

    print 'Reading labels for %d train and %d test docs' %(len(train_fnames), len(test_fnames))
    test_labels = { os.path.basename(f).split('_')[0]: open(f).read().split(',') for f in test_fnames }
    train_labels = { os.path.basename(f).split('_')[0]: open(f).read().split(',') for f in train_fnames }

    train_pmids = sorted(train_labels.keys())
    test_pmids = test_labels.keys()

    assert len(set(test_pmids) & set(train_pmids)) == 0

    vocabs = { 'lemmas': [UNK, BOS, EOS], 'pos': [UNK, BOS, EOS] }
    for etype, cutoff in [('lemmas', 20), ('pos', 0)]:
      counts = defaultdict(lambda: 0)
      for pmid in train_pmids:
        for e in docs[pmid][etype]:
          counts[e] += 1
      vocabs[etype] += [e for e, count in sorted(counts.items(), key = itemgetter(1), reverse = True) \
                            if count >= cutoff]
      vocabs[etype] = { e: i for i, e in enumerate(vocabs[etype]) }
      print 'Constructed vocab for %s: %d elements' %(etype, len(vocabs[etype]))

    # Balanced class weights significantly improve recall for participants and outcomes
    # No class weights are better for interventions
    model = linear_model.LogisticRegression(class_weight = 'balanced')
    print 'Buidling matrices'
    X = get_X(train_pmids, vocabs, docs)
    print 'Built X, shape = %s' %(str(X.shape))
    Y = get_Y(train_pmids, train_labels)
    labels = sorted(list(set(Y) - set('0')))
    print 'Built Y, shape = (%d, 1)' %(len(Y))
    
    print 'Fitting model'
    model.fit(X,Y)

    print 'Evaluating on test set'
    Xt = get_X(test_pmids, vocabs, docs)
    Yt = get_Y(test_pmids, test_labels)
    Yp = model.predict(Xt)
    prec=precision_score(Yt,Yp,labels,average='macro')
    rec=recall_score(Yt,Yp,labels,average='macro')
    f1=2*prec*rec/(prec+rec)
    print 'Results for %s' %pio
    print 'f1 = %.2f, precision = %.2f, recall = %.2f' %(f1, prec, rec)
    print 'Class precision: ', precision_score(Yt,Yp,labels,average=None)
    print 'Class recall:    ', recall_score(Yt,Yp,labels,average=None)
    print

if __name__ == '__main__':
  docs = None
  if not os.path.isfile(DOC_PKL):
    print 'Building data file...'
    docs = build_data()
  logreg(docs = docs)
