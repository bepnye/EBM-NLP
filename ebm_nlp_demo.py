import os
from glob import glob
from itertools import groupby, combinations
from sklearn.metrics import cohen_kappa_score

DATA_DIR = 'ebm_nlp_1_00/'
PHASES = ('starting_spans', 'hierarchical_labels')
ELEMENTS = ('participants', 'interventions', 'outcomes')

LABEL_DECODERS = { \
  PHASES[0] : { \
      'participants':  { 0: 'No Label', 1: 'p' },
      'interventions': { 0: 'No Label', 1: 'i' },
      'outcomes':      { 0: 'No Label', 1: 'o' }
    },
  PHASES[1]: { \
      'participants': { \
        0: 'No label',
        1: 'Age',
        2: 'Sex',
        3: 'Sample-size',
        4: 'Condition' },

      'interventions': { \
        0: 'No label',
        1: 'Surgical',
        2: 'Physical',
        3: 'Pharmacological',
        4: 'Educational',
        5: 'Psychological',
        6: 'Other',
        7: 'Control' },

      'outcomes': { \
        0: 'No label',
        1: 'Physical',
        2: 'Pain',
        3: 'Mortality',
        4: 'Adverse-effects',
        5: 'Mental',
        6: 'Other' }
    }
}

def rpad(inp, n, min_buf=0):
  s = str(inp)[:n - min_buf]
  return (s+' '*(n-len(s)))

def lpad(inp, n, min_buf=0):
  s = str(inp)[:n - min_buf]
  return (' '*(n-len(s))+s)

class Doc:
  def __init__(self, pmid, phase, element):
    with open(os.path.join(DATA_DIR, 'documents', '%s.text' %pmid)) as fp:
      self.text = fp.read()
    with open(os.path.join(DATA_DIR, 'documents', '%s.tokens' %pmid)) as fp:
      self.tokens = fp.read().split(' ')
    self.pmid = pmid
    self.decoder = LABEL_DECODERS[phase][element]
    self.anns = {}

class Worker:
  def __init__(self, wid):
    self.wid = wid
    self.pmids = []

def get_pmids():
  doc_fnames = glob(os.path.join(DATA_DIR, 'documents', '*.text'))
  pmids = [os.path.basename(f).split('.')[0] for f in doc_fnames]
  return pmids

def read_anns(phase, element, ann_type = 'aggregated', model_phase = 'train'):
  workers = {}
  docs = {}

  fdir = os.path.join(DATA_DIR, 'annotations', ann_type, phase, element, model_phase)
  fnames = glob(os.path.join(fdir, '*.ann'))
  print('Found %d files in %s' %(len(fnames), fdir))

  for fname in fnames:
    labels = map(int, open(fname).read().strip().split(','))
    pmid, wid = os.path.basename(fname).split('.')[0].split('_')
    if pmid not in docs:
      docs[pmid] = Doc(pmid, phase, element)
    if wid not in workers:
      workers[wid] = Worker(wid)
    docs[pmid].anns[wid] = labels
    workers[wid].pmids.append(pmid)

  print('Loaded annotations for %d documents from %d worker%s' %(len(docs), len(workers), 's' if len(workers) != 1 else ''))
  return workers, docs

def print_token_labels(doc, width = 80): 
  t_str = '' 
  l_str = '' 
  for wid, labels in doc.anns.items():
    for t, l in zip(doc.tokens, labels):
      if l != 0:
        l_s = doc.decoder[l]
      else:
        l_s = ' '*len(t)
      slen = max(len(t), len(l_s))
      if len(t_str) + slen > width:
        if any([c != ' ' for c in l_str]):
          print(l_str)
        print(t_str)
        t_str = '' 
        l_str = '' 
      t_str += ' ' + rpad(t, slen)
      l_str += ' ' + rpad(l_s, slen)
    print(l_str)
    print(t_str)

def condense_labels(labels):
  groups = [(k, sum(1 for _ in g)) for k,g in groupby(labels)]
  spans = []
  i = 0
  for label, length in groups:
    if label != 0:
      spans.append((label, i, i+length))
    i += length
  return spans

def print_labeled_spans(doc):
  for wid, labels in doc.anns.items():
    label_spans = condense_labels(labels)
    print('Label spans for wid = %s' %wid)
    for label, token_i, token_f in label_spans:
      print('[%s]: %s ' %(doc.decoder[label], ' '.join(doc.tokens[token_i:token_f])))
    print()

def compute_worker_kappas(workers, docs):
  wids = sorted(workers.keys())
  worker_pairs = list(combinations(wids, 2))
  worker_kappas = [['' for _ in wids] for __ in wids]
  for (wid1, wid2) in worker_pairs:
    pmids = list(set(workers[wid1].pmids).intersection(workers[wid2].pmids))
    if len(pmids) > 0:
      l1 = sum([docs[pmid].anns[wid1] for pmid in pmids], [])
      l2 = sum([docs[pmid].anns[wid2] for pmid in pmids], [])
      kappa = cohen_kappa_score(l1, l2)
      idx1 = wids.index(wid1)
      idx2 = wids.index(wid2)
      worker_kappas[idx1][idx2] = kappa
      worker_kappas[idx2][idx1] = kappa

  print_matrix(worker_kappas, wids, 'Pairwise Cohen\'s Kappa')
  return worker_kappas

def print_matrix(matrix, row_names, title):
  row_names = row_names or ['' for row in matrix]
  title = title or 'Table'
  llen = max(map(len, row_names))
  print('%s:' %title)
  print('%s  %s' %(lpad('', llen), ' '.join([lpad(n, llen) for n in row_names])))
  for row,name in zip(matrix, row_names):
    print('%s: %s' %(lpad(name, llen), ' '.join([lpad(x if type(x) is str else '%.2f' %x, llen) for x in row])))

