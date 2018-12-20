import os, random
from glob import glob
from itertools import groupby, combinations
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, precision_recall_fscore_support

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
    with open(os.path.join(DATA_DIR, 'documents', '%s.txt' %pmid)) as fp:
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
    labels = [int(i) for i in open(fname).read().strip().split(',')]
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

def add_dicts(d1, d2):
  d = d1.copy()
  d.update(d2)
  return d

def combine_model_phases(p1_data, p2_data):
  w1, d1 = p1_data
  w2, d2 = p2_data
  workers = add_dicts(w1, w2)
  docs = {}
  for pmid, d in d1.items():
    docs[pmid] = d
  for pmid, d in d2.items():
    if pmid not in docs:
      docs[pmid] = d
    else:
      docs[pmid].anns = dict(list(docs[pmid].anns.items()) + list(d.anns.items()))
  return workers, docs

def get_multiple_model_phases(phase, element, ann_type, phase1, phase2):
  p1 = read_anns(phase, element, ann_type, model_phase = phase1)
  p2 = read_anns(phase, element, ann_type, model_phase = phase2)
  return combine_model_phases(p1, p2)

def get_wid_color(wid):
  if wid == 'AGGREGATED':
    r = 0
    g = 150
    b = 50
  elif wid == 'UNION':
    r = 0
    g = 250
    b = 150
  else:
    r = int(random.random()*255)
    g = int(random.random()*126)
    b = 255
  color = '{:02x}{:02x}{:02x}'.format(r, g, b)
  return color

def write_brat_files(docs):
  wid_translator = { 'AGGREGATED': 'Upwork', 'UNION': 'Student' }
  fdir = 'brat/'
  while True:
    if not os.path.isdir(fdir):
      print('Please create the target directory: %s' %fdir)
      input('press [enter] when done  ')
    else:
      break
  wids = set()
  for pmid, doc in docs.items():
    offsets = [(0, len(doc.tokens[0]))]
    text = doc.tokens[0]
    for token in doc.tokens[1:]:
      spaced_token = ' ' + token
      offsets.append((len(text) + 1, len(text) + len(spaced_token)))
      text += spaced_token
      assert text[offsets[-1][0]:offsets[-1][1]] == token
    with open('%s/%s.txt' %(fdir, pmid), 'w') as fp:
      fp.write(text)
    with open('%s/%s.test.ann' %(fdir, pmid), 'w') as fp:
      tid = 0
      doc_wids = sorted(doc.anns.keys(), reverse = True)
      for wid in doc_wids:
        wids.add(wid)
        label_spans = condense_labels(doc.anns[wid])
        for label, token_i, token_f in label_spans:
          char_i = offsets[token_i][0]
          char_f = offsets[token_f-1][1]
          fp.write('T%d\t%s %d %d\t%s\n' %(tid, wid_translator.get(wid, wid), char_i, char_f, text[char_i:char_f]))
          tid += 1
  with open('%s/annotation.conf' %fdir, 'w') as fp:
    fp.write('[entities]\n\n')
    for wid in wids:
      wid = wid_translator.get(wid,wid)
      fp.write(wid+'\n')
    fp.write('[relations]\n\n')
    fp.write('<OVERLAP> Arg1:<ENTITY>, Arg2:<ENTITY>, <OVL-TYPE>:<ANY>\n\n')
    fp.write('[events]\n\n')
    fp.write('[attributes]\n\n')
  with open('%s/visual.conf' %fdir, 'w') as fp:
    fp.write('[drawing]\n\n')
    for wid in wids:
      color = get_wid_color(wid)
      wid = wid_translator.get(wid,wid)
      fp.write('%s bgColor:#%s\n' %(wid, color))
