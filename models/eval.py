from glob import glob
from itertools import groupby, combinations
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, precision_recall_fscore_support

from nltk.corpus import stopwords
stops = stopwords.words('english')

PHASES = ('starting_spans', 'hierarchical_labels')
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

EBM_NLP = ''

def fname_to_pmid(f):
  return f.split('/')[-1].split('.')[0]
def apply_bitmask(arr, mask):
  return [x for x, m in zip(arr, mask) if m]
def condense_labels(labels):
  groups = [(k, sum(1 for _ in g)) for k,g in groupby(labels)]
  spans = []
  i = 0
  for label, length in groups:
    if label != 0:
      spans.append((label, i, i+length))
    i += length
  return spans

def get_test_labels(ebm_nlp_dir, phase, pio):
  test_dir = '%s/annotations/aggregated/%s/%s/test/gold/' %(EBM_NLP, phase, pio)
  test_fnames = glob('%s/*.ann' %test_dir)
  print('Loading %d anns from %s' %(len(test_fnames), test_dir))
  return { fname_to_pmid(fname): open(fname).read().split() for fname in test_fnames }

#def span_overlap(pmids, pred_labels, test_labels, labels):
#  for pmid in pmids:
#    test_spans = condense_labels(test_labels[pmid])
#    pred_spans = condense_labels(pred_labels[pmid])
#    for tspan in test_spans:
#      for pspan in pred_spans:
#        pass
  
def vanilla_tokens(pmids, pred_labels, test_labels, labels):
  y_pred = []
  y_test = []
  for pmid in pmids:
    assert len(pred_labels[pmid]) == len(test_labels[pmid])
    y_pred += pred_labels[pmid]
    y_test += test_labels[pmid]
  token_f1(true = y_test, pred = y_pred, labels = labels)

def sw_tokens(pmids, pred_labels, test_labels, labels):
  y_pred = []
  y_test = []
  for pmid in pmids:
    assert len(pred_labels[pmid]) == len(test_labels[pmid])
    tokens = open('%s/documents/%s.tokens' %(EBM_NLP, pmid)).read().split()
    token_mask = [t in stops for t in tokens]
    y_pred += apply_bitmask(pred_labels[pmid], token_mask)
    y_test += apply_bitmask(test_labels[pmid], token_mask)
  token_f1(true = y_test, pred = y_pred, labels = labels)

def eval_labels(ebm_nlp_dir, pred_labels, phase, pio, eval_func = vanilla_tokens):
  global EBM_NLP
  EBM_NLP = ebm_nlp_dir

  print('Evaluating labels for %s %s' %(phase, pio))
  test_labels = get_test_labels(EBM_NLP, phase, pio)
  pmids = set(test_labels.keys()) & set(pred_labels.keys())
  print('Checking labels for %d pmids (out of %d possible test docs)' %(len(pmids), len(test_labels)))

  labels = set(LABEL_DECODERS[phase][pio].keys())
  labels.remove(0)
  labels = [str(l) for l in labels]

  eval_func(pmids, pred_labels, test_labels, labels)

def get_f1(prec, rec):
  return 2*prec*rec/(prec+rec)

def token_f1(true, pred, labels):

  print(true[:30])
  print(pred[:30])
  print(labels)

  prec = precision_score(true, pred, labels = labels, average='micro')
  rec = recall_score(true, pred, labels = labels, average='micro')
  f1 = get_f1(prec, rec)
  print('f1        = %.2f' %f1)
  print('precision = %.2f' %prec)
  print('recall    = %.2f' %rec)
  class_scores = zip(labels, precision_score(true,pred,labels,average=None), recall_score(true,pred,labels,average=None))
  for label, prec, rec in class_scores:
    print('Label: %s' %label)
    print('\tf1        = %.2f' %get_f1(prec, rec))
    print('\tprecision = %.2f' %prec)
    print('\trecall    = %.2f' %rec)
  return { 'f1': f1, 'precision': prec, 'recall': rec }
