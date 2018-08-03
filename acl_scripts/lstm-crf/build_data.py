from model.config import Config
from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word
import json, re
import nltk
from nltk import pos_tag, sent_tokenize
from nltk import ngrams

path = '/nlp/data/romap/lstm-tuning/joint/'

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
    print 'Config'
    processing_word = get_processing_word(lowercase=True)
    print 'Processing_word'

    # Generators
    dev   = CoNLLDataset(config.filename_dev, processing_word)
    test  = CoNLLDataset(config.filename_test, processing_word)
    train = CoNLLDataset(config.filename_train, processing_word)
    print 'Loaded dev, test, train'

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])
    print 'Loading vocab_words'
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

if __name__ == "__main__":
    main()

    '''train, test, dev = [], [], []
    path = '/Users/romapatel/Desktop/set/'
    f = open(path + 'data/docids/gold.txt', 'r')
    for line in f: test.append(line.strip())

    f = open(path + 'data/docids/dev.txt', 'r')
    for line in f: dev.append(line.strip())
    
    f = open(path + 'data/docids/train.txt', 'r')
    for line in f: train.append(line.strip())

    f = open(path + 'data/docids/test.txt', 'r')
    for line in f: train.append(line.strip())'''

    '''crowd, gold = {}, {}
    f = open('/Users/romapatel/Desktop/PICO-data/annotations/PICO-annos-crowd-hmm-mv-union.json', 'r')
    for line in f:
        temp = json.loads(line); gold[temp['docid']] = temp
    f = open('/Users/romapatel/Desktop/PICO-data/annotations/PICO-annos-prof-hmm-mv-union.json', 'r')
    for line in f:
        temp = json.loads(line); gold[temp['docid']] = temp

    path = '/Users/romapatel/Desktop/lstm-tuning/joint/'

    finf = open(path + 'data/dev.txt', 'w+')
    dict = {'Participants': {'gold': 'mv', 'crowd': 'hmm'}, 'Intervention': {'gold': 'union', 'crowd': 'union'}, 'Outcome': {'gold': 'union', 'crowd': 'mv'}}
    fin_dict = {}
    #docid -> [word, pos, p, i, o]
    for docid in train:
        fin_dict[docid] = []
        finf.write('-DOCSTART- -X- O O\n\n')
        text = ''
        f = open('/Users/romapatel/Desktop/PICO-data/docs/' + docid + '.txt', 'r')
        for line in f: text += line
        tokens = tokenize(text)
        sents, sent = [], []
        for item in tokens:
            token = item[0]
            if len(token) < 1: continue
            sent.append(token)
            if token == '.':
                sents.append(sent)
                sent = []
        tags = []
        for sent in sents:
            pos_tags = pos_tag(sent)
            tags.extend([item[1] for item in pos_tags])

        indices = [[item[1], item[2]] for item in tokens]
        anno_tags = [['N' for item in tags], ['N' for item in tags], ['N' for item in tags]]
        for i in range(len(dict.keys())):
            annotype = dict.keys()[i]
            anno = annotype[0]
            if annotype not in dict.keys(): continue

            print annotype
            agg = dict[annotype]['crowd']
            print agg
            if agg not in gold[docid][annotype].keys(): continue
            spans = gold[docid][annotype][agg]
            for span in spans:
                begin, end = index_map(span, indices);
                for j in range(begin, end):
                    if j >= len(anno_tags[i]): continue
                    anno_tags[i][j] = anno
        for i in range(len(tags)):
            val = [tokens[i][0], tags[i], anno_tags[0][i], anno_tags[2][i], anno_tags[1][i]]
            fin_dict[docid].append(val)'''

    '''path = '/Users/romapatel/Desktop/lstm-tuning/joint/'

    f = open(path + '/data/train.json', 'r')
    for line in f: dict = json.loads(line)
    f = open(path + '/data/train.txt', 'w+')
    for docid in dict:
        f.write('-DOCSTART- -X- O O\n\n')
        for vec in dict[docid]:
            val = 'N'
            if vec[2] == 'P': val = 'P'
            elif vec[4] == 'O': val = 'O'
            elif vec[3] == 'I': val = 'I'
            print vec
            flag = False
            word = ''
            for char in vec[0]:
                if ord(char) > 128: continue
                word += char
            
            f.write(word + ' ' + vec[1] + ' ' + val + '\n')
            if vec[0] == '.': f.write('\n')
        f.write('\n')'''
        

    
                
            
        




    
