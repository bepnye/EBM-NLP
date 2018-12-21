import os
from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.base_model import BaseModel
from model.config import Config


def main(data_prefix = None):
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")
    
    if data_prefix:
      cwd = os.getcwd()
      config.filename_dev   = os.path.join(cwd, 'data', data_prefix + '_' + os.path.basename(config.filename_dev))
      config.filename_test  = os.path.join(cwd, 'data', data_prefix + '_' + os.path.basename(config.filename_test))
      config.filename_train = os.path.join(cwd, 'data', data_prefix + '_' + os.path.basename(config.filename_train))

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    # train model
    print('training')
    model.train(train, dev)

if __name__ == "__main__":
  try:
    data_prefix = sys.argv[1]
  except IndexError:
    data_prefix = None
  main(data_prefix)
