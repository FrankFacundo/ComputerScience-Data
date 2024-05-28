import sys
import json

import torch
from torch.utils import data

# = = = = = = = = = = =

path_root = "./"# fill me !

path_to_model = path_root + 'code/'
sys.path.insert(0, path_to_model)

from model import seq2seqModel

path_to_data = path_root + 'data/'
path_to_save_models = path_root + 'models/'

# = = = = = = = = = = =

class Dataset(data.Dataset):
  def __init__(self, pairs):
        self.pairs = pairs

  def __len__(self):
        return len(self.pairs) # total nb of observations

  def __getitem__(self, idx):
        source, target = self.pairs[idx] # one observation
        return torch.LongTensor(source), torch.LongTensor(target)


def load_pairs(train_or_test):
    with open(path_to_data + 'pairs_' + train_or_test + '_ints.txt', 'r', encoding='utf-8') as file:
        pairs_tmp = file.read().splitlines()
    pairs_tmp = [elt.split('\t') for elt in pairs_tmp]
    pairs_tmp = [[[int(eltt) for eltt in elt[0].split()],[int(eltt) for eltt in \
                  elt[1].split()]] for elt in pairs_tmp]
    return pairs_tmp


# = = = = = = = = = = =

do_att = True # should always be set to True
is_prod = True # production mode or not

if not is_prod:
        
    pairs_train = load_pairs('train')
    pairs_test = load_pairs('test')
    
    with open(path_to_data + 'vocab_source.json','r') as file:
        vocab_source = json.load(file) # word -> index
    
    with open(path_to_data + 'vocab_target.json','r') as file:
        vocab_target = json.load(file) # word -> index
    
    vocab_target_inv = {v:k for k,v in vocab_target.items()} # index -> word
    
    print('data loaded')
        
    training_set = Dataset(pairs_train)
    test_set = Dataset(pairs_test)
    
    print('data prepared')
    
    
    print('= = = attention-based model?:',str(do_att),'= = =')
    
    model = seq2seqModel(vocab_s=vocab_source,
                         source_language='english',
                         vocab_t_inv=vocab_target_inv,
                         embedding_dim_s=40,
                         embedding_dim_t=40,
                         hidden_dim_s=30,
                         hidden_dim_t=30,
                         hidden_dim_att=20,
                         do_att=do_att,
                         padding_token=0,
                         oov_token=1,
                         sos_token=2,
                         eos_token=3,
                         max_size=30) # max size of generated sentence in prediction mode
    
    model.fit(training_set,test_set,lr=0.001,batch_size=64,n_epochs=40,patience=2)
    model.save(path_to_save_models + 'my_model.pt')

else:
    model = seq2seqModel.load(path_to_save_models + 'pretrained_moodle.pt')
    
    to_test = ['I am a student.',
               'I have a red car.',  # inversion captured
               'I love playing video games.',
                'This river is full of fish.', # plein vs pleine (accord)
                'The fridge is full of food.',
               'The cat fell asleep on the mat.',
               'my brother likes pizza.', # pizza is translated to 'la pizza'
               'I did not mean to hurt you', # translation of mean in context
               'She is so mean',
               'Help me pick out a tie to go with this suit!', # right translation
               "I can't help but smoking weed", # this one and below: hallucination
               'The kids were playing hide and seek',
               'The cat fell asleep in front of the fireplace']
    
    for elt in to_test:
        print('= = = = = \n','%s -> %s' % (elt,model.predict(elt)))
