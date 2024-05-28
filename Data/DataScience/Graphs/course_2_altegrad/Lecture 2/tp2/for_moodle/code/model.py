import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

from nltk import word_tokenize


class Encoder(nn.Module):
    '''
    to be passed the entire source sequence at once
    we use padding_idx in nn.Embedding so that the padding vector does not take gradient (always zero)
    https://pytorch.org/docs/stable/nn.html#gru
    '''
    def __init__(self, vocab_size, embedding_dim, hidden_dim, padding_idx):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.rnn = nn.GRU(embedding_dim, hidden_dim)
    
    def forward(self, input):
        # fill the gaps # (transform input into embeddings and pass embeddings to RNN)
        # you should return a tensor of shape (seq,batch,feat)
        emb = self.embedding(input)
        hs, _ = self.rnn(emb)
        return hs


class seq2seqAtt(nn.Module):
    '''
    concat global attention a la Luong et al. 2015 (subsection 3.1)
    https://arxiv.org/pdf/1508.04025.pdf
    '''
    def __init__(self, hidden_dim, hidden_dim_s, hidden_dim_t):
        super(seq2seqAtt, self).__init__()
        self.ff_concat = nn.Linear(hidden_dim_s+hidden_dim_t,hidden_dim)
        self.ff_score = nn.Linear(hidden_dim,1,bias=False) # just a dot product here
    
    def forward(self,target_h,source_hs):
        target_h_rep = target_h.repeat(source_hs.size(0),1,1) # (1,batch,feat) -> (seq,batch,feat)
        # fill the gaps #
        # implement the score computation part of the concat formulation (see section 3.1. of Luong 2015)
        concat_output = self.ff_concat(torch.cat((target_h_rep, source_hs), dim=2))
        scores = self.ff_score(torch.tanh(concat_output)) # should be of shape (seq,batch,1)
        scores = scores.squeeze(dim=2) # (seq,batch,1) -> (seq,batch). dim=2 because we don't want to squeeze the batch dim if batch size = 1
        norm_scores = torch.softmax(scores,0)
        source_hs_p = source_hs.permute((2,0,1)) # (seq,batch,feat) -> (feat,seq,batch)
        weighted_source_hs = (norm_scores * source_hs_p) # (seq,batch) * (feat,seq,batch) (* checks from right to left that the dimensions match)
        ct = torch.sum(weighted_source_hs.permute((1,2,0)),0,keepdim=True) # (feat,seq,batch) -> (seq,batch,feat) -> (1,batch,feat); keepdim otherwise sum squeezes 
        return ct, norm_scores


class Decoder(nn.Module):
    '''to be used one timestep at a time
       see https://pytorch.org/docs/stable/nn.html#gru'''
    def __init__(self, vocab_size, embedding_dim, hidden_dim, padding_idx):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.rnn = nn.GRU(embedding_dim, hidden_dim)
        self.ff_concat = nn.Linear(2*hidden_dim,hidden_dim)
        self.predict = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input, source_context, h):
        # fill the gaps #
        # transform input into embeddings, pass embeddings to RNN, concatenate with source_context and apply tanh, and make the prediction
        # prediction should be of shape (1,batch,vocab), h and tilde_h of shape (1,batch,feat)
        emb = self.embedding(input)
        ht, h = self.rnn(emb, h)
        concat = torch.cat((source_context, ht),2)
        tilde_h = torch.tanh(self.ff_concat(concat))
        prediction = self.predict(tilde_h)
        return prediction, h


class seq2seqModel(nn.Module):
    '''the full seq2seq model'''
    ARGS = ['vocab_s','source_language','vocab_t_inv','embedding_dim_s','embedding_dim_t',
     'hidden_dim_s','hidden_dim_t','hidden_dim_att','do_att','padding_token',
     'oov_token','sos_token','eos_token','max_size']
    def __init__(self, vocab_s, source_language, vocab_t_inv, embedding_dim_s, embedding_dim_t, 
                 hidden_dim_s, hidden_dim_t, hidden_dim_att, do_att, padding_token,
                 oov_token, sos_token, eos_token, max_size):
        super(seq2seqModel, self).__init__()
        self.vocab_s = vocab_s
        self.source_language = source_language
        self.vocab_t_inv = vocab_t_inv
        self.embedding_dim_s = embedding_dim_s
        self.embedding_dim_t = embedding_dim_t
        self.hidden_dim_s = hidden_dim_s
        self.hidden_dim_t = hidden_dim_t
        self.hidden_dim_att = hidden_dim_att
        self.do_att = do_att # should attention be used?
        self.padding_token = padding_token
        self.oov_token = oov_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.max_size = max_size
        
        self.max_source_idx = max(list(vocab_s.values()))
        print('max source index',self.max_source_idx)
        print('source vocab size',len(vocab_s))
        
        self.max_target_idx = max([int(elt) for elt in list(vocab_t_inv.keys())])
        print('max target index',self.max_target_idx)
        print('target vocab size',len(vocab_t_inv))
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder = Encoder(self.max_source_idx+1,self.embedding_dim_s,self.hidden_dim_s,self.padding_token).to(self.device)
        self.decoder = Decoder(self.max_target_idx+1,self.embedding_dim_t,self.hidden_dim_t,self.padding_token).to(self.device)
        
        if self.do_att:
            self.att_mech = seq2seqAtt(self.hidden_dim_att,self.hidden_dim_s,self.hidden_dim_t).to(self.device)
    
    def my_pad(self,my_list):
        '''my_list is a list of tuples of the form [(tensor_s_1,tensor_t_1),...,(tensor_s_batch,tensor_t_batch)]
        the <eos> token is appended to each sequence before padding
        https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pad_sequence'''
        batch_source = pad_sequence([torch.cat((elt[0],torch.LongTensor([self.eos_token]))) for elt in my_list],batch_first=True,padding_value=self.padding_token)
        batch_target = pad_sequence([torch.cat((elt[1],torch.LongTensor([self.eos_token]))) for elt in my_list],batch_first=True,padding_value=self.padding_token)
        return batch_source,batch_target
    
    def forward(self,input,max_size,is_prod):
        
        if is_prod: 
            input = input.unsqueeze(1) # (seq) -> (seq,1) 1D input <=> we receive just one sentence as input (predict/production mode)
        
        current_batch_size = input.size(1)
    
        # fill the gap #
        # use the encoder
        source_hs = self.encoder(input)
        
        # = = = decoder part (one timestep at a time)  = = =
        
        target_h = torch.zeros(size=(1,current_batch_size,self.hidden_dim_t)).to(self.device) # init (1,batch,feat)
        
        # fill the gap #
        # (initialize target_input with the proper token)
        target_input = torch.LongTensor([self.sos_token]).repeat(current_batch_size).unsqueeze(0).to(self.device) # init (1,batch)
        
        pos = 0
        eos_counter = 0
        logits = []
        attention_weights = []

        while True:
            
            if self.do_att:
                source_context, weights = self.att_mech(target_h,source_hs) # (1,batch,feat)
                attention_weights.append(weights.squeeze().tolist())
            else:
                source_context = source_hs[-1,:,:].unsqueeze(0) # (1,batch,feat) last hidden state of encoder
            
            # fill the gap #
            # use the decoder
            prediction, target_h = self.decoder(target_input,source_context,target_h)
            
            logits.append(prediction) # (1,batch,vocab)
            
            # fill the gap #
            # get the next input to pass the decoder
            target_input = torch.argmax(prediction, dim=2)
            
            eos_counter += torch.sum(target_input==self.eos_token).item()
            
            pos += 1
            if pos>=max_size or (eos_counter == current_batch_size and is_prod):
                break
        
        to_return = torch.cat(logits,0) # logits is a list of tensors -> (seq,batch,vocab)
        
        if is_prod:
            to_return = to_return.squeeze(dim=1) # (seq,vocab)

        ## visualization
        self.visualize(input.squeeze().tolist(), to_return, attention_weights)

        return to_return

    
    def fit(self, trainingDataset, testDataset, lr, batch_size, n_epochs, patience):
        
        parameters = [p for p in self.parameters() if p.requires_grad]
        
        optimizer = optim.Adam(parameters, lr=lr)
        
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.padding_token) # the softmax is inside the loss!
        
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        # we pass a collate function to perform padding on the fly, within each batch
        # this is better than truncation/padding at the dataset level
        train_loader = data.DataLoader(trainingDataset, batch_size=batch_size, 
                                       shuffle=True, collate_fn=self.my_pad) # returns (batch,seq)
        
        test_loader = data.DataLoader(testDataset, batch_size=512,
                                      collate_fn=self.my_pad)
        
        tdqm_dict_keys = ['loss', 'test loss']
        tdqm_dict = dict(zip(tdqm_dict_keys,[0.0,0.0]))
        
        patience_counter = 1
        patience_loss = 99999
        
        for epoch in range(n_epochs):
                   
            with tqdm(total=len(train_loader),unit_scale=True,postfix={'loss':0.0,'test loss':0.0},
                      desc="Epoch : %i/%i" % (epoch, n_epochs-1),ncols=100) as pbar:
                for loader_idx, loader in enumerate([train_loader, test_loader]):
                    total_loss = 0
                    # set model mode (https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
                    if loader_idx == 0:
                        self.train()
                    else:
                        self.eval()
                    for i, (batch_source,batch_target) in enumerate(loader):
                        batch_source = batch_source.transpose(1,0).to(self.device) # RNN needs (seq,batch,feat) but loader returns (batch,seq)                        
                        batch_target = batch_target.transpose(1,0).to(self.device) # (seq,batch)
                        
                        # are we using the model in production / as an API?
                        is_prod = len(batch_source.shape)==1 # if False, 2D input (seq,batch), i.e., train or test
                        
                        if is_prod:
                            max_size = self.max_size
                            self.eval()
                        else:
                            max_size = batch_target.size(0) # no need to continue generating after we've exceeded the length of the longest ground truth sequence
                        
                        unnormalized_logits = self.forward(batch_source,max_size,is_prod)

                        sentence_loss = criterion(unnormalized_logits.flatten(end_dim=1),batch_target.flatten())

                        total_loss += sentence_loss.item()
                        
                        tdqm_dict[tdqm_dict_keys[loader_idx]] = total_loss/(i+1)
                        
                        pbar.set_postfix(tdqm_dict)
                        
                        if loader_idx == 0:
                            optimizer.zero_grad() # flush gradient attributes
                            sentence_loss.backward() # compute gradients
                            optimizer.step() # update
                            pbar.update(1)
            
            if total_loss > patience_loss:
                patience_counter += 1
            else:
                patience_loss = total_loss
                patience_counter = 1 # reset
            
            if patience_counter>patience:
                break

    def sourceNl_to_ints(self,source_nl):
        '''converts natural language source sentence into source integers'''
        source_nl_clean = source_nl.lower().replace("'",' ').replace('-',' ')
        source_nl_clean_tok = word_tokenize(source_nl_clean,self.source_language)
        source_ints = [int(self.vocab_s[elt]) if elt in self.vocab_s else \
                       self.oov_token for elt in source_nl_clean_tok]
        
        source_ints = torch.LongTensor(source_ints).to(self.device)
        return source_ints 
    
    def targetInts_to_nl(self,target_ints):
        '''converts integer target sentence into target natural language'''
        return ['<PAD>' if elt==self.padding_token else '<OOV>' if elt==self.oov_token \
                else '<EOS>' if elt==self.eos_token else '<SOS>' if elt==self.sos_token\
                else self.vocab_t_inv[elt] for elt in target_ints]
    
    def predict(self,source_nl):
        source_ints = self.sourceNl_to_ints(source_nl)
        logits = self.forward(source_ints,self.max_size,True) # (seq) -> (<=max_size,vocab)
        target_ints = logits.argmax(-1).squeeze() # (<=max_size,1) -> (<=max_size)
        target_nl = self.targetInts_to_nl(target_ints.tolist())
        
        return ' '.join(target_nl)
        
    def save(self,path_to_file):
        attrs = {attr:getattr(self,attr) for attr in self.ARGS}
        attrs['state_dict'] = self.state_dict()
        torch.save(attrs,path_to_file)
    
    def visualize(self,source_ints,target_int,attention_weights):
        import numpy as np
        vocab_source_inv = {v:k for k,v in self.vocab_s.items()}
        target_ints = target_int.argmax(-1).squeeze() # (<=max_size,1) -> (<=max_size)
        
        target_nl = self.targetInts_to_nl(target_ints.tolist())
        source_nl = [vocab_source_inv[i] for i in source_ints if i in vocab_source_inv]

        target_size = len(target_nl)
        if("." in target_nl):
            target_size = target_nl.index('.')+1

        fig, ax = plt.subplots(figsize=(7,7))
        im = ax.imshow(attention_weights[:target_size],cmap=plt.cm.gray)
        fig.colorbar(im)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(source_nl)))
        ax.set_yticks(np.arange(len(target_nl[:target_size])))

        # ... and label them with the respective list entries
        ax.set_xticklabels(source_nl, rotation = 90)
        ax.set_yticklabels(target_nl[:target_size])

        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        # save figure
        path_to_save = '.' + '/data/' + " ".join(source_nl[0:3])
        fig.savefig(path_to_save)

    @classmethod # a class method does not see the inside of the class (a static method does not take self as first argument)
    def load(cls,path_to_file):
        attrs = torch.load(path_to_file, map_location=lambda storage, loc: storage) # allows loading on CPU a model trained on GPU, see https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/6
        state_dict = attrs.pop('state_dict')
        new = cls(**attrs) # * list and ** names (dict) see args and kwargs
        new.load_state_dict(state_dict)
        return new        
