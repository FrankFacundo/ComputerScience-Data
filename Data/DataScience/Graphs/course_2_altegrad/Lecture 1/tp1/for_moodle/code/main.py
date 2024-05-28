import json
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import islice
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity as cosine


def get_windows(seq, n):
    '''
    returns a sliding window (of width n) over data from the iterable
    taken from: https://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator/6822773#6822773
    '''
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def sample_examples(docs, max_window_size, n_windows):
    '''generate target,context pairs and negative examples'''
    windows = []
    for i, doc in enumerate(docs):
        ### fill the gaps (get windows of size 'window_size' from the current document. Sample 'window_size' uniformly in {1,...,max_window_size}) ###
        window_size = np.random.randint(1, max_window_size + 1)
        windows.append(list(get_windows(doc, 2*window_size + 1)).copy())
    windows = [elt for sublist in windows for elt in sublist]  # flatten
    windows = list(np.random.choice(windows, size=n_windows))  # select a subset
    
    ### fill the gap (sample n_negs*len(windows) negatives according to some probability distribution ###
    all_negs = list(np.random.choice(token_ints, size=n_negs*len(windows), p=neg_distr))
    
    return windows, all_negs


def compute_dot_products(pos, negs, target):
    prods = Wc[pos+negs, ] @ Wt[target, ]  # (n_pos+n_negs,d) X (d,) -> (n_pos+n_negs,)
    return prods


def compute_loss(prodpos, prodnegs):
    '''prodpos and prodnegs are numpy vectors containing the dot products of the context word vectors with the target word vector'''
    ### fill the gaps ###
    term_pos = np.log(1+np.exp(-prodpos))
    term_negs = np.log(1+np.exp(prodnegs))
    return np.sum(term_pos) + np.sum(term_negs)


def compute_gradients(pos, negs, target, prodpos, prodnegs):
    factors_pos = 1/(np.exp(prodpos)+1)
    factors_negs = 1/(np.exp(-prodnegs)+1)
    
    ### fill the gaps ### 
    partials_pos = -(np.tile(Wt[target, ], (len(pos), 1)).T*factors_pos).T
    partials_negs = (np.tile(Wt[target, ], (len(negs), 1)).T*factors_negs).T
    term_pos = -(Wc[pos, ].T * factors_pos).T
    term_negs = (Wc[negs, ].T * factors_negs).T
    partial_target = np.sum(term_pos, axis=0) + np.sum(term_negs, axis=0)
    
    return partials_pos, partials_negs, partial_target


def my_cos_similarity(word1, word2):
    sim = cosine(Wt[vocab[word1], ].reshape(1, -1), Wt[vocab[word2], ].reshape(1, -1))
    return round(float(sim), 4)

# = = = = = = = = = = = = = = = = = = = = = 


path_read = "./data/"
path_write = path_read

stpwds = set(stopwords.words('english'))

max_window_size = 5  # extends on both sides of the target word
n_windows = int(1e6)  # number of windows to sample at each epoch
n_negs = 5  # number of negative examples to sample for each positive
d = 30  # dimension of the embedding space
n_epochs = 15
lr_0 = 0.025
decay = 1e-6

train = False

with open(path_read + 'doc_ints.txt', 'r') as file:
    docs = file.read().splitlines()

docs = [[int(eltt) for eltt in elt.split()] for elt in docs]

with open(path_read + 'vocab.json', 'r') as file:
    vocab = json.load(file)

vocab_inv = {v: k for k, v in vocab.items()}

with open(path_read + 'counts.json', 'r') as file:
    counts = json.load(file)

token_ints = range(1, len(vocab)+1)
neg_distr = [counts[vocab_inv[elt]] for elt in token_ints]
neg_distr = np.sqrt(neg_distr)
neg_distr = neg_distr/sum(neg_distr)  # normalize

# ========== train model ==========

if train:
    
    total_its = 0
    
    Wt = np.random.normal(size=(len(vocab)+1, d))  # + 1 is for the OOV token
    Wc = np.random.normal(size=(len(vocab)+1, d))
    
    for epoch in range(n_epochs):
        
        windows, all_negs = sample_examples(docs, max_window_size, n_windows)
        print('training examples sampled')
        
        np.random.shuffle(windows)
        
        total_loss = 0
        
        with tqdm(total=len(windows), unit_scale=True, postfix={'loss':0.0,'lr':lr_0}, desc="Epoch : %i/%i" % (epoch+1, n_epochs), ncols=50) as pbar:
            for i, w in enumerate(windows):
                
                target = w[int(len(w)/2)]  # elt at the center
                pos = list(w)
                del pos[int(len(w)/2)]  # all elts but the center one
                
                negs = all_negs[n_negs*i:n_negs*i+n_negs]
                
                prods = compute_dot_products(pos, negs, target)
                prodpos = prods[0:len(pos), ]
                prodnegs = prods[len(pos):(len(pos)+len(negs)), ]
                
                partials_pos, partials_negs, partial_target = compute_gradients(pos, negs, target, prodpos, prodnegs)
                
                lr = lr_0 * 1/(1+decay*total_its)
                total_its += 1
                
                ### fill the gaps (perform the updates) ###
                Wt[target, ] -= lr * partial_target.copy()
                Wc[pos, ] -= lr * partials_pos.copy()
                Wc[negs, ] -= lr * partials_negs.copy()
                
                total_loss += compute_loss(prodpos, prodnegs)
                pbar.set_postfix({'loss': total_loss/(i+1), 'lr': lr})
                pbar.update(1)
                
    np.save(path_write + 'input_vecs', Wt, allow_pickle=False)  # pickle disabled for portability reasons
    np.save(path_write + 'output_vecs', Wc, allow_pickle=False)
    
    print('word vectors saved to disk')
    
else:
    Wt = np.load(path_write + 'input_vecs.npy')
    Wc = np.load(path_write + 'output_vecs.npy')
    

# ========== sanity checks ==========

if not train:

    # = = some similarities = = 
    ### fill the gaps (compute the cosine similarity between some (un)related words, like movie/film/banana ###
    print("cosinus similarity [movie,film] : ", my_cos_similarity("movie", "film"))
    print("cosinus similarity [movie,banana] : ", my_cos_similarity("movie", "banana"))
    print("cosinus similarity [banana,film] : ", my_cos_similarity("banana", "film"))
    # = = visualization of most frequent tokens = =

    n_plot = 500
    mft = [vocab_inv[elt] for elt in range(1, n_plot+1)]

    # exclude stopwords and punctuation
    keep_idxs = [idx for idx, elt in enumerate(mft) if len(elt) > 3 and elt not in stpwds]
    mft = [mft[idx] for idx in keep_idxs]
    keep_ints = [list(range(1, n_plot+1))[idx] for idx in keep_idxs]
    Wt_freq = Wt[keep_ints, ]
    
    ### fill the gaps (perfom PCA (10D) followed by t-SNE (2D). For t-SNE, you can use a perplexity of 5.) ###
    ### for t-SNE, see https://lvdmaaten.github.io/tsne/#faq ###
    my_pca = PCA(n_components=10)
    my_tsne = TSNE(n_components=2, perplexity=5)

    my_pca_fit = my_pca.fit_transform(Wt_freq)
    my_tsne_fit = my_tsne.fit_transform(my_pca_fit)

    fig, ax = plt.subplots()
    ax.scatter(my_tsne_fit[:, 0], my_tsne_fit[:, 1], s=3)  ### fill the gap ###
    for x, y, token in zip(my_tsne_fit[:, 0], my_tsne_fit[:, 1], mft):  ### fill the gap ###
        ax.annotate(token, xy=(x, y), size=8)

    fig.suptitle('t-SNE visualization of word embeddings', fontsize=20)
    fig.set_size_inches(11, 7)
    fig.savefig(path_write + 'word_embeddings.pdf', dpi=300)
    fig.show()


