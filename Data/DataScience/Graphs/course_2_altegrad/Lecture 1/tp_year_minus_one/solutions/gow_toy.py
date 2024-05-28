import string
import matplotlib.pyplot as plt  
from nltk.corpus import stopwords

from library import clean_text_simple, terms_to_graph, core_dec

stpwds = stopwords.words('english')
punct = string.punctuation.replace('-', '')

my_doc = 'A method for solution of systems of linear algebraic equations \
with m-dimensional lambda matrices. A system of linear algebraic \
equations with m-dimensional lambda matrices is considered. \
The proposed method of searching for the solution of this system \
lies in reducing it to a numerical system of a special kind.'

my_doc = my_doc.replace('\n', '')

# pre-process document
my_tokens = clean_text_simple(my_doc,my_stopwords=stpwds,punct=punct)

g = terms_to_graph(my_tokens, 4)

# number of edges
print(len(g.es))

# the number of nodes should be equal to the number of unique terms
len(g.vs) == len(set(my_tokens))

edge_weights = []
for edge in g.es:
    source = g.vs[edge.source]['name']
    target = g.vs[edge.target]['name']
    weight = edge['weight']
    edge_weights.append([source, target, weight])

print(edge_weights)

densities = []
window_sizes = range(2,14)
for w in window_sizes:
    g = terms_to_graph(my_tokens, w)
    g_dens = g.density()
    densities.append(g_dens)
    print(g_dens)

plt.plot(list(window_sizes),densities)  
plt.xlabel('window size')  
plt.ylabel('density')  
plt.title('density VS window size')   
plt.grid(True)  
plt.savefig('density_window.pdf')  
plt.show()  
 
# decompose g
core_numbers = core_dec(g,False)
print(core_numbers)

# compare with igraph method
print(dict(zip(g.vs['name'],g.coreness())))

# retain main core as keywords
max_c_n = max(core_numbers.values())
keywords = [kwd for kwd, c_n in core_numbers.items() if c_n == max_c_n]
print(keywords)
