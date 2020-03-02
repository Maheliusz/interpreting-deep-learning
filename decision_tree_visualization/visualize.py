import torch
import torch.nn as nn
from SDT import SDT
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np 
from torchvision import utils
from matplotlib.colors import NoNorm
import networkx as nx
import argparse
import numpy as np

from main import learner_args

try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydot
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("This example needs Graphviz and either "
                          "PyGraphviz or pydot")


parser = argparse.ArgumentParser()
parser.add_argument('model_path')
args = parser.parse_args()

tree = torch.load(args.path)
tree.eval()

weights = tree.inner_nodes._modules['linear'].weight.detach()
images = torch.reshape(weights[:, 1:], (-1,)+learner_args['input_dim'])

leaf_weights = tree.leaf_nodes.weight.detach()
leaves = leaf_weights.max(0)[1].tolist()
G = nx.balanced_tree(2, 5)
pos=graphviz_layout(G, prog='dot')
relabel = dict(zip(range(G.size() - len(leaves)+1, G.size()+1), leaves))
nx.draw(G, pos, width=2, alpha=0.6)
nx.draw_networkx_labels(G, pos, relabel)
ax=plt.gca()
fig=plt.gcf()
fig.set_figwidth(fig.get_figwidth()*3)
trans = ax.transData.transform
trans2 = fig.transFigure.inverted().transform
imsize = 0.1 # this is the image size
for i, n in enumerate(G.nodes()):
    (x,y) = pos[n]
    xx,yy = trans((x,y)) # figure coordinates
    xa,ya = trans2((xx,yy)) # axes coordinates
    a = plt.axes([xa-imsize/2.0,ya-imsize/2.0, imsize, imsize])
    if i<images.shape[0]:
        a.imshow(images[i], cmap='gray')
    a.set_aspect('equal')
    a.axis('off')
plt.savefig('graph.png')
