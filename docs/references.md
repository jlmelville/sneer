---
title: "References"
output: html_document
---

Previous: [Visualization](visualization.html). Up: [Index](index.html).

Here's a list of things (mainly papers) that provided the various options 
available in `sneer`. I've tried to link to the offical publication websites
rather than to PDFs directly, so it's a bit easier to find or confirm
bibliographic information.

### Embedding Methods

#### Metric Multi Dimensional Scaling (MDS)
Borg, I., & Groenen, P. J. (2005). 
*Modern multidimensional scaling: Theory and applications.*
Springer Science & Business Media.
https://dx.doi.org/10.1007/0-387-28981-X

Hughes, N. P., & Lowe, D. (2002).
Artefactual Structure from Least-Squares Multidimensional Scaling.
In *Advances in Neural Information Processing Systems* (pp. 913-920).
https://papers.nips.cc/paper/2239-artefactual-structure-from-least-squares-multidimensional-scaling

#### Sammon Mapping
Sammon, J. W. (1969). 
A nonlinear mapping for data structure analysis. 
*IEEE Transactions on computers*, *18*(5), 401-409.
https://dx.doi.org/10.1109/T-C.1969.222678

#### Stochastic Neighbor Embedding (SNE)
Hinton, G. E., & Roweis, S. T. (2002).
Stochastic neighbor embedding.
In *Advances in neural information processing systems* (pp. 833-840).
https://papers.nips.cc/paper/2276-stochastic-neighbor-embedding

#### Symmetric Stochastic Neighbor Embedding (SSNE)
Cook, J., Sutskever, I., Mnih, A., & Hinton, G. E. (2007).
Visualizing similarity data with a mixture of maps.
In *International Conference on Artificial Intelligence and Statistics* (pp. 67-74).
https://www.cs.toronto.edu/~amnih/papers/sne_am.pdf

#### t-Distributed Stochastic Neighbor Embedding (t-SNE)
Van der Maaten, L., & Hinton, G. (2008).
Visualizing data using t-SNE.
*Journal of Machine Learning Research*, *9* (2579-2605).
http://www.jmlr.org/papers/v9/vandermaaten08a.html

#### Heavy-Tailed Symmetric Stochastic Neighbor Embedding (HSSNE)
Yang, Z., King, I., Xu, Z., & Oja, E. (2009).
Heavy-tailed symmetric stochastic neighbor embedding.
In *Advances in neural information processing systems* (pp. 2169-2177).
https://papers.nips.cc/paper/3770-heavy-tailed-symmetric-stochastic-neighbor-embedding

#### Inhomogeneous t-SNE
Kitazono, J., Grozavu, N., Rogovschi, N., Omori, T., & Ozawa, S. (2016, October).
t-Distributed Stochastic Neighbor Embedding with Inhomogeneous Degrees of Freedom. 
In *Proceedings of the 23rd International Conference on Neural Information Processing (ICONIP 2016)* 
(pp. 119-128).
http://dx.doi.org/10.1007/978-3-319-46675-0_14

#### Weighted Symmetric Stochastic Neighbor Embedding (ws-SNE)
Yang, Z., Peltonen, J., & Kaski, S. (2014).
Optimization equivalence of divergences improves neighbor embedding.
In *Proceedings of the 31st International Conference on Machine Learning (ICML-14)*
(pp. 460-468).
http://jmlr.org/proceedings/papers/v32/yange14.html

#### Neighbor Retrieval Visualizer (NeRV)
Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010).
Information retrieval perspective to nonlinear dimensionality reduction for 
data visualization.
*Journal of Machine Learning Research*, *11*, 451-490.
http://www.jmlr.org/papers/v11/venna10a.html

#### Jensen-Shannon Embedding (JSE)
Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013).
Type 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in
dimensionality reduction based on similarity preservation.
*Neurocomputing*, *112*, 92-108.
https://dx.doi.org/10.1016/j.neucom.2012.12.036

### Perplexity and Input Initialization

Vladymyrov, M., & Carreira-Perpinán, M. A. (2013, June). 
Entropic Affinities: Properties and Efficient Numerical Computation. 
*Proceedings of the 30th international conference on machine learning (ICML-13)*,
pp. 477-485.
http://jmlr.org/proceedings/papers/v28/vladymyrov13.html

Not implemented in `sneer`, but describes fast ways to calculate the input
probabilities for large data sets. Also, coins the term "entropic affinities" 
to describe the process of calibrating probabilities via setting a perplexity 
value.

#### Multiscale Embedding
Lee, J. A., Peluffo-Ordónez, D. H., & Verleysen, M. (2014).
Multiscale stochastic neighbor embedding: Towards parameter-free
dimensionality reduction. In *Proceedings of 2014 European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2014)* (pp. 177-182), 
https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2014-64.pdf

Lee, J. A., Peluffo-Ordónez, D. H., & Verleysen, M. (2015).
Multi-scale similarities in stochastic neighbour embedding: Reducing
dimensionality while preserving both local and global structure.
*Neurocomputing*, *169*, 246-261.
https://dx.doi.org/10.1016/j.neucom.2014.12.095

Fun fact: just noticed that this is the only paper on the list where 
"neighbour" is spelt in the British English style.

### Optimization

#### Jacobs Method for Adaptive Step Size
Jacobs, R. A. (1988).
Increased rates of convergence through learning rate adaptation.
*Neural networks*, *1*(4), 295-307.
https://dx.doi.org/10.1016/0893-6080(88)90003-2

Janet, J. A., Scoggins, S. M., Schultz, S. M., Snyder, W. E., White, M. W.,
& Sutton, J. C. (1998, May).
Shocking: An approach to stabilize backprop training with greedy adaptive learning rates.
In *1998 IEEE International Joint Conference on Neural Networks Proceedings.*
(Vol. 3, pp. 2218-2223). IEEE.
https://dx.doi.org/10.1109/IJCNN.1998.687205

#### Spectral Directions
Vladymyrov, M., & Carreira-Perpiñán, M. A. (2012).
Partial-Hessian Strategies for Fast Learning of Nonlinear Embeddings.
In *Proceedings of the 29th International Conference on Machine Learning (ICML-12)*
(pp. 345-352).
https://arxiv.org/abs/1206.4646

#### Nesterov Accelerated Gradient
On the importance of initialization and momentum in deep learning.
In *Proceedings of the 30th international conference on machine learning (ICML-13)*
(pp. 1139-1147).
http://www.jmlr.org/proceedings/papers/v28/sutskever13.html

#### Adaptive Restart
O'Donoghue, B., & Candes, E. (2013).
Adaptive restart for accelerated gradient schemes.
*Foundations of computational mathematics*, *15*(3), 715-732.
https://dx.doi.org/10.1007/s10208-013-9150-3, https://arxiv.org/abs/1204.3982

Su, W., Boyd, S., & Candes, E. J. (2016). 
A differential equation for modeling nesterov’s accelerated gradient method: theory and insights. 
*Journal of Machine Learning Research*, *17*(153), 1-43.
http://jmlr.org/papers/v17/15-084.html

### Evaluation

#### Precision-Recall AUC and Receiver Operating Characteristic Area Under the Curve
Davis, J., & Goadrich, M. (2006, June).
The relationship between Precision-Recall and ROC curves.
In *Proceedings of the 23rd international conference on Machine learning*
(pp. 233-240). ACM.
http://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf

Keilwagen, J., Grosse, I., & Grau, J. (2014).
Area under precision-recall curves for weighted and unweighted data.
*PloS One*, *9*(3), e92209.
https://dx.doi.org/10.1371/journal.pone.0092209.

#### Neighborhood Preservation
Lee, J. A., & Verleysen, M. (2009).
Quality assessment of dimensionality reduction: Rank-based criteria.
*Neurocomputing*, *72*(7), 1431-1443.
https://dx.doi.org/10.1016/j.neucom.2008.12.017

### Miscellany

Things that aren't in `sneer`, but piqued my interest, or don't fit into the
other categories.

#### Distance Approximations

The bottle-neck in embedding methods is the requirement to calculate the 
all-against-all distance matrix. A variety of work-arounds have been suggested.
The most popular is to group distant points together and treat them as a single
point via a multipole method.

This is not currently implemented in `sneer`, and I suggest looking at the 
[Rtsne](https://cran.r-project.org/package=Rtsne) package for the most popular
implementation using the Barnes-Hut algorithm. It's a wrapper around C++ code, 
so it's fast too! Might as well throw in a few references, while we're here:

Van Der Maaten, L. (2013). 
Barnes-Hut-SNE.
*arXiv preprint* *arXiv*:1301.3342.
https://arxiv.org/abs/1301.3342

Yang, Z., Peltonen, J., & Kaski, S. (2013, June). 
Scalable Optimization of Neighbor Embedding for Visualization. 
In *Proceedings of the 30th international conference on machine learning (ICML-13)*
(pp. 127-135).
http://www.jmlr.org/proceedings/papers/v28/yang13b.html

Van Der Maaten, L. (2014). 
Accelerating t-SNE using tree-based algorithms. 
*Journal of machine learning research*, *15*(1), 3221-3245.
http://www.jmlr.org/papers/v15/vandermaaten14a.html

Vladymyrov, M., & Carreira-Perpinán, M. A. (2014). 
Linear-time training of nonlinear low-dimensional embeddings. 
In *17th International Conference on Artificial Intelligence and Statistics (AISTATS 2014)* 
(pp. 968-977).
http://jmlr.org/proceedings/papers/v33/vladymyrov14.html

Parviainen, E. (2016). 
A graph-based N-body approximation with application to stochastic neighbor 
embedding. 
*Neural Networks*, *75*, 1-11.
http://dx.doi.org/10.1016/j.neunet.2015.11.007

#### Divergences

Cichocki, A., Cruces, S., & Amari, S. I. (2011). 
Generalized alpha-beta divergences and their application to robust nonnegative 
matrix factorization. 
*Entropy*, *13*(1), 134-170.
https://dx.doi.org/10.3390/e13010134

Generalizes divergences (including the Kullback-Leibler divergence used in the 
SNE family) to non-normalized weights.

Bunte, K., Haase, S., Biehl, M., & Villmann, T. (2012). 
Stochastic neighbor embedding (SNE) for dimension reduction and visualization 
using arbitrary divergences. 
*Neurocomputing*, *90*, 23-45.
http://dx.doi.org/10.1016/j.neucom.2012.02.034

If you like divergences, this paper has you covered.

#### Why Do Probability-Based Embeddings Work?

Lee, J. A., & Verleysen, M. (2011). 
Shift-invariant similarities circumvent distance concentration in stochastic 
neighbor embedding and variants. 
*Procedia Computer Science*, 4, 538-547.
http://dx.doi.org/10.1016/j.procs.2011.04.056

Lee, J. A., & Verleysen, M. (2014, December). 
Two key properties of dimensionality reduction methods. 
In *Computational Intelligence and Data Mining (CIDM), 2014 IEEE Symposium on* 
(pp. 163-170). IEEE.
http://dx.doi.org/10.1109/CIDM.2014.7008663

Two papers which try and work out why the probability-based embedding methods
work so much better than other algorithms. They suggest that it's because
of the normalization that occurs when converting weights to probabilities, and
the ability to ignore or modify specific input distances (which they term
"plasticity").

Demartines, P., & Hérault, J. (1997). 
Curvilinear component analysis: A self-organizing neural network for nonlinear 
mapping of data sets. 
*IEEE Transactions on neural networks*, *8*(1), 148-154.
http://dx.doi.org/10.1109/72.554199

CCA is interesting for two reasons: first, it uses a cutoff parameter, so that
distances beyond a certain value are ignored in the gradient. Second, it uses
a random pair-wise update to optimize the embedding, effectively using 
stochastic gradient descent. The Lee & Verleysen 2014 paper points to the first
property as an important component in dimensionality reduction.

Agrafiotis, D. K., & Xu, H. (2002). 
A self-organizing principle for learning nonlinear manifolds. 
*Proceedings of the National Academy of Sciences*, *99*(25), 15869-15872.
http://dx.doi.org/10.1073/pnas.242424399

This paper describes Stochastic Proximity Embedding (SPE), which despite its
name is much more related to CCA than SNE. It seems to be an independent
rediscovery of many of the principles of CCA, although there are some important
differences (particularly with respect to the Lee & Verleysen concept of 
plasticity mentioned above). There is an R 
[spe package](https://cran.r-project.org/package=spe) to play with if you're
interested.

#### Majorization-Minimization

Yang, Z., Peltonen, J., & Kaski, S. (2015). 
Majorization-Minimization for Manifold Embedding. In
*Proceedings of the 18th International Conference on Artificial Intelligence and Statistics (AISTATS 2015)*
(pp. 1088-1097).
http://www.jmlr.org/proceedings/papers/v38/yang15a.html

#### Two excellent interactive web pages

Visualizing MNIST: An Exploration of Dimensionality Reduction
http://colah.github.io/posts/2014-10-Visualizing-MNIST/

How to Use t-SNE Effectively
http://distill.pub/2016/misread-tsne/

Previous: [Visualization](visualization.html). Up: [Index](index.html).
