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

#### Weighted Symmetric Stochastic Neighbor Embedding (ws-SNE)
Yang, Z., Peltonen, J., & Kaski, S. (2014).
Optimization equivalence of divergences improves neighbor embedding.
In *Proceedings of the 31st International Conference on Machine Learning (ICML-14)*
(pp. 460-468).
http://jmlr.org/proceedings/papers/v32/yange14.pdf

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

#### Multiscale Embedding
Lee, J. A., Peluffo-Ordónez, D. H., & Verleysen, M. (2014).
Multiscale stochastic neighbor embedding: Towards parameter-free
dimensionality reduction. In ESANN.
https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2014-64.pdf

Lee, J. A., Peluffo-Ordónez, D. H., & Verleysen, M. (2015).
Multi-scale similarities in stochastic neighbour embedding: Reducing
dimensionality while preserving both local and global structure.
*Neurocomputing*, *169*, 246-261.
https://dx.doi.org/10.1016/j.neucom.2014.12.095

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
http://www.jmlr.org/proceedings/papers/v28/sutskever13.pdf

#### Adaptive Restart
O'Donoghue, B., & Candes, E. (2013).
Adaptive restart for accelerated gradient schemes.
*Foundations of computational mathematics*, *15*(3), 715-732.
https://arxiv.org/abs/1204.3982

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
Two excellent interactive web pages:

Visualizing MNIST: An Exploration of Dimensionality Reduction
http://colah.github.io/posts/2014-10-Visualizing-MNIST/

How to Use t-SNE Effectively
http://distill.pub/2016/misread-tsne/

Previous: [Visualization](visualization.html). Up: [Index](index.html).
