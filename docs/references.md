---
title: "References"
output:
  html_document:
    theme: cosmo
    toc: true
    toc_float:
      collapsed: false
---

Previous: [Visualization](visualization.html). Up: [Index](index.html).

Here's a list of things (mainly papers) that provided the various options
available in `sneer` or I just think are interesting. I haven't tried to be
exhaustive. Some areas I haven't had time or inclination to investigate,
including out-of-sample mapping and parametric t-SNE (e.g.
[Lion t-SNE](https://arxiv.org/abs/1708.04983)), or
graph layout (e.g. [tsNET](https://doi.org/10.1111/cgf.13187)).

I've tried to link to the offical publication websites rather than to PDFs
directly, so it's a bit easier to find or confirm bibliographic information.

## Embedding Methods

### Review

van der Maaten, L., Postma, E., & van der Herik, J. (2009).
*Dimensionality Reduction: A Comparative Review*
(TiCC TR 2009-005). Tilburg: Tilburg University.
<https://www.tilburguniversity.edu/upload/59afb3b8-21a5-4c78-8eb3-6510597382db_TR2009005.pdf>

Covers the state of the art prior to t-SNE.

Peel, D., & McLachlan, G. J. (2000).
Robust mixture modelling using the t distribution.
*Statistics and Computing*, *10*(4), 339-348.
<https://doi.org/10.1023/A:1008981510081>

de Ridder, D., & Franc, V. (2003).
Robust subspace mixture models using t-distributions.
In *Proceedings of the British Machine Vision Conference* (pp. 319–328).
<https://doi.org/10.5244/C.17.35>

Two examples of the use of t-distributions as replacement for Gaussians in
mixture modeling for use in clustering (the Peel & McLachlan paper) and manifold
learning (the de Ridder & Franc) paper. The t-SNE paper doesn't cite either of
these, but the van der Maaten review paper cites the de Ridder & Franc paper,
which in turn cites Peel & McLachlan.

Priam, R. (2018, January).
Symmetric Generative Methods and tSNE: A Short Survey.
In *VISIGRAPP (3: IVAPP)* (pp. 356-363).

Covers a lot of the more recent literature, and draws some connections with
generative and probabilistic models.

### Metric Multi Dimensional Scaling (MDS)

Borg, I., & Groenen, P. J. (2005).
*Modern multidimensional scaling: Theory and applications.*
Springer Science & Business Media.
<https://dx.doi.org/10.1007/0-387-28981-X>

Hughes, N. P., & Lowe, D. (2002).
Artefactual Structure from Least-Squares Multidimensional Scaling.
In *Advances in Neural Information Processing Systems* (pp. 913-920).
<https://papers.nips.cc/paper/2239-artefactual-structure-from-least-squares-multidimensional-scaling>

### Sammon Mapping

Sammon, J. W. (1969).
A nonlinear mapping for data structure analysis.
*IEEE Transactions on computers*, *18*(5), 401-409.
<https://dx.doi.org/10.1109/T-C.1969.222678>

### Stochastic Neighbor Embedding (SNE)

Hinton, G. E., & Roweis, S. T. (2002).
Stochastic neighbor embedding.
In *Advances in neural information processing systems* (pp. 833-840).
<https://papers.nips.cc/paper/2276-stochastic-neighbor-embedding>

### Symmetric Stochastic Neighbor Embedding (SSNE)

Cook, J., Sutskever, I., Mnih, A., & Hinton, G. E. (2007).
Visualizing similarity data with a mixture of maps.
In *International Conference on Artificial Intelligence and Statistics* (pp. 67-74).
<https://www.cs.toronto.edu/~amnih/papers/sne_am.pdf>

### t-Distributed Stochastic Neighbor Embedding (t-SNE)

Van der Maaten, L., & Hinton, G. (2008).
Visualizing data using t-SNE.
*Journal of Machine Learning Research*, *9* (2579-2605).
<http://www.jmlr.org/papers/v9/vandermaaten08a.html>

### Heavy-Tailed Symmetric Stochastic Neighbor Embedding (HSSNE)

Yang, Z., King, I., Xu, Z., & Oja, E. (2009).
Heavy-tailed symmetric stochastic neighbor embedding.
In *Advances in neural information processing systems* (pp. 2169-2177).
<https://papers.nips.cc/paper/3770-heavy-tailed-symmetric-stochastic-neighbor-embedding>

van der Maaten, L. (2009).
Learning a Parametric Embedding by Preserving Local Structure
In *Proceedings of the 12th International Conference on Artificial Intelligence and Statistics (AISTATS)* (pp 384-391).
<http://proceedings.mlr.press/v5/maaten09a>

Independently proposes a very similar technique to HSSNE, and provides three
strategies for determining the heavy tailedness, including minimizing it along
with the coordinates.

De Bodt, C., Mulders, D., Verleysen, M., & Lee, J. A. (2018).
Perplexity-free t-SNE and twice Student tt-SNE.
In *European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2018)*
(pp. 123-128).
<http://hdl.handle.net/2078.1/200844>

The tt-SNE method proposes using a heavy-tailed kernel in not only the
low-dimensional output space, but in the input dimension too. The
heavy-tailedness is related to the estimate of intrinsic dimensionality, which
is described in the multi-scale JSE paper (Lee and co-workers, 2015, see below
in the multi-scale section). In the output space, if embedding into 2D, you end
up with something very close to the usual t-SNE kernel.

Kobak, D., Linderman, G., Steinerberger, S., Kluger, Y., & Berens, P. (2019)
Heavy-tailed kernels reveal a finer cluster structure in t-SNE visualisations
*arXiv preprint* *arXiv*:1902.05804.
<https://arxiv.org/abs/1902.05804>
<https://github.com/dkobak/finer-tsne>

Describes implementing HSSNE in the 
[FIt-SNE](https://github.com/KlugerLab/FIt-SNE) software (see below), and
some interesting discussion on effective values for the heavy tailedness. They
also note that optimization time increases with heavier tails.

Kahloot, K., & Ekler, P. (2019).
Improving t-SNE clustering and visualization.
<http://real.mtak.hu/100794/1/AACS19_paper_12.pdf>
<https://github.com/kkahloots/Improving-t-SNE-Visualization-and-Clustering>

This paper applies the heavy-tailed t-SNE to the
[small NORB](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/) and
[CIFAR 10 and 100](https://www.cs.toronto.edu/~kriz/cifar.html) datasets and
suggest a slightly heavier tail than mentioned in the Kobak paper, although they
don't touch on optimization time.

### Inhomogeneous t-SNE

Kitazono, J., Grozavu, N., Rogovschi, N., Omori, T., & Ozawa, S. (2016, October).
t-Distributed Stochastic Neighbor Embedding with Inhomogeneous Degrees of Freedom.
In *Proceedings of the 23rd International Conference on Neural Information Processing (ICONIP 2016)*
(pp. 119-128).
<http://dx.doi.org/10.1007/978-3-319-46675-0_14>

### Weighted Symmetric Stochastic Neighbor Embedding (ws-SNE)

Yang, Z., Peltonen, J., & Kaski, S. (2014).
Optimization equivalence of divergences improves neighbor embedding.
In *Proceedings of the 31st International Conference on Machine Learning (ICML-14)*
(pp. 460-468).
<http://jmlr.org/proceedings/papers/v32/yange14.html>

### Neighbor Retrieval Visualizer (NeRV)

Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010).
Information retrieval perspective to nonlinear dimensionality reduction for
data visualization.
*Journal of Machine Learning Research*, *11*, 451-490.
<http://www.jmlr.org/papers/v11/venna10a.html>

### Jensen-Shannon Embedding (JSE)

Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013).
Type 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in
dimensionality reduction based on similarity preservation.
*Neurocomputing*, *112*, 92-108.
<https://dx.doi.org/10.1016/j.neucom.2012.12.036>

### Elastic Embedding (EE)

Carreira-Perpinán, M. A. (2010, June).
The Elastic Embedding Algorithm for Dimensionality Reduction.
In *Proceedings of the 27th International Conference on Machine Learning (ICML-10)*
(pp. 167-174).
<http://faculty.ucmerced.edu/mcarreira-perpinan/papers/icml10.pdf> (PDF)

Not implemented in `sneer`, but is related to SSNE. Also interesting for having
a separable cost function, i.e. no normalization of weights occurs. The
"Optimization equivalence of divergences improves neighbor embedding" paper,
(see above, under the ws-SNE heading), goes into more detail on the connection
between SSNE and EE. Also, see the
[experimental gradients](http://jlmelville.github.io/sneer/experimental-gradients.html#elastic_embedding)
page for details on the gradient.

### LargeVis

Tang, J., Liu, J., Zhang, M., & Mei, Q. (2016, April).
Visualizing large-scale and high-dimensional data.
In *Proceedings of the 25th International Conference on World Wide Web*
(pp. 287-297). International World Wide Web Conferences Steering Committee.
<https://arxiv.org/abs/1602.00370>

Not implemented in `sneer`, but another method that doesn't use normalization of
the output weights. Despite this, it performs very well. The cost function shows
some resemblance to Elastic Embedding. It also goes to a lot of effort to be
scalable to large datasets, and the lack of normalization is critical in this
effort. Also, see the
[experimental gradients](http://jlmelville.github.io/sneer/experimental-gradients.html#largevis)
page for details on the gradient. The
[source code](https://github.com/lferry007/LargeVis) is on GitHub,
and there is also a [CRAN package](https://cran.r-project.org/package=largeVis).

### UMAP

McInnes, L., & Healey, J. (2018).
UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction
*arXiv preprint*, *arXiv*:1802.03426.
<https://arxiv.org/abs/1802.03426>
<https://github.com/lmcinnes/umap>

Also not in `sneer`. Uses a similar optimization scheme to LargeVis but the
theory behind it is based on fuzzy sets. Also uses non-normalized weights.

### NCVis

Artemenkov, A., & Panov, M. (2020, April). 
NCVis: Noise Contrastive Approach for Scalable Visualization. 
In *Proceedings of The Web Conference 2020* (pp. 2941-2947).
<https://doi.org/10.1145/3366423.3380061>
<https://github.com/stat-ml/ncvis>

A method that turns out to be very similar to LargeVis and UMAP in
implementation, but which uses Noise Contrastive Estimation (a method to
estimate the parameters of probabilistic models commonly used in language
modeling) to motivate its cost function and the sampling strategy.

### xSNE

Strickert, M. (2012, August).
No Perplexity in Stochastic Neighbor Embedding.
In *Workshop New Challenges in Neural Computation 2012* (p. 68).
<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.6041&rep=rep1&type=pdf#page=68>
<http://mloss.org/software/view/418/>

Not implemented in `sneer`. Suggests using rank-based data to avoid perplexity
calculations.

### Doubly Stochastic Neighbor Embedding on Spheres (DOSNES)

Lu, Y., Yang, Z., & Corander, J. (2016).
Doubly Stochastic Neighbor Embedding on Spheres.
*arXiv preprint* *arXiv*:1609.01977.
<https://arxiv.org/abs/1609.01977>

Not implemented in `sneer`. Notes that even t-SNE can show a crowding problem if
embedding graph data with a large disparity in the connectivities of the nodes:
highly connected nodes are placed in the center of the plot, lower connectivity
nodes are pushed to the edges. The proposed solution is to project the
probability matrices to a doubly stochastic version and then view the embedding
on a spherical projection.

### Global t-SNE (g-SNE)

Zhou, Y., & Sharpee, T. (2018).
Using global t-SNE to preserve inter-cluster data structure.
*bioRxiv*, 331611.
<https://doi.org/10.1101/331611>
<https://github.com/gyrheart/gsne>

Not implemented in `sneer`. Like NeRV and JSE, suggests adding a second
divergence to encourage long range structure. In this case, it's also the KL
divergence but with both the input and output kernels being the reciprocal of
the typical t-SNE weight function. You can see a derivation of the gradient (at
least up to the force constant) at the
[gradients](http://jlmelville.github.io/sneer/gradients.html) page.

## Perplexity

Vladymyrov, M., & Carreira-Perpinán, M. A. (2013, June).
Entropic Affinities: Properties and Efficient Numerical Computation.
*Proceedings of the 30th international conference on machine learning (ICML-13)*,
pp. 477-485.
<http://jmlr.org/proceedings/papers/v28/vladymyrov13.html>

Not implemented in `sneer`, but describes fast ways to calculate the input
probabilities for large data sets. Also, coins the term "entropic affinities" to
describe the process of calibrating probabilities via setting a perplexity
value.

Cao, Y., & Wang, L. (2017).
Automatic Selection of t-SNE Perplexity.
*arXiv preprint* *arXiv*:1708.03229.
<https://arxiv.org/abs/1708.03229>

Suggests that you can choose between embeddings with different perplexities by a
regularized version of the final error.

Schubert, E., & Gertz, M. (2017, October).
Intrinsic t-Stochastic Neighbor Embedding for Visualization and Outlier
Detection.
In *International Conference on Similarity Search and Applications*
(pp. 188-203).
Springer, Cham.
<https://doi.org/10.1007/978-3-319-68474-1_13>

Not exactly about perplexity, but connected: discusses intrinsic dimensionality,
and correcting input distances in high dimensional spaces. Also suggests an
alternative input probability normalization: the geometric mean rather than
arithmetric mean, which effectively converts the input probability from a
symmetric knn graph to a mutual knn graph.

## Multiscale Embedding

Lee, J. A., Peluffo-Ordónez, D. H., & Verleysen, M. (2014).
Multiscale stochastic neighbor embedding: Towards parameter-free
dimensionality reduction.
In *Proceedings of 2014 European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2014)* (pp. 177-182).
<https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2014-64.pdf> (PDF)

Lee, J. A., Peluffo-Ordónez, D. H., & Verleysen, M. (2015).
Multi-scale similarities in stochastic neighbour embedding: Reducing
dimensionality while preserving both local and global structure.
*Neurocomputing*, *169*, 246-261.
<https://dx.doi.org/10.1016/j.neucom.2014.12.095>

Fun fact: just noticed that this is the only paper on the list where "neighbour"
is spelt in the British English style.

De Bodt, C., Mulders, D., Verleysen, M., & Lee, J. A. (2018).
Perplexity-free t-SNE and twice Student tt-SNE.
In *European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2018)*
(pp. 123-128).
<http://hdl.handle.net/2078.1/200844>

This describes multiscale t-SNE, which is radically simpler than the multiscale
SNE and JSE methods in the previous two papers: the probabilities are averaged
without any mention of slowing adding them over the course of the embedding, and
no modification is made to the output kernel.

tt-SNE is discussed in the section on HSSNE above.

## Optimization

### Delta-Bar-Delta

Jacobs, R. A. (1988).
Increased rates of convergence through learning rate adaptation.
*Neural networks*, *1*(4), 295-307.
<https://dx.doi.org/10.1016/0893-6080(88)90003-2>

Janet, J. A., Scoggins, S. M., Schultz, S. M., Snyder, W. E., White, M. W.,
& Sutton, J. C. (1998, May).
Shocking: An approach to stabilize backprop training with greedy adaptive learning rates.
In *1998 IEEE International Joint Conference on Neural Networks Proceedings.*
(Vol. 3, pp. 2218-2223). IEEE.
<https://dx.doi.org/10.1109/IJCNN.1998.687205>

### Early Exaggeration

Linderman, G. C., & Steinerberger, S. (2017).
Clustering with t-SNE, provably.
*arXiv preprint* *arXiv*:1706.02582.
<https://arxiv.org/abs/1706.02582>

Suggests that the early exaggeration phase of t-SNE optimization effectively
carries out spectral clustering. Based on this relationship, the authors also
suggest that using standard gradient descent (no momentum and a fixed learning
rate of 1), and a much larger exaggeration factor of *n* / 10, where *n* is the
number of points in the dataset, may be more effective than the usual early
exaggeration settings.

Belkina, A. C., Ciccolella, C. O., Anno, R., Halpert, R., Spidlen, J., & Snyder-Cappione, J. E. (2019). 
Automated optimized parameters for T-distributed stochastic neighbor embedding
improve visualization and analysis of large datasets. 
*Nature communications*, *10*(1), 1-12.
<https://doi.org/10.1038/s41467-019-13055-y>
*bioRxiv*, 451690.
<https://doi.org/10.1101/451690>

Recommends monitoring the KL divergence to determine when to turn off the
exaggeration factor, rather than using a fixed number of iteration. Also for the
large datasets studied (transcriptomics and cytometry), they don't notice a big
difference when using an exaggeration factor between 4-20.

Additionally, the recommend setting the learning rate to $n / \alpha$, where
$n$ is the size of the dataset, and $\alpha$ is the exaggeration factor. For
typical exaggeration factors of 4-12, this would mean that with larger datasets
(more than n ~ 2000), you can get away with a larger learning rate than is
typical (it's 200 in BH t-SNE).

For another paper that discusses learning rate, see bigMap by Garriga and 
Bartumeus, below.

Böhm, J. N., Berens, P., & Kobak, D. (2020). 
A Unifying Perspective on Neighbor Embeddings along the Attraction-Repulsion 
Spectrum.
*arXiv preprint* *arXiv*:2007.08902.
<https://arxiv.org/abs/2007.08902>

This paper extends the observation of Linderman and Steinerberger that t-SNE
with increasing early exaggeration reduces to Laplacian Eigenmaps, to place
t-SNE in a family of algorithms which differ only in the relative weighting
of attraction vs repulsions. This family contains not just LE, but also UMAP,
i.e. despite the apparent differences in theoretical underpinnings, UMAP is 
effectively t-SNE with a moderate exaggeration factor applied throughout the
optimization.

### Spectral Directions

Vladymyrov, M., & Carreira-Perpiñán, M. A. (2012).
Partial-Hessian Strategies for Fast Learning of Nonlinear Embeddings.
In *Proceedings of the 29th International Conference on Machine Learning (ICML-12)*
(pp. 345-352).
<https://arxiv.org/abs/1206.4646>

Van Der Maaten, L. (2010).
Fast optimization for t-SNE.
In *NIPS Workshop on Challenges in Data Visualization.*
<http://cseweb.ucsd.edu/~lvdmaaten/workshops/nips2010/papers/vandermaaten.pdf> (PDF)

This advocates a similar approach to spectral directions, although with a
slightly different line search and without enforcing positive definiteness of
the inverse Hessian approximation.

### Nesterov Accelerated Gradient

On the importance of initialization and momentum in deep learning.
In *Proceedings of the 30th international conference on machine learning (ICML-13)*
(pp. 1139-1147).
<http://www.jmlr.org/proceedings/papers/v28/sutskever13.html>

### Adaptive Restart

O'Donoghue, B., & Candes, E. (2013).
Adaptive restart for accelerated gradient schemes.
*Foundations of computational mathematics*, *15*(3), 715-732.
<https://dx.doi.org/10.1007/s10208-013-9150-3>
<https://arxiv.org/abs/1204.3982>

Su, W., Boyd, S., & Candes, E. J. (2016).
A differential equation for modeling nesterov’s accelerated gradient method: theory and insights.
*Journal of Machine Learning Research*, *17*(153), 1-43.
<http://jmlr.org/papers/v17/15-084.html>

## Evaluation

### Precision-Recall AUC and Receiver Operating Characteristic Area Under the Curve

Davis, J., & Goadrich, M. (2006, June).
The relationship between Precision-Recall and ROC curves.
In *Proceedings of the 23rd international conference on Machine learning*
(pp. 233-240). ACM.
<http://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf> (PDF)

Keilwagen, J., Grosse, I., & Grau, J. (2014).
Area under precision-recall curves for weighted and unweighted data.
*PloS One*, *9*(3), e92209.
<https://dx.doi.org/10.1371/journal.pone.0092209.>

### Neighborhood Preservation

Lee, J. A., & Verleysen, M. (2009).
Quality assessment of dimensionality reduction: Rank-based criteria.
*Neurocomputing*, *72*(7), 1431-1443.
<https://dx.doi.org/10.1016/j.neucom.2008.12.017>

## Miscellany

Things that aren't in `sneer`, but piqued my interest, or don't fit into the
other categories.

### Distance Approximations

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
<https://arxiv.org/abs/1301.3342>

Yang, Z., Peltonen, J., & Kaski, S. (2013, June).
Scalable Optimization of Neighbor Embedding for Visualization.
In *Proceedings of the 30th international conference on machine learning (ICML-13)*
(pp. 127-135).
<http://www.jmlr.org/proceedings/papers/v28/yang13b.html>

Van Der Maaten, L. (2014).
Accelerating t-SNE using tree-based algorithms.
*Journal of machine learning research*, *15*(1), 3221-3245.
<http://www.jmlr.org/papers/v15/vandermaaten14a.html>

Vladymyrov, M., & Carreira-Perpinán, M. A. (2014).
Linear-time training of nonlinear low-dimensional embeddings.
In *17th International Conference on Artificial Intelligence and Statistics (AISTATS 2014)*
(pp. 968-977).
<http://jmlr.org/proceedings/papers/v33/vladymyrov14.html>

Parviainen, E. (2016).
A graph-based N-body approximation with application to stochastic neighbor
embedding.
*Neural Networks*, *75*, 1-11.
<http://dx.doi.org/10.1016/j.neunet.2015.11.007>

Linderman, G. C., Rachh, M., Hoskins, J. G., Steinerberger, S., & Kluger, Y. (2017).
Efficient Algorithms for t-distributed Stochastic Neighborhood Embedding.
*arXiv preprint* *arXiv*:1712.09005.
<https://arxiv.org/abs/1712.09005>
<https://doi.org/10.1038/s41592-018-0308-4>
<https://github.com/KlugerLab/FIt-SNE>

Modified Barnes-Hut t-SNE by using interpolation onto a grid and using a Fast
Fourier Transform to give $O(N)$ scaling.

De Bodt, C., Mulders, D., Verleysen, M., & Lee, J. A. (2018).
Extensive assessment of Barnes-Hut t-SNE
In *European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2018)*
(pp. 135-140).
<http://hdl.handle.net/2078.1/200843>

Compares BH t-SNE with standard t-SNE in terms of neighbor retrieval over
several datasets. They confirm that it works very well, and recommend the
current default setting for the hyperparameter $\theta = 0.5$.

Chan, D. M., Rao, R., Huang, F., & Canny, J. F. (2018).
t-SNE-CUDA: GPU-Accelerated t-SNE and its Applications to Modern Data.
*arXiv preprint* *arXiv*:1807.11824.
<https://github.com/CannyLab/tsne-cuda>

BH t-SNE implemented on a GPU.

Policar, P. G., Strazar, M., & Zupan, B. (2019).
openTSNE: a modular Python library for t-SNE dimensionality reduction and embedding.
*bioRxiv*, 731877.
<https://doi.org/10.1101/731877>
<https://github.com/pavlin-policar/openTSNE>

Describes a Python package that implements a lot of state-of-the-art practices:
Barnes-Hut of FFT, PCA initialization, perplexity annealing, and allows for
embedding new points.

### Other Fast t-SNE Methods

Many of these methods are intended for interactive data analysis.

Kim, M., Choi, M., Lee, S., Tang, J., Park, H., & Choo, J. (2016).
PixelSNE: Visualizing Fast with Just Enough Precision via Pixel-Aligned
Stochastic Neighbor Embedding.
*arXiv preprint* *arXiv*:1611.02568.
<https://arxiv.org/abs/1611.02568>
<https://github.com/awesome-davian/PixelSNE>

Achieves a speed up by using the fact that screen resolution means that the
usual precision in the calculations is completely unnecessary.

Pezzotti, N., Höllt, T., Lelieveldt, B., Eisemann, E., & Vilanova, A. (2016, June).
Hierarchical stochastic neighbor embedding.
In *Computer Graphics Forum* (Vol. 35, No. 3, pp. 21-30).
<https://doi.org/10.1111/cgf.12878>

Approximated and User Steerable tSNE for Progressive Visual Analytics
Pezzotti, N., Lelieveldt, B. P., van der Maaten, L., Höllt, T., Eisemann, E., & Vilanova, A. (2017).
*IEEE transactions on visualization and computer graphics*, *23*(7), 1739-1752.

Describes Approximated t-SNE (At-SNE). Suggests using a forest of randomized
KD-Trees as the approximate nearest neighbors calculation for speeding up the
perplexity calibration.

Pezzotti, N., Mordvintsev, A., Hollt, T., Lelieveldt, B. P., Eisemann, E., & Vilanova, A. (2018).
Linear tSNE optimization for the Web.
*arXiv preprint* *arXiv*:1805.10817.
<https://github.com/tensorflow/tfjs-tsne>

Uses WebGL textures to calculate approximate repulsive interactions and then
uses tensorflowjs to optimize.

bigMap: Big Data Mapping with Parallelized t-SNE
Garriga, J., & Bartumeus, F. (2018).
*arXiv preprint* *arXiv*:1812.09869.
<https://cran.r-project.org/package=bigMap>

An R package implementing a parallelized t-SNE algorithm intended to be
distributed across multiple machines in a cluster. This requires a smoother
optimization scheme than the effect of early exaggeration causes, so it must do
without that (and also the momentum switching that occurs with the usual DBD
optimizer).

There is also an interesting discussion of what they call "the big crowd
problem", which arises due to the normalization of affinities into
probabilities: there is only a fixed amount of probability mass that can be
partitioned to each pair of points, so as the dataset grows this will naturally
lead to lowered average similarities. This results in the cost function having a
dependence on the size of the dataset (as in `sneer` there is an attempt to
normalize for this). Additionally, this means that the $p_{ij} - q_{ij}$
term in the gradient tends to zero as the dataset sizes increase, and because of
the Student-t distribution, the other bit of the t-SNE gradient term
$w_{ij} (\mathbf{y}_i - \mathbf{y}_j)$ *also* tends to zero as the embedding
area grows, i.e. over the course of the optimization. An adaptive learning rate
is suggested to counteract these tendencies:
$\eta = \log(N - 1) (\mathbf{y}_{max} - \mathbf{y}_{min}) / 2$. That learning
rate does however suggest that with the standard t-SNE initialization, where
$(\mathbf{y}_{max} - \mathbf{y}_{min})$ is very close to 0, would result in
a very small learning rate initially, and indeed the default random
initialization is a random distribution over a circle of radius 1, which
suggests the typical t-SNE initalization would be a problem.

### Divergences

The "Optimization equivalence of divergences improves neighbor embedding"
paper of Yang and co-workers (2014), listed under the "ws-SNE" section also
contains an extensive discussion of divergences.

Cichocki, A., Cruces, S., & Amari, S. I. (2011).
Generalized alpha-beta divergences and their application to robust nonnegative
matrix factorization.
*Entropy*, *13*(1), 134-170.
<https://dx.doi.org/10.3390/e13010134>

Generalizes divergences (including the Kullback-Leibler divergence used in the
SNE family) to non-normalized weights.

Bunte, K., Haase, S., Biehl, M., & Villmann, T. (2012).
Stochastic neighbor embedding (SNE) for dimension reduction and visualization
using arbitrary divergences.
*Neurocomputing*, *90*, 23-45.
<http://dx.doi.org/10.1016/j.neucom.2012.02.034>

If you like divergences, this paper has you covered. Applies Bregman, f-, and
$\gamma$- divergences to COIL-20 and the Olivetti faces.

Narayan, K. S., Punjani, A., & Abbeel, P. (2015, June).
Alpha-Beta Divergences Discover Micro and Macro Structures in Data.
In *Proceedings of the 32nd International Conference on Machine Learning (ICML-14)*
(pp 796-804).
<http://proceedings.mlr.press/v37/narayan15.html>

Considers using the generalized $\alpha, \beta$-divergence (as discussed in the
Cichocki paper above) as a generalization of t-SNE.

Im, D. J., Verma, N., & Branson, K. (2018).
Stochastic Neighbor Embedding under f-divergences.
*arXiv preprint* *arXiv*:1811.01247.
<https://arxiv.org/abs/1811.01247>
<https://github.com/jiwoongim/ft-SNE>

Looks at a subset of divergences, the f-divergences, which includes the
Kullback-Leibler and Jensen-Shannon divergences, suggesting that while clustered
data is best served by the KL divergence, other types of data (manifolds,
hierarchical data) would benefit from other divergences.

The code at the github page is an implementation in Theano, and does not use
early exaggeration or the DBD optimization method. Instead it uses a learning
rate and momentum decay, although by default the momentum switches back to a
large value after the first 10% of the optimization, which is a little like the
DBD method.

### Why Do Probability-Based Embeddings Work?

Lee, J. A., & Verleysen, M. (2011).
Shift-invariant similarities circumvent distance concentration in stochastic
neighbor embedding and variants.
*Procedia Computer Science*, 4, 538-547.
<http://dx.doi.org/10.1016/j.procs.2011.04.056>

Lee, J. A., & Verleysen, M. (2014, December).
Two key properties of dimensionality reduction methods.
In *Computational Intelligence and Data Mining (CIDM), 2014 IEEE Symposium on*
(pp. 163-170). IEEE.
<http://dx.doi.org/10.1109/CIDM.2014.7008663>

Two papers which try and work out why the probability-based embedding methods
work so much better than other algorithms. They suggest that it's because of the
normalization that occurs when converting weights to probabilities, and the
ability to ignore or modify specific input distances (which they term
"plasticity").

Demartines, P., & Hérault, J. (1997).
Curvilinear component analysis: A self-organizing neural network for nonlinear
mapping of data sets.
*IEEE Transactions on neural networks*, *8*(1), 148-154.
<http://dx.doi.org/10.1109/72.554199>

CCA is interesting for two reasons: first, it uses a cutoff parameter, so that
distances beyond a certain value are ignored in the gradient. Second, it uses a
random pair-wise update to optimize the embedding, effectively using stochastic
gradient descent. The Lee & Verleysen 2014 paper points to the first property as
an important component in dimensionality reduction.

Agrafiotis, D. K., & Xu, H. (2002).
A self-organizing principle for learning nonlinear manifolds.
*Proceedings of the National Academy of Sciences*, *99*(25), 15869-15872.
<http://dx.doi.org/10.1073/pnas.242424399>

This paper describes Stochastic Proximity Embedding (SPE), which despite its
name is much more related to CCA than SNE. It seems to be an independent
rediscovery of many of the principles of CCA, although there are some important
differences (particularly with respect to the Lee & Verleysen concept of
plasticity mentioned above). There is an R
[spe package](https://cran.r-project.org/package=spe) to play with if you're
interested.

### Majorization-Minimization

Yang, Z., Peltonen, J., & Kaski, S. (2015).
Majorization-Minimization for Manifold Embedding. In
*Proceedings of the 18th International Conference on Artificial Intelligence and Statistics (AISTATS 2015)*
(pp. 1088-1097).
<http://www.jmlr.org/proceedings/papers/v38/yang15a.html>

### Initialization

Kobak, D., & Berens, P. (2019).
The art of using t-SNE for single-cell transcriptomics.
*Nature communications*, *10*(1), 1-14.
<https://doi.org/10.1038/s41467-019-13056-x>
*bioRxiv*, 453449.
<https://doi.org/10.1101/453449>

Lots to mull over in this paper, but particularly interesting in advocating the
use of the first two principal components (suitably scaled) as a deterministic
initialization for t-SNE that retains global structure well.

Becht, E., McInnes, L., Healy, J., Dutertre, C. A., Kwok, I. W., Ng, L. G., ... & Newell, E. W. (2019). 
Dimensionality reduction for visualizing single-cell data using UMAP.
*Nature biotechnology*, *37*(1), 38.
<https://doi.org/10.1038/nbt.4314>

This argues that UMAP is superior to t-SNE for some biological datasets, because
among other things, it is better at preserving global structure.

Kobak, D., & Linderman, G. C. (2019). 
UMAP does not preserve global structure any better than t-SNE when using the same initialization. 
*bioRxiv*.
<https://doi.org/10.1101/2019.12.19.877522>

Kobak and Linderman take issue with the Becht and co-workers paper above, 
finding that t-SNE initialized with PCA does as good a job as UMAP in preserving
global structure.

### Adding New Points

Poličar, P. G., Stražar, M., & Zupan, B. (2019).
Embedding to Reference t-SNE Space Addresses Batch Effects in Single-Cell Classification. 
*bioRxiv*, 671404.
<https://doi.org/10.1101/671404>

Dealing with adding new points to an existing t-SNE embedding is also discussed
in "The art of using t-SNE for single-cell transcriptomics". This paper 
considers it as a way to remove batch effects when visualizing data from
multiple sources.


### Web Pages

Visualizing MNIST: An Exploration of Dimensionality Reduction
<http://colah.github.io/posts/2014-10-Visualizing-MNIST/>

How to Use t-SNE Effectively
<http://distill.pub/2016/misread-tsne/>

Previous: [Visualization](visualization.html). Up: [Index](index.html).
