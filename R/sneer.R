#' sneer: Stochastic Neighbor Embedding Experiments in R.
#'
#' A package for exploring probability-based embedding and related forms
#' of dimensionality reduction. Its main goal is to implement multiple
#' embedding methods within a single framework so comparison between them is
#' easier, without worrying about the effect of differences in preprocessing,
#' optimization and heuristics. Additionally it provides a way to explore the
#' importance of those effects on the performance of individual methods. For
#' example, how much of an effect does the early exaggeration trick in t-SNE
#' have on the embedding?
#'
#' @seealso
#' In writing this package, the emphasis has been on making it possible to
#' implement multiple different methods and apply different optimization and
#' preprocessing techniques to them. This has two downsides: no effort has been
#' made to make this package fast or memory efficient; and the entire workflow
#' of an embedding algorithm is spread out over multiple functions. If you
#' just want to understand how, for example, t-SNE is implemented, it's better
#' to look at the code in the
#' \href{https://github.com/jdonaldson/rtsne}{'tsne' R package}.
#'
#' @section Embedding:
#' Currently, the following techniques are implemented:
#' \itemize{
#'  \item{Metric MDS using the STRESS and SSTRESS functions}
#'  \item{Sammon Mapping}
#'  \item{t-Distributed Stochastic Neighbor Embedding (t-SNE)}
#'  \item{Asymmetric SNE (ASNE, what was originally just called SNE)}
#'  \item{Symmetic SNE (SSNE)}
#' }
#'
#' The first two methods are traditional non-linear mapping techniques that
#' attempt to reproduce the input distances in the output space. These are older
#' techniques that only work well when the input dimension is not too large.
#' They're useful as benchmarks for testing more advanced methods, though.
#'
#' The SNE family of algorithms are focussed on preserving similarity: they
#' don't attempt to preserve the input distances but input probabilities, which
#' are based on the distances by some non-linear weighting function.
#'
#' To carry out a distance-based embedding, use the \code{\link{embed_dist}}
#' function. The entry point for probability-based embeddings is the
#' function \code{\link{embed_prob}}. They have a similar signature, but
#' provide useful defaults.
#'
#' Apart from the description of the various parts of the embedding below, you
#' can get more info and examples by looking up the help page for that function
#' and then further help by looking up the functions mentioned in the
#' \code{See Also} section.
#'
#' To carry out an embedding, you need to provide quite a bit of information
#' to parameterise it correctly. Rather then requiring it all be passed as
#' multiple arguments to the function, logically related pieces are separated
#' into separate components. These could be as simple as a single function,
#' or be a list of related functions. Each component has its own factory
#' function, so you don't need to worry about the underlying structure.
#'
#' @section Preprocessing:
#' Entirely optional, but provides some way to preprocess the input data, e.g.
#' various scaling methods, filtering of zero-variance columns, whitening.
#' See \code{\link{make_preprocess}} for more details.
#'
#' @section Input Initialization:
#' Generates the input matrices used in the embedding. For SNE and related
#' methods, a probability matrix is needed. There's really only one way of
#' doing this, which involves specifying a target perplexity for row-wise
#' probabilities. See \code{\link{make_init_inp}} and
#' \code{\link{make_init_inp_dist}} for more details.
#'
#' @section Output Initialization:
#' Generates the initial embedded coordinates. A random initialization from
#' a gaussian distribution with a very small standard deviation (e.g. 1e-4) is
#' very popular. But that makes comparing different results difficult. Using
#' PCA to generate a two-component scores matrix gives repeatable results with
#' a computational cost that is entirely reasonable. See
#' \code{\link{make_init_out}} for more details.
#'
#' @section Embedding Method:
#' I take the view that the essentials of any embedding method (or at least the
#' ones this package can deal with) is the definition of the gradient. This
#' in turn depends on the cost function used. For probability-based methods
#' like SNE, the exact form of the gradient is also dependent on the weighting
#' function that converts distances to probabilities, and how the probability
#' is defined (e.g. joint versus conditional probabilities). The embedding
#' method defines the functions needed to do this conversion. If the embedding
#' can have a stiffness matrix defined for it it should be possible to
#' implement it as a method in sneer, although it requires a little knowledge
#' of the internals of the package. See \code{\link{embedding_methods}} for
#' the list of available embedding methods.
#'
#' @section Optimization:
#' The optimization methods in sneer are composed of four sub-components:
#' \itemize{
#'  \item Gradient position: 'Classic' or Nesterov.
#'  \item Direction: currently only steepest descent is supported.
#'  \item Step Size: two adaptive step size methods are available:
#'  the Jacobs method used in the t-SNE paper (also known as the delta-bar-delta
#'  method), and the bold driver method.
#'  \item Update: various momentum schedules.
#' }
#' This might seem a bit overengineered, but it allows for several optimization
#' methods: including the Jacobs method used in the t-SNE paper, and Nesterov
#' Accelerated Gradient, which seems to work very well for embeddings. You could
#' also implement conjugate gradient within this framework, including backwards
#' stepping Wolfe line search for step size selection, but you would also need
#' a good guess for the trace of the Hessian, and NAG works well enough with
#' the bold driver method for step size selection without all that.
#' See \code{\link{optimization_methods}} for more details.
#'
#' @section Tricks:
#' Various embedding methods use different heuristics to speed up convergence.
#' Only the "early exaggeration" method described in the t-SNE paper is
#' currently implemented. See \code{\link{make_tricks}} for more details.
#'
#' @section Reporter:
#' Optional functions that will run on a regular schedule during the
#' optimization. Used for keeping track of the progress of the optimization,
#' and stopping early if necessary. Also, you can plot the current state of the
#' embedding. See \code{\link{make_reporter}} for more details.
#'
#' @section Synthetic Dataset:
#' There's a synthetic dataset in this package, called \code{s1k}. It consists
#' of a 1000 points representing a fuzzy 9D simplex. It's intended to
#' demonstrate the "crowding effect" and require the sort of
#' probability-based embedding methods provided in this package (PCA does a
#' horrible job of separated the 10 clusters in the data). See \code{\link{s1k}}
#' for more details.
#'
#' @examples
#' \dontrun{
#' # Do t-SNE on the iris dataset with the same options as the t-SNE paper
#' # and initialize from random. Use generic plot function, displaying the first
#' # two characters of the "Species" factor for the points. Explicitly choose
#' # t-SNE as the method.
#' tsne_iris <- embed_prob(iris[, 1:4],
#'                        method = tsne(),
#'                        opt = tsne_opt(),
#'                        init_inp = make_init_inp(perplexity = 25),
#'                        init_out = make_init_out(stdev = 1e-4),
#'                        tricks = tsne_tricks(),
#'                        reporter = make_reporter(plot_fn = make_plot(
#'                          iris, "Species", make_label(2))))
#'
#' # Do ASNE on the synthetic the s1k dataset (10 overlapping 9D Gaussian blobs),
#' # Autoscale the input, use PCA to initialize the embedding, and use
#' # Nesterov Accelerated Gradient to do the optimization.
#' asne_s1k <- embed_prob(s1k[, 1:9],
#'                       method = asne(),
#'                       preprocess = make_preprocess(auto_scale = TRUE),
#'                       init_inp = make_init_inp(perplexity = 50),
#'                       init_out = make_init_out(from_PCA = TRUE),
#'                       opt = bold_nag_opt(),
#'                       reporter = make_reporter(
#'                        plot_fn = make_plot(s1k, "Label")))
#'
#' # Same as the previous example but creating the NAG optimizer explicitly with
#' # generic make_opt method
#' asne_s1k <- embed_prob(s1k[, 1:9],
#'                       method = asne(),
#'                       preprocess = make_preprocess(auto_scale = TRUE),
#'                       init_inp = make_init_inp(perplexity = 50),
#'                       init_out = make_init_out(from_PCA = TRUE),
#'                       opt = make_opt(gradient = nesterov_gradient(),
#'                                      step_size = bold_driver(),
#'                                      update = nesterov_nsc_momentum()),
#'                       reporter = make_reporter(
#'                        plot_fn = make_plot(s1k, "Label")))
#'
#' # Do metric MDS on the iris data set
#' # In addition to the STRESS loss function also report the Kruskal Stress
#' # (often used in MDS applications) and the mean relative error, which can
#' # be multiplied by 100 and interpreted as a percentage error. Also, use
#' # the make_iris_plot function, which wrap the make_plot function specifically
#' # for the iris dataset, which is quite handy for testing.
#' mds_iris <- embed_dist(iris[, 1:4],
#'                        method = mmds(),
#'                        opt = bold_nag_opt(),
#'                        reporter = make_reporter(
#'                          extra_costs = c("kruskal_stress",
#'                                          "mean_relative_error")),
#'                                          plot_fn = make_iris_plot())
#'
#' # Sammon map the iris data set, which turns out to be a surprisingly tough
#' # assignment. Increase epsilon substantially to 1e-4 to avoid the gradient
#' # being overwhelmed by zero distances in the input space. Additionally, we
#' # report two other normalized stress functions used in MDS. The Sammon
#' # mapping cost function is already normalized, so tell the make_reporter
#' # function not to report an automatically normalized version in the logging
#' # output.
#' sammon_iris <- embed_dist(iris[, 1:4],
#'                           method = sammon_map(eps = 1e-4),
#'                           opt = bold_nag_opt(),
#'                           init_out = make_init_out(stdev = 1e-4),
#'                           reporter = make_reporter(normalize_cost = FALSE,
#'                                        extra_costs = c("normalized_stress",
#'                                                        "kruskal_stress"),
#'                                        plot_fn = make_iris_plot()))
#' }
#' @references
#'
#' MDS
#' Jan de Leeuw, Patrick Mair (2009).
#' Multidimensional Scaling Using Majorization: SMACOF in R.
#' Journal of Statistical Software, 31(3), 1-30.
#' URL http://www.jstatsoft.org/v31/i03/.
#'
#' Ingwer Borg, Patrick J. F. Groenen
#' Modern Multidimensional Scaling: Theory and Applications
#' Springer Science & Business Media, Aug 4, 2005
#'
#' Sammon Mapping
#' Sammon, J. W. (1969)
#' A non-linear mapping for data structure analysis.
#' IEEE Trans. Comput., C-18 401-409.
#'
#' t-SNE, SNE and ASNE:
#' Laurens van der Maarten, Geoffrey Hinton.
#' Visualizing Data using t-SNE.
#' Journal of Machine Learning Research, 2008, 9, 2579-2605.
#'
#' Nesterov Accelerated Gradient:
#' Sutskever, I., Martens, J., Dahl, G. and Hinton, G. E.
#' On the importance of momentum and initialization in deep learning.
#' 30th International Conference on Machine Learning, Atlanta, USA, 2013.
#' JMLR: W&CP volume 28.
#'
#' @docType package
#' @name sneer
NULL
