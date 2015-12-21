#' sneer: Stochastic Neighbor Embedding Experiments in R.
#'
#' A package for exploring similarity-preserving embedding and related forms
#' of dimensionality reduction. Its main goal is to implement multiple
#' embedding methods within a single framework so comparison between them is
#' easier, without worrying about the effect of differences in preprocessing,
#' optimization and heuristics. Additionally it provides a way to explore the
#' importance of those effects on the performance of individual methods. For
#' example, how much of an effect does the early exaggeration trick in t-SNE
#' have on the embedding?
#'
#' First, some examples of how to use the package for embedding. A longer
#' discussion of the package structure follows.
#'
#' @examples
#' # Do t-SNE on the iris dataset with the same options as the t-SNE paper
#' # and initialize from random. Use generic plot function, displaying the first
#' # two characters of the "Species" factor for the points. Explicitly choose
#' # t-SNE as the method.
#' tsne_iris <- embed_sim(iris[, 1:4],
#'                        method = tsne(),
#'                        opt = tsne_opt(),
#'                        init_inp = make_init_inp(perplexity = 25),
#'                        init_out = make_init_out(stdev = 1e-4)
#'                        tricks = tsne_tricks(),
#'                        reporter = make_reporter(plot_fn = make_plot(
#'                          iris, "Species", make_label(2))))
#'
#' # Do ASNE on the synthetic the s1k dataset (10 overlapping 9D Gaussian blobs),
#' # Autoscale the input, use PCA to initialize the embedding, and use
#' # Nesterov Accelerated Gradient to do the optimization.
#' asne_s1k <- embed_sim(s1k[, 1:9],
#'                       method = asne(),
#'                       preprocess = make_preprocess(auto_scale = TRUE),
#'                       init_inp = make_init_inp(perplexity = 50),
#'                       init_out = make_init_out(from_PCA = TRUE),
#'                       opt = bold_nag_opt(),
#'                       reporter = make_reporter(
#'                        plot_fn = make_plot(s1k, "Label")))
#'
#' @seealso
#' In writing this package, the emphasis has been on making it possible to
#' implement multiple different methods and apply different optimization and
#' preprocessing techniques to them. This has two downsides: no effort has been
#' made to make this package fast or memory efficient; and the entire workflow
#' of an embedding algorithm is spread out over multiple functions. If you
#' just want to understand how, for example, t-SNE is implemented, it's better
#' to look at the code in 'tsne' package
#' \link{https://github.com/jdonaldson/rtsne}.
#'
#' @section Embedding:
#' Currently, only the similarity-based embedding techniques related to
#' Stochastic Neighbor Embedding (SNE) are supported: the popular t-Distributed
#' Stochastic Neighbor Embedding (t-SNE) technique, and two older (and
#' generally less useful) methods asymmetric SNE (ASNE, what was originally just
#' called SNE) and symmetric SNE (SSNE).
#'
#' A characteristic quality of these methods is that they don't attempt to
#' preserve the input distances but input probabilities, which are based on
#' the distances by some non-linear weighting function.
#'
#' The entry point for these similarity
#' preserving embeddings is the function \code{embed_sim}. Apart from the
#' description of the various parts of the embedding below, you can get more info
#' and examples by looking up the help by running
#'
#' \code{?embed_sim}
#'
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
#' See \code{?make_preprocess} for more details.
#'
#' @section Input Initialization:
#' Generates the input matrices used in the embedding. For SNE and related
#' methods, a probability matrix is needed. There's really only one way of
#' doing this, which involves specifying a target perplexity for row-wise
#' probabilities. See \code{?make_init_inp} for more details.
#'
#' @section Output Initialization:
#' Generates the initial embedded coordinates. A random initialization from
#' a gaussian distribution with a very small standard deviation (e.g. 1e-4) is
#' very popular. But that makes comparing different results difficult. Using
#' PCA to generate a two-component scores matrix gives repeatable results with
#' a computational cost that is entirely reasonable. See \code{?make_init_out}
#' for more details.
#'
#' @section Embedding Method:
#' I take the view that the essentials of any embedding method (or at least the
#' ones this package can deal with) is the definition of the gradient. This
#' in turn depends on the cost function used. For similarity preserving methods
#' like SNE, the exact form of the gradient is also dependent on the weighting
#' function that converts distances to probabilities, and how the probability
#' is defined (e.g. joint versus conditional probabilities). The embedding
#' method defines the functions needed to do this conversion. For now, the
#' only available methods are ASNE, SSNE and TSNE. More complicated embedding
#' methods such as JSE and NeRV can be implemented in this framework, and also
#' distance-preserving embeddings like MDS and Sammon mapping, which don't
#' involve probabilities at all. These are all coming soon. If the embedding can
#' have a stiffness matrix defined for it it should be possible to implement it
#' as a method in sneer, although it requires a little knowledge of the
#' internals.
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
#' See \code{?make_opt} for more details.
#'
#' @section Tricks:
#' Various embedding methods use different heuristics to speed up convergence.
#' Only the "early exaggeration" method described in the t-SNE paper is
#' currently implemented. See \code{?make_tricks} for more details.
#'
#' @section Reporter:
#' Optional functions that will run on a regular schedule during the
#' optimization. Used for keeping track of the progress of the optimization,
#' and stopping early if necessary. Also, you can plot the current state of the
#' embedding. See \code{?make_reporter} for more details.
#'
#' @docType package
#' @name sneer
NULL
