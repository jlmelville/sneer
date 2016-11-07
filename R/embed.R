# Functions to do the embedding. The interface to sneer. Also, core
# functions to do with gradient calculation, optimization, updating and so on
# that don't need to be overridden or changed.

# Embedding Methods
#
# Links to all available embedding methods.
#
# @examples
# \dontrun{
# embed_prob(method = tsne())
# embed_dist(method = mmds())
# }
# @keywords internal
# @name embedding_methods
# @family sneer embedding methods
NULL

#' Probability-based Embedding
#'
#' Carries out an embedding of a high-dimensional dataset into a two dimensional
#' scatter plot, based on distance-based methods (e.g. Sammon maps) and
#' probability-based methods (e.g. t-distributed Stochastic Neighbor Embedding).
#'
#' The embedding methods available are:
#' \itemize{
#'  \item \code{"pca"} The first two principal components.
#'  \item \code{"mmds"} Metric multidimensional scaling.
#'  \item \code{"sammon"} Sammon map.
#'  \item \code{"tsne"} t-Distributed Stochastic Neighbor Embedding of van der
#'   Maaten and Hinton (2008).
#'  \item \code{"asne"} Asymmetric Stochastic Neighbor Embedding of Hinton and
#'   Roweis (2002).
#'  \item \code{"ssne"} Symmetric Stochastic Neighbor Embedding of Cook et al
#'   (2007).
#'  \item \code{"wssne"} Weighted Symmetric Stochastic Neighbor Embedding of
#'   Yang et al (2014). Note that despite its name this version is a
#'   modification of t-SNE, not SSNE.
#'  \item \code{"hssne"} Heavy-tailed Symmetric Stochastic Neighbor Embedding of
#'   Yang et al (2009).
#'  \item \code{"nerv"} Neighbor Retrieval Visualizer of Venna et al (2010).
#'  \item \code{"jse"} Jensen-Shannon Embedding of Lee at al (2013).
#' }
#'
#' The following scaling options can be applied via the \code{scale_type}
#' parameter:
#' \itemize{
#'  \item \code{"m"} Range scale the entire data so that the maximum value is
#'   1 and the minimum 0.
#'  \item \code{"r"} Range scale each column that the maximum value in each
#'   column is 1 and the minimum 0.
#'  \item \code{"a"} Scale each column so that its mean is 0 and variance is
#'   1.
#' }
#' Default is to do no scaling. Zero variance columns will be removed even if no
#' preprocessing is carried out.
#'
#' The \code{perplexity} parameter is used in combination with the
#' \code{perp_scale} parameter, which can take the following values:
#' \itemize{
#'  \item \code{"single"} \code{perplexity} should be a single value, which
#'    will be used over the entire course of the embedding.
#'  \item \code{"step"} \code{perplexity} should be a vector of
#'     perplexity values. Each perplexity will be used in turn over the course
#'     of the embedding, in sequential order. By starting with a large
#'     perplexity, and ending with the desired perplexity, it has been
#'     suggested by some researchers that local minima can be avoided.
#' \item \code{"multi"} The multiscaling method of Lee et al (2015).
#'     \code{perplexity} should be a vector of perplexity values. Each
#'     perplexity will be used in turn over the course of the embedding, in
#'     sequential order. Unlike with the \code{"step"} method, probability
#'     matrices from earlier perplexities are retained and combined by
#'     averaging.
#' }
#'
#' For \code{perp_scale} values that aren't \code{"single"}, if a non-vector
#' argument is suppied to the \code{perplexity} argument, it will be ignored,
#' and a suitable vector of perplexity values will be used instead. For
#' \code{"multi"} these will range from the the number of observations in the
#' dataset divided by four down to 2, in descending powers of 2. For
#' \code{"step"}, 5 equally spaced values ranging from the number of
#' observations divided by 2 down to 32 (or the number of observations divided
#' by 4, if the dataset is smaller than 65 observations.)
#'
#' The \code{prec_scale} parameter determines if the input weighting kernel
#' precision parameters should be used to modify the output kernel parameter
#' after the input probability calculation for a given perplexity value
#' completes values are:
#' \itemize{
#'  \item \code{"n"} Do nothing. Most embedding methods follow this strategy,
#'    leaving the output similarity kernels to all have unit precision.
#'  \item \code{"t"} Transfer the input similarity kernel parameters to the
#'    output similarity kernel. This method was suggesed by Venna et al (2010).
#'  \item \code{"s"} Scale the output kernel precisions based on the target
#'    \code{perplexity} and the intrinsic dimensionality of the input data. This
#'    method is part of the multiscaling technique proposed by Lee et al (2015).
#' }
#'
#' The \code{prec_scale} parameter will be ignored if the \code{method} used
#' does not use an output similarity kernel with a free parameter, e.g.
#' \code{tsne} or \code{wtsne}. Also, because the input and output similarity
#' kernels must be of the same type, \code{prec_scale} is incompatible with
#' setting \code{perp_kernel_fun} to "step".
#'
#' For initializing the output coordinates, the options for the
#' \code{init} parameter are:
#' \itemize{
#'  \item \code{"p"} Initialize using the first two scores of the PCA (using
#'  classical MDS if \code{df} is a distance matrix). Data will be centered,
#'  but not scaled unless the \code{scale_type} parameter is used.
#'  \item \code{"r"} Initialize each coordinate value from a normal random
#'  distribution with a standard deviation of 1e-4, as suggested by van der
#'  Maaten and Hinton (2008).
#'  \item \code{"u"} Initialize each coordinate value from a uniform random
#'  distribution between 0 and 1 as suggested by Venna et al (2010).
#'  \item \code{"m"} Initialize the coordinates from a user-supplied matrix.
#'   Supply the coordinates as the \code{init_config} parameter.
#'}
#'
#' For configuring the optimization method, the options for the \code{opt}
#' parameter are:
#' \itemize{
#'  \item \code{"TSNE"} The optimization method used in the original t-SNE
#'    paper: the Jacobs method for step size selection and a step function
#'    for the momentum: switching from 0.4 to 0.8 after 250 steps. You may need
#'    to modify the \code{"epsilon"} parameter to get good results, depending
#'    on how you have scaled and preprocessed your data, and the embedding
#'    method used.
#'  \item \code{"L-BFGS"} The low-memory BFGS method.
#'  \item \code{"NAG-BOLD"} Nesterov Accelerated Gradient with bold driver step
#'    size selection.
#'  \item \code{"NAG-MT"} Nesterov Accelerated Gradient with More-Thuente step
#'    size selection.
#'  \item \code{"NAG-R"} Nesterov Accelerated Gradient with Rasmussen step size
#'   selection.
#'  \item \code{"CG-MT"} Conjugate Gradient with More-Thuente step size
#'  selection.
#'  \item \code{"CG-R"} Conjugate Gradient with Rasmussen step size selection.
#'  \item \code{"SPEC-BOLD"} Spectral Direction method of Vladymyrov and
#'  Carreira-Perpinan (2012) with bold driver step size selection.
#'  \item \code{"SPEC-MT"} Spectral Direction with More-Thuente step size
#'  selection.
#'  \item \code{"SPEC-R"} Spectral Direction with Rasmussen step size selection.
#' }
#'
#' There are some caveats to using these optimization routines:
#'
#' \itemize{
#'  \item To use the conjugate gradient method or the
#'   Rasmussen or More-Thuente step size methods, you must install and load the
#'   \code{rconjgrad} package from \url{https://github.com/jlmelville/rconjgrad}.
#'  \item The external optimization routines (\code{L-BFGS} and \code{CG-}
#'   methods) run in batches of \code{report_every}. For example, if you want to
#'   report every 50 iterations, the optimization routine will be run for 50
#'   iterations, the cost is logged to screen, and then a new batch of 50
#'   iterations are run, losing any memory of the previous direction or other
#'   state, effectively "resetting" the search. Therefore, do not set
#'   \code{report_every} too low in this case, or the optimization will
#'   approach the behavior of steepest descent.
#'  \item Use of the external optimization routines is incompatible with
#'   \code{perp_scale} settings that need to update the input probabilities at
#'   certain iterations (e.g. multiscaling), because that iteration number might
#'   have been "lost" inside the optimization routine.
#'  \item The spectral direction method requires a probability-based embedding
#'   method and that the input probability matrix be symmetric. Some
#'   probability-based methods are not compatible (e.g. NeRV and JSE; t-SNE
#'   works with it, however). Also, while it works with the dense matrices used
#'   by sneer, because this method uses a Cholesky decomposition of the input
#'   probability matrix which has a complexity of O(N^3), it is intended
#'   to be used with sparse matrices. Its inclusion here is suitable for use
#'   with smaller datasets.
#' }
#'
#' The default is to use NAG with the bold driver step size, and adaptive
#' restarting. This is not quite as fast as using the Jacobs method for some
#' datasets, but is more robust across different embedding methods and scaling,
#' and doesn't require fiddling with the learning rate.
#'
#' For the \code{quality_measures} argument, a vector with one or more of the
#' following options can be supplied:
#' \itemize{
#'  \item \code{"r"} Calculate the area under the ROC curve, averaged over
#'   each observation, using the output distance matrix to rank each
#'   observation. Observations are partitioned into the positive and negative
#'   class depending upon the value of the label determined by the
#'   \code{label_name} argument. Only calculated if the \code{label_name}
#'   parameter is supplied.
#'  \item \code{"p"} Calculate the area under the Precision-Recall curve.
#'   Only calculated if the \code{label_name} parameter is supplied.
#'  \item \code{"n"} Calculate the area under the RNX curve, using the
#'   method of Lee et al (2015).
#' }
#'
#' Progress of the embedding is logged to the standard output every 50
#' iterations. The raw cost of the embedding will be provided along with some
#' tolerances of either how the embedding or the cost has changed.
#'
#' Because the different costs are not always scaled in a way that makes it
#' obvious how well the embedding has performed, a normalized cost is also
#' shown, where 0 is the minimum possible cost (coinciding with the
#' probabilities or distances in the input and output space being matched), and
#' a normalized cost of 1 is what you would get if you just set all the
#' distances and probabilities to be equal to each other (i.e. ignoring any
#' information from the input space).
#'
#' Also, the embedding will be plotted. Plotting can be done
#' with either the standard \code{\link[graphics]{plot}} function (the default
#' or by explicitly providing \code{plot_type = "p"}) or with the \code{ggplot2}
#' library (which you need to install and load yourself), by using
#' \code{plot_type = "g"}. The goal has been to provide enough customization to
#' give intelligible results for most datasets. The following are things to
#' consider:
#'
#' \itemize{
#'  \item The plot symbols are normally filled circles. However, if you
#'  set the \code{plot_text} argument to \code{TRUE}, the \code{labels}
#'  argument can be used to provide a factor vector that provides a meaningful
#'  label for each data point. In this case, the text of each factor level will
#'  be used as a level. This creates a mess with all but the shortest labels
#'  and smallest datasets. There's also a \code{label_fn} parameter that lets
#'  you provide a function to convert the vector of labels to a different
#'  (preferably shorter) form, but you may want to just do it yourself ahead
#'  of time and add it to the data frame.
#'  \item Points are colored using two strategies. The most straightforward way
#'  is to provide a vector of rgb color strings as an argument to \code{colors}.
#'  Each element of \code{colors} will be used to color the equivalent point
#'  in the data frame. Note, however, this is currently ignored when plotting
#'  with ggplot2.
#'  \item The second way to color the embedding plot uses the \code{labels}
#'  parameter mentioned above. Each level of the factor used for \code{labels}
#'  will be mapped to a color and that used to color each point. The mapping
#'  is handled by the \code{color_scheme} parameter. It can be either a color
#'  ramp function like \code{\link[grDevices]{rainbow}} or the name of a color
#'  scheme in the \code{RColorBrewer} package (e.g. \code{"Set3"}). The latter
#'  requires the \code{RColorBrewer} package to have been installed and loaded.
#'  Unlike with using \code{colors}, providing a \code{labels} argument works
#'  with ggplot2 plots. In fact, you may find it preferable to use ggplot2,
#'  because if the \code{legend} argument is \code{TRUE} (the default), you
#'  will get a legend with the plot. Unfortunately, getting a legend with an
#'  arbitary number of elements to fit on an image created with the
#'  \code{graphics::plot} function and for it not to obscure the points proved
#'  beyond my capabilities. Even with ggplot2, a dataset with a large number
#'  of categories can generate a large and unwieldy legend.
#' }
#'
#' Additionally, instead of providing the vectors directly, there are
#' \code{color_name} and \code{label_name} arguments that take a string
#' containing the name of a column in the data frame, e.g. you can use
#' \code{labels = iris$Species} or \code{label_name = "Species"} and get the
#' same result.
#'
#' If you don't care that much about the colors, provide none of these options
#' and sneer will try and work out a suitable column to use. If it finds at
#' least one color column in the data frame (i.e. a string column where every
#' element can be parsed as a color), it will use the last column found as
#' if you had provided it as the \code{colors} argument.
#'
#' Otherwise, it will repeat the process but looking for a vector of factors.
#' If it finds one, it will map it to colors via the \code{color_scheme}, just
#' as if you had provided the \code{labels} argument. The default color scheme
#' is to use the \code{rainbow} function so you should normally get a colorful,
#' albeit potentially garish, result.
#'
#' For the \code{ret} argument, a vector with one or more of the
#' following options can be supplied:
#' \itemize{
#'  \item \code{"x"} The input coordinates after scaling and column filtering.
#'  \item \code{"dx"} The input distance matrix. Calculated if not present.
#'  \item \code{"dy"} The output distance matrix. Calculated if not present.
#'  \item \code{"p"} The input probability matrix.
#'  \item \code{"q"} The output probability matrix.
#'  \item \code{"prec"} The input kernel precisions (inverse of the squared
#'  bandwidth).
#'  \item \code{"dim"} The intrinsic dimensionality for each observation,
#'  calculated according to the method of Lee et al (2015). These are
#'  meaningless if not using the default exponential \code{perp_kernel_fun}.
#'  \item \code{"deg"} Degree centrality of the input probability. Calculated
#'  if not present.
#' }
#'
#' The \code{color_scheme} parameter is used to set the color scheme for the
#' embedding plot that is displayed during the optimization. It can be one of
#' either a color ramp function (e.g. \code{grDevices::rainbow}), accepting an
#' integer n as an argument and returning n colors, or the name of a ColorBrewer
#' color scheme (e.g. "Spectral"). Using a ColorBrewer scheme requires the
#' \code{RColorBrewer} package be installed.
#'
#' For some applicable color ramp functions, see the \code{Palettes} help page
#' in the \code{grDevices} package (e.g. by running the \code{?rainbow} command).
#'
#' @param df Data frame or distance matrix (as dist object) to embed.
#' @param indexes Indexes of the columns of the numerical variables to use in
#'  the embedding. The default of \code{NULL} will use all the numeric
#'  variables.
#' @param ndim Number of output dimensions (normally 2).
#' @param method Embedding method. See 'Details'.
#' @param alpha Heavy tailedness parameter. Used only if the method is
#'  \code{"hssne"}.
#' @param lambda NeRV parameter. Used only if the method is \code{"nerv"}.
#' @param kappa JSE parameter. Used only if the method is \code{"jse"}.
#' @param scale_type Type of scaling to carry out on the input data. See
#'  'Details'.
#' @param perplexity Target perplexity or vector of trial perplexities (if
#'  \code{perp_scale} is set). Applies to probability-based embedding methods
#'  only (i.e. anything that isn't PCA, MDS or Sammon mapping).
#' @param perp_scale Type of perplexity scaling to apply. See 'Details'. Ignored
#'  by non-probability based methods.
#' @param perp_scale_iter Number of iterations to scale perplexity values over.
#'  Must be smaller than the \code{max_iter} parameter. Default is to use
#'  \code{max_iter / 5}. Ignored by non-probability based methods or if
#'  \code{perp_scale} is not set.
#' @param perp_kernel_fun The input data weight function. Either \code{"exp"}
#'  to use exponential function (the default) or \code{"step"} to use a step
#'  function. The latter emulates a k-nearest neighbor graph, but does not
#'  provide any of the efficiency advantages of a sparse matrix.
#' @param prec_scale Whether to scale the output kernel precision based on
#'  perplexity results. See 'Details'. Ignored by non-probability based methods.
#'  Can't be used if \code{perp_kernel_fun} is set to \code{"step"}.
#' @param init Type of initialization of the output coordinates. See 'Details'.
#' @param init_config Coordinates to use for initial configuration. Used only
#'  if \code{init} is \code{"m"}.
#' @param opt Type of optimizer. See 'Details'.
#' @param epsilon Learning rate when \code{opt} is set to \code{"TSNE"} and
#'  the initial step size for the bold driver and back tracking step search
#'  methods.
#' @param max_iter Maximum number of iterations to carry out optimization of
#'  the embedding. Ignored if the \code{method} is \code{"pca"}.
#' @param report_every Frequency (in terms of iteration number) with which to
#'  update plot and report the cost function.
#' @param tol Tolerance for comparing cost change (calculated according to the
#'  interval determined by \code{report_every}). If the change falls below this
#'  value, the optimization stops early.
#' @param exaggerate If non-\code{NULL}, scales input probabilities by this
#'  value from iteration 0 until \code{exaggerate_off_iter}. Normally a value
#'  of \code{4} is used. Has no effect with PCA, Sammon mapping or metric MDS.
#'  Works best when using random initialization with (\code{init = "r"} or
#'  \code{"u"}. You probably don't want to use it if you are providing your own
#'  initial configuration (\code{init = "m"}).
#' @param exaggerate_off_iter Iteration number to stop the "early exaggeration"
#'  scaling specified \code{exaggerate}. Has no effect if \code{exaggerate} is
#'  \code{NULL}.
#' @param plot_type String code indicating the type of plot of the embedding
#'  to display: \code{"p"} to use the usual \code{\link[graphics]{plot}}
#'  function; \code{"g"} to use the \code{ggplot2} package. You are responsible
#'  for installing and loading the ggplot2 package yourself.
#' @param colors Vector of colors to use to color each point in the embedding
#'  plot.
#' @param color_name Name of column of colors in \code{df} to be used to color
#'  the points directly. Ignored if \code{colors} is provided.
#' @param labels Factor vector associated with (but not necessarily in)
#'  \code{df}. Used to map from factor levels to colors in the embedding plot
#'  (if no \code{color} or \code{color_name} is provided), and as text labels
#'  in the plot if \code{plot_labels} is \code{TRUE}. Ignored if \code{colors}
#'  or \code{color_name} is provided.
#' @param label_name Name of a factor column in \code{df}, to be used like
#'  \code{labels}. Ignored if \code{labels} is provided.
#' @param label_chars Number of characters to use for the labels in the
#'  embedding plot. Applies only when \code{plot_type} is set to \code{"p"}.
#' @param point_size Size of the points (or text) in the embedding plot.
#' @param plot_labels If \code{TRUE} and either \code{labels} or
#'  \code{label_name} is provided, then the specified factor column will be used
#'  to provide a text label associated with each point in the plot. Only useful
#'  for small dataset with short labels. Ignored if \code{plot_type} is not
#'  set to \code{"p"}.
#' @param color_scheme Either a color ramp function, or the name of a Color
#'  Brewer palette name to use for mapping the factor specified by
#'  \code{labels} or \code{label_name}. Ignored if not using \code{labels}
#'  or \code{label_name}.
#' @param equal_axes If \code{TRUE}, the embedding plot will have the axes
#'  scaled so that both the X and Y axes have the same extents. Only applies if
#'  \code{plot_type} is set to \code{"p"}.
#' @param legend if \code{TRUE}, display the legend in the embedding plot.
#'  Applies when \code{plot_type} is \code{"g"} only.
#' @param legend_rows Number of rows to use for displaying the legend in
#'  an embedding plot. Applies when \code{plot_type} is \code{"g"} only.
#' @param quality_measures Vector of names of quality measures to apply to the
#'  finished embedding. See 'Details'. Values of the quality measures will
#'  be printed to screen after embedding and retained in the list that is
#'  returned from this function.
#' @param ret Vector of names of extra data to return from the embedding. See
#'  'Details',
#' @return List with the following elements:
#' \itemize{
#' \item \code{coords} Embedded coordinates.
#' \item \code{cost} Cost function value for the embedded coordinates. The
#'  type of the cost depends on the method, but the lower the better.
#' \item \code{norm_cost} \code{cost}, normalized so that a perfect embedding
#'  gives a value of 0 and one where all the distances were equal would have
#'  a value of 1.
#' \item \code{method} String giving the method used for the embedding.
#' }
#' Additional elements will be in the list if \code{ret} or
#' \code{quality_measures} are non-empty.
#' @references
#' Cook, J., Sutskever, I., Mnih, A., & Hinton, G. E. (2007).
#' Visualizing similarity data with a mixture of maps.
#' In \emph{International Conference on Artificial Intelligence and Statistics} (pp. 67-74).
#'
#' Hinton, G. E., & Roweis, S. T. (2002).
#' Stochastic neighbor embedding.
#' In \emph{Advances in neural information processing systems} (pp. 833-840).
#'
#' Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013).
#' Type 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in
#' dimensionality reduction based on similarity preservation.
#' \emph{Neurocomputing}, \emph{112}, 92-108.
#'
#' Lee, J. A., Peluffo-Ordo'nez, D. H., & Verleysen, M. (2015).
#' Multi-scale similarities in stochastic neighbour embedding: Reducing
#' dimensionality while preserving both local and global structure.
#' \emph{Neurocomputing}, \emph{169}, 246-261.
#'
#' Van der Maaten, L., & Hinton, G. (2008).
#' Visualizing data using t-SNE.
#' \emph{Journal of Machine Learning Research}, \emph{9}(2579-2605).
#'
#' Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010).
#' Information retrieval perspective to nonlinear dimensionality reduction for
#' data visualization.
#' \emph{Journal of Machine Learning Research}, \emph{11}, 451-490.
#'
#' Vladymyrov, M., & Carreira-Perpinan, M. A. (2012).
#' Partial-Hessian Strategies for Fast Learning of Nonlinear Embeddings.
#' In \emph{Proceedings of the 29th International Conference on Machine Learning (ICML-12)}
#' (pp. 345-352).
#'
#' Yang, Z., King, I., Xu, Z., & Oja, E. (2009).
#' Heavy-tailed symmetric stochastic neighbor embedding.
#' In \emph{Advances in neural information processing systems} (pp. 2169-2177).
#'
#' Yang, Z., Peltonen, J., & Kaski, S. (2014).
#' Optimization equivalence of divergences improves neighbor embedding.
#' In \emph{Proceedings of the 31st International Conference on Machine Learning (ICML-14)}
#' (pp. 460-468).
#' @examples
#' \dontrun{
#'   # PCA on iris dataset and plot result using Species label name
#'   res <- sneer(iris, indexes = 1:4, label_name = "Species", method = "pca")
#'   # Same as above, but with sensible defaults (use all numeric columns, plot
#'   # with first factor column found)
#'   res <- sneer(iris, method = "pca")
#'
#'   # Can use a distance matrix as input with external vector of labels
#'   res <- sneer(dist(iris[1:4]), method = "pca", labels = iris$Species)
#'
#'   # scale columns so each one has mean 0 and variance 1
#'   res <- sneer(iris, method = "pca", scale_type = "a")
#'   # full species name on plot is cluttered, so just use the first two
#'   # letters and half size
#'   res <- sneer(iris, method = "pca", scale_type = "a", label_chars = 2,
#'                point_size = 0.5)
#'
#'   library(ggplot2)
#'   library(RColorBrewer)
#'   # Use ggplot2 and RColorBrewer palettes for the plot
#'   res <- sneer(iris, method = "pca", scale_type = "a", plot_type = "g")
#'   # Use a different ColorBrewer palette, bigger points, and range scale each
#'   # column
#'   res <- sneer(iris, method = "pca", scale_type = "r", plot_type = "g",
#'                color_scheme = "Dark2", point_size = 2)
#'
#'   # metric MDS starting from the PCA
#'   res <- sneer(iris, method = "mmds", scale_type = "a", init = "p")
#'   # Sammon map starting from random distribution
#'   res <- sneer(iris, method = "sammon", scale_type = "a", init = "r")
#'
#'   # TSNE with a perplexity of 32, initialize from PCA
#'   res <- sneer(iris, method = "tsne", scale_type = "a", init = "p",
#'                perplexity = 32)
#'   # default settings are to use TSNE with perplexity 32 and initialization
#'   # from PCA so the following is the equivalent of the above
#'   res <- sneer(iris, scale_type = "a")
#'
#'   # Use the standard tSNE optimization method (Jacobs step size method) with
#'   # step momentum. Range scale the matrix and use an aggressive learning
#'   # rate (epsilon).
#'   res <- sneer(iris, scale_type = "m", perplexity = 25, opt = "tsne",
#'                epsilon = 500)
#'
#'   # Use the L-BFGS optimization method
#'   res <- sneer(iris, scale_type = "a", opt = "L-BFGS")
#'   # Use the Spectral Directions method with bold driver
#'   res <- sneer(iris, scale_type = "a", opt = "SPEC-BOLD")
#'
#'   # Load the rconjgrad library: make use of other line search algorithms and
#'   # conjugate gradient optimizer
#'   install.packages("devtools")
#'   devtools::install_github("jlmelville/rconjgrad")
#'   library("rconjgrad")
#'   # Use More-Thuente line search with NAG optimizer instead of bold driver
#'   res <- sneer(iris, scale_type = "a", opt = "NAG-MT")
#'   # Use Rasmussen line search
#'   res <- sneer(iris, scale_type = "a", opt = "NAG-R")
#'   # Use Conjugate Gradient with More-Thuente line search
#'   res <- sneer(iris, scale_type = "a", opt = "CG-MT")
#'
#'   # Use the Spectral Direction method with More-Thuente line search
#'   res <- sneer(iris, scale_type = "a", opt = "SPEC-MT")
#'
#'   # NeRV method, starting at a more global perplexity and slowly stepping
#'   # towards a value of 32 (might help avoid local optima)
#'   res <- sneer(iris, scale_type = "a", method = "nerv", perp_scale = "step")
#'
#'   # NeRV method has a lambda parameter - closer to 1 it gets, the more it
#'   # tries to avoid false positives (close points in the map that aren't close
#'   # in the input space):
#'   res <- sneer(iris, scale_type = "a", method = "nerv", perp_scale = "step",
#'                lambda = 1)
#'
#'   # Original NeRV paper transferred input exponential similarity kernel
#'   # precisions to the output kernel, and initialized from a uniform random
#'   # distribution
#'   res <- sneer(iris, scale_type = "a", method = "nerv", perp_scale = "step",
#'                lambda = 1, prec_scale = "t", init = "u")
#'
#'   # Like NeRV, the JSE method also has a controllable parameter that goes
#'   # between 0 and 1, called kappa. It gives similar results to NeRV at 0 and
#'   # 1 but unfortunately the opposite way round! The following gives similar
#'   # results to the NeRV embedding above:
#'   res <- sneer(iris, scale_type = "a", method = "jse", perp_scale = "step",
#'                kappa = 0)
#'
#'   # Rather than step perplexities, use multiscaling to combine and average
#'   # probabilities across multiple perplexities. Output kernel precisions
#'   # can be scaled based on the perplexity value (compare to NeRV example
#'   # which transferred the precision directly from the input kernel)
#'   res <- sneer(iris, scale_type = "a", method = "jse", perp_scale = "multi",
#'                prec_scale = "s")
#'
#'   # HSSNE has a controllable parameter, alpha, that lets you control how
#'   # much extra space to give points compared to the input distances.
#'   # Setting it to 1 is equivalent to TSNE, so 1.1 is a bit of an extra push:
#'   res <- sneer(iris, scale_type = "a", method = "hssne", alpha = 1.1)
#'
#'   # wTSNE treats the input probability like a graph where the probabilities
#'   # are weighted edges and adds extra repulsion to nodes with higher degrees
#'   res <- sneer(iris, scale_type = "a", method = "wtsne")
#'
#'   # can use a step-function input kernel to make input probability more like
#'   # a k-nearest neighbor graph (but note that we don't take advantage of the
#'   # sparsity for performance purposes, sadly)
#'   res <- sneer(iris, scale_type = "a", method = "wtsne",
#'                perp_kernel_fun = "step")
#'
#'   # Some quality measures are available to quantify embeddings
#'   # The area under the RNX curve measures whether neighbors in the input
#'   # are still neighors in the output space
#'   res <- sneer(iris, scale_type = "a", method = "wtsne",
#'                quality_measures =  c("n"))
#'
#'   # Create a 5D gaussian with its own column specifying colors to use
#'   # for each point (in this case, random)
#'   g5d <- data.frame(matrix(rnorm(100 * 5), ncol = 5),
#'                     color = rgb(runif(100), runif(100), runif(100)),
#'                     stringsAsFactors = FALSE)
#'   # Specify the name of the color column and the plot will use it rather than
#'   # trying to map factor levels to colors
#'   res <- sneer(g5d, method = "pca", color_name = "color")
#'
#'   # If your dataset labels divide the data into natural classes, can
#'   # calculate average area under the ROC and/or precision-recall curve too,
#'   # but you need to have installed the PRROC package.
#'   # All these techniques can be slow (scale with the square of the number of
#'   # observations).
#'   library(PRROC)
#'   res <- sneer(iris, scale_type = "a", method = "wtsne",
#'                quality_measures =  c("n", "r", "p"))
#'
#'   # export the distance matrices and do whatever quality measures we
#'   # want at our leisure
#'   res <- sneer(iris, scale_type = "a", method = "wtsne", ret = c("dx", "dy"))
#'
#'   # Calculate the Area Under the Precision Recall Curve for the embedding
#'   pr <- pr_auc_embed(res$dy, iris$Species)
#'
#'   # Similarly, for the ROC curve:
#'   roc <- roc_auc_embed(res$dy, iris$Species)
#'
#'   # export degree centrality, input weight function precision parameters,
#'   # and intrinsic dimensionality
#'   res <- sneer(iris, scale_type = "a", method = "wtsne",
#'                ret = c("deg", "prec", "dim"))
#'
#'   # Plot the embedding as points colored by category, using the rainbow
#'   # color ramp function:
#'   embed_plot(res$coords, iris$Species, color_scheme = rainbow)
#'
#'   # Load the RColorBrewer Library
#'   library(RColorBrewer)
#'
#'   # Use a ColorBrewer Qualitative color scheme name (pass a string, not
#'   # a function!)
#'   embed_plot(res$coords, iris$Species, color_scheme = "Dark2")
#'
#'   # Visualize embedding colored by various values:
#'   # Degree centrality
#'   embed_plot(res$coords, x = res$deg)
#'   # Intrinsic Dimensionality using the PRGn palette
#'   embed_plot(res$coords, x = res$dim, color_scheme = "PRGn")
#'   # Input weight function precision parameter with the Spectral palette
#'   embed_plot(res$coords, x = res$prec, color_scheme = "Spectral")
#'
#'   # calculate the 32-nearest neighbor preservation for each observation
#'   # 0 means no neighbors preserved, 1 means all of them
#'   pres32 <- nbr_pres(res$dx, res$dy, 32)
#'   embed_plot(res$coords, x = pres32, cex = 1.5)
#' }
#' @export
sneer <- function(df,
                  indexes = NULL,
                  ndim = 2,
                  method = "tsne",
                  alpha = 0.5,
                  lambda = 0.5,
                  kappa = 0.5,
                  scale_type = "",
                  perplexity = 32, perp_scale = "single",
                  perp_scale_iter = NULL,
                  perp_kernel_fun = "exp",
                  prec_scale = "",
                  init = "p", init_config = NULL,
                  opt = "NAG-BOLD",
                  epsilon = 1,
                  max_iter = 1000,
                  report_every = 50,
                  tol = 1e-4,
                  exaggerate = NULL,
                  exaggerate_off_iter = 50,
                  plot_type = "p",
                  colors = NULL,
                  color_name = NULL,
                  labels = NULL,
                  label_name = NULL,
                  label_chars = NULL,
                  point_size = 1,
                  plot_labels = FALSE,
                  color_scheme = grDevices::rainbow,
                  equal_axes = FALSE,
                  legend = TRUE,
                  legend_rows = NULL,
                  quality_measures = NULL,
                  ret = c()) {

  if (class(df) != "dist" && class(df) != "data.frame") {
    stop("df should be a data frame or dist object")
  }
  if (class(df) != "dist" && is.null(indexes)) {
    indexes <- which(vapply(df, is.numeric, logical(1)))
    message("Found ", length(indexes), " numeric columns")
  }

  normalize_cost <- TRUE

  embed_methods <- list(
    pca = function() { mmds() },
    mmds = function() { mmds() },
    sammon = function() { sammon_map() },
    tsne = function() { tsne() },
    ssne = function() { ssne() },
    asne = function() { asne() },
    wtsne = function() { importance_weight(tsne()) },
    hssne = function() { hssne(alpha = alpha) },
    nerv = function() { unerv(lambda = lambda) },
    jse = function() { jse(kappa = kappa) },
    tasne = function() { tasne() },
    tpsne = function() { tpsne() },
    tpsne_plugin = function() { tpsne_plugin() },
    ssne_plugin = function() { ssne_plugin() },
    asne_plugin = function() { asne_plugin() },
    hssne_plugin = function() { hssne_plugin(alpha = alpha) },
    nerv_plugin = function() { unerv_plugin(lambda = lambda) },
    jse_plugin = function() { jse_plugin(kappa = kappa) }
  )

  extra_costs <- NULL
  # Usually, method is a name of a method that needs to be created
  if (class(method) == "character") {
    method <- tolower(method)
    if (!method %in% names(embed_methods)) {
      stop("Unknown embedding method '",
           method,
           "', should be one of: ",
           paste(names(embed_methods), collapse = ", "))
    }

    # Need to use plugin method if precisions can be non-uniform
    if (prec_scale == "t") {
      new_method <- paste0(method, "_plugin")
      if (!new_method %in% names(embed_methods)) {
        stop("Method '", method, "' is not compatible with prec_scale option 't'")
      }
      message("Switching to plugin method for non-uniform output precisions")
      embed_method <- embed_methods[[new_method]]()
    }
    else {
      embed_method <- embed_methods[[method]]()
    }

    # special casing for different methods
    if (method == "pca") {
      max_iter <- 0
      perplexity <- NULL
      init <- "p"
      if (is.null(extra_costs)) {
        extra_costs <- c("kruskal_stress")
      }
    }
    else if (method == "mmds") {
      perplexity <- NULL
      if (is.null(extra_costs)) {
        extra_costs <- c("kruskal_stress")
      }
    }
    else if (method == "sammon") {
      perplexity <- NULL
      if (is.null(extra_costs)) {
        extra_costs <- c("kruskal_stress")
      }
      normalize_cost <- FALSE
    }
  }
  else {
    # Allow masters of the dark arts to pass in a method directly
    embed_method <- method
  }

  preprocess <- make_preprocess()
  if (!is.null(scale_type)) {
    if (scale_type == "m") {
      preprocess <- make_preprocess(range_scale_matrix = TRUE)
    }
    else if (scale_type == "r") {
      preprocess <- make_preprocess(range_scale = TRUE)
    }
    else if (scale_type == "a") {
      preprocess <- make_preprocess(auto_scale = TRUE)
    }
  }

  init_inp <- NULL
  if (!is.null(perplexity)) {
    if (!is.null(perp_kernel_fun)) {
      if (perp_kernel_fun == "exp") {
        weight_fn <- exp_weight
      }
      else if (perp_kernel_fun == "step") {
        weight_fn <- step_weight
      }
      else if (perp_kernel_fun == "sqrt_exp") {
        weight_fn <- sqrt_exp_weight
      }
      else {
        stop("Unknown perplexity kernel function '", perp_kernel_fun, "'")
      }
    }

    modify_kernel_fn <- NULL
    if ((prec_scale) != "") {
      if (perp_kernel_fun == "step") {
        stop("Can't use precision scaling with step input weight function")
      }
      if (prec_scale == "s") {
        modify_kernel_fn <- scale_prec_to_perp
      }
      else if (prec_scale == "t") {
        modify_kernel_fn <- transfer_kernel_precisions
      }
      else {
        stop("Unknown prec_scale value: '", prec_scale, "'")
      }
    }

    if (!is.null(perp_scale) && perp_scale != "single") {
      if (is.null(perp_scale_iter)) {
        perp_scale_iter <- ceiling(max_iter / 5)
      }
      else {
        if (perp_scale_iter > max_iter) {
          stop("Parameter perp_scale_iter must be <= max_iter")
        }
      }
      if (perp_scale == "max") {
        if (length(perplexity) == 1) {
          perplexity <- ms_perps(df)
        }
        init_inp <- inp_from_dint_max(perplexities = perplexity,
                                      modify_kernel_fn = modify_kernel_fn,
                                      input_weight_fn = weight_fn)
      }
      else if (perp_scale == "multi") {
        if (length(perplexity) == 1) {
          perplexity <- ms_perps(df)
        }
        init_inp <- inp_from_perps_multi(perplexities = perplexity,
                                         num_scale_iters = perp_scale_iter,
                                         modify_kernel_fn = modify_kernel_fn,
                                         input_weight_fn = weight_fn)
      }
      else if (perp_scale == "multil") {
        if (length(perplexity) == 1) {
          perplexity <- ms_perps(df)
        }
        init_inp <- inp_from_perps_multil(perplexities = perplexity,
                                          num_scale_iters = perp_scale_iter,
                                          modify_kernel_fn = modify_kernel_fn,
                                          input_weight_fn = weight_fn)
      }
      else if (perp_scale == "step") {
        if (length(perplexity) == 1) {
          perplexity = step_perps(df)
        }
        init_inp <- inp_from_step_perp(perplexities = perplexity,
                                       num_scale_iters = perp_scale_iter,
                                       modify_kernel_fn = modify_kernel_fn,
                                       input_weight_fn = weight_fn)
      }
      else {
        stop("No perplexity scaling method '", perp_scale, "'")
      }
    }
    else {
      # no perplexity scaling asked for
      if (length(perplexity) == 1) {
        if (class(df) == "dist") {
          # length = (nr * (nr + 1)) / 2; solve for nr by quadratic equation
          nr <- (1 + sqrt(1 + (8 * length(df)))) / 2
        }
        else {
          nr <- nrow(df)
        }
        if (perplexity >= nr) {
          perplexity <- nr / 4
          message("Setting perplexity to ", perplexity)
        }
        init_inp <- inp_from_perp(perplexity = perplexity,
                                  modify_kernel_fn = modify_kernel_fn,
                                  input_weight_fn = weight_fn)
      }
      else {
        stop("Must provide 'perp_scale' argument if using multiple perplexity ",
             "values")
      }
    }
  }

  init_out <- NULL
  if (init == "p") {
    init_out <- out_from_PCA(k = ndim)
  }
  else if (init == "r") {
    init_out <- out_from_rnorm(k = ndim)
  }
  else if (init == "u") {
    init_out <- out_from_runif(k = ndim)
  }
  else if (init == "m") {
    init_out <- out_from_matrix(init_config = init_config, k = ndim)
  }
  else {
    stop("No initialization method '", init, "'")
  }

  embed_plot <- NULL
  if (ndim == 2) {
    color_res <- process_color_options(df,
                                       colors = colors, color_name = color_name,
                                       labels = labels, label_name = label_name,
                                       color_scheme = color_scheme,
                                       verbose = TRUE)
    colors <- color_res$colors
    labels <- color_res$labels

    if (is.null(plot_type)) { plot_type <- "n" }
    if (plot_type == "g") {
      if (!requireNamespace("ggplot2", quietly = TRUE, warn.conflicts = FALSE)) {
        stop("plot type 'g' requires 'ggplot2' package")
      }
      if (!requireNamespace("RColorBrewer",
                            quietly = TRUE,
                            warn.conflicts = FALSE)) {
        stop("plot type 'g' requires 'RColorBrewer' package")
      }
      embed_plot <-
        make_qplot(
          df,
          labels = labels, label_name = label_name,
          color_scheme = color_scheme,
          size = point_size,
          legend = legend,
          legend_rows = legend_rows,
          equal_axes = equal_axes
        )
    }
    else if (plot_type == 'p') {
      label_fn <- NULL
      if (!is.null(label_chars)) {
        label_fn <- make_label(label_chars)
      }

      embed_plot <- make_plot(x = df,
                              colors = colors,
                              labels = labels,
                              label_fn = label_fn,
                              color_scheme = color_scheme,
                              cex = point_size,
                              show_labels = plot_labels,
                              equal_axes = equal_axes)
    }
  }

  after_embed <- NULL
  if (!is.null(quality_measures)) {
    qs <- c()

    for (name in unique(quality_measures)) {
      if (name == 'r') {
        if (!requireNamespace("PRROC", quietly = TRUE, warn.conflicts = FALSE)) {
          stop("Calculating ROC AUC requires 'PRROC' package")
        }
        qs <- c(qs, roc_auc)
      }
      else if (name == 'p') {
        if (!requireNamespace("PRROC", quietly = TRUE, warn.conflicts = FALSE)) {
          stop("Calculating PR AUC requires 'PRROC' package")
        }
        qs <- c(qs, pr_auc)
      }
      else if (name == 'n') {
        qs <- c(qs, rnx_auc)
      }
      else {
        stop("No quality measure '", name,"'")
      }
    }
    if (length(qs) > 0 && !is.null(labels)) {
      after_embed <- make_quality_reporter(qs, labels)
    }
  }

  ok_rets <- c("x", "dx", "dy", "p", "q", "w", "prec", "dim", "deg")
  ret <- unique(ret)
  for (r in (ret)) {
    if (!r %in% ok_rets) {
      stop("Invalid return name: '", r,"'")
    }
  }

  # c1 = 1e-4 suggested by Nocedal & Wright
  c1 <- 1e-4
  # 0.9 for Newton and Quasi-Newton, 0.1 for CG
  c2 <- 0.9
  # Adaptive restart decrement
  dec_mult <- 0.1
  # check if we are using an external optimization method
  ext_opt <- FALSE
  burn_in <- 3
  if (toupper(opt) == "L-BFGS") {
    message("Optimizing with L-BFGS")
    optimizer <- ropt(method = "L-BFGS-B", batch_iter = report_every,
                      inc_iter = TRUE)
    ext_opt <- TRUE
  }
  else if (toupper(opt) == "BFGS") {
    message("Optimizing with BFGS")
    optimizer <- ropt(method = "BFGS", batch_iter = report_every,
                      inc_iter = TRUE)
    ext_opt <- TRUE
  }
  else if (toupper(opt) == "NAG-BOLD") {
    message("Optimizing with Adaptive NAG and bold driver step size")
    optimizer <- bold_nag_adapt(dec_mult = dec_mult, burn_in = burn_in,
                                init_step_size = epsilon)
  }
  else if (toupper(opt) == "NAG-BACK") {
    message("Optimizing with Adaptive NAG and backstepping step size")
    optimizer <- back_nag_adapt(dec_mult = dec_mult, burn_in = burn_in,
                                init_step_size = epsilon)
  }
  else if (toupper(opt) == "NAG-MT") {
    if (!requireNamespace("rconjgrad",
                          quietly = TRUE,
                          warn.conflicts = FALSE)) {
      stop("Using More-Thuente line search requires 'rconjgrad' package")
    }
    message("Optimizing with Adaptive NAG and MT line search")
    optimizer <- back_nag_adapt(dec_mult = dec_mult, burn_in = burn_in,
                                init_step_size = epsilon)
    optimizer$step_size <- more_thuente_ls(c1 = c1, c2 = c2)
  }
  else if (toupper(opt) == "NAG-R") {
    if (!requireNamespace("rconjgrad",
                          quietly = TRUE,
                          warn.conflicts = FALSE)) {
      stop("Using Rasmussen line search requires 'rconjgrad' package")
    }
    message("Optimizing with Adaptive NAG and Rasmussen line search")
    optimizer <- back_nag_adapt(dec_mult = dec_mult, burn_in = burn_in,
                                init_step_size = epsilon)
    optimizer$step_size <- rasmussen_ls(c1 = c1, c2 = c2)
  }
  else if (toupper(opt) == "CG-R") {
    if (!requireNamespace("rconjgrad",
                          quietly = TRUE,
                          warn.conflicts = FALSE)) {
      stop("Using conjugate gradient optimizer requires 'rconjgrad' package")
    }
    message("Optimizing with CG and Rasmussen line search")
    c2 <- 0.1
    optimizer <- optim_rcg(line_search = "r", batch_iter = report_every,
                           c1 = c1, c2 = c2,
                           inc_iter = TRUE)
    ext_opt <- TRUE
  }
  else if (toupper(opt) == "CG-MT") {
    if (!requireNamespace("rconjgrad",
                          quietly = TRUE,
                          warn.conflicts = FALSE)) {
      stop("Using conjugate gradient optimizer requires 'rconjgrad' package")
    }
    message("Optimizing with CG and More-Thuente line search")
    c2 <- 0.1
    optimizer <- optim_rcg(line_search = "mt", batch_iter = report_every,
                           c1 = c1, c2 = c2,
                           inc_iter = TRUE)
    ext_opt <- TRUE
  }
  else if (toupper(opt) == "SPEC-R") {
    if (!requireNamespace("rconjgrad",
                          quietly = TRUE,
                          warn.conflicts = FALSE)) {
      stop("Using spectral direction optimizer requires 'rconjgrad' package")
    }
    message("Optimizing with Spectral Direction and Rasmussen line search")
    optimizer <- optim_spectral(line_search = "r", c1 = c1, c2 = c2)
  }
  else if (toupper(opt) == "SPEC-MT") {
    if (!requireNamespace("rconjgrad",
                          quietly = TRUE,
                          warn.conflicts = FALSE)) {
      stop("Using More-Thuente optimizer requires 'rconjgrad' package")
    }
    message("Optimizing with Spectral Direction and More-Thuente line search")
    optimizer <- optim_spectral(line_search = "mt", c1 = c1, c2 = c2)
  }
  else if (toupper(opt) == "SPEC-BOLD") {
    message("Optimizing with Spectral Direction and Bold Driver line search")
    optimizer <- optim_spectral(line_search = "bold")
  }
  else if (toupper(opt) == "SPEC-BACK") {
    message("Optimizing with Spectral Direction and Backstepping line search")
    optimizer <- optim_spectral(line_search = "back", c1 = c1)
  }
  else if (toupper(opt) == "TSNE") {
    optimizer <- tsne_opt(epsilon = epsilon)
  }
  else {
    stop("Unrecognized optimizer option '", opt, "'")
  }

  # Ensure optimizer can work with perp scaling
  if (ext_opt && !is.null(perp_scale) &&
      perp_scale %in% c('multi', 'multil', 'step')) {
    stop("optimizer '", opt, "' is incompatible with perplexity scale option '",
         perp_scale, "'")
  }

  # Ensure that if Spectral Direction optimizer is chosen, it can be used with
  # the chosen embedding method
  if (substr(opt, 1, 5) == "SPEC-" &&
      (is.null(embed_method$prob_type) || embed_method$prob_type != "joint")) {
    stop("Spectral direction optimizer is only compatible with ",
         "probability-based embedding methods that use symmetric input ",
         "probabilities (e.g. t-SNE), not '", method, "'")
  }

  tricks <- NULL
  if (!is.null(exaggerate)) {
    tricks <- make_tricks(early_exaggeration(exaggeration = exaggerate,
                                             off_iter = exaggerate_off_iter,
                                             verbose = TRUE))
  }

  if (class(df) == 'dist') {
    xm <- df
  }
  else {
    xm <- df[, indexes]
  }
  embed_result <- embed_main(
    xm,
    method = embed_method,
    opt = optimizer,
    preprocess = preprocess,
    init_inp = init_inp,
    init_out = init_out,
    tricks = tricks,
    reporter = make_reporter(
      normalize_cost = normalize_cost,
      report_every = report_every,
      extra_costs = extra_costs,
      plot = embed_plot,
      reltol = tol
    ),
    after_embed = after_embed,
    max_iter = max_iter,
    export = c("report", "inp", "out", "method")
  )

  result <- list(coords = embed_result$ym,
                 cost = embed_result$cost,
                 method = method)
  if (ndim == 2) {
    colnames(result$coords) <- c("X", "Y")
  }
  else {
    colnames(result$coords) <- paste0("X", 1:ndim)
  }

  if (!is.null(embed_result$quality)) {
    for (qual_name in names(embed_result$quality)) {
      result[[qual_name]] <- embed_result$quality[[qual_name]]
    }
  }
  for (extra_cost in extra_costs) {
    if (!is.null(embed_result$report[[extra_cost]])) {
      result[[extra_cost]] <- embed_result$report[[extra_cost]]
    }
  }

  if (!is.null(embed_result$report$norm)) {
    result$norm_cost <- embed_result$report$norm
  }

  inp <- embed_result$inp
  out <- embed_result$out
  for (r in unique(ret)) {
    if (r == "x") {
      if (!is.null(inp$xm)) {
        result$x <- inp$xm
      }
    }
    else if (r == "dx") {
      if (!is.null(inp$dm)) {
        result$dx <- inp$dm
      }
      else if (is.null(inp$dm) && !is.null(inp$xm)) {
        result$dx <- distance_matrix(inp$xm)
      }
    }
    else if (r == "dy") {
      if (!is.null(out$dm)) {
        result$dy <- out$dm
      }
      else if (is.null(out$dm) && !is.null(out$ym)) {
        result$dy <- distance_matrix(out$ym)
      }
    }
    else if (r == "p") {
      if (!is.null(inp$pm)) {
        result$p <- inp$pm
      }
    }
    else if (r == "w") {
      if (!is.null(out$wm)) {
        result$w <- out$wm
      }
    }
    else if (r == "q") {
      if (!is.null(out$qm)) {
        result$q <- out$qm
      }
    }
    else if (r == "prec") {
      if (!is.null(inp$beta)) {
        result$prec <- 2 * inp$beta
      }
    }
    else if (r == "dim") {
      if (!is.null(inp$dims)) {
        result$dim <- inp$dims
      }
    }
    else if (r == "deg") {
      if (!is.null(inp$deg)) {
        result$deg  <- inp$deg
      }
      else {
        if (!is.null(inp$pm)) {
          result$deg <- centrality(inp, embed_result$method)
        }
      }
    }
    else {
      warning("Skipping return of unknown result type '", r, "'")
    }
  }

  result
}

# Probability-Based Embedding
#
# Carry out an embedding of a dataset using a probability-based method
# (e.g. t-SNE), with some useful default parameters.
#
# @param xm A matrix or data frame to embed.
# @param method Embedding method. Set by assigning the result value of one of
#   the configuration functions listed in
#   \code{\link{probability_embedding_methods}}.
# @param preprocess Input data preprocess callback. Set by assigning the
#   result value of \code{\link{make_preprocess}}.
# @param init_inp Input initializer. Set by assigning the result value of
#   calling one of the available \code{\link{input_initializers}}.
# @param init_out Output initializer. Set by assigning the result value of
#   calling one of the available \code{\link{output_initializers}}.
# @param opt Optimization method. Set by assigning the result of value of one
#   of the configuration functions listed in
#   \code{\link{optimization_methods}}.
# @param max_iter Maximum number of optimization steps to take.
# @param tricks Optional collection of heuristics. Set by assigning the result
#  value of \code{\link{make_tricks}} or a related wrapper. See
#  \code{\link{tricks}} for the available tricks.
# @param reporter Reporter callback. Set by assigning the result value of
#   \code{\link{make_reporter}}.
# @param export Vector of names, specifying data in addition to of the output
#   data to export provide in the return value. Possible names are:
#   \describe{
#     \item{"\code{inp}"}{The input data, based on the argument passed to
#       \code{init_inp}. See the help text for the specific
#       \code{\link{input_initializers}} for details.}
#     \item{"\code{report}"}{The result of the last report. See the help
#       text for \code{\link{make_reporter}} for details.}
#     \item{"\code{opt}"}{The optimizer. See the help text for the optimizer
#     passed to \code{opt} for details.}
#   }
#   For each name supplied, an item with that name will appear in the return
#   value list.
# @param after_embed Callback to run on input and output data before output
#   data is returned.
# @param verbose If \code{TRUE} display messages about the embedding progress.
# @return The output data. A list containing:
#  \itemize{
#   \item \code{ym} Embedded coordinates.
#   \item \code{cost} The cost value associated with \code{ym}.
#   If the parameter \code{export} was used, additional elements will be
#   present. See the help text for the \code{export} parameter for more
#   details.
#   }
# @seealso
# \itemize{
# \item \code{\link{probability_embedding_methods}} for configuring
#   \code{method}
# \item \code{\link{make_preprocess}} for configuring \code{preprocess}
# \item \code{\link{input_initializers}} for configuring \code{init_inp}
# \item \code{\link{output_initializers}} for configuring \code{init_out}
# \item \code{\link{optimization_methods}} for configuring \code{opt}
# \item \code{\link{make_tricks}} for configuring \code{tricks}
# \item \code{\link{make_reporter}} for configuring \code{reporter}
# }
#
# @examples
# \dontrun{
# # Do t-SNE on the iris dataset with the same options as the t-SNE paper
# # except initialize from PCA so output is repeatable.
# # plot 2D result during embedding with convenience function for iris plot.
# # Default method is tsne. Set perplexity to 25.
# tsne_iris <- embed_prob(iris[, 1:4], opt = tsne_opt(),
#                init_inp = inp_from_perp(perplexity = 25),
#                tricks = tsne_tricks(),
#                reporter = make_reporter(plot =
#                                           make_plot(iris,
#                                                     labels = iris$Species)))
#
# # Do t-SNE on the iris dataset with the same options as the t-SNE paper
# # and initialize from a random normal distribution. Use generic plot
# # function, displaying the first two characters of the "Species" factor for
# # the points. Explicitly choose t-SNE as the method.
# tsne_iris <- embed_prob(iris[, 1:4],
#                method = tsne(),
#                opt = tsne_opt(),
#                init_inp = inp_from_perp(perplexity = 25),
#                init_out = out_from_rnorm(sd = 1e-4),
#                tricks = tsne_tricks(),
#                reporter = make_reporter(
#                  plot = make_plot(iris, label_name = "Species",
#                                   label_fn = make_label(2))))
#
# # Use the SSNE method, and preprocess input data by range scaling. t-SNE
# # tricks and optimization are reasonable defaults for other probability-based
# # embeddings. Initialize from a uniform distribution.
# ssne_iris <- embed_prob(iris[, 1:4],
#                method = ssne(),
#                opt = tsne_opt(),
#                init_inp = inp_from_perp(perplexity = 25),
#                preprocess = make_preprocess(range_scale_matrix = TRUE),
#                init_out = out_from_runif(),
#                tricks = tsne_tricks(),
#                reporter = make_reporter(
#                  plot = make_plot(iris, label_name = "Species",
#                                   label_fn = make_label(2))))
#
# # ASNE method on the s1k dataset (10 overlapping 9D Gaussian blobs),
# # Set perplexity for input initialization to 50, initialize with PCA scores,
# # preprocess by autoscaling columns, optimize
# # with Nesterov Accelrated Gradient and bold driver step size
# # (highly recommended as an optimizer). Labels for s1k are one digit, so
# # can use simplified plot function.
# asne_s1k <- embed_prob(s1k[, 1:9], method = asne(),
#  preprocess = make_preprocess(auto_scale = TRUE),
#  init_inp = inp_from_perp(perplexity = 50),
#  init_out = out_from_PCA(),
#  opt = make_opt(gradient = nesterov_gradient(), step_size = bold_driver(),
#   update = nesterov_nsc_momentum()),
#   reporter = make_reporter(plot = make_plot(s1k, label_name = "Label")))
#
# # Same as above, but using convenience method to create optimizer with less
# # typing
# asne_s1k <- embed_prob(s1k[, 1:9], method = asne(),
#  preprocess = make_preprocess(auto_scale = TRUE),
#  init_inp = inp_from_perp(perplexity = 50),
#  init_out = out_from_PCA(),
#  opt = bold_nag(),
#  reporter = make_reporter(plot = make_plot(s1k, label_name = "Label")))
# }
# @family sneer embedding functions
embed_prob <- function(xm,
                      method = tsne(),
                      preprocess = make_preprocess(verbose = verbose),
                      init_inp = inp_from_perp(perplexity = 30,
                                           input_weight_fn = exp_weight,
                                           verbose = verbose),
                      init_out = out_from_PCA(verbose = verbose),
                      opt = make_opt(),
                      max_iter = 1000,
                      tricks = make_tricks(),
                      reporter = make_reporter(verbose = verbose),
                      export = NULL,
                      after_embed = NULL,
                      verbose = TRUE) {
  embed_main(xm, method, init_inp, init_out, opt, max_iter, tricks,
        reporter, preprocess, export, after_embed)
}

# Distance-Based Embedding.
#
# Carry out an embedding of a dataset using a distance-based method
# (e.g. Sammon Mapping), with some useful default parameters.
#
# @param xm A matrix or data frame to embed.
# @param method Embedding method. Set by assigning the result value of one of
#   the configuration functions listed in
#   \code{\link{distance_embedding_methods}}.
# @param preprocess Input data preprocess callback. Set by assigning the
#   result value of \code{\link{make_preprocess}}.
# @param init_inp Input initializer. Set by assigning the result value of
#   calling one of the available \code{\link{input_initializers}}.
# @param init_out Output initializer. Set by assigning the result value of
#   calling one of the available \code{\link{output_initializers}}.
# @param opt Optimization method. Set by assigning the result of value of one
#   of the configuration functions listed in
#   \code{\link{optimization_methods}}.
# @param max_iter Maximum number of optimization steps to take.
# @param tricks Optional collection of heuristics. Set by assigning the result
#  value of \code{\link{make_tricks}} or a related wrapper. See
#  \code{\link{tricks}} for the available tricks.
# @param reporter Reporter callback. Set by assigning the result value of
#   \code{\link{make_reporter}}.
# @param export Vector of names, specifying data in addition to of the output
#   data to export provide in the return value. Possible names are:
#   \describe{
#     \item{"\code{inp}"}{The input data, based on the argument passed to
#       \code{init_inp}. See the help text for the specific
#       \code{\link{input_initializers}} for details.}
#     \item{"\code{report}"}{The result of the last report. See the help
#       text for \code{\link{make_reporter}} for details.}
#     \item{"\code{opt}"}{The optimizer. See the help text for the optimizer
#     passed to \code{opt} for details.}
#   }
# @param after_embed Callback to run on input and output data before output
#   data is returned.
# @param verbose If \code{TRUE} display messages about the embedding progress.
# @return The output data. A list containing:
#  \itemize{
#   \item \code{ym} Embedded coordinates.
#   \item \code{cost} The cost value associated with \code{ym}.
#   If the parameter \code{export} was used, additional elements will be
#   present. See the help text for the \code{export} parameter for more
#   details.
#  }
# @seealso
# \itemize{
# \item \code{\link{distance_embedding_methods}} for configuring
#   \code{method}
# \item \code{\link{make_preprocess}} for configuring \code{preprocess}
# \item \code{\link{input_initializers}} for configuring \code{init_inp}
# \item \code{\link{output_initializers}} for configuring \code{init_out}
# \item \code{\link{optimization_methods}} for configuring \code{opt}
# \item \code{\link{make_tricks}} for configuring \code{tricks}
# \item \code{\link{make_reporter}} for configuring \code{reporter}
# }
#
# @examples
# \dontrun{
# # Do metric MDS on the iris data set
# # In addition to the STRESS loss function also report the Kruskal Stress
# # (often used in MDS applications) and the mean relative error, which can
# # be multiplied by 100 and interpreted as a percentage error.
# mds_iris <-
#   embed_dist(iris[, 1:4],
#              method = mmds(),
#              opt = bold_nag(),
#              reporter = make_reporter(
#                          extra_costs = c("kruskal_stress",
#                                          "mean_relative_error")),
#                          plot = make_plot(iris, labels = iris$Species))
#
# # Sammon map the autoscaled iris data set, which turns out to be a
# # surprisingly tough assignment. Increase epsilon substantially to 1e-4 to
# # avoid the gradient being overwhelmed by zero distances in the input space.
# # Additionally, we report two other normalized stress functions often used
# # in MDS. The Sammon mapping cost function is already normalized, so tell the
# # make_reporter function not to report an automatically normalized version in
# # the output.
# sammon_iris <-
#   embed_dist(iris[, 1:4],
#              method = sammon_map(eps = 1e-4),
#              opt = bold_nag(),
#              preprocess = make_preprocess(auto_scale = TRUE),
#              init_out = out_from_rnorm(sd = 1e-4),
#              reporter = make_reporter(
#                 normalize_cost = FALSE,
#                 extra_costs = c("normalized_stress", "kruskal_stress"),
#                 plot = make_plot(iris, label_name = "Species",
#                                  label_fn = make_label())))
# }
# @family sneer embedding functions
embed_dist <- function(xm,
                       method = mmds(),
                       preprocess = make_preprocess(verbose = verbose),
                       init_inp = NULL,
                       init_out = out_from_PCA(verbose = verbose),
                       opt = make_opt(),
                       max_iter = 1000,
                       tricks = make_tricks(),
                       reporter = make_reporter(verbose = verbose),
                       export = NULL,
                       after_embed = NULL,
                       verbose = TRUE) {
  embed_main(xm, method, init_inp, init_out, opt, max_iter, tricks,
        reporter, preprocess, export, after_embed)
}


# Generic Embedding
#
# Carry out an embedding of a dataset using any embedding method. The most
# generic embedding function, and conversely the one with the fewest helpful
# defaults.
#
# @param xm A matrix or data frame to embed.
# @param method Embedding method. Set by assigning the result value of one of
#   the configuration functions listed in
#   \code{\link{embedding_methods}}.
# @param preprocess Input data preprocess callback. Set by assigning the
#   result value of \code{\link{make_preprocess}}.
# @param init_inp Input initializer. Set by assigning the result value of
#   calling one of the available \code{\link{input_initializers}}.
# @param init_out Output initializer. Set by assigning the result value of
#   calling one of the available \code{\link{output_initializers}}.
# @param opt Optimization method. Set by assigning the result of value of one
#   of the configuration functions listed in
#   \code{\link{optimization_methods}}.
# @param max_iter Maximum number of optimization steps to take.
# @param tricks Optional collection of heuristics. Set by assigning the result
#  value of \code{\link{make_tricks}} or a related wrapper. See
#  \code{\link{tricks}} for the available tricks.
# @param reporter Reporter callback. Set by assigning the result value of
#   \code{\link{make_reporter}}.
# @param export Vector of names, specifying data in addition to of the output
#   data to export provide in the return value. Possible names are:
#   \describe{
#     \item{"\code{inp}"}{The input data, based on the argument passed to
#       \code{init_inp}. See the help text for the specific
#       \code{\link{input_initializers}} for details.}
#     \item{"\code{report}"}{The result of the last report. See the help
#       text for \code{\link{make_reporter}} for details.}
#     \item{"\code{opt}"}{The optimizer. See the help text for the optimizer
#     passed to \code{opt} for details.}
#   }
# @param after_embed Callback to run on input and output data before output
#   data is returned.
# @return The output data. A list containing:
#  \itemize{
#   \item \code{ym} Embedded coordinates.
#   \item \code{cost} The cost value associated with \code{ym}.
#   If the parameter \code{export} was used, additional elements will be
#   present. See the help text for the \code{export} parameter for more
#   details.
# }
# @seealso
# \itemize{
# \item \code{\link{embedding_methods}} for configuring \code{method}
# \item \code{\link{make_preprocess}} for configuring \code{preprocess}
# \item \code{\link{input_initializers}} for configuring \code{init_inp}
# \item \code{\link{output_initializers}} for configuring \code{init_out}
# \item \code{\link{optimization_methods}} for configuring \code{opt}
# \item \code{\link{make_tricks}} for configuring \code{tricks}
# \item \code{\link{make_reporter}} for configuring \code{reporter}
# }
embed_main <- function(xm, method, init_inp, init_out, opt, max_iter = 1000,
                  tricks = NULL, reporter = NULL,
                  preprocess = make_preprocess(),
                  export = NULL, after_embed = NULL) {

  opt$max_iter <- max_iter
  init_result <- init_embed(xm, method, preprocess, init_inp, init_out, opt)
  inp <- init_result$inp
  out <- init_result$out
  method <- init_result$method
  opt <- init_result$opt
  report <- init_result$report

  iter <- 0
  while (iter <= max_iter) {
    # For convenience we already did the 0 iter initialization in init_embed,
    # so we skip calling init_inp on the first iteration. Cheesy, I know.
    if (iter != 0 && !is.null(init_inp)) {
      inp_result <- init_inp(inp, method, opt, iter, out)
      inp <- inp_result$inp
      out <- inp_result$out
      method <- inp_result$method
      opt <- inp_result$opt
    }

    if (!is.null(tricks)) {
      tricks_result <- tricks(inp, out, method, opt, iter)
      inp <- tricks_result$inp
      out <- tricks_result$out
      method <- tricks_result$method
      opt <- tricks_result$opt
    }

    out <- update_out_if_necessary(inp, out, method)

    if (!is.null(reporter)) {
      report <- reporter(iter, inp, out, method, opt, report)
      if (report$stop_early) {
        break
      }
    }

    if (!is.null(opt$stop_early) && opt$stop_early) {
      message("Optimizer no longer making progress, stopping early")
      break
    }

    opt_result <- opt$optimize_step(opt, method, inp, out, iter)
    out <- opt_result$out
    opt <- opt_result$opt
    if (!is.null(opt_result$iter)) {
      iter <- opt_result$iter
    }

    iter <- iter + 1
  }

  if (!is.null(after_embed)) {
    out <- after_embed(inp, out)
  }

  # Force an update on the report if it wasn't triggered on the final iteration
  if ((is.null(report$iter) || report$iter != iter - 1) &&
      (!report$stop_early || (!is.null(opt$stop_early) && opt$stop_early))) {
    if (!is.null(reporter)) {
      report <- reporter(iter, inp, out, method, opt, report, force = TRUE)
    }
  }

  if (!is.null(report$cost)) {
    out$cost <- report$cost
  }
  else {
    out$cost <- calculate_cost(method, inp, out)
  }

  for (obj_name in export) {
    out[[obj_name]] <- get(obj_name)
  }

  out
}

# Post Initialization
#
# Function called after input and output data have been initialized. Useful for
# doing data-dependent initialization of stiffness parameters, e.g. based on
# the parameterization of the input probabilities.
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return List consisting of:
#  \itemize{
#   \item \code{inp} Updated input data.
#   \item \code{out} Updated output data.
#   \item \code{method} Updated embedding method.
#  }
after_init <- function(inp, out, method) {
  if (!is.null(method$after_init_fn)) {
    result <- method$after_init_fn(inp, out, method)
    if (!is.null(result$inp)) {
      inp <- result$inp
    }
    if (!is.null(result$out)) {
      out <- result$out
    }
    if (!is.null(result$method)) {
      method <- result$method
    }
  }
  list(inp = inp, out = out, method = method)
}

# Embedding Initialziation
#
# Initializes all data, ready to be optimized.
#
# @param xm A matrix or data frame to embed.
# @param method Embedding method. Set by assigning the result value of one of
#   the configuration functions listed in
#   \code{\link{embedding_methods}}.
# @param preprocess Input data preprocess callback. Does any necessary
#  filtering and scaling of the input data.
# @param init_inp Input initializer. Creates the input data required by
#  embedding algorithm.
# @param init_out Output initializer. Creates the initial set of output
#  coordinates.
# @param opt Optimization method.
# @return A list containing:
#  \itemize{
#   \item \code{inp} Initialized input.
#   \item \code{out} Initialized output.
#   \item \code{method} Initialized embedding method.
#   \item \code{opt} Initialized optimizer.
#   \item \code{report} Initialized report.
#  }
init_embed <- function(xm, method, preprocess, init_inp, init_out, opt) {
  inp <- preprocess(xm)

  # Output initialization normally only needs to make us of input coordinates
  # or distance matrix, which preprocessing gives us, not things like the input
  # probabilities. So we can do output initialization here.
  out <- init_out(inp)

  # Input initialization handles input probability and other related data.
  # As we often only indirectly control the shape of the input data, we may want
  # to adjust some method parameters in the light of this data and the output
  # data. So it's most convenient to allow input initialization to assume output
  # initialization has already happened, rather than vice versa.
  if (!is.null(init_inp)) {
    inp_result <- init_inp(inp, method, opt, iter = 0, out)
    inp <- inp_result$inp
    method <- inp_result$method
    opt <- inp_result$opt
  }

  # do late initialization that relies on input or output initialization
  # being completed
  after_init_result <- after_init(inp, out, method)
  inp <- after_init_result$inp
  out <- after_init_result$out
  method <- after_init_result$method

  # initialize matrices needed for gradient calculation
  out <- update_out(inp, out, method)

  # reuse reports from old invocation of reporter, so we can use info
  # to determine whether to stop early (e.g. relative convergence tolerance)
  report <- list()

  list(
    inp = inp,
    out = out,
    method = method,
    opt = opt,
    report = report
  )
}

# Set Output Data Coordinates
#
# This function sets the embedded coordinates in the output data, as well as
# recalculating any auxiliary output data that is dependent on the coordinates
# (e.g. distances and probabilities)
#
# @param inp Input data.
# @param coords Matrix of coordinates.
# @param method Embedding method.
# @param mat_name Name of the \code{out} entry to store \code{coords} in.
#  Defaults to \code{ym}.
# @param out Existing output data. Optional.
# @return \code{out} list with updated with coords and auxiliary data.
set_solution <- function(inp, coords, method, mat_name = "ym",
                         out = NULL) {
  if (is.null(out)) {
    out <- list()
  }
  out[[mat_name]] <- coords
  update_out(inp, out, method)
}

# Check if Output Data Needs Updating
#
# When the coordinates of an embedding are updated via
# \code{\link{set_solution}}, the other matrices and values that depend on
# those coordinates should be updated automatically. If a change has occurred
# to another part of the embedding data (most likely the input probabilities,
# or the output kernel parameters), then you may need to call this function to
# see if the output data needs updating before calculating the cost or
# gradient.
#
# @param out Output data.
# @return \code{TRUE} if the output data matrices need updating.
should_update <- function(out) {
  !is.null(out$dirty) && out$dirty
}

# Mark Output Data as Up to Date
#
# This function should be called when the output data has just been updated.
# Some parts of the code will avoid unnecessarily repeating the update, but
# only if they can detect that there's no change to make.
#
# @param out Output data.
# @return Output data, now marked as being up to date.
undirty <- function(out) {
  out$dirty <- FALSE
  out
}

# Update if Something Has Changed
#
# This function calls \code{\link{update_out}} only if it detects that there
# are changes that needs to be made.
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return (Possible) updated output data. Otherwise, \code{out} is returned
# unchanged.
update_out_if_necessary <- function(inp, out, method) {
  if (should_update(out)) {
    out <- update_out(inp, out, method)
  }
  out
}

# Update Output Data
#
# Updates the output data (matrices dependent on the embedding coordinates,
# such as the distance matrix, weight matrix, output probability matrix).
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return Updated output data.
update_out <- function(inp, out, method) {
  if (!is.null(method$update_out_fn)) {
    out <- method$update_out_fn(inp, out, method)
    out <- undirty(out)
  }
  out
}

