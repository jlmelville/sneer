#' sneer: Stochastic Neighbor Embedding Experiments in R.
#'
#' A package for exploring probability-based embedding and related forms
#' of dimensionality reduction. Its main goal is to implement multiple
#' embedding methods within a single framework so comparison between them is
#' easier, without worrying about the effect of differences in preprocessing,
#' optimization and heuristics.
#'
#' @section Embedding:
#'
#' The \code{\link{sneer}} function provides a variety of methods for embedding,
#' including:
#'
#' \itemize{
#'  \item{Stochastic Neighbor Embedding and variants (ASNE, SSNE and TSNE)}
#'  \item{Metric MDS using the STRESS and SSTRESS functions}
#'  \item{Sammon Mapping}
#'  \item{Heavy-tailed Symmetric Stochastic Neighbor Embedding (HSSNE)}
#'  \item{Neigbor Retrieval Visualizer (NeRV)}
#'  \item{Jensen-Shannon Embedding (JSE)}
#'  \item{Inhomogeneous t-SNE}
#' }
#'
#' See the documentation for the function for the exact list of methods
#' and variations. If you want to create variations on these methods by
#' trying different cost functions, weighting functions and normalization
#' schemes, see the \code{\link{embedder}} function.
#'
#' Optimization is carried out with the mize package
#' (\url{https://github.com/jlmelville/mize}) with the limited memory BFGS.
#' Other optimization methods include the Nesterov Accelerated Gradient method
#' (Sutskever et al 2013) with an adaptive restart (O'Donoghue and Candes 2013),
#' which is a bit more robust compared to the usual t-SNE optimization method
#' across the different methods exposed by sneer.
#'
#' @section Visualization:
#'
#' The \code{\link{embed_plot}} function will take the output of the
#' \code{\link{sneer}} function and provide a visualization of the embedding.
#' If you install the \code{RColorBrewer} package installed, you can use the
#' ColorBrewer palettes by name.
#'
#' @section Quantifying embedding quality:
#'
#' Some functions are available for attempting to quantify embedding quality,
#' independent of the particular loss function used for an embedding method.
#' The \code{\link{nbr_pres}} function will measure how well the embedding
#' preserves a neighborhood of a given size around each observation. The
#' \code{\link{rnx_auc_embed}} function implements the Area Under the Curve
#' of the RNX curve (Lee et al. 2015), which generalizes the neighborhood
#' preservation to account for all neighborhood sizes, with a bias towards
#' smaller neighborhoods.
#'
#' If your observations have labels which could be used for a classification
#' task, then there are also functions which will use these labels to calculate
#' the Area Under the ROC or PR (Precision/Recall) Curve, using the embedded
#' distances to rank each observation: these are \code{\link{roc_auc_embed}}
#' and \code{\link{pr_auc_embed}} functions, respectively. Note that to use
#' these two functions, you must have the \code{PRROC} package installed.
#'
#' @section Synthetic Dataset:
#' There's a synthetic dataset in this package, called \code{s1k}. It consists
#' of a 1000 points representing a fuzzy 9D simplex. It's intended to
#' demonstrate the "crowding effect" and require the sort of
#' probability-based embedding methods provided in this package (PCA does a
#' horrible job of separated the 10 clusters in the data). See \code{s1k}
#' for more details.
#'
#' @examples
#' \dontrun{
#' # Do t-SNE on the iris dataset, scaling columns to zero mean and
#' # unit variance.
#' res <- sneer(iris, scale_type = "a")
#'
#' # Use the weighted TSNE variant and export the input and output distance
#' # matrices.
#' res <- sneer(iris, scale_type = "a", method = "wtsne", ret = c("dx", "dy"))
#'
#' # calculate the 32-nearest neighbor preservation for each observation
#' # 0 means no neighbors preserved, 1 means all of them
#' pres32 <- nbr_pres(res$dx, res$dy, 32)
#'
#' # Calculate the Area Under the RNX Curve
#' rnx_auc <- rnx_auc_embed(res$dx, res$dy)
#'
#' # Load the PRROC library
#' library(PRROC)
#'
#' # Calculate the Area Under the Precision Recall Curve for the embedding
#' pr <- pr_auc_embed(res$dy, iris$Species)
#'
#' # Similarly, for the ROC curve:
#' roc <- roc_auc_embed(res$dy, iris$Species)
#'
#' # Load the RColorBrewer library
#' library(RColorBrewer)
#' # Plot the embedding, with points colored by the neighborhood preservation
#' embed_plot(res$coords, x = pres32, color_scheme = "Blues")
#' }
#' @references
#'
#' t-SNE, SNE and ASNE
#' Van der Maaten, L., & Hinton, G. (2008).
#' Visualizing data using t-SNE.
#' \emph{Journal of Machine Learning Research}, \emph{9}(2579-2605).
#'
#' NeRV
#' Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010).
#' Information retrieval perspective to nonlinear dimensionality reduction for
#' data visualization.
#' \emph{Journal of Machine Learning Research}, \emph{11}, 451-490.
#'
#' JSE
#' Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013).
#' Type 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in
#' dimensionality reduction based on similarity preservation.
#' \emph{Neurocomputing}, \emph{112}, 92-108.
#'
#' Inhomogeneous t-SNE
#' Kitazono, J., Grozavu, N., Rogovschi, N., Omori, T., & Ozawa, S.
#' (2016, October).
#' t-Distributed Stochastic Neighbor Embedding with Inhomogeneous Degrees of
#' Freedom.
#' In \emph{International Conference on Neural Information Processing (ICONIP 2016)} (pp. 119-128).
#' Springer International Publishing.
#'
#' Nesterov Accelerated Gradient:
#' Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013).
#' On the importance of initialization and momentum in deep learning.
#' In \emph{Proceedings of the 30th international conference on machine learning (ICML-13)}
#' (pp. 1139-1147).
#'
#' O'Donoghue, B., & Candes, E. (2013).
#' Adaptive restart for accelerated gradient schemes.
#' \emph{Foundations of computational mathematics}, \emph{15}(3), 715-732.
#'
#' Spectral Direction:
#' Vladymyrov, M., & Carreira-Perpinan, M. A. (2012).
#' Partial-Hessian Strategies for Fast Learning of Nonlinear Embeddings.
#' In \emph{Proceedings of the 29th International Conference on Machine Learning (ICML-12)}
#' (pp. 345-352).
#'
#' @docType package
#' @name sneer
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
#'   \emph{NB}: The original paper suggests setting the output weight function
#'   precisions to be equal to those of the input weights. Later papers don't
#'   mention this. For consistency with other embedding methods, the default
#'   behavior is \emph{not} to transfer the precisions to the output function.
#'   To transfer precisions, set \code{prec_scale = "transfer"}.
#'  \item \code{"jse"} Jensen-Shannon Embedding of Lee at al (2013).
#'  \item \code{"itsne"} Inhomogeneous t-SNE method of Kitazono et al (2016).
#'  \item \code{"dhssne"} A "dynamic" version of HSSNE, inspired by the
#'  inhomogeneous t-SNE Method of Kitazono et al.
#' }
#'
#' Custom embedding methods can also be used, via the \code{\link{embedder}}
#' function.
#'
#' The \code{"dyn"} parameter allows for kernel parameters to be optimized, if
#' the output kernel is exponential or heavy-tailed, i.e. methods \code{asne},
#' \code{ssne}, \code{nerv} and \code{jse} (which use the exponential kernel)
#' and \code{hssne} (which uses the heavy-tailed kernel). The parameter
#' should be a list consisting of the following names:
#'
#' \itemize{
#'   \item For exponential kernels, \code{"beta"} (the precision of the
#'   exponential.)
#'   \item For the heavy-tailed kernel, \code{"alpha"} (the heavy-tailedness),
#'   and \code{"beta"} (analogous to the precision of the exponential).
#'   \item \code{alt_opt} If \code{TRUE}, then optimize non-coordinates
#'   separately from coordinates.
#'   \item \code{"kernel_opt_iter"} Wait this number of iterations before
#'   beginning to optimize non-coordinate parameters.
#' }
#'
#' The values of the list \code{"beta"} and \code{"alpha"} items should be one
#' of:
#'
#' \itemize{
#'   \item \code{"global"} The parameter is the same for every point.
#'   \item \code{"point"} The value is applied per point, and can be different
#'   for each point.
#'   \item \code{"static"} The value is fixed at its initial value and is not
#'   optimized.
#' }
#'
#' Setting a value to \code{"static"} only makes sense for kernels where there
#' is more than one parameter that could be optimized and you don't want all of
#' them optimized (e.g. you may only want to optimize alpha in the heavy-tailed
#' kernel). It's an error to specify all parameters as \code{"static"}.
#'
#' The methods \code{"dhssne"} and \code{"itsne"} already use dynamic kernel
#' optimization and don't require any further specification, but specifying the
#' \code{alt_opt} and \code{kernel_opt_iter} list members will affect their
#' behavior.
#'
#' The following scaling options can be applied via the \code{scale_type}
#' parameter:
#' \itemize{
#'  \item \code{"none"} Do nothing. The default.
#'  \item \code{"matrix"} Range scale the entire data so that the maximum value is
#'   1 and the minimum 0.
#'  \item \code{"range"} Range scale each column that the maximum value in each
#'   column is 1 and the minimum 0.
#'  \item \code{"auto"} Scale each column so that its mean is 0 and variance is
#'   1.
#' }
#' These arguments can be abbreviated. Default is to do no scaling. Zero
#' variance columns will be removed even if no preprocessing is carried out.
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
#'     averaging. \strong{N.B.} Multiscaling is not compatible with
#'     \code{method}s \code{"itsne"} or \code{"dhssne"}.
#' }
#' These arguments can be abbreviated.
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
#'  \item \code{"none"} Do nothing. Most embedding methods follow this strategy,
#'    leaving the output similarity kernels to all have unit precision.
#'  \item \code{"transfer"} Transfer the input similarity kernel parameters to the
#'    output similarity kernel. This method was suggesed by Venna et al (2010).
#'    This is only compatible with methods \code{"asne"}, \code{"jse"} and
#'    \code{"nerv"}.
#'  \item \code{"scale"} Scale the output kernel precisions based on the target
#'    \code{perplexity} and the intrinsic dimensionality of the input data. This
#'    method is part of the multiscaling technique proposed by Lee et al (2015).
#' }
#' These arguments can be abbreviated.
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
#'  \item \code{"pca"} Initialize using the first two scores of the PCA (using
#'  classical MDS if \code{df} is a distance matrix). Data will be centered,
#'  but not scaled unless the \code{scale_type} parameter is used.
#'  \item \code{"random"} Initialize each coordinate value from a normal random
#'  distribution with a standard deviation of 1e-4, as suggested by van der
#'  Maaten and Hinton (2008).
#'  \item \code{"uniform"} Initialize each coordinate value from a uniform random
#'  distribution between 0 and 1 as suggested by Venna et al (2010).
#'  \item Coordinates may also be passed directly as a \code{matrix}. The
#'  dimensions must be correct for the input data.
#'}
#' Character arguments can be abbreviated.
#'
#' For configuring the optimization method, the options for the \code{opt}
#' parameter are:
#' \itemize{
#'  \item \code{"TSNE"} The optimization method used in the original t-SNE
#'    paper: the Jacobs method for step size selection and a step function
#'    for the momentum: switching from 0.4 to 0.8 after 250 steps. You may need
#'    to modify the \code{"eta"} parameter to get good results, depending
#'    on how you have scaled and preprocessed your data, and the embedding
#'    method used.
#'  \item \code{"BFGS"} The Broyden-Fletcher-Goldfarb-Shanno (BFGS) method.
#'  Requires storing an approximation to the Hessian, so not good for large
#'  datasets.
#'  \item \code{"L-BFGS"} The limited-memory BFGS method (using the last ten
#'  updates). Default method.
#'  \item \code{"NEST"} Momentum emulating Nesterov Accelerated Gradient
#'  (Sutskever and co-workers 2013).
#'  \item \code{"CG"} Conjugate Gradient.
#'  \item \code{"SPEC"} Spectral Direction partial Hessian method of Vladymyrov
#'  and Carreira-Perpinan (2012). Requires a probability-based embedding
#'   method and that the input probability matrix be symmetric. Some
#'   probability-based methods are not compatible (e.g. NeRV and JSE; t-SNE
#'   works with it, however). Also, while it works with the dense matrices used
#'   by sneer, because this method uses a Cholesky decomposition of the input
#'   probability matrix which has a complexity of O(N^3), it is intended
#'   to be used with sparse matrices. Its inclusion here is suitable for use
#'   with smaller datasets.
#' }
#'
#' For the \code{quality_measures} argument, a vector with one or more of the
#' following options can be supplied:
#' \itemize{
#'  \item \code{"rocauc"} Calculate the area under the ROC curve, averaged over
#'   each observation, using the output distance matrix to rank each
#'   observation. Observations are partitioned into the positive and negative
#'   class depending upon the value of the label determined by the
#'   \code{label_name} argument. Only calculated if the \code{label_name}
#'   parameter is supplied.
#'  \item \code{"prauc"} Calculate the area under the Precision-Recall curve.
#'   Only calculated if the \code{label_name} parameter is supplied.
#'  \item \code{"rnxauc"} Calculate the area under the RNX curve, using the
#'   method of Lee et al (2015).
#' }
#' Options may be abbreviated.
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
#' or by explicitly providing \code{plot_type = "plot"}) or with the
#' \code{ggplot2} library (which you need to install and load yourself), by
#' using \code{plot_type = "ggplot2"} (you may abbreviate these arguments). The
#' goal has been to provide enough customization to give intelligible results
#' for most datasets. The following are things to consider:
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
#'  \item \code{"pcost"} The final cost function value, decomposed into n
#'  contributions, where n is the number of points embedded.
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
#'  \item \code{"dyn"} "Dynamic" parameters, i.e. any non-coordinate parameters
#'  which were optimized. Only used if the \code{"dyn"} input parameter was
#'  non-\code{NULL}, in which case the values of the parameters specified
#'  are returned.
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
#'  \code{"hssne"} or \code{"dhssne"}. For the latter, it specifies the initial
#'  value of the parameter.
#' @param dof Initial number of degrees of freedom. Used only if the method is
#' \code{"itsne"}. A value of 1 gives initial behavior like t-ASNE, and values
#' approaching infinity behave like ASNE.
#' @param dyn List containing kernel parameters to be optimized. See "Details".
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
#' @param opt Type of optimizer. See 'Details'.
#' @param eta Learning rate when \code{opt} is set to \code{"TSNE"} and
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
#' Kitazono, J., Grozavu, N., Rogovschi, N., Omori, T., & Ozawa, S.
#' (2016, October).
#' t-Distributed Stochastic Neighbor Embedding with Inhomogeneous Degrees of
#' Freedom.
#' In \emph{International Conference on Neural Information Processing (ICONIP 2016)} (pp. 119-128).
#' Springer International Publishing.
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
#'   # rate (eta).
#'   res <- sneer(iris, scale_type = "m", perplexity = 25, opt = "tsne",
#'                eta = 500)
#'
#'   # Use the L-BFGS optimization method
#'   res <- sneer(iris, scale_type = "a", opt = "L-BFGS")
#'   # Use the Spectral Directions method
#'   res <- sneer(iris, scale_type = "a", opt = "SPEC")
#'
#'   # Use Conjugate Gradient
#'   res <- sneer(iris, scale_type = "a", opt = "CG")
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
#'
#'   # DHSSNE is a "dynamic" extension to HSSNE which will modify alpha from
#'   # its starting point, similar to how it-SNE works (except there's
#'   # only one global value being optimized)
#'   # Setting alpha simply chooses the initial value
#'   res <- sneer(iris, method = "dhssne", alpha = 0.5)
#'
#'   # Can make other embedding methods "dynamic" in the style of it-SNE and
#'   # DSSNE. Here we let the ASNE output kernel have different precision
#'   # parameters:
#'   res <- sneer(iris, method = "asne", dyn = list(beta = "point"))
#'
#'   # DHSSNE could be defined manually like this: alpha is optimized as a single
#'   # global parameter, while the beta parameters are not optimized
#'   res <- sneer(iris, method = "hssne",
#'                dyn = list(alpha = "global", beta = "static"))
#'
#'   # Allow both alpha and beta in the heavy-tailed function to vary per-point:
#'   res <- sneer(iris, method = "hssne",
#'                dyn = list(alpha = "point", beta = "point"))
#'
#'   # it-SNE has a similar degree of freedom parameter to HSSNE's alpha, but
#'   # applies independently to each point and is optimized as part of the
#'   # embedding.
#'   # Setting dof chooses the initial value (1 is like t-SNE, large values
#'   # approach ASNE)
#'   # kernel_opt_iter sets how many iterations with just coordinate
#'   # optimization before including dof optimization too.
#'   res <- sneer(iris, method = "itsne", dof = 10,
#'                dyn = list(kernel_opt_iter = 50))
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
#'                quality_measures =  c("rnxauc"))
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
#'                quality_measures =  c("rnx", "roc", "pr"))
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
#'   # export per-point error, degree centrality, input weight function
#'   # precision parameters and intrinsic dimensionality
#'   res <- sneer(iris, scale_type = "a", method = "wtsne",
#'                ret = c("pcost", "deg", "prec", "dim"))
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
#'   # Per-point embedding error
#'   embed_plot(res$coords, x = res$pcost)
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
                  dof = 10,
                  dyn = c(),
                  lambda = 0.5,
                  kappa = 0.5,
                  scale_type = "none",
                  perplexity = 32, perp_scale = "single",
                  perp_scale_iter = NULL,
                  perp_kernel_fun = "exp",
                  prec_scale = "none",
                  init = "pca",
                  opt = "L-BFGS",
                  eta = 1,
                  max_iter = 1000,
                  report_every = 50,
                  tol = 1e-4,
                  exaggerate = NULL,
                  exaggerate_off_iter = 50,
                  plot_type = "plot",
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

  if (!methods::is(df, "dist") && !methods::is(df, "data.frame")) {
    stop("df should be a data frame or dist object")
  }
  if (!methods::is(df, "dist") && is.null(indexes)) {
    indexes <- which(vapply(df, is.numeric, logical(1)))
    message("Found ", length(indexes), " numeric columns")
  }

  if (any(alpha < 0)) {
    stop("alpha must be non-negative")
  }
  if (any(dof < 0)) {
    stop("dof must be non-negative")
  }

  kernel_opt_iter <- 50
  if (!is.null(dyn) && (!is.null(dyn$kernel_opt_iter))) {
    kernel_opt_iter <- dyn$kernel_opt_iter
    if (kernel_opt_iter < 0) {
      stop("kernel_opt_iter must be non-negative")
    }
  }

  alt_opt <- TRUE
  if (!is.null(dyn) && (!is.null(dyn$alt_opt))) {
    alt_opt <- dyn$alt_opt
  }

  normalize_cost <- TRUE

  embed_methods <- list(
    pca = function() { mmds() },
    mmds = function() { mmds() },
    sammon = function() { sammon_map() },
    tsne = function() { tsne() },
    ssne = function() { ssne() },
    asne = function() { asne() },
    wtsne = function() { imp_weight_method(tsne()) },
    hssne = function() { hssne(alpha = alpha) },
    nerv = function() { nerv(lambda = lambda) },
    jse = function() { jse(kappa = kappa) },
    tasne = function() { tasne() },
    itsne = function() { itsne(dof = dof, opt_iter = kernel_opt_iter,
                               alt_opt = alt_opt) },
    dhssne = function() { dhssne(alpha = alpha, opt_iter = kernel_opt_iter,
                                 alt_opt = alt_opt) }
  )

  extra_costs <- NULL
  # Usually, method is a name of a method that needs to be created
  if (methods::is(method, "character")) {
    method <- tolower(method)
    if (!method %in% names(embed_methods)) {
      stop("Unknown embedding method '",
           method,
           "', should be one of: ",
           paste(
             Filter(function(x) { !endsWith(x, '_plugin') },
                    names(embed_methods)),
             collapse = ", "))
    }
    embed_method <- embed_methods[[method]]()

    # special casing for different methods
    if (method == "pca") {
      max_iter <- -1
      perplexity <- NULL
      init <- "p"
      if (is.null(extra_costs)) {
        extra_costs <- c("kruskal_stress")
      }
    }
    else if (method == "mmds" || method == "sammon") {
      perplexity <- NULL
      if (is.null(extra_costs)) {
        extra_costs <- c("kruskal_stress")
      }
      if (method == "sammon") {
        normalize_cost <- FALSE
      }
    }
  }
  else {
    # Allow masters of the dark arts to pass in a method directly
    embed_method <- method
    if (method$kernel$name == "none") {
      perplexity <- NULL
    }
  }
  embed_method$verbose <- TRUE

  # Don't check convergence until we've finished exaggeration or
  # perplexity scaling or until we've started kernel optimization, whichever
  # is later
  convergence_iter <- 0

  if (!is.null(dyn)) {
    embed_method$dynamic_kernel <- TRUE
    embed_method$dyn <- dyn
    embed_method$opt_iter <- kernel_opt_iter
    embed_method$switch_iter <- kernel_opt_iter
    embed_method$xi_eps <- .Machine$double.eps
    embed_method$alt_opt <- alt_opt
  }
  convergence_iter <- max(convergence_iter, kernel_opt_iter)


  preprocess <- make_preprocess()
  if (!is.null(scale_type)) {
    scale_type <- match.arg(tolower(scale_type),
                            c("none", "matrix", "range", "auto"))
    preprocess <- switch(scale_type,
                         none = make_preprocess(),
                         matrix = make_preprocess(range_scale_matrix = TRUE),
                         range = make_preprocess(range_scale = TRUE),
                         auto = make_preprocess(auto_scale = TRUE)
    )
  }

  ok_rets <- c("x", "dx", "dy", "p", "q", "w", "prec", "dim", "deg", "degs",
               "v", "dyn", "pcost", "method")
  ret <- unique(ret)
  for (r in (ret)) {
    match.arg(tolower(r), ok_rets)
    if (r == "v") {
      embed_method$keep_inp_weights <- TRUE
    }
  }

  init_inp <- NULL
  if (!is.null(perplexity)) {

    if (!is.null(perp_kernel_fun)) {
      perp_kernel_fun <- match.arg(tolower(perp_kernel_fun),
                                   c("exp", "step", "sqrt_exp"))
      weight_fn <- switch(perp_kernel_fun,
                          exp = exp_weight,
                          step = step_weight,
                          sqrt_exp = sqrt_exp_weight
      )
    }

    prec_scale <- match.arg(tolower(prec_scale), c("none", "scale", "transfer"))
    modify_kernel_fn <- NULL
    if (prec_scale != "none") {
      if (perp_kernel_fun == "step") {
        stop("Can't use precision scaling with step input weight function")
      }
      modify_kernel_fn <- switch(prec_scale,
                                 scale = scale_prec_to_perp,
                                 transfer = transfer_kernel_precisions
      )
    }

    if (!is.null(perp_scale) && methods::is(perp_scale, "function")) {
      init_inp <- perp_scale
    }
    else {
      if (!is.null(perp_scale) && perp_scale != "single") {
        perp_scale <- match.arg(tolower(perp_scale),
                                c("single", "max", "multi", "multil", "step"))
        if (!is.null(embed_method$extra_gr) &&
            perp_scale %in% c("multi", "multil")) {
          stop("Multiscaling perplexities is incompatible with embedding ",
               "method '", method, "'")
        }
        if (is.null(perp_scale_iter)) {
          perp_scale_iter <- ceiling(max_iter / 5)
        }
        else {
          if (perp_scale_iter > max_iter) {
            stop("Parameter perp_scale_iter must be <= max_iter")
          }
        }
        switch(perp_scale,
               max = {
                 if (length(perplexity) == 1) {
                   perplexity <- ms_perps(df)
                 }
                 init_inp <- inp_from_dint_max(
                   perplexities = perplexity,
                   modify_kernel_fn = modify_kernel_fn,
                   input_weight_fn = weight_fn)
               },
               multi = {
                 if (length(perplexity) == 1) {
                   perplexity <- ms_perps(df)
                 }
                 init_inp <- inp_from_perps_multi(
                   perplexities = perplexity,
                   num_scale_iters = perp_scale_iter,
                   modify_kernel_fn = modify_kernel_fn,
                   input_weight_fn = weight_fn)
               },
               multil = {
                 if (length(perplexity) == 1) {
                   perplexity <- ms_perps(df)
                 }
                 init_inp <- inp_from_perps_multil(
                   perplexities = perplexity,
                   num_scale_iters = perp_scale_iter,
                   modify_kernel_fn = modify_kernel_fn,
                   input_weight_fn = weight_fn
                   )
               },
               step = {
                 if (length(perplexity) == 1) {
                   perplexity = step_perps(df)
                 }
                 init_inp <- inp_from_step_perp(
                   perplexities = perplexity,
                   num_scale_iters = perp_scale_iter,
                   modify_kernel_fn = modify_kernel_fn,
                   input_weight_fn = weight_fn)
               }
        )
      }
      else {
        # no perplexity scaling asked for
        if (length(perplexity) == 1) {
          if (methods::is(df, "dist")) {
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
          init_inp <- inp_from_perp(
            perplexity = perplexity,
            modify_kernel_fn = modify_kernel_fn,
            input_weight_fn = weight_fn)
        }
        else {
          stop("Must provide 'perp_scale' argument if using multiple perplexity ",
               "values")
        }
      }
    }
  }
  convergence_iter <- max(convergence_iter, perp_scale_iter)

  if (methods::is(init, "matrix")) {
    init_config <- init
    init <- "matrix"
  }
  init <- match.arg(tolower(init),
                    c("pca", "random", "uniform", "matrix"))

  init_out <- switch(init,
                     pca = out_from_PCA(k = ndim),
                     random = out_from_rnorm(k = ndim),
                     uniform = out_from_runif(k = ndim),
                     matrix = out_from_matrix(init_config = init_config, k = ndim)
  )

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

    plot_type <- match.arg(tolower(plot_type), c("none", "ggplot2", "plot"))
    switch(plot_type,
           ggplot2 = {
             if (!requireNamespace("ggplot2", quietly = TRUE,
                                   warn.conflicts = FALSE)) {
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
           },
           plot = {
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
    )
  }

  after_embed <- NULL
  if (!is.null(quality_measures)) {
    qs <- c()

    for (name in unique(quality_measures)) {
      name <- match.arg(tolower(name), c("rocauc", "prauc", "rnxauc"))
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

  tricks <- NULL
  if (!is.null(exaggerate)) {
    tricks <- make_tricks(early_exaggeration(exaggeration = exaggerate,
                                             off_iter = exaggerate_off_iter,
                                             verbose = TRUE))
    convergence_iter <- max(convergence_iter, exaggerate_off_iter)
  }

  # Ensure that if Spectral Direction optimizer is chosen, it can be used with
  # the chosen embedding method
  if (methods::is(opt, "character") && opt == "SPEC" &&
      (is.null(embed_method$prob_type) || embed_method$prob_type != "joint")) {
    stop("Spectral direction optimizer is only compatible with ",
         "probability-based embedding methods that use symmetric input ",
         "probabilities (e.g. t-SNE), not '", method, "'")
  }
  optimizer <- opt_sneer(opt, embed_method)

  if (methods::is(df, "dist")) {
    xm <- df
  }
  else {
    xm <- df[, indexes]
  }

  optimizer$convergence_iter <- convergence_iter
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
      reltol = tol,
      convergence_iter = convergence_iter
    ),
    after_embed = after_embed,
    max_iter = max_iter,
    export = c("report", "inp", "out", "method")
  )

  result <- list(coords = embed_result$ym,
                 cost = embed_result$cost)
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

  for (r in (ret)) {
    match.arg(tolower(r), ok_rets)
    switch(r,
      method = {
        result$method <- embed_result$method
      },
      pcost = {
        if (!is.null(embed_result$method$cost$point)) {
          result$pcost <- embed_result$method$cost$point(inp, out, embed_result$method)
        }
      },
      x = {
       if (!is.null(inp$xm)) {
         result$x <- inp$xm
       }
      },
      dx = {
        if (!is.null(inp$dm)) {
          result$dx <- inp$dm
        }
        else if (is.null(inp$dm) && !is.null(inp$xm)) {
          result$dx <- distance_matrix(inp$xm)
        }
      },
      dy = {
        if (!is.null(out$dm)) {
          result$dy <- out$dm
        }
        else if (is.null(out$dm) && !is.null(out$ym)) {
          result$dy <- distance_matrix(out$ym)
        }
      },
      p = {
        if (!is.null(inp$pm)) {
          result$p <- inp$pm
        }
      },
      w = {
        if (!is.null(out$wm)) {
          result$w <- out$wm
        }
      },
      q = {
        if (!is.null(out$qm)) {
          result$q <- out$qm
        }
      },
      prec = {
        if (!is.null(inp$beta)) {
          result$prec <- 2 * inp$beta
        }
      },
      dim = {
        if (!is.null(inp$dims)) {
          result$dim <- inp$dims
        }
      },
      deg = {
        if (!is.null(inp$deg)) {
          result$deg  <- inp$deg
        }
        else {
          if (!is.null(inp$pm)) {
            result$deg <- centrality(inp, embed_result$method)
          }
        }
      },
      degs = {
        if (!is.null(inp$pm)) {
          deg_res <- centralities(inp, embed_result$method)
          result$deg <- deg_res$deg
          result$indeg <- deg_res$indeg
          result$outdeg <- deg_res$outdeg
        }
      },
      v = {
        result$v <- inp$wm
      },
      dyn = {
        if (!is.null(embed_result$method$export_extra_par)) {
          result$dyn <- embed_result$method$export_extra_par(embed_result$method)
        }
        else if (!is.null(embed_result$method$get_extra_par)) {
          result$dyn <- embed_result$method$get_extra_par(embed_result$method)
        }
      }
    )
  }
  result
}

#' Create an Embedding Method
#'
#' Creates an embedding method to be used in the \code{\link{sneer}} function,
#' allowing arbitrary combinations of cost function, kernel and normalization
#' schemes. Several embedding methods from the literature (e.g. SNE, t-SNE, JSE,
#' NeRV) can be created.
#'
#' The \code{cost} parameter is the cost function to minimize, one of:
#'
#' \itemize{
#' \item \code{"KL"} Kullback-Leibler divergence, as used in the asymmetric
#' Stochastic Neighbor Embedding (SNE) method (Hinton and Roweis, 2002) and
#' Symmetric Stochastic Neighbor Embedding (SSNE) method (Cook et al., 2007),
#' and t-distributed SNE (van der Maaten and Hinton,, 2008).
#' \item \code{"reverse-KL"} Kullback-Leibler divergence, with the output
#' probability as the reference distribution. Part of the cost function used in
#' the Neighbor Retrieval Visualizer (NeRV) method (Venna et al., 2010).
#' \item \code{"nerv"} Cost function used in the (NeRV) method (Venna et al.,
#' 2010).
#' \item \code{"JS"} Jensen-Shannon divergence, as used in the Jensen-Shannon
#' Embedding (JSE) method (Lee et al., 2013).
#' }
#'
#' The \code{transform} will carry out a transformation on the distances. One
#' of:
#' \itemize{
#' \item code{"none"} No transformation. As used in distance-based embeddings
#' such as metric MDS.
#' \item code{"square"} Square the distances. As used in probablity-based
#' embeddings (e.g. t-SNE).
#' }
#'
#' The \code{kernel} is a function to convert the transformed output distances
#' into weights. Must be one of:
#'
#' \itemize{
#' \item \code{"exponential"} Exponential function as used in the asymmetric
#' Stochastic Neighbor Embedding (SNE) method (Hinton and Roweis, 2002) and
#' Symmetric Stochastic Neighbor Embedding (SSNE) method (Cook et al., 2007).
#' \item \code{"t-distributed"} The t-distribution with one degree of freedom,
#' as used in t-distributed SNE (van der Maaten and Hinton,, 2008).
#' \item \code{"heavy-tailed"}. Heavy-tailedness function used in Heavy-tailed
#' SSNE (Zhang et al. 2009).
#' \item \code{"inhomogeneous"}. The function used in inhomogeneous t-SNE
#' (Kitazono et al. 2016).
#' }
#'
#' The \code{norm} determines how weights are converted to probabilities. Must
#' be one of:
#'
#' \itemize{
#'   \item \code{"none"} No normalization, as used in metric MDS. Only the
#'   \code{"square"} and \code{"kl"} \code{cost} functions are compatible with
#'   this option.
#'   \item \code{"point"} Point-wise normalization, as used in asymmetric SNE,
#'   NeRV and JSE.
#'   \item \code{"pair"} Pair-wise normalization.
#'   \item \code{"joint"} Pair-wise normalization, plus enforcing the
#'   probabilities to be joint by averaging, as used in symmetric SNE and
#'   t-distributed SNE. Output probabilities will only be averaged if the
#'   \code{kernel} has non-uniform parameters.
#' }
#'
#' You may also specify a vector of size 2, where the first member is the input
#' normalization, and the second the output normalization. This should only be
#' used to mix \code{"pair"} and \code{"joint"} normalization schemes.
#'
#' @param cost The cost function to optimize. See 'Details'. Can be abbreviated.
#' @param kernel The function used to convert squared distances to weights. See
#'   'Details'. Can be abbreviated.
#' @param transform Transformation to apply to distances before applying
#'   \code{kernel}. See 'Details'. Can be abbreviated.
#' @param beta Precision (narrowness) of the \code{"exponential"} and
#'   \code{"heavy-tailed"} kernels.
#' @param alpha Heavy tailedness of the \code{"heavy-tailed"} kernel. A value of
#'   0 makes the kernel behave like \code{"exponential"}, and a value of 1
#'   behaves like \code{"heavy-tailed"}.
#' @param dof Degrees of freedom of the \code{"inhomogeneous"} kernel. A value
#'   of 1 makes the kernel behave like \code{"t-distributed"}, and a value
#'   approaching approaching infinity behaves like \code{"exponential"}.
#' @param norm Weight normalization to carry out. See 'Details'. Can be
#'   abbreviated.
#' @param lambda Controls the weighting of the \code{"nerv"} cost function. Must
#'   take a value between 0 (where it behaves like \code{"reverse-KL"}) and 1
#'   (where it behaves like \code{"KL"}).
#' @param kappa Controls the weighting of the \code{"js"} cost function. Must
#'   take a value between 0 (where it behaves like \code{"KL"}) and 1 (where it
#'   behaves like \code{"reverse-KL"}).
#' @param importance_weight If \code{TRUE}, modify the embedder to use the
#'   importance weighting method (Yang et al. 2014).
#' @param verbose If \code{TRUE}, log information about the embedding method to
#'   the console.
#' @return An embedding method, to be passed as an argment to the \code{method}
#'   parameter of \code{\link{sneer}}.
#'
#' @references Cook, J., Sutskever, I., Mnih, A., & Hinton, G. E. (2007).
#' Visualizing similarity data with a mixture of maps. In \emph{International
#' Conference on Artificial Intelligence and Statistics} (pp. 67-74).
#'
#' Hinton, G. E., & Roweis, S. T. (2002). Stochastic neighbor embedding. In
#' \emph{Advances in neural information processing systems} (pp. 833-840).
#'
#' Kitazono, J., Grozavu, N., Rogovschi, N., Omori, T., & Ozawa, S. (2016,
#' October). t-Distributed Stochastic Neighbor Embedding with Inhomogeneous
#' Degrees of Freedom. In \emph{International Conference on Neural Information
#' Processing (ICONIP 2016)} (pp. 119-128). Springer International Publishing.
#'
#' Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013). Type
#' 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in
#' dimensionality reduction based on similarity preservation.
#' \emph{Neurocomputing}, \emph{112}, 92-108.
#'
#' Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE.
#' \emph{Journal of Machine Learning Research}, \emph{9}(2579-2605).
#'
#' Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010). Information
#' retrieval perspective to nonlinear dimensionality reduction for data
#' visualization. \emph{Journal of Machine Learning Research}, \emph{11},
#' 451-490.
#'
#' Yang, Z., King, I., Xu, Z., & Oja, E. (2009). Heavy-tailed symmetric
#' stochastic neighbor embedding. In \emph{Advances in neural information
#' processing systems} (pp. 2169-2177).
#'
#' Yang, Z., Peltonen, J., & Kaski, S. (2014). Optimization equivalence of
#' divergences improves neighbor embedding. In \emph{Proceedings of the 31st
#' International Conference on Machine Learning (ICML-14)} (pp. 460-468).
#'
#' @seealso For literature embedding methods, \code{\link{sneer}} will generate
#'   the method for you, by passing its name (e.g. \code{method = "tsne"}). This
#'   function is only strictly necessary for experimentation purposes.
#'
#' @examples
#' # t-SNE
#' embedder(cost = "kl", kernel = "t-dist", norm = "joint")
#'
#' # NeRV
#' embedder(cost = "nerv", kernel = "exp", norm = "point")
#'
#' # JSE
#' embedder(cost = "JS", kernel = "exp", norm = "point")
#'
#' # weighted SSNE
#' embedder(cost = "kl", kernel = "exp", norm = "joint", importance_weight = TRUE)
#'
#' # SSNE where the input probabilities are averaged, but output probabilites
#' # are not. This only has an effect if the kernel parameters are set to be
#' # non-uniform.
#' embedder(cost = "kl", kernel = "exp", norm = c("joint", "pair"))
#'
#' # MDS
#' embedder(cost = "square", transform = "none", kernel = "none", norm = "none")
#'
#' # un-normalized version of t-SNE
#' embedder(cost = "kl", kernel = "t-dist", norm = "none")
#'
#' \dontrun{
#' # Pass result of calling embedder to the sneer function's method parameter
#' sneer(iris, method = embedder(cost = "kl", kernel = "t-dist", norm = "joint"))
#' }
#' @export
embedder <- function(cost, kernel, transform = "square",
                     kappa = 0.5, lambda = 0.5,
                     beta = 1, alpha = 0, dof = 10,
                     norm = "joint",
                     importance_weight = FALSE,
                     verbose = TRUE) {
  cost <- match.arg(tolower(cost),
                    c("kl", "revkl", "js", "nerv", "square"))
  cost <- switch(cost,
                 kl = kl_fg(),
                 "reverse-kl" = reverse_kl_fg(),
                 js = jse_fg(kappa = kappa),
                 nerv = nerv_fg(lambda = lambda),
                 square = sum2_fg())

  kernel <- match.arg(tolower(kernel),
                      c("exponential", "heavy-tailed", "inhomogeneous",
                        "t-distributed", "none"))
  kernel <- switch(kernel,
                   exponential = exp_kernel(beta = beta),
                   "t-distributed" = tdist_kernel(),
                   "heavy-tailed" = heavy_tail_kernel(beta = beta,
                                                      alpha = alpha),
                   inhomogeneous = itsne_kernel(dof = dof),
                   none = no_kernel())

  transform <- match.arg(tolower(transform),
                         c("none", "square"))
  transform <- switch(transform,
                      square = d2_transform(),
                      none = no_transform() )

  if (length(norm) > 2) {
    stop("Normalization must contain one or two values only")
  }

  # If we have a kernel but are not using probabilities, swap to row-based
  # probabilities in the input space for calibration
  if (length(norm) == 1 && norm == "none" && kernel$name != "none") {
    norm <- c("point", "none")
    cost$keep_weights <- TRUE
    cost$replace_probs_with_weights <- TRUE
  }

  prob_type <- c()
  out_prob_type <- NULL
  for (n in norm) {
    n <- match.arg(tolower(n), c("none", "point", "pair", "joint"))
    prob_type <- c(prob_type, switch(n,
                                     "none" = "un",
                                     "point" = "row",
                                     "pair" = "cond",
                                     "joint" = "joint"))
  }

  if (any(prob_type == "point") && !all(prob_type == "point")) {
    stop("Can't mix point with non-point normalization")
  }
  if (length(prob_type) == 2) {
    out_prob_type <- prob_type[2]
    prob_type <- prob_type[1]
  }

  # Account for un-normalized version of cost functions
  is_un_norm <- FALSE
  if (!is.null(out_prob_type)) {
    is_un_norm <- out_prob_type == "un"
  }
  else {
    is_un_norm <- prob_type == "un"
  }

  if (is_un_norm) {
    if (cost$name == "KL") {
      cost <- unkl_fg()
    }
    else if (cost$name != sum2_fg()$name) {
      stop("Non-normalized embedders can only be used with the ",
           "'KL' and 'square' cost functions")
    }
  }

  embedder <- prob_embedder(cost = cost, kernel = kernel, transform = transform,
                            prob_type = prob_type,
                            out_prob_type = out_prob_type,
                            eps = .Machine$double.eps, verbose = verbose)

  if (importance_weight) {
    embedder <- imp_weight_method(embedder)
  }
  if (!is.null(embedder$cost$keep_weights) && embedder$cost$keep_weights) {
    embedder$keep_inp_weights <- TRUE
  }
  if (!is.null(embedder$cost$replace_probs_with_weights) &&
      embedder$cost$replace_probs_with_weights) {
    embedder$replace_probs_with_weights <- TRUE
  }
  embedder
}


opt_sneer <- function(opt, method, eta = 500) {
  if (methods::is(opt, "function")) {
    return(opt())
  }

  if (methods::is(opt, "list")) {
    return(mize_opt(opt))
  }

  opt <- tolower(opt)
  if (!is.null(method$alt_opt) && method$alt_opt) {
    ctor <- mize_opt_alt
  }
  else {
    ctor <- mize_opt
  }

  if (opt == "tsne") {
    optimizer <- ctor(
      "DBD",
      step_up_fun = "+", step_up = 0.2, step_down = 0.8, step0 = eta,
      mom_type = "classical", mom_schedule = "switch",
      mom_init = 0.4, mom_final = 0.8, mom_switch_iter = 250
    )
  }
  else if (opt == "nest") {
    optimizer <- ctor(
      "SD", norm_direction = TRUE,
      line_search = "bold", step0 = eta,
      mom_schedule = "nesterov", mom_type = "nesterov",
      nest_convex_approx = FALSE, nest_burn_in = 1,
      use_nest_mu_zero = FALSE, restart = "fn")
  }
  else if (opt == "l-bfgs") {
    optimizer <- ctor(
      "L-BFGS", c1 = 1e-4, c2 = 0.1,
      abs_tol = 0, rel_tol = 0, step_tol = NULL,
      step_next_init = "quad", line_search = "mt",
      step0 = "ras")
  }
  else if (opt == "bfgs") {
    optimizer <- ctor(
      "BFGS", c1 = 1e-4, c2 = 0.9,
      step0 = "scipy", step_next_init = "quad")
  }
  else if (opt == "spec") {
    optimizer <- ctor(
      "PHESS", c1 = 1e-4, c2 = 0.9,
      step0 = "scipy", step_next_init = "quad", try_newton_step = TRUE)
  }
  else if (opt == "cg") {
    optimizer <- ctor(
      "CG", c1 = 1e-4, c2 = 0.1,
      step0 = "rasmussen", step_next_init = "slope ratio")
  }
  else {
    stop("Unknown optimization method '", opt, "'")
  }

  optimizer$convergence_iter <- 0

  optimizer
}
