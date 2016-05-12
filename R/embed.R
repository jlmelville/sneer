# Functions to do the embedding. The interface to sneer. Also, core
# functions to do with gradient calculation, optimization, updating and so on
# that don't need to be overridden or changed.

#' Embedding Methods
#'
#' Links to all available embedding methods.
#'
#' @examples
#' \dontrun{
#' embed_prob(method = tsne())
#' embed_dist(method = mmds())
#' }
#' @keywords internal
#' @name embedding_methods
#' @family sneer embedding methods
NULL

#' Probability-based Embedding
#'
#' Carries out an embedding of a high-dimensional dataset into a two dimensional
#' scatter plot, based on distance-based methods (e.g. Sammon maps) and
#' probability-based methods (e.g. t-distributed Stochastic Neighbor Embedding).
#'
#' @details
#'
#' The embedding methods available are:
#' \itemize{
#'  \item{\code{"pca"}}{The first two principal components.}
#'  \item{\code{"mmds"}}{Metric multidimensional scaling.}
#'  \item{\code{"sammon"}}{Sammon map.}
#'  \item{\code{"tsne"}}{t-Distributed Stochastic Neighbor Embedding of van der
#'   Maaten and Hinton (2008).}
#'  \item{\code{"asne"}}{Asymmetric Stochastic Neighbor Embedding of Hinton and
#'   Roweis (2002)}
#'  \item{\code{"ssne"}}{Symetric Stochastic Neighbor Embedding of Cook et al
#'   (2007).}
#'  \item{\code{"wssne"}}{Weighted Symmetric Stochastic Neighbor Embedding of
#'   Yang et al (2014). Note that despite its name this version is a
#'   modification of t-SNE, not SSNE.}
#'  \item{\code{"hssne"}}{Heavy-tailed Symmetric Stochastic Neighbor Embedding of
#'   Yang et al (2009).}
#'  \item{\code{"nerv"}}{Neighbor Retrieval Visualizer of Venna et al (2010).}
#'  \item{\code{"jse"}}{Jensen-Shannon Embedding of Lee at al (2013).}
#' }
#'
#' The following scaling options can be applied via the \code{scale_type}
#' parameter:
#' \itemize{
#'  \item{\code{"m"}}{Range scale the entire data so that the maximum value is
#'   1 and the minimum 0.}
#'  \item{\code{"r"}}{Range scale each column that the maximum value in each
#'   column is 1 and the minimum 0.}
#'  \item{\code{"a"}}{Scale each column so that its mean is 0 and variance is
#'   1.}
#' }
#' Default is to do no scaling. Zero variance columns will be removed even if no
#' preprocessing is carried out.
#'
#' The \code{perplexity} parameter is used in combination with the
#' \code{perp_scale} parameter, which can take the following values:
#' \itemize{\item{"single"}}{\code{perplexity} should be a single value, which
#' will be used over the entire course of the embedding.}
#' \itemize{\item{"step"}}{\code{perplexity} should be a vector of
#' perplexity values. Each perplexity will be used in turn over the course of
#' the embedding, in sequential order. By starting with a large perplexity, and
#' ending with the desired perplexity, it has been suggested by some
#' researchers that local minima can be avoided.}
#' \itemize{\item{"multi"}}{The multiscaling method of Lee et al (2015).
#' \code{perplexity} should be a vector of  perplexity values. Each perplexity
#' will be used in turn over the course of the embedding, in sequential order.
#' Unlike with the \code{"step"} method, probability matrices from earlier
#' perplexities are retained and combined by averaging.}
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
#' parameters should be used to modify the output kernel parameter after the
#' input probability calculation for a given perplexity value completes.
#' values are:
#' \itemize{
#'  \item{"n"}{Do nothing. Most embedding methods follow this strategy, leaving
#'  the output similarity kernels to all have unit precision parameters.}
#'  \item{"t"}{Transfer the input similarity kernel parameters to the
#'  output similarity kernel. This method was suggesed by Venna et al (2010).}
#'  \item{"s"}{Scale the output kernel precisions based on the target
#'  \code{perplexity} and the intrinsic dimensionality of the input data. This
#'  method is part of the multiscaling technique proposed by Lee et al (2015).}
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
#'  \item{\code{"p"}}{Initialize using the first two scores of the PCA.
#'  Data will be centered, but not scaled unless the \code{scale_type} parameter
#'  is used.}
#' \code{"r"}{Initialize each coordinate vlaue from a normal random
#' distribution with a standard deviation of 1e-4, as suggested by van der Maaten
#' and Hinton (2008).}
#' \code{"u"}{Initialize each coordinate value from a uniform random
#' distribution between 0 and 1 as suggested by Venna et al (2010).}
#'   init_out <- NULL
#' \code{"u"}{Initialize the coordinates from a user-supplied matrix. Supply
#' the coordinates as the \code{init_config} parameter.}
#'}
#'
#' For the \code{quality_measures} argument, a vector with one or more of the
#' following options can be supplied:
#' \itemize{
#'  \item{\code{"r"}}{Calculate the area under the ROC curve, averaged over
#'  each observation, using the output distance matrix to rank each
#'  observation. Observations are partitioned into the positive and negative
#'  class depending upon the value of the label determined by the
#'  \code{label_name} argument. Only calculated if the \code{label_name}
#'  parameter is supplied.}
#'  \item{\code{"p"}}{Calculate the area under the Precision-Recall curve.
#'   Only calculated if the \code{label_name} parameter is supplied.}
#'  \item{\code{"n"}}{Calculate the area under the RNX curve, using the
#'  method of Lee et al (2015).}
#' }
#'
#' For the \code{ret} argument, a vector with one or more of the
#' following options can be supplied:
#' \itemize{
#'  \item{\code{"x"}}{The input coordinates after scaling and column filtering.}
#'  \item{\code{"dx"}}{The input distance matrix. Calculated if not present.}
#'  \item{\code{"dy"}}{The output distance matrix. Calculated if not present.}
#'  \item{\code{"p"}}{The input probability matrix.}
#'  \item{\code{"q"}}{The output probability matrix.}
#'  \item{\code{"prec"}}{The input similarity kernel precisions.}
#'  \item{\code{"dim"}}{The intrinsic dimensionality for each observation,
#'  calculated according to the method of Lee et al (2015). These are
#'  meaningless if not using the default exponential \code{perp_kernel_fun}}.
#'  \item{\code{"deg"}}{Degree centrality of the input probability. Calculated
#'  if not present.}
#' }
#'
#' @param df Data frame to embed.
#' @param indexes Indexes of the columns of the numerical variables to use in
#'  the embedding. The default of \code{NULL} will use all the numeric
#'  variables.
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
#' @param max_iter Maximum number of iterations to carry out optimization of
#'  the embedding. Ignored if the \code{method} is \code{"pca"}.
#' @param report_every Frequency (in terms of iteration number) with which to
#'  update plot and report the cost function.
#' @param tol Tolerance for comparing cost change (calculated according to the
#'  interval determined by \code{report_every}). If the change falls below this
#'  value, the optimization stops early.
#' @param plot_type String code indicating the type of plot of the embedding
#'  to display: \code{"p"} to use the usual \code{\link[graphics]{plot}}
#'  function; \code{"g"} to use the \code{ggplot2} package, with the
#'  \code{RColorBrewer} palettes. You are responsible for installing and
#'  loading these packages yourself.
#' @param label_name Name of factor-typed column in \code{df} to be used to
#'  color the points in the embedding plot (for \code{plot_type}
#'  \code{"g"}) or to color the text associated with each plotted observation
#'  {\code{plot_type "p"}}. If not specified, then the first factor column
#'  will be used. If no suitable column can be found, then no plotting
#'  is carried out.
#' @param label_chars Number of characters to use for the labels in the
#'  embedding plot. Applies only when \code{plot_type} is set to \code{"p"}.
#' @param label_size Size of the points in the embedding plot.
#' @param palette Color Brewer palette name to use for coloring points in embedding
#'  plot. Applies to \code{plot_type} type \code{"g"} only.
#' @param legend if \code{TRUE}, display the legend in the embedding plot.
#'  Applies when \code{plot_type} is \code{"g"} only.
#' @param legend_rows Number of rows to use for displaying the legend in
#'  an embedding plot.
#' @param quality_measures Vector of names of quality measures to apply to the
#'  finished embedding. See 'Details'. Values of the quality measures will
#'  be printed to screen after embedding and retained in the list that is
#'  returned from this function.
#' @param ret Vector of names of extra data to return from the embedding. See
#'  'Details',
#' @return List with the following elements:
#' \itemize{
#' \item{\code{coords}}{Embedded coordinates}
#' \item{\code{cost}}{Cost function value for the embedded coordinates. The
#' type of the cost depends on the method, but the lower the better.}
#' \item{\code{norm_cost}}{\code{cost}, normalized so that a perfect embedding
#' gives a value of 0 and one where all the distances were equal would have
#' a value of 1.}
#' \item{\code{method}}{String giving the method used for the embedding.}
#' }
#' Additional elements will be in the list if \code{ret} or
#' \code{quality_measures} are non-empty.
#' @references
#'
#' Hinton, G. E., & Roweis, S. T. (2002).
#' Stochastic neighbor embedding.
#' In \emph{Advances in neural information processing systems} (pp. 833-840).
#'
#' Cook, J., Sutskever, I., Mnih, A., & Hinton, G. E. (2007).
#' Visualizing similarity data with a mixture of maps.
#' In \emph{International Conference on Artificial Intelligence and Statistics} (pp. 67-74).
#'
#' Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013).
#' Type 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in
#' dimensionality reduction based on similarity preservation.
#' \emph{Neurocomputing}, \emph{112}, 92-108.
#'
##' Lee, J. A., Peluffo-Ordo'nez, D. H., & Verleysen, M. (2015).
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
#'   res <- embed(iris, indexes = 1:4, label_name = "Species", method = "pca")
#'   # Same as above, but with sensible defaults (use all numeric columns, plot
#'   # with first factor column found)
#'   res <- embed(iris, method = "pca")
#'   # scale columns so each one has mean 0 and variance 1
#'   res <- embed(iris, method = "pca", scale_type = "a")
#'   # full species name on plot is cluttered, so just use the first two
#'   # letters and half size
#'   res <- embed(iris, method = "pca", scale_type = "a", label_chars = 2,
#'                label_size = 0.5)
#'
#'   library(ggplot2)
#'   library(RColorBrewer)
#'   # Use ggplot2 and RColorBrewer palettes for the plot
#'   res <- embed(iris, method = "pca", scale_type = "a", plot_type = "g")
#'   # Use a different ColorBrewer palette, bigger points, and range scale each
#'   # column
#'   res <- embed(iris, method = "pca", scale_type = "r", plot_type = "g",
#'                palette = "Dark2", label_size = 2)
#'
#'   # metric MDS starting from the PCA
#'   res <- embed(iris, method = "mmds", scale_type = "a", init = "p")
#'   # Sammon map starting from random distribution
#'   res <- embed(iris, method = "sammon", scale_type = "a", init = "r")
#'
#'   # TSNE with a perplexity of 32, initialize from PCA
#'   res <- embed(iris, method = "tsne", scale_type = "a", init = "p",
#'                perplexity = 32)
#'   # default settings are to use TSNE with perplexity 32 and initialization
#'   # from PCA so the following is the equivalent of the above
#'   res <- embed(iris, scale_type = "a")
#'
#'   # NeRV method, starting at a more global perplexity and slowly stepping
#'   # towards a value of 32 (might help avoid local optima)
#'   res <- embed(iris, scale_type = "a", method = "nerv", perp_scale = "step")
#'
#'   # NeRV method has a lambda parameter - closer to 1 it gets, the more it
#'   # tries to avoid false positives (close points in the map that aren't close
#'   # in the input space):
#'   res <- embed(iris, scale_type = "a", method = "nerv", perp_scale = "step",
#'                lambda = 1)
#'
#'   # Original NeRV paper transferred input exponential similarity kernel
#'   # precisions to the output kernel, and initialized from a uniform random
#'   # distribution
#'   res <- embed(iris, scale_type = "a", method = "nerv", perp_scale = "step",
#'                lambda = 1, prec_scale = "t", init = "u")
#'
#'   # Like NeRV, the JSE method also has a controllable parameter that goes
#'   # between 0 and 1, called kappa. It gives similar results to NeRV at 0 and
#'   # 1 but unfortunately the opposite way round! The following gives similar
#'   # results to the NeRV embedding above:
#'   res <- embed(iris, scale_type = "a", method = "jse", perp_scale = "step",
#'                kappa = 0)
#'
#'   # Rather than step perplexities, use multiscaling to combine and average
#'   # probabilities across multiple perplexities. Output kernel precisions
#'   # can be scaled based on the perplexity value (compare to NeRV example
#'   # which transferred the precision directly from the input kernel)
#'   res <- embed(iris, scale_type = "a", method = "jse", perp_scale = "multi",
#'                prec_scale = "s")
#'
#'   # HSSNE has a controllable parameter, alpha, that lets you control how
#'   # much extra space to give points compared to the input distances.
#'   # Setting it to 1 is equivalent to TSNE, so 1.1 is a bit of an extra push:
#'   res <- embed(iris, scale_type = "a", method = "hssne", alpha = 1.1)
#'
#'   # wTSNE treats the input probability like a graph where the probabilities
#'   # are weighted edges and adds extra repulsion to nodes with higher degrees
#'   res <- embed(iris, scale_type = "a", method = "wtsne")
#'
#'   # can use a step-function input kernel to make input probability more like
#'   # a k-nearest neighbor graph (but note that we don't take advantage of the
#'   # sparsity for performance purposes, sadly)
#'   res <- embed(iris, scale_type = "a", method = "wtsne",
#'                perp_kernel_fun = "step")
#'
#'   # Some quality measures are available to quantify embeddings
#'   # The area under the RNX curve measures whether neighbors in the input
#'   # are still neighors in the output space
#'   res <- embed(iris, scale_type = "a", method = "wtsne",
#'                quality_measures =  c("n"))
#'
#'   # If your dataset labels divide the data into natural classes, can
#'   # calculate average area under the ROC and/or precision-recall curve too,
#'   # but you need to have installed the PRROC package.
#'   # All these techniques can be slow (scale with the square of the number of
#'   # observations).
#'   library(PRROC)
#'   res <- embed(iris, scale_type = "a", method = "wtsne",
#'                quality_measures =  c("n", "r", "p"))
#'
#'   # export the distance matrices and do whatever quality measures we
#'   # want at our leisure
#'   res <- embed(iris, scale_type = "a", method = "wtsne", ret = c("dx", "dy"))
#'
#'   # calculate the 32-nearest neighbor preservation for each observation
#'   # 0 means no neighbors preserved, 1 means all of them
#'   pres32 <- nbr_pres(res$dx, res$dy, 32)
#'
#'   # use map2color helper function with diverging or sequential color palettes
#'   # to map values onto the embedded points
#'   plot(res$coords, col = map2color(pres32), pch = 20, cex = 1.5)
#'
#'   # export degree centrality, input beta values and intrinsic dimensionality
#'   res <- embed(iris, scale_type = "a", method = "wtsne",
#'                ret = c("deg", "prec", "dim"))
#'
#'   plot(res$coords, col = map2color(res$deg), pch = 20, cex = 1.5)
#'   plot(res$coords, col = map2color(res$dim, name = "PRGn"), pch = 20,
#'        cex = 1.5)
#'   plot(res$coords, col = map2color(res$prec, name = "Spectral"), pch = 20,
#'        cex = 1.5)
#' }
#' @export
embed <- function(df,
                  indexes = NULL,
                  method = "tsne",
                  alpha = 1,
                  lambda = 0.5,
                  kappa = 0.5,
                  scale_type = "",
                  perplexity = 32, perp_scale = "single",
                  perp_scale_iter = NULL,
                  perp_kernel_fun = "exp",
                  prec_scale = NULL,
                  init = "p", init_config = NULL,
                  max_iter = 1000,
                  report_every = 50,
                  tol = 1e-4,
                  plot_type = "p",
                  label_name = NULL,
                  label_chars = NULL,
                  label_size = 1,
                  palette = "Set1",
                  legend = TRUE,
                  legend_rows = NULL,
                  quality_measures = NULL,
                  ret = c()) {
  if (is.null(indexes)) {
    indexes <- sapply(df, is.numeric)
  }

  if (is.null(label_name)) {
    factor_names <- names(df)[(sapply(df, is.factor))]
    if (length(factor_names) > 0) {
      label_name <- factor_names[length(factor_names)]
      message("Using '", label_name, "' as the label")
    }
    else {
      message("No label found")
    }
  }

  if (is.null(df[[label_name]])) {
    stop("Data frame does not have a '",label_name,
         "' column for use as a label")
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
    ssne_plugin = function() { ssne_plugin() },
    asne_plugin = function() { asne_plugin() },
    hssne_plugin = function() { hssne_plugin(alpha = alpha) },
    nerv_plugin = function() { unerv_plugin(lambda = lambda) },
    jse_plugin = function() { jse_plugin(kappa = kappa) }
  )

  if (!method %in% names(embed_methods)) {
    stop("Unknown embedding method '",
         method,
         "', should be one of: ",
         paste(names(embed_methods), collapse = ", "))
  }

  # Need to use plugin method if precisions can be non-uniform
  if (!is.null(prec_scale) && prec_scale == "t") {
    new_method <- paste0(method, "_plugin")
    if (!new_method %in% names(embed_methods)) {
      stop("Method '", method, "' is not compatible with prec_scale option 't'")
    }
    embed_method <- embed_methods[[new_method]]()
  }
  else {
    embed_method <- embed_methods[[method]]()
  }

  extra_costs <- NULL
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
    weight_fn <- exp_weight
    if (!is.null(perp_kernel_fun) && perp_kernel_fun == "step") {
      weight_fn <- step_weight
    }

    modify_kernel_fn <- NULL
    if (!is.null(prec_scale)) {
      if (perp_kernel_fun == "step") {
        stop("Can't use precision scaling with step input weight function")
      }
      if (prec_scale == "s") {
        modify_kernel_fn <- scale_prec_to_perp
      }
      else if (prec_scale == "t") {
        modify_kernel_fn <- transfer_kernel_bandwidths
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
        if (perplexity >= nrow(df)) {
          perplexity <- nrow(df) / 4
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
    init_out <- out_from_PCA()
  }
  else if (init == "r") {
    init_out <- out_from_rnorm()
  }
  else if (init == "u") {
    init_out <- out_from_runif()
  }
  else if (init == "m") {
    init_out <- out_from_matrix(init_config = init_config)
  }
  else {
    stop("No initialization method '", init, "'")
  }

  embed_plot <- NULL
  if (is.null(plot_type)) { plot_type <- "n" }
  if (!is.null(label_name)) {
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
          attr_name = label_name,
          size = label_size,
          palette = palette,
          legend = legend,
          legend_rows = legend_rows
        )
    }
    else if (plot_type == 'p') {
      if (!is.null(label_chars)) {
        embed_plot <- make_plot(df,
                                label_name,
                                cex = label_size,
                                label_fn = make_label(label_chars))
      }
      else {
        embed_plot <- make_plot(df, label_name, cex = label_size)
      }
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
    if (length(qs) > 0) {
      after_embed <- make_quality_reporter(qs, df[[label_name]])
    }
  }

  ok_rets <- c("x", "dx", "dy", "p", "q", "prec", "dim", "deg")
  ret <- unique(ret)
  for (r in (ret)) {
    if (!r %in% ok_rets) {
      stop("Invalid return name: '", r,"'")
    }
  }

  embed_result <- embed_main(
    df[, indexes],
    method = embed_method,
    opt = bold_nag_adapt(),
    preprocess = preprocess,
    init_inp = init_inp,
    init_out = init_out,
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
  colnames(result$coords) <- c("X", "Y")

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
  result$inp <- inp
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
    else if (r == "q") {
      if (!is.null(out$qm)) {
        result$q <- out$qm
      }
    }
    else if (r == "prec") {
      if (!is.null(inp$beta)) {
        result$prec <- inp$beta
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
        result$deg <- centrality(inp, embed_result$method)
      }
    }
  }

  result
}

#' Probability-Based Embedding
#'
#' Carry out an embedding of a dataset using a probability-based method
#' (e.g. t-SNE), with some useful default parameters.
#'
#' @param xm A matrix or data frame to embed.
#' @param method Embedding method. Set by assigning the result value of one of
#'   the configuration functions listed in
#'   \code{\link{probability_embedding_methods}}.
#' @param preprocess Input data preprocess callback. Set by assigning the
#'   result value of \code{\link{make_preprocess}}.
#' @param init_inp Input initializer. Set by assigning the result value of
#'   calling one of the available \code{\link{input_initializers}}.
#' @param init_out Output initializer. Set by assigning the result value of
#'   calling one of the available \code{\link{output_initializers}}.
#' @param opt Optimization method. Set by assigning the result of value of one
#'   of the configuration functions listed in
#'   \code{\link{optimization_methods}}.
#' @param max_iter Maximum number of optimization steps to take.
#' @param tricks Optional collection of heuristics. Set by assigning the result
#'  value of \code{\link{make_tricks}} or a related wrapper. See
#'  \code{\link{tricks}} for the available tricks.
#' @param reporter Reporter callback. Set by assigning the result value of
#'   \code{\link{make_reporter}}.
#' @param export Vector of names, specifying data in addition to of the output
#'   data to export provide in the return value. Possible names are:
#'   \describe{
#'     \item{"\code{inp}"}{The input data, based on the argument passed to
#'       \code{init_inp}. See the help text for the specific
#'       \code{\link{input_initializers}} for details.}
#'     \item{"\code{report}"}{The result of the last report. See the help
#'       text for \code{\link{make_reporter}} for details.}
#'     \item{"\code{opt}"}{The optimizer. See the help text for the optimizer
#'     passed to \code{opt} for details.}
#'   }
#'   For each name supplied, an item with that name will appear in the return
#'   value list.
#' @param after_embed Callback to run on input and output data before output
#'   data is returned.
#' @param verbose If \code{TRUE} display messages about the embedding progress.
#' @return The output data. A list containing:
#'   \item{\code{ym}}{Embedded coordinates.}
#'   \item{\code{cost}}{The cost value associated with \code{ym}.}
#'   If the parameter \code{export} was used, additional elements will be
#'   present. See the help text for the \code{export} parameter for more
#'   details.
#' @seealso
#' \itemize{
#' \item{\code{\link{probability_embedding_methods}}} for configuring
#'   \code{method}
#' \item{\code{\link{make_preprocess}}} for configuring \code{preprocess}
#' \item{\code{\link{input_initializers}}} for configuring \code{init_inp}
#' \item{\code{\link{output_initializers}}} for configuring \code{init_out}
#' \item{\code{\link{optimization_methods}}} for configuring \code{opt}
#' \item{\code{\link{make_tricks}}} for configuring \code{tricks}
#' \item{\code{\link{make_reporter}}} for configuring \code{reporter}
#' }
#'
#' @examples
#' \dontrun{
#' # Do t-SNE on the iris dataset with the same options as the t-SNE paper
#' # except initialize from PCA so output is repeatable.
#' # plot 2D result during embedding with convenience function for iris plot.
#' # Default method is tsne. Set perplexity to 25.
#' tsne_iris <- embed_prob(iris[, 1:4], opt = tsne_opt(),
#'                init_inp = inp_from_perp(perplexity = 25),
#'                tricks = tsne_tricks(),
#'                reporter = make_reporter(plot = make_iris_plot()))
#'
#' # Do t-SNE on the iris dataset with the same options as the t-SNE paper
#' # and initialize from a random normal distribution. Use generic plot
#' # function, displaying the first two characters of the "Species" factor for
#' # the points. Explicitly choose t-SNE as the method.
#' tsne_iris <- embed_prob(iris[, 1:4],
#'                method = tsne(),
#'                opt = tsne_opt(),
#'                init_inp = inp_from_perp(perplexity = 25),
#'                init_out = out_from_rnorm(sd = 1e-4),
#'                tricks = tsne_tricks(),
#'                reporter = make_reporter(
#'                  plot = make_plot(iris, "Species", make_label(2))))
#'
#' # Use the SSNE method, and preprocess input data by range scaling. t-SNE
#' # tricks and optimization are reasonable defaults for other probability-based
#' # embeddings. Initialize from a uniform distribution.
#' ssne_iris <- embed_prob(iris[, 1:4],
#'                method = ssne(),
#'                opt = tsne_opt(),
#'                init_inp = inp_from_perp(perplexity = 25),
#'                preprocess = make_preprocess(range_scale_matrix = TRUE),
#'                init_out = out_from_runif(),
#'                tricks = tsne_tricks(),
#'                reporter = make_reporter(
#'                  plot = make_plot(iris, "Species", make_label(2))))
#'
#' # ASNE method on the s1k dataset (10 overlapping 9D Gaussian blobs),
#' # Set perplexity for input initialization to 50, initialize with PCA scores,
#' # preprocess by autoscaling columns, optimize
#' # with Nesterov Accelrated Gradient and bold driver step size
#' # (highly recommended as an optimizer). Labels for s1k are one digit, so
#' # can use simplified plot function.
#' asne_s1k <- embed_prob(s1k[, 1:9], method = asne(),
#'  preprocess = make_preprocess(auto_scale = TRUE),
#'  init_inp = inp_from_perp(perplexity = 50),
#'  init_out = out_from_PCA(),
#'  opt = make_opt(gradient = nesterov_gradient(), step_size = bold_driver(),
#'   update = nesterov_nsc_momentum()),
#'   reporter = make_reporter(plot = make_plot(s1k, "Label")))
#'
#' # Same as above, but using convenience method to create optimizer with less
#' # typing
#' asne_s1k <- embed_prob(s1k[, 1:9], method = asne(),
#'  preprocess = make_preprocess(auto_scale = TRUE),
#'  init_inp = inp_from_perp(perplexity = 50),
#'  init_out = out_from_PCA(),
#'  opt = bold_nag(),
#'  reporter = make_reporter(plot = make_plot(s1k, "Label")))
#' }
#' @family sneer embedding functions
#' @export
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

#' Distance-Based Embedding.
#'
#' Carry out an embedding of a dataset using a distance-based method
#' (e.g. Sammon Mapping), with some useful default parameters.
#'
#' @param xm A matrix or data frame to embed.
#' @param method Embedding method. Set by assigning the result value of one of
#'   the configuration functions listed in
#'   \code{\link{distance_embedding_methods}}.
#' @param preprocess Input data preprocess callback. Set by assigning the
#'   result value of \code{\link{make_preprocess}}.
#' @param init_inp Input initializer. Set by assigning the result value of
#'   calling one of the available \code{\link{input_initializers}}.
#' @param init_out Output initializer. Set by assigning the result value of
#'   calling one of the available \code{\link{output_initializers}}.
#' @param opt Optimization method. Set by assigning the result of value of one
#'   of the configuration functions listed in
#'   \code{\link{optimization_methods}}.
#' @param max_iter Maximum number of optimization steps to take.
#' @param tricks Optional collection of heuristics. Set by assigning the result
#'  value of \code{\link{make_tricks}} or a related wrapper. See
#'  \code{\link{tricks}} for the available tricks.
#' @param reporter Reporter callback. Set by assigning the result value of
#'   \code{\link{make_reporter}}.
#' @param export Vector of names, specifying data in addition to of the output
#'   data to export provide in the return value. Possible names are:
#'   \describe{
#'     \item{"\code{inp}"}{The input data, based on the argument passed to
#'       \code{init_inp}. See the help text for the specific
#'       \code{\link{input_initializers}} for details.}
#'     \item{"\code{report}"}{The result of the last report. See the help
#'       text for \code{\link{make_reporter}} for details.}
#'     \item{"\code{opt}"}{The optimizer. See the help text for the optimizer
#'     passed to \code{opt} for details.}
#'   }
#' @param after_embed Callback to run on input and output data before output
#'   data is returned.
#' @param verbose If \code{TRUE} display messages about the embedding progress.
#' @return The output data. A list containing:
#'   \item{\code{ym}}{Embedded coordinates.}
#'   \item{\code{cost}}{The cost value associated with \code{ym}.}
#'   If the parameter \code{export} was used, additional elements will be
#'   present. See the help text for the \code{export} parameter for more
#'   details.
#' @seealso
#' \itemize{
#' \item{\code{\link{distance_embedding_methods}}} for configuring
#'   \code{method}
#' \item{\code{\link{make_preprocess}}} for configuring \code{preprocess}
#' \item{\code{\link{input_initializers}}} for configuring \code{init_inp}
#' \item{\code{\link{output_initializers}}} for configuring \code{init_out}
#' \item{\code{\link{optimization_methods}}} for configuring \code{opt}
#' \item{\code{\link{make_tricks}}} for configuring \code{tricks}
#' \item{\code{\link{make_reporter}}} for configuring \code{reporter}
#' }
#'
#' @examples
#' \dontrun{
#' # Do metric MDS on the iris data set
#' # In addition to the STRESS loss function also report the Kruskal Stress
#' # (often used in MDS applications) and the mean relative error, which can
#' # be multiplied by 100 and interpreted as a percentage error. Also, use
#' # the make_iris_plot function, which wrap the make_plot function specifically
#' # for the iris dataset, which is quite handy for testing.
#' mds_iris <- embed_dist(iris[, 1:4],
#'                        method = mmds(),
#'                        opt = bold_nag(),
#'                        reporter = make_reporter(
#'                          extra_costs = c("kruskal_stress",
#'                                          "mean_relative_error")),
#'                                          plot = make_iris_plot())
#'
#' # Sammon map the autoscaled iris data set, which turns out to be a
#' # surprisingly tough assignment. Increase epsilon substantially to 1e-4 to
#' # avoid the gradient being overwhelmed by zero distances in the input space.
#' # Additionally, we report two other normalized stress functions often used
#' # in MDS. The Sammon mapping cost function is already normalized, so tell the
#' # make_reporter function not to report an automatically normalized version in
#' # the output.
#' sammon_iris <- embed_dist(iris[, 1:4],
#'                           method = sammon_map(eps = 1e-4),
#'                           opt = bold_nag(),
#'                           preprocess = make_preprocess(auto_scale = TRUE),
#'                           init_out = out_from_rnorm(sd = 1e-4),
#'                           reporter = make_reporter(normalize_cost = FALSE,
#'                                        extra_costs = c("normalized_stress",
#'                                                        "kruskal_stress"),
#'                                        plot = make_plot(iris, "Species",
#'                                                            make_label())))
#' }
#' @family sneer embedding functions
#' @export
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


#' Generic Embedding
#'
#' Carry out an embedding of a dataset using any embedding method. The most
#' generic embedding function, and conversely the one with the fewest helpful
#' defaults.
#'
#' @param xm A matrix or data frame to embed.
#' @param method Embedding method. Set by assigning the result value of one of
#'   the configuration functions listed in
#'   \code{\link{embedding_methods}}.
#' @param preprocess Input data preprocess callback. Set by assigning the
#'   result value of \code{\link{make_preprocess}}.
#' @param init_inp Input initializer. Set by assigning the result value of
#'   calling one of the available \code{\link{input_initializers}}.
#' @param init_out Output initializer. Set by assigning the result value of
#'   calling one of the available \code{\link{output_initializers}}.
#' @param opt Optimization method. Set by assigning the result of value of one
#'   of the configuration functions listed in
#'   \code{\link{optimization_methods}}.
#' @param max_iter Maximum number of optimization steps to take.
#' @param tricks Optional collection of heuristics. Set by assigning the result
#'  value of \code{\link{make_tricks}} or a related wrapper. See
#'  \code{\link{tricks}} for the available tricks.
#' @param reporter Reporter callback. Set by assigning the result value of
#'   \code{\link{make_reporter}}.
#' @param export Vector of names, specifying data in addition to of the output
#'   data to export provide in the return value. Possible names are:
#'   \describe{
#'     \item{"\code{inp}"}{The input data, based on the argument passed to
#'       \code{init_inp}. See the help text for the specific
#'       \code{\link{input_initializers}} for details.}
#'     \item{"\code{report}"}{The result of the last report. See the help
#'       text for \code{\link{make_reporter}} for details.}
#'     \item{"\code{opt}"}{The optimizer. See the help text for the optimizer
#'     passed to \code{opt} for details.}
#'   }
#' @param after_embed Callback to run on input and output data before output
#'   data is returned.
#' @return The output data. A list containing:
#'   \item{\code{ym}}{Embedded coordinates.}
#'   \item{\code{cost}}{The cost value associated with \code{ym}.}
#'   If the parameter \code{export} was used, additional elements will be
#'   present. See the help text for the \code{export} parameter for more
#'   details.
#' @seealso
#' \itemize{
#' \item{\code{\link{embedding_methods}}} for configuring \code{method}
#' \item{\code{\link{make_preprocess}}} for configuring \code{preprocess}
#' \item{\code{\link{input_initializers}}} for configuring \code{init_inp}
#' \item{\code{\link{output_initializers}}} for configuring \code{init_out}
#' \item{\code{\link{optimization_methods}}} for configuring \code{opt}
#' \item{\code{\link{make_tricks}}} for configuring \code{tricks}
#' \item{\code{\link{make_reporter}}} for configuring \code{reporter}
#' }
embed_main <- function(xm, method, init_inp, init_out, opt, max_iter = 1000,
                  tricks = NULL, reporter = NULL,
                  preprocess = make_preprocess(),
                  export = NULL, after_embed = NULL) {

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

#' Post Initialization
#'
#' Function called after input and output data have been initialized. Useful for
#' doing data-dependent initialization of stiffness parameters, e.g. based on
#' the parameterization of the input probabilities.
#'
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @return List consisting of:
#'   \item{\code{inp}}{Updated input data.}
#'   \item{\code{out}}{Updated output data.}
#'   \item{\code{method}}{Updated embedding method.}
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

#' Embedding Initialziation
#'
#' Initializes all data, ready to be optimized.
#'
#' @param xm A matrix or data frame to embed.
#' @param method Embedding method. Set by assigning the result value of one of
#'   the configuration functions listed in
#'   \code{\link{embedding_methods}}.
#' @param preprocess Input data preprocess callback. Does any necessary
#'  filtering and scaling of the input data.
#' @param init_inp Input initializer. Creates the input data required by
#'  embedding algorithm.
#' @param init_out Output initializer. Creates the initial set of output
#'  coordinates.
#' @param opt Optimization method.
#' @return A list containing:
#'   \item{\code{inp}}{Initialized input}
#'   \item{\code{out}}{Initialized output}
#'   \item{\code{method}}{Initialized embedding method}
#'   \item{\code{opt}}{Initialized optimizer}
#'   \item{\code{report}}{Initialized report}
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

#' Set Output Data Coordinates
#'
#' This function sets the embedded coordinates in the output data, as well as
#' recalculating any auxiliary output data that is dependent on the coordinates
#' (e.g. distances and probabilities)
#'
#' @param inp Input data.
#' @param coords Matrix of coordinates.
#' @param method Embedding method.
#' @param mat_name Name of the \code{out} entry to store \code{coords} in.
#'  Defaults to \code{ym}.
#' @param out Existing output data. Optional.
#' @return \code{out} list with updated with coords and auxiliary data.
set_solution <- function(inp, coords, method, mat_name = "ym",
                         out = NULL) {
  if (is.null(out)) {
    out <- list()
  }
  out[[mat_name]] <- coords
  update_out(inp, out, method)
}

#' Check if Output Data Needs Updating
#'
#' When the coordinates of an embedding are updated via
#' \code{\link{set_solution}}, the other matrices and values that depend on
#' those coordinates should be updated automatically. If a change has occurred
#' to another part of the embedding data (most likely the input probabilities,
#' or the output kernel parameters), then you may need to call this function to
#' see if the output data needs updating before calculating the cost or
#' gradient.
#'
#' @param out Output data.
#' @return \code{TRUE} if the output data matrices need updating.
should_update <- function(out) {
  !is.null(out$dirty) && out$dirty
}

#' Mark Output Data as Up to Date
#'
#' This function should be called when the output data has just been updated.
#' Some parts of the code will avoid unnecessarily repeating the update, but
#' only if they can detect that there's no change to make.
#'
#' @param out Output data.
#' @return Output data, now marked as being up to date.
undirty <- function(out) {
  out$dirty <- FALSE
  out
}

#' Update if Something Has Changed
#'
#' This function calls \code{\link{update_out}} only if it detects that there
#' are changes that needs to be made.
#'
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @return (Possible) updated output data. Otherwise, \code{out} is returned
#' unchanged.
update_out_if_necessary <- function(inp, out, method) {
  if (should_update(out)) {
    out <- update_out(inp, out, method)
  }
  out
}

#' Update Output Data
#'
#' Updates the output data (matrices dependent on the embedding coordinates,
#' such as the distance matrix, weight matrix, output probability matrix).
#'
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @return Updated output data.
update_out <- function(inp, out, method) {
  if (!is.null(method$update_out_fn)) {
    out <- method$update_out_fn(inp, out, method)
    out <- undirty(out)
  }
  out
}

