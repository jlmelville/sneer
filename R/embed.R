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
  embed(xm, method, init_inp, init_out, opt, max_iter, tricks,
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
  embed(xm, method, init_inp, init_out, opt, max_iter, tricks,
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
embed <- function(xm, method, init_inp, init_out, opt, max_iter = 1000,
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
    out$cost <- method$cost$fn(inp, out, method)
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
  out <- method$update_out_fn(inp, out, method)

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
  method$update_out_fn(inp, out, method)
}
