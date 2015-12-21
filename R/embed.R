# Functions to do the embedding. The interface to sneer. Also, core
# functions to do with gradient calculation, optimization, updating and so on
# that don't need to be overridden or changed.

#' Similarity-based embedding.
#'
#' Carry out an embedding of a dataset using a similarity-based method
#' (e.g. t-SNE), with some useful default parameters.
#'
#' @param xm A matrix or data frame to embed.
#' @param mat_name Name of the matrix in the output data list that will contain
#' the embedded coordinates.
#' @param preprocess Input data preprocess callback. Create by assigning the
#' result of \code{make_preprocess}.
#' @param init_inp Input initialization callback. Create by assigning the
#' result of \code{make_init_inp}.
#' @param init_out Output initialization callback. Create by assigning the
#' result of \code{make_init_out}.
#' @param method Embedding method. Set by calling a configuration
#' function:
#' \itemize{
#'  \item \code{tsne()} t-Distributed Stochastic Neighbor Embedding.
#'  \item \code{ssne()} Symmetric Stochastic Neighbor Embedding.
#'  \item \code{asne()} Asymmetric Stochastic Neighbor Embedding.
#' }
#' @param opt Optimization method. Create by assigning the result of
#' \code{make_opt}.
#' @param max_iter Maximum number of optimization steps to take.
#' @param tricks Tricks callback. Create by assigning the result of
#'  \code{make_tricks}.
#' @param reporter Reporter callback. Create by assigning the result of
#' \code{make_reporter}.
#' @param export Vector of names to export. Possible names are:
#' \itemize{
#'  \item "\code{inp}" The input data.
#'  \item "\code{report}" The result of the last report.
#' }
#' @param after_embed Callback to run on input and output data before output
#' data is returned.
#' @return The output data. A list containing:
#' \itemize{
#'  \item \code{ym} Embedded coordinates. This name can be changed by
#'  specifying \code{mat_name}.
#'  \item \code{qm} Probability matrix generated from the weight matrix
#'  \code{wm}.
#'  \item \code{wm} Weight matrix generated from the distances between points
#'  in \code{ym}.
#'  \item \code{inp} The input data, if "\code{inp}" is included in the
#'  \code{export} list parameter.
#'  \item \code{report} Most recent report, if
#'  "\code{report}" is included in the \code{export} list parameter.
#' }
#' If the \code{inp} list is present, it contains:
#' \itemize{
#'  \item \code{xm} The (potentially preprocessed) input coordinates if the
#'  input data was not a distance matrix.
#'  \item \code{dm} Input distance matrix.
#'  \item \code{pm} Input probabilities.
#'  \item \code{beta} Input weighting parameters. Only present if
#'  \code{make_init_inp} is called with \code{keep_all_results} set to
#'  \code{TRUE} in when creating the callback \code{init_inp}.
#' }
#' If the \code{report} list is present, it contains:
#' \itemize{
#'  \item \code{stop_early} If \code{TRUE}, the optimization stopped before
#'  \code{max_iters} was reached.
#'  \item \code{cost} Cost of the embedded configuration in the most recent
#'  iteration.
#'  \item \code{costs} Matrix of all report costs and the iterations at which
#'  they occurred. Only present if \code{keep_costs} is set to \code{TRUE}
#'  when the \code{make_reporter} factory function is called.
#'  \item \code{reltol} Relative tolerance of the difference between present
#'  cost and the cost from the previous report.
#'  \item \code{stress} Stress for the most recent iteration. Only present if
#'  \code{calc_stress} is set to \code{TRUE} when the \code{make_reporter}
#'  factory function is called.
#'  \item \code{iter} The iteration at which the report is generated.
#' }
#' @seealso
#' \code{\link{make_preprocess}} for configuring \code{preprocess},
#' \code{\link{make_init_inp}} for configuring \code{init_inp},
#' \code{\link{make_init_out}} for configuring \code{init_out},
#' \code{\link{make_opt}} for configuring \code{opt},
#' \code{\link{make_tricks}} for configuring \code{tricks},
#' \code{\link{make_reporter}} for configuring \code{reporter}.
#'
#' @examples
#' # Do t-SNE on the iris dataset with the same options as the t-SNE paper
#' # except initialize from PCA so output is repeatable.
#' # plot 2D result during embedding with convenience function for iris plot.
#' # Default method is tsne. Set perplexity to 25.
#' tsne_iris <- embed_sim(iris[, 1:4], opt = tsne_opt(),
#'                init_inp = make_init_inp(perplexity = 25),
#'                tricks = tsne_tricks(),
#'                reporter = make_reporter(plot_fn = make_iris_plot()))
#'
#' # Do t-SNE on the iris dataset with the same options as the t-SNE paper
#' # and initialize from random. Use generic plot function, displaying the first
#' # two characters of the "Species" factor for the points. Explicitly choose
#' # t-SNE as the method.
#' tsne_iris <- embed_sim(iris[, 1:4],
#'                method = tsne(),
#'                opt = tsne_opt(),
#'                init_inp = make_init_inp(perplexity = 25),
#'                init_out = make_init_out(stdev = 1e-4)
#'                tricks = tsne_tricks(),
#'                reporter = make_reporter(
#'                  plot_fn = make_plot(iris, "Species", make_label(2))))
#'
#' # Use the SSNE method, and preprocess input data by range scaling. t-SNE
#' # tricks and optimization are reasonable defaults for other similarity
#' # embeddings.
#' ssne_iris <- embed_sim(iris[, 1:4],
#'                method = ssne(),
#'                opt = tsne_opt(),
#'                init_inp = make_init_inp(perplexity = 25),
#'                preprocess = make_preprocess(range_scale_matrix = TRUE)
#'                init_out = make_init_out(stdev = 1e-4)
#'                tricks = tsne_tricks(),
#'                reporter = make_reporter(
#'                  plot_fn = make_plot(iris, "Species", make_label(2))))
#'
#' # ASNE method on the s1k dataset (10 overlapping 9D Gaussian blobs),
#' # Set perplexity for input initialization to 50, initialize with PCA scores,
#' # preprocess by autoscaling columns, optimize
#' # with Nesterov Accelrated Gradient and bold driver step size
#' # (highly recommended as an optimizer). Labels for s1k are one digit, so
#' # can use simplified plot function.
#' asne_s1k <- embed_sim(s1k[, 1:9], method = asne(),
#'  preprocess = make_preprocess(auto_scale = TRUE),
#'  init_inp = make_init_inp(perplexity = 50),
#'  init_out = make_init_out(from_PCA = TRUE),
#'  opt = make_opt(gradient = nesterov_gradient(), step_size = bold_driver(),
#'   update = nesterov_nsc_momentum()),
#'   reporter = make_reporter(plot_fn = make_plot(s1k, "Label")))
#'
#' # Same as above, but using convenience method to create optimizer with less
#' # typing
#' asne_s1k <- embed_sim(s1k[, 1:9], method = asne(),
#'  preprocess = make_preprocess(auto_scale = TRUE),
#'  init_inp = make_init_inp(perplexity = 50),
#'  init_out = make_init_out(from_PCA = TRUE),
#'  opt = bold_nag_opt(),
#'  reporter = make_reporter(plot_fn = make_plot(s1k, "Label")))
embed_sim <- function(xm,
                      mat_name = "ym",
                      preprocess = make_preprocess(verbose = verbose),
                      init_inp = make_init_inp(perplexity = 30,
                                               input_weight_fn = exp_weight,
                                               verbose = verbose),
                      init_out = make_init_out(from_PCA = TRUE,
                                               mat_name = mat_name,
                                               verbose = verbose),
                      method = tsne(),
                      opt = make_opt(mat_name = mat_name),
                      max_iter = 1000,
                      tricks = NULL,
                      reporter = make_reporter(verbose = verbose),
                      export = NULL,
                      after_embed = NULL,
                      verbose = TRUE) {
  embed(xm, init_inp, init_out, method, opt, max_iter, tricks,
        reporter, preprocess, export, after_embed)
}

#' Generic embedding.
#'
#' Carrying out an embedding of a dataset.
#'
#' @param xm A matrix or data frame to embed.
#' @param mat_name Name of the matrix in the output data list that will contain
#' the embedded coordinates.
#' @param Input data preprocess callback. Create by assigning the result of
#' \code{make_preprocess}.
#' @param init_inp Input initialization callback. Create by assigning the
#' result of \code{make_init_inp}.
#' @param init_out Output initialization callback. Create by assigning the
#' result of \code{make_init_out}.
#' @param method Embedding method. Assign result of calling one of the following
#' functions: \itemize{
#'  \item \code{tsne()} t-Distributed Stochastic Neighbor Embedding.
#'  \item \code{ssne()} Symmetric Stochastic Neighbor Embedding.
#'  \item \code{asne()} Asymmetric Stochastic Neighbor Embedding.
#' }
#' @param opt Optimization method. Create by assigning the result of
#' \code{make_opt}.
#' @param max_iter Maximum number of optimization steps to take.
#' @param tricks Tricks callback. Create by assigning the result of
#'  \code{make_tricks}.
#' @param reporter Report callback. Create by assigning the result of
#' \code{make_reporter}.
#' @param export Vector of names to export. Possible names are:
#' \itemize{
#'  \item "\code{inp}" The input data.
#'  \item "\code{report}" The most recent report.
#' }
#' @param after_embed Callback to run on input and output data before output
#' data is returned.
#' @return The output data. A list containing:
#' \itemize{
#'  \item \code{ym} Embedded coordinates. This name can be changed by
#'  specifying \code{mat_name}.
#'  \item \code{qm} Probability matrix generated from the weight matrix
#'  \code{wm}.
#'  \item \code{wm} Weight matrix generated from the distances between points
#'  in \code{ym}.
#'  \item \code{inp} The input data, if "\code{inp}" is included in the
#'  \code{export} list parameter.
#'  \item \code{report} Most recent report, if "\code{report}" is included in
#'  the \code{export} list parameter.
#' }
#' If the \code{inp} list is present, it contains:
#' \itemize{
#'  \item \code{xm} The (potentially preprocessed) input coordinates if the
#'  input data was not a distance matrix.
#'  \item \code{dm} Input distance matrix.
#'  \item \code{pm} Input probabilities.
#'  \item \code{beta} Input weighting parameters. Only present if
#'  \code{make_init_inp} is called with \code{keep_all_results} set to
#'  \code{TRUE} in when creating the callback \code{init_inp}.
#' }
#' If the \code{report} list is present, it contains:
#' \itemize{
#'  \item \code{stop_early} If \code{TRUE}, the optimization stopped before
#'  \code{max_iters} was reached.
#'  \item \code{cost} Cost of the configuration at iteration \code{iter}.
#'  \item \code{costs} Matrix of all costs and the iterations at which
#'  the reports were made. Only present if \code{keep_costs} is set to
#'  \code{TRUE} when the \code{make_reporter} factory function is called.
#'  \item \code{reltol} Relative tolerance of the difference between present
#'  cost and the cost evaluted by the previous report
#'  \item \code{stress} Stress for the current configuration Only present if
#'  \code{calc_stress} is set to \code{TRUE} when the \code{make_reporter}
#'  factory function is called.
#'  \item \code{iter} The iteration at which the report is generated.
#' }
#' @seealso
#' \code{\link{make_preprocess}} for configuring \code{preprocess},
#' \code{\link{make_init_inp}} for configuring \code{init_inp},
#' \code{\link{make_init_out}} for configuring \code{init_out},
#' \code{\link{make_opt}} for configuring \code{opt},
#' \code{\link{make_tricks}} for configuring \code{tricks},
#' \code{\link{make_reporter}} for configuring \code{reporter}.
embed <- function(xm, init_inp, init_out, method, opt, max_iter = 1000,
                  tricks = NULL, reporter = NULL, preprocess = NULL,
                  export = NULL, after_embed = NULL) {
  if (!is.null(preprocess)) {
    xm <- preprocess(xm)
  }
  inp <- init_inp(xm)
  out <- init_out(inp)

  # do late initialization that relies on input or output initialization
  # being completed
  after_init_result <- after_init(inp, out, method)
  inp <- after_init_result$inp
  out <- after_init_result$out
  method <- after_init_result$method

  # initialize matrices needed for gradient calculation
  out <- update_out(inp, out, method, opt$mat_name)

  # reuse reports from old invocation of reporter, so we can use info
  # to determine whether to stop early (e.g. relative convergence tolerance)
  report <- list()

  iter <- 0
  while (iter <= max_iter) {
    if (!is.null(tricks)) {
      tricks_result <- tricks(inp, out, method, opt, iter)
      inp <- tricks_result$inp
      out <- tricks_result$out
      method <- tricks_result$method
      opt <- tricks_result$opt
    }

    if (!is.null(reporter)) {
      report <- reporter(iter, inp, out, method, report)
      if (report$stop_early) {
        break
      }
    }

    opt_result <- optimize_step(opt, method, inp, out, iter)
    out <- opt_result$out
    opt <- opt_result$opt

    iter <- iter + 1
  }

  for (obj_name in export) {
    out[[obj_name]] <- get(obj_name)
  }

  if (!is.null(after_embed)) {
    out <- after_embed(inp, out)
  }

  out
}

#' Function called after initialization of input and output.
#'
#' Useful for doing data-dependent initialization of stiffness
#' parameters, e.g. based on the parameterization of the input probabilities.
#'
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @return List consisting of:
#' \itemize{
#'  \item \code{inp} Updated input data.
#'  \item \code{out} Updated output data.
#'  \item \code{method} Updated embedding method.
#' }
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


#' One step of optimization.
#'
#' @param opt Optimizer
#' @param method Embedding method.
#' @param inp Input data.
#' @param out Output data.
#' @param iter Iteration number.
#' @return List consisting of:
#' \itemize{
#'  \item \code{opt} Updated optimizer.
#'  \item \code{inp} Updated input.
#'  \item \code{out} Updated output.
#' }
optimize_step <- function(opt, method, inp, out, iter) {
  if (iter == 0) {
    opt <- opt$init(opt, inp, out, method)
  }

  grad_result <- opt$grad_pos_fn(opt, inp, out, method)

  if (any(is.nan(grad_result$gm))) {
    stop("NaN in grad. descent at iter ", iter)
  }
  opt$gm <- grad_result$gm

  if (opt$normalize_grads) {
    opt$gm <- normalize(opt$gm)
  }

  direction_result <-
    opt$direction_method$get_direction(opt, inp, out, method, iter)

  if (!is.null(direction_result$opt)) {
    opt <- direction_result$opt
  }
  opt$direction_method$direction <- direction_result$direction

  opt$step_size_method$step_size <-
    opt$step_size_method$get_step_size(opt, inp, out, method)

  opt$update_method$update <-
    opt$update_method$get_update(opt, inp, out, method)

  new_out <- update_solution(opt, inp, out, method)

  # intercept whether we want to accept the new solution e.g. bold driver
  ok <- TRUE
  if (!is.null(opt$validate)) {
    validation_result <- opt$validate(opt, inp, out, new_out, method)
    opt <- validation_result$opt
    inp <- validation_result$inp
    out <- validation_result$out
    new_out <- validation_result$new_out
    method <- validation_result$method
    ok <- validation_result$ok
  }

  if (!ok) {
    new_out <- out
  }

  if (!is.null(opt$after_step)) {
    after_step_result <- opt$after_step(opt, inp, out, new_out, ok, iter)
    opt <- after_step_result$opt
    inp <- after_step_result$inp
    out <- after_step_result$out
    new_out <- after_step_result$new_out
  }

  list(opt = opt, inp = inp, out = new_out)
}

#' Update output data.
#'
#' This function updates the embedded coordinates in the output data, based
#' on the update information in the Optimizer, as well as updating any
#' auxiliary output data that is dependent on the coordinates (e.g. distances
#' and probabilities)
#'
#' @param opt Optimizer.
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @return Updated \code{out}.
update_solution <- function(opt, inp, out, method) {
  new_out <- out
  new_solution <- new_out[[opt$mat_name]] + opt$update_method$update
  new_out[[opt$mat_name]] <- new_solution
  update_out(inp, new_out, method, opt$mat_name)
}


#' Calculate the gradient of the cost function for the current configuration.
#'
#' @param inp Input data.
#' @param out Output data.
#' @param method Embedding method.
#' @return List containing:
#' \itemize{
#'  \item \code{km} Stiffness matrix.
#'  \item \code{gm} Gradient matrix.
#' }
gradient <- function(inp, out, method, mat_name = "ym") {
  km <- method$stiffness_fn(method, inp, out)
  gm <- stiff_to_grads(out[[mat_name]], km)
  list(km = km, gm = gm)
}

#' Convert stiffness matrix to gradient matrix.
#'
#' @param ym Embedded coordinates.
#' @param km Stiffness matrix.
#' @return Gradient matrix.
stiff_to_grads <- function(ym, km) {
  gm <- matrix(0, nrow(ym), ncol(ym))
  for (i in 1:nrow(ym)) {
    disp <- sweep(-ym, 2, -ym[i, ]) #  matrix of y_ik - y_jk
    gm[i, ] <- apply(disp * km[, i], 2, sum) # row is sum_j (km_ji * disp)
  }
  gm
}

#' Creates a matrix of squared Euclidean distances from a coordinate matrix.
#'
#' Similarity-preserving embedding techniques use the squared Euclidean distance
#' as input to their weighting functions.
#'
#' @param xm a matrix of coordinates
#' @return Squared distance matrix.
coords_to_dist2 <- function(xm) {
  sumsq <- apply(xm ^ 2, 1, sum)  # sum of squares of each row of xm
  d2m <- -2 * xm %*% t(xm)  # cross product
  d2m <- sweep(d2m, 2, -t(sumsq))  # adds sumsq[j] to D2[i,j]
  sumsq + d2m  # add sumsq[i] to D2[i,j]
}
