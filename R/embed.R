#' Partially apply a function.
#'
#' @param f Function to partially apply.
#' @param ... params of \code{f} to apply.
#' @return Partially applied version of \code{f}.
partial <- function(f, ...) {
  args <- list(...)
  function(...) {
    do.call(f, c(args, list(...)))
  }
}

#' Similarity-based embedding.
#'
#'
embed_sim <- function(xm,
                      mat_name = "ym",
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
                      epoch = make_epoch(verbose = verbose),
                      preprocess = make_preprocess(verbose = verbose),
                      export = NULL,
                      after_embed = NULL,
                      verbose = TRUE) {
  embed(xm, init_inp, init_out, method, opt, max_iter, tricks,
        epoch, preprocess, export, after_embed)
}

#' Embed coordinates
#'
embed <- function(xm, init_inp, init_out, method, opt, max_iter = 1000,
                  tricks = NULL, epoch = NULL, preprocess = NULL, export = NULL,
                  after_embed = NULL) {
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

  # epoch result may contain info from previous epoch that controls
  # whether to stop early (e.g. relative convergence tolerance)
  epoch_result <- list()

  iter <- 0
  while (iter <= max_iter) {
    if (!is.null(tricks)) {
      tricks_result <- tricks(inp, out, method, opt, iter)
      inp <- tricks_result$inp
      out <- tricks_result$out
      method <- tricks_result$method
      opt <- tricks_result$opt
    }

    if (!is.null(epoch)) {
      epoch_result <- epoch(iter, inp, out, method, epoch_result)
      if (epoch_result$stop_early) {
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
