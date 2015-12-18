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

#' Combine two functions into one with the effect of applying one after the
#' other.
#'
#' @param f Function. Should have a return type compatible with the parameters
#' to \code{g}.
#' @param g Function. Should have parameters compatible with the return type of
#' \code{f}.
#' @return Composed function with output equivalent to calling \code{f(g(...))}.
compose <- function(f, g) {
  function(...) f(g(...))
}


embed_sim <- function(xm,
                      mat_name = "ym",
                      perplexity = 30,
                      input_weight_fn = exp_weight,
                      init_out = make_init_out(from_PCA = TRUE,
                                               mat_name = mat_name,
                                               verbose = verbose),
                      method = tsne(),
                      opt = make_opt(),
                      max_iter = 1000,
                      tricks = make_tricks(verbose = verbose),
                      epoch = make_epoch(verbose = verbose),
                      preprocess = make_preprocess(verbose = verbose),
                      export = NULL,
                      after_embed = NULL,
                      verbose = TRUE) {

  d_to_p_fn <- partial(method$prob_in_fn, perplexity = perplexity,
                       weight_fn = input_weight_fn,
                       verbose = verbose)

  init_inp <- make_init_inp(d_to_p_fn)

  opt$mat_name <- mat_name

  stiffness <- method$stiffness

  opt$update_method$max_iter <- max_iter

  embed(xm, init_inp, init_out, stiffness, opt, max_iter, tricks,
        epoch, preprocess, export, after_embed)
}

embed <- function(xm, init_inp, init_out, stiffness, opt, max_iter = 1000,
                  tricks = NULL, epoch = NULL, preprocess = NULL, export = NULL,
                  after_embed = NULL) {
  if (!is.null(preprocess)) {
    xm <- preprocess(xm)
  }
  inp <- init_inp(xm)
  out <- init_out(inp)

  # do late initialization that relies on input or output initialization
  # being completed
  stiffness <- after_init(stiffness, inp, out)

  # initialize matrices needed for gradient calculation
  out <- update_out(inp, out, stiffness, opt$mat_name)

  # epoch result may contain info from previous epoch that controls
  # whether to stop early (e.g. relative convergence tolerance)
  epoch_result <- list()

  iter <- 0
  while (iter <= max_iter) {
    if (!is.null(tricks)) {
      tricks_result <- tricks(inp, out, stiffness, opt, iter)
      if (!is.null(tricks_result$inp)) {
        inp <- tricks_result$inp
      }
      if (!is.null(tricks_result$out)) {
        out <- tricks_result$out
      }
      if (!is.null(tricks_result$stiffness)) {
        stiffness <- tricks_result$stiffness
      }
      if (!is.null(tricks_result$opt)) {
        opt <- tricks_result$opt
      }
    }

    if (!is.null(epoch)) {
      epoch_result <- epoch(iter, inp, out, stiffness, epoch_result)
      if (epoch_result$stop_early) {
        break
      }
    }

    opt_result <- optimize_step(opt, stiffness, inp, out, iter)
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
#' @param stiffness Stiffness configuration.
#' @param inp Input data.
#' @param out Output data.
#' @return updated stiffness configuration.
after_init <- function(stiffness, inp, out) {
  if (!is.null(stiffness$after_init)) {
    stiffness <- stiffness$after_init(stiffness, inp, out)
  }

  stiffness
}

#' Calculate the gradient of the cost function for the current configuration.
#'
#' @param inp Input data.
#' @param out Output data.
#' @param stiffness Stiffness configuration.
#' @return a list with two items: \code{km} stiffness matrix,
#' \code{gm}, the gradient matrix.
gradient <- function(inp, out, stiffness, mat_name = "ym") {
  km <- stiffness$stiffness_fn(stiffness, inp, out)
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
