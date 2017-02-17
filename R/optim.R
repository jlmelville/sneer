# Bridges Sneer and Mize

# Called at every iteration to convert sneer data structures into mize form
make_optim_fg <- function(opt, inp, out, method, iter) {
  if (!is.null(inp$xm)) {
    nr <- nrow(inp$xm)
  }
  else {
    nr <- nrow(inp$dm)
  }
  dout <- out$dim

  res <- list(
    fn = function(par) {
      if (!is.null(method$set_extra_par)) {
        extra_par <- par[(dout * nr + 1):length(par)]
        par <- par[1:(dout * nr)]
        method <- method$set_extra_par(method, extra_par)
      }
      res <- par_to_out(par, opt, inp, out, method, nr)
      calculate_cost(method, res$inp, res$out)
    },
    gr = function(par) {
      if (!is.null(method$extra_gr)) {
        extra_par <- par[(dout * nr + 1):length(par)]
        par <- par[1:(dout * nr)]
      }
      res <- par_to_out(par, opt, inp, out, method, nr)
      out <- res$out
      inp <- res$inp
      grvec <- mat_to_par(gradient(inp, out, method, opt$mat_name)$gm)
      if (!is.null(method$extra_gr)) {
        extra_grvec <- method$extra_gr(opt, inp, out, method, iter, extra_par)
        grvec <- c(grvec, extra_grvec)
      }

      grvec
    },
    fg = function(par) {

      if (!is.null(method$set_extra_par)) {
        extra_par <- par[(dout * nr + 1):length(par)]
        par <- par[1:(dout * nr)]
        method <- method$set_extra_par(method, extra_par)
      }
      res <- par_to_out(par, opt, inp, out, method, nr)
      out <- res$out
      inp <- res$inp
      grvec <- mat_to_par(gradient(inp, out, method, opt$mat_name)$gm)
      if (!is.null(method$extra_gr)) {
        extra_grvec <- method$extra_gr(opt, inp, out, method, iter, extra_par)
        grvec <- c(grvec, extra_grvec)
      }

      list(
        fn = calculate_cost(method, inp, out),
        gr = grvec
      )
    }
  )

  res
}

# Convert 1D Parameter to Sneer Output
#
# This function converts the 1D parameter format expected by
# \code{\link[stats]{optim}} into the matrix format used by sneer.
#
# @param par Vector of embedded coordinates.
# @param opt Optimize
# @param inp Input data.
# @param method Embedding method.
# @param nrow Number of rows in the sneer output matrix.
# @return Output data with coordinates converted from \code{par}.
par_to_out <- function(par, opt, inp, out, method, nrow) {
  dim(par) <- c(nrow, length(par) / nrow)
  res <- set_solution(inp, par, method, mat_name = opt$mat_name, out = out)
  out <- res$out
  out$dirty <- TRUE
  list(inp = res$inp, out = out)
}

# Convert Matrix to 1D Parameter Vector
#
# This function takes a matrix used by sneer internally (e.g. output
# coordinates or gradient matrix) and converts into a 1D vector, as used
# by \code{\link[stats]{optim}}. The matrix is converted columnwise.
#
# @param mat Matrix to convert.
# @return Matrix in vector form.
mat_to_par <- function(mat) {
  dim(mat) <- NULL
  mat
}

mize_opt_step <- function(opt, method, inp, out, iter) {

  fg <- make_optim_fg(opt, inp, out, method, iter)
  if ((iter == 0 && toupper(opt$name) == "PHESS") || toupper(opt$name) == "NEWTON") {
    # Using Newton is super pointless with standard spectral direction
    # Get the same Hessian approximation at every iteration
    # D+
    dm <- diag(indegree_centrality(inp$pm))
    # Graph Laplacian , L+ = D+ - W+
    lm <- dm - inp$pm
    # enforce positive definiteness by adding a small number
    mu <- min(lm[lm > 0]) * 1e-10
    # Cholesky decomposition of graph Laplacian
    fg$hs <- function(par) { 4 * (lm + mu) }
  }

  par <- mat_to_par(out$ym)
  # Add extra parameters
  if (!is.null(method$get_extra_par)) {
    par <- c(par, method$get_extra_par(method))
  }

  mize <- opt$mize

  if (iter == 0) {
    mize <- opt$mize_module$opt_init(mize, par, fg,
                                     step_tol = .Machine$double.eps,
                                     max_iter = Inf)
    opt$mize <- mize
  }

  if (!is.null(opt$old_cost_dirty) && opt$old_cost_dirty) {
    mize <- opt$mize_module$opt_clear_cache(mize)
    opt$old_cost_dirty <- FALSE
  }
  step_res <- opt$mize_module$opt_step(mize, par, fg)
  mize <- step_res$opt
  step_info <- opt$mize_module$mize_step_summary(mize, step_res$par, fg,
                                                 par_old = par)
  # message("nf = ", step_info$nf, " ng = ", step_info$ng
  #         , " f = ", formatC(step_info$f)
  #         , " step = ", step_info$step, " alpha = ", step_info$alpha)

  mize <- opt$mize_module$check_mize_convergence(step_info)

  opt$mize <- mize
  par <- step_res$par

  if (!is.null(inp$xm)) {
    nr <- nrow(inp$xm)
  }
  else {
    nr <- nrow(inp$dm)
  }
  dout <- out$dim

  # remove extra parameters that were pushed (and update method)
  if (!is.null(method$set_extra_par)) {
    extra_par <- par[(dout * nr + 1):length(par)]
    par <- par[1:(dout * nr)]
    method <- method$set_extra_par(method, extra_par)
  }

  # convert y coord par into sneer form
  res <- par_to_out(par, opt, inp, out, method, nr)
  out <- res$out
  inp <- res$inp
  if (opt$mize$is_terminated) {
    if (opt$verbose) {
      message("Optimizer reports termination due to: ", opt$mize$terminate$what)
    }
    opt$stop_early <- TRUE
  }

  list(opt = opt, inp = inp, out = out, method = method, iter = iter)
}

mize_opt <- function(opt_name, verbose = FALSE, ...) {
  mize_module <- mize()

  if (class(opt_name) == "list") {
    opt <- opt_name
  }
  else {
    opt <- mize_module$make_mize(method = opt_name, ...)
  }

  list(
    name = opt_name,
    mat_name = "ym",
    optimize_step = mize_opt_step,
    mize_module = mize_module,
    mize = opt,
    verbose = verbose
  )
}

mize_opt_alt <- function(opt_name, verbose = FALSE, ...) {
  mize_module <- mize()

  if (class(opt_name) == "list") {
    stop("Can't use list for alternating optimization")
  }
  else {
    opt <- mize_module$make_mize(method = opt_name, ...)
    mize_alt <- mize_module$make_mize(method = opt_name, ...)
  }

  list(
    name = opt_name,
    mat_name = "ym",
    optimize_step = mize_opt_alt_step,
    mize_module = mize_module,
    mize = opt,
    mize_alt = mize_alt,
    verbose = verbose
  )
}

mize_bold_nag_adapt <- function() {
  mize_opt("SD", line_search = "bold", norm_direction = TRUE,
            mom_schedule = "nsconvex", mom_type = "nesterov",
            mom_linear_weight = TRUE, nest_convex_approx = TRUE,
            nest_burn_in = 1, use_init_mom = TRUE, restart = "fn")
}

mize_bold_nag <- function() {
  mize_opt("SD", line_search = "bold", norm_direction = TRUE,
            mom_schedule = "nsconvex", mom_type = "nesterov",
            mom_linear_weight = TRUE, nest_convex_approx = TRUE,
            nest_burn_in = 1, use_init_mom = TRUE)
}

mize_back_nag <- function() {
  mize_opt("SD", line_search = "back", norm_direction = TRUE,
            mom_schedule = "nsconvex", mom_type = "nesterov",
            mom_linear_weight = TRUE, nest_convex_approx = TRUE,
            nest_burn_in = 1, use_init_mom = TRUE,
            c1 = 0.1, step_down = 0.8)
}

mize_back_nag_adapt <- function() {
  mize_opt("SD", line_search = "back", norm_direction = TRUE,
            mom_schedule = "nsconvex", mom_type = "nesterov",
            mom_linear_weight = TRUE, nest_convex_approx = TRUE,
            nest_burn_in = 1, use_init_mom = TRUE, restart = "fn",
            c1 = 0.1, step_down = 0.8)
}

mize_grad_descent <- function() {
  mize_opt("SD", line_search = "bold", norm_direction = TRUE)
}


mize_opt_alt_step <- function(opt, method, inp, out, iter) {

  fg_coord <- make_optim_coord_fg(opt, inp, out, method, iter)
  fg_alt <- make_optim_alt_fg(opt, inp, out, method, iter)

  if ((iter == 0 && toupper(opt$name) == "PHESS") || toupper(opt$name) == "NEWTON") {
    # Using Newton is super pointless with standard spectral direction
    # Get the same Hessian approximation at every iteration
    # D+
    dm <- diag(indegree_centrality(inp$pm))
    # Graph Laplacian , L+ = D+ - W+
    lm <- dm - inp$pm
    # enforce positive definiteness by adding a small number
    mu <- min(lm[lm > 0]) * 1e-10
    # Cholesky decomposition of graph Laplacian
    fg_coord$hs <- function(par) { 4 * (lm + mu) }
  }

  par <- mat_to_par(out$ym)

  mize <- opt$mize
  mize_alt <- opt$mize_alt
  if (iter == 0) {
    mize <- opt$mize_module$opt_init(mize, par, fg_coord,
                                     step_tol = .Machine$double.eps,
                                     max_iter = Inf)
    opt$mize <- mize
  }

  # Can't reuse any old gradients
  mize <- opt$mize_module$opt_clear_cache(mize)

  opt$old_cost_dirty <- FALSE


  step_res <- opt$mize_module$opt_step(mize, par, fg_coord)
  mize <- step_res$opt
  step_info <- opt$mize_module$mize_step_summary(mize, step_res$par, fg_coord,
                                                 par_old = par)
  # message("coord nf = ", step_info$nf, " ng = ", step_info$ng
  #         , " f = ", formatC(step_info$f)
  #         , " step = ", step_info$step, " alpha = ", step_info$alpha)

  mize <- opt$mize_module$check_mize_convergence(step_info)

  opt$mize <- mize
  par <- step_res$par

  if (!is.null(inp$xm)) {
    nr <- nrow(inp$xm)
  }
  else {
    nr <- nrow(inp$dm)
  }
  # convert y coord par into sneer form
  res <- par_to_out(par, opt, inp, out, method, nr)
  out <- res$out
  inp <- res$inp
  if (opt$mize$is_terminated) {
    if (opt$verbose) {
      message("Optimizer reports termination due to: ", opt$mize$terminate$what)
    }
    opt$stop_early <- TRUE
  }

  # Optimize parameters
  do_init <- FALSE
  if (!is.null(method$opt_iter) && method$opt_iter == iter) {
    do_init <- TRUE
  }
  else if (iter == 0) {
    do_init <- TRUE
  }

  if (do_init) {
    mize_alt <- opt$mize_module$opt_init(mize_alt, par, fg_alt,
                                         step_tol = .Machine$double.eps,
                                         max_iter = Inf)
    opt$mize_alt <- mize_alt
  }

  do_opt_alt <- FALSE
  if (!is.null(method$opt_iter)) {
    if (iter >= method$opt_iter) {
      do_opt_alt <- TRUE
    }
  }
  else {
    do_opt_alt <- TRUE
  }

  if (do_opt_alt) {
    if (!is.null(method$get_extra_par)) {
      par <- method$get_extra_par(method)
    }
    else {
      stop("No extra parameters for alternating optimization")
    }
    mize_alt <- opt$mize_module$opt_clear_cache(mize_alt)
    step_res <- opt$mize_module$opt_step(mize_alt, par, fg_alt)
    mize_alt <- step_res$opt
    step_info <- opt$mize_module$mize_step_summary(mize_alt, step_res$par,
                                                   fg_alt,
                                                   par_old = par)
    # message("iter = ", iter,
    #   " param nf = ", step_info$nf, " ng = ", step_info$ng
    #         , " f = ", formatC(step_info$f)
    #         , " step = ", step_info$step, " alpha = ", step_info$alpha)

    mize_alt <- opt$mize_module$check_mize_convergence(step_info)

    opt$mize_alt <- mize_alt
    par <- step_res$par

    method <- method$set_extra_par(method, par)
  }
  list(opt = opt, inp = inp, out = out, method = method, iter = iter)
}

make_optim_alt_fg <- function(opt, inp, out, method, iter) {
  if (!is.null(inp$xm)) {
    nr <- nrow(inp$xm)
  }
  else {
    nr <- nrow(inp$dm)
  }

  res <- list(
    fn = function(par) {
      if (!is.null(method$set_extra_par)) {
        method <- method$set_extra_par(method, par)
        out <- set_solution(inp, out$ym, method, out = out)$out
      }
      else {
        stop("No parameters available for alternating optimization")
      }
      calculate_cost(method, inp, out)
    },
    gr = function(par) {
      if (!is.null(method$extra_gr)) {
        grvec <- method$extra_gr(opt, inp, out, method, iter, par)
      }
      else {
        stop("No parameters available for alternating optimization")
      }

      grvec
    },
    fg = function(par) {
      if (!is.null(method$set_extra_par)) {
        method <- method$set_extra_par(method, par)
      }
      else {
        stop("No parameters available for alternating optimization")
      }
      out <- set_solution(inp, out$ym, method, out = out)$out

      grvec <- method$extra_gr(opt, inp, out, method, iter, par)
      list(
        fn = calculate_cost(method, inp, out),
        gr = grvec
      )
    }
  )

  res
}

make_optim_coord_fg <- function(opt, inp, out, method, iter) {
  if (!is.null(inp$xm)) {
    nr <- nrow(inp$xm)
  }
  else {
    nr <- nrow(inp$dm)
  }

  res <- list(
    fn = function(par) {
      res <- par_to_out(par, opt, inp, out, method, nr)
      calculate_cost(method, res$inp, res$out)
    },
    gr = function(par) {
      res <- par_to_out(par, opt, inp, out, method, nr)
      out <- res$out
      inp <- res$inp
      grvec <- mat_to_par(gradient(inp, out, method, opt$mat_name)$gm)
      grvec
    },
    fg = function(par) {
      res <- par_to_out(par, opt, inp, out, method, nr)
      out <- res$out
      inp <- res$inp
      grvec <- mat_to_par(gradient(inp, out, method, opt$mat_name)$gm)

      list(
        fn = calculate_cost(method, inp, out),
        gr = grvec
      )
    }
  )

  res
}
