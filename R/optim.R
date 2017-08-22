# Bridges Sneer and Mize

# Called at every iteration to convert sneer data structures into mize form
make_optim_fg <- function(opt, inp, out, method, iter) {
  if (!is.null(method$gradient) && !is.null(method$gradient$fn)) {
    grad_fn <- method$gradient$fn
  }
  else {
    grad_fn <- dist2_gradient
  }

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
      grvec <- mat_to_par(grad_fn(inp, out, method, opt$mat_name)$gm)
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
      grvec <- mat_to_par(grad_fn(inp, out, method, opt$mat_name)$gm)
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
# @return Output data with coordinates converted from \code{par} and recentered.
par_to_out <- function(par, opt, inp, out, method, nrow) {
  dim(par) <- c(nrow, length(par) / nrow)

  # Recenter the coordinates
  par <- sweep(par, 2, colMeans(par))

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
                                     step_tol = opt$step_tol,
                                     max_iter = Inf,
                                     max_fn = opt$max_fn,
                                     max_gr = opt$max_gr,
                                     max_fg = opt$max_fg)
    opt$mize <- mize
    opt$restarted_once <- FALSE
  }

  if (!is.null(opt$old_cost_dirty) && opt$old_cost_dirty) {
    mize <- opt$mize_module$opt_clear_cache(mize)
    opt$old_cost_dirty <- FALSE
  }

  if (!is.null(opt$do_opt_step) && !opt$do_opt_step) {
    restart_res <- restart_if_possible(opt, inp, iter, mize, par, fg,
                                       is_before_step = TRUE)
    opt <- restart_res$opt
    mize <- restart_res$mize
  }

  if (is.null(opt$do_opt_step) || opt$do_opt_step) {
    step_res <- opt$mize_module$opt_step(mize, par, fg)
    mize <- step_res$opt
    step_info <- opt$mize_module$mize_step_summary(mize, step_res$par, fg,
                                                   par_old = par)
    # message("iter = ", iter, " nf = ", step_info$nf, " ng = ", step_info$ng
    #         , " f = ", formatC(step_info$f)
    #         , " step = ", step_info$step, " alpha = ", step_info$alpha)
    mize <- opt$mize_module$check_mize_convergence(step_info)
    opt$nf <- step_info$nf
    opt$ng <- step_info$ng

    if (mize$is_terminated && mize$terminate$what == "step_tol") {
      restart_res <- restart_if_possible(opt, inp, iter, mize, par, fg,
                                         is_before_step = FALSE)
      opt <- restart_res$opt
      mize <- restart_res$mize
    }

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
  }

  opt$mize <- mize

  if (opt$mize$is_terminated) {
    if (opt$verbose) {
      message("Optimizer reports termination due to: ", opt$mize$terminate$what)
    }
    opt$stop_early <- TRUE
  }

  list(opt = opt, inp = inp, out = out, method = method, iter = iter)
}

mize_opt <- function(opt_name,
                     max_fn = Inf,
                     max_gr = Inf,
                     max_fg = Inf,
                     verbose = FALSE, ...) {
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
    convergence_iter = 0,
    verbose = verbose,
    nf = 0,
    ng = 0,
    max_fn = max_fn,
    max_gr = max_gr,
    max_fg = max_fg,
    step_tol = sqrt(.Machine$double.eps)
  )
}

mize_opt_alt <- function(opt_name,
                         max_fn = Inf,
                         max_gr = Inf,
                         max_fg = Inf,
                         verbose = FALSE, ...) {
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
    convergence_iter = 0,
    nf = 0,
    ng = 0,
    max_fn = max_fn,
    max_gr = max_gr,
    max_fg = max_fg,
    step_tol = sqrt(.Machine$double.eps),
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

mize_bfgs <- function() {
  mize_opt("BFGS", c1 = 1e-4, c2 = 0.1,
           abs_tol = 0, rel_tol = 0, step_tol = NULL,
           step_next_init = "quad", line_search = "mt",
           step0 = "ras")
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
                                     step_tol = opt$step_tol,
                                     max_iter = Inf,
                                     max_fn = opt$max_fn,
                                     max_gr = opt$max_gr,
                                     max_fg = opt$max_fg)
    opt$mize <- mize
    opt$restarted_once <- FALSE
  }

  # Can't reuse any old gradients
  mize <- opt$mize_module$opt_clear_cache(mize)
  opt$old_cost_dirty <- FALSE


  ###
  # Optimize Coords
  ###
  if (!is.null(opt$do_opt_step) && !opt$do_opt_step) {
    restart_res <- restart_if_possible(opt, inp, iter, mize, par, fg_coord,
                                       is_before_step = TRUE)
    opt <- restart_res$opt
    mize <- restart_res$mize
  }

  if (is.null(opt$do_opt_step) || opt$do_opt_step) {
    step_res <- opt$mize_module$opt_step(mize, par, fg_coord)
    mize <- step_res$opt
    step_info <- opt$mize_module$mize_step_summary(mize, step_res$par, fg_coord,
                                                   par_old = par)
    # message("coord nf = ", step_info$nf, " ng = ", step_info$ng
    #         , " f = ", formatC(step_info$f)
    #         , " step = ", step_info$step, " alpha = ", step_info$alpha)

    mize <- opt$mize_module$check_mize_convergence(step_info)
    opt$nf <- step_info$nf
    opt$ng <- step_info$ng

    if (mize$is_terminated && mize$terminate$what == "step_tol") {
      restart_res <- restart_if_possible(opt, inp, iter, mize, par, fg_coord,
                                         is_before_step = FALSE)
      opt <- restart_res$opt
      mize <- restart_res$mize
    }

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
  }
  opt$mize <- mize

  if (opt$mize$is_terminated) {
    if (opt$verbose) {
      message("Optimizer reports termination due to: ", opt$mize$terminate$what)
    }
    opt$stop_early <- TRUE
  }


  ####
  # Optimize parameters
  ####
  do_init <- FALSE
  if (!is.null(method$opt_iter) && method$opt_iter == iter) {
    do_init <- TRUE
  }
  else if (iter == 0) {
    do_init <- TRUE
  }

  if (do_init) {
    mize_alt <- opt$mize_module$opt_init(mize_alt, par, fg_alt,
                                         step_tol = opt$step_tol,
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

  # If we are refraining from bothering with coordinate optimization on this
  # iteration also don't do parameter optimization
  if (!is.null(opt$do_opt_step) && !opt$do_opt_step) {
    do_opt_alt <- FALSE
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

    # Lack of progress in parameter optimization is not cause for stopping
    # embedding: just always restart
    mize_alt <- opt$mize_module$check_mize_convergence(step_info)
    if (mize_alt$is_terminated && mize_alt$terminate$what == "step_tol") {
      mize_alt$is_terminated <- FALSE
      mize_alt$terminate <- NULL
      mize_alt <- opt$mize_module$opt_init(mize_alt, par, fg_alt,
                                       step_tol = opt$step_tol,
                                       max_iter = Inf)
    }

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
  if (!is.null(method$gradient) && !is.null(method$gradient$fn)) {
    grad_fn <- method$gradient$fn
  }
  else {
    grad_fn <- dist2_gradient
  }
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
      grvec <- mat_to_par(grad_fn(inp, out, method, opt$mat_name)$gm)
      grvec
    },
    fg = function(par) {
      res <- par_to_out(par, opt, inp, out, method, nr)
      out <- res$out
      inp <- res$inp
      grvec <- mat_to_par(grad_fn(inp, out, method, opt$mat_name)$gm)

      list(
        fn = calculate_cost(method, inp, out),
        gr = grvec
      )
    }
  )

  res
}

# Generally, we allow sneer to restart once during optimization. After that,
# if the optimizer stops making progress (step size <~ 1e-8), we stop optimizing
# and end the embedding early.
# However, sometimes we will want to ignore any early convergence because we
# have temporarily changed some parameterized state that isn't being directly
# optimized: e.g. we are in the middle of multiscaling or stepping through
# perplexity, doing early exaggeration or going to optimize kernel parameters.
# During these periods we want to allow as many restarts as needed when that
# "indirect" state changes, but also to skip pointless optimization steps if
# we know we aren't going to make progress.
# is_before_step is a flag that checks whether we are attempting to restart
# before this iteration's optimization step or after. If before, and we are now
# at the first iteration where early stopping could happen, we don't want to
# count this restart as our one restart attempt, because whatever happened to
# cause the optimization to stall occurred during the pre-convergence
# iterations.
restart_if_possible <- function(opt, inp, iter, mize, par, fg, is_before_step) {
  if (!opt$restarted_once) {
    mize$is_terminated <- FALSE
    mize$terminate <- NULL

    worth_restarting <- FALSE
    if (!is.null(inp$updated_iter)) {
      if (is.null(opt$restart_iter)) {
        opt$restart_iter <- 0
      }
      worth_restarting <- inp$updated_iter > opt$restart_iter
    }
    if (worth_restarting) {
      mize <- opt$mize_module$opt_init(mize, par, fg,
                                       step_tol = opt$step_tol,
                                       max_iter = Inf,
                                       max_fn = opt$max_fn,
                                       max_gr = opt$max_gr,
                                       max_fg = opt$max_fg)
      # Restarting doesn't count towards convergence during any iterations where
      # we don't want to stop yet
      # If we are restarting *before* this iteration's optimization step
      # then we are restarting because of convergence during a pre-termination
      # iteration: e.g. we converged during early exaggeration and this is the
      # first iteration where early exaggeration is off. This restart shouldn't
      # count as the final restart.
      # Otherwise, we are checking *after* this iteration's optimization step
      # and a restart that occurs on the first non-termination iteration should
      # count as a final restart, e.g. this could be as early as iteration 0 if
      # we aren't doing any pre-convergence stuff.
      if (is_before_step) {
        is_final_restart <- iter > opt$convergence_iter
      }
      else {
        is_final_restart <- iter >= opt$convergence_iter
      }
      if (is_final_restart) {
        opt$restarted_once <- TRUE
        if (opt$verbose) {
          message("Restarting optimizer one last time at iter ", iter)
        }
      }
      else {
        if (opt$verbose) {
          message("Restarting optimizer at iter ", iter)
        }
      }
      opt$restart_iter <- iter
      opt$do_opt_step <- TRUE
    }
    else {
      opt$do_opt_step <- FALSE
    }
  }
  list(opt = opt, mize = mize)
}
