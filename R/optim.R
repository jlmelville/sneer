# Bridges Sneer and Mizer
make_optim_fg <- function(opt, inp, out, method, iter) {
  if (!is.null(inp$xm)) {
    nr <- nrow(inp$xm)
  }
  else {
    nr <- nrow(inp$dm)
  }

  list(
    fn = function(par) {
      out <- par_to_out(par, opt, inp, out, method, nr)
      calculate_cost(method, inp, out)
    },
    gr = function(par) {
      out <- par_to_out(par, opt, inp, out, method, nr)
      mat_to_par(gradient(inp, out, method, opt$mat_name)$gm)
    },
    fg = function(par) {
      out <- par_to_out(par, opt, inp, out, method, nr)
      list(
        fn = calculate_cost(method, inp, out),
        gr = mat_to_par(gradient(inp, out, method, opt$mat_name)$gm)
      )
    }
  )
}

# Convert 1D Parameter to Sneer Output
#
# This function converts the 1D parameter format expected by
# \code{\link[stats]{optim}} into the matrix format used by sneer.
#
# @param par Vector of embedded coordinates.
# @param opt Optimizer
# @param inp Input data.
# @param method Embedding method.
# @param nrow Number of rows in the sneer output matrix.
# @return Output data with coordinates converted from \code{par}.
par_to_out <- function(par, opt, inp, out, method, nrow) {
  dim(par) <- c(nrow, length(par) / nrow)
  out <- set_solution(inp, par, method, mat_name = opt$mat_name, out = out)
  out$dirty <- TRUE
  out
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

mizer_opt_step <- function(opt, method, inp, out, iter) {

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
  # par0 <- par
  mizer <- opt$mizer

  if (iter == 0) {
    mizer <- opt$mizer_module$opt_init(mizer, par, fg)
    opt$mizer <- mizer
  }

  if (!is.null(opt$old_cost_dirty) && opt$old_cost_dirty) {
    mizer <- opt$mizer_module$opt_clear_cache(mizer)
  }
  step_res <- opt$mizer_module$opt_step(mizer, par, fg, iter)
  mizer <- step_res$opt
  opt$mizer <- mizer

  par <- step_res$par

  if (!is.null(inp$xm)) {
    nr <- nrow(inp$xm)
  }
  else {
    nr <- nrow(inp$dm)
  }
  out <- par_to_out(par, opt, inp, out, method, nr)

  if (!is.null(opt$mizer$cache$gr_curr) &&
      length_vec(opt$mizer$cache$gr_curr) < sqrt(.Machine$double.eps)) {
    opt$stop_early <- TRUE
  }

  list(opt = opt, inp = inp, out = out, iter = iter)
}

mizer_opt <- function(opt_name, ...) {
  mizer_module <- mizer()

  if (class(opt_name) == "list") {
    opt <- opt_name
  }
  else {
    opt <- mizer_module$make_mizer(method = opt_name, ...)
  }

  list(
    name = opt_name,
    mat_name = "ym",
    optimize_step = mizer_opt_step,
    mizer_module = mizer_module,
    mizer = opt
  )
}

mizer_bold_nag_adapt <- function() {
  mizer_opt("SD", line_search = "bold", norm_direction = TRUE,
            mom_schedule = "nesterov", mom_type = "nesterov",
            mom_linear_weight = TRUE, nest_convex_approx = TRUE,
            nest_burn_in = 1, use_nest_mu_zero = TRUE, restart = "fn")
}

mizer_bold_nag <- function() {
  mizer_opt("SD", line_search = "bold", norm_direction = TRUE,
            mom_schedule = "nesterov", mom_type = "nesterov",
            mom_linear_weight = TRUE, nest_convex_approx = TRUE,
            nest_burn_in = 1, use_nest_mu_zero = TRUE)
}

mizer_back_nag <- function() {
  mizer_opt("SD", line_search = "back", norm_direction = TRUE,
            mom_schedule = "nesterov", mom_type = "nesterov",
            mom_linear_weight = TRUE, nest_convex_approx = TRUE,
            nest_burn_in = 1, use_nest_mu_zero = TRUE,
            c1 = 0.1, step_down = 0.8)
}

mizer_back_nag_adapt <- function() {
  mizer_opt("SD", line_search = "back", norm_direction = TRUE,
            mom_schedule = "nesterov", mom_type = "nesterov",
            mom_linear_weight = TRUE, nest_convex_approx = TRUE,
            nest_burn_in = 1, use_nest_mu_zero = TRUE, restart = "fn",
            c1 = 0.1, step_down = 0.8)
}

mizer_grad_descent <- function() {
  mizer_opt("SD", line_search = "bold", norm_direction = TRUE)
}
