conjugate_gradient <- function(nu = 0.1, update = pr_plus_update, eps = .Machine$double.eps) {
  list(
    calculate = function(opt, inp, out, method, iter) {
      if (is.null(opt$direction$value) || cg_restart(opt, nu)) {
        opt$direction$value <- -opt$gm
      }
      else {
        beta <- update(opt, eps)
        opt$direction$value <- -opt$gm + (beta * opt$direction$value)
      }
      d <- dot(opt$gm, opt$direction$value)
      if (d > 0) {
#        message("Next CG direction is not a descent direction, resetting to SD")
        opt$direction$value <- -opt$gm
      }
      list(opt = opt)
    },
    after_step = function(opt, inp, out, new_out, ok, iter) {
      #message("Step size = ", formatC(opt$step_size$value))
      opt$direction$g_old <- opt$gm
      list(opt = opt)
    }
  )
}

pr_update <- function(opt, eps = .Machine$double.eps) {
  g_old <- opt$direction$g_old
  g_new <- opt$gm
  (dot(g_new, g_new) - dot(g_new, g_old)) / (dot(g_old, g_old) + eps)
}

pr_plus_update <- function(opt, eps = .Machine$double.eps) {
  beta <- pr_update(opt, eps)
#  if (beta < 0) {
#    message("PR+: Not a descent direction, restarting")
#  }
  max(0, beta)
}

cg_restart <- function(opt, nu = 0.1) {
  g_old <- opt$direction$g_old
  g_new <- opt$gm

  ortho_test <- abs(dot(g_new, g_old)) / dot(g_new, g_new)
  should_restart <- ortho_test >= nu
#  if (should_restart) {
#    message("New CG direction not sufficiently orthogonal: ", formatC(ortho_test), " >= ", formatC(nu), " restarting")
#  }
  should_restart
}
