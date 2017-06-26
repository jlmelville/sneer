# Mize is distributed under this License:
#
# Copyright (c) 2017, James Melville
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
mize <- function() {
# Adaptive Restart --------------------------------------------------------

# Adds Adaptive Restart for optimizers which are using a momentum scheme.
#
# Candes and O'Donoghue suggested a restart scheme for Nesterov Accelerated
# Gradient schemes to avoid oscillatory behavior. It effectively restarts the
# momentum part of the optimization and can hence be applied to any optimization
# that uses momentum.
#
# There are two ways to check if a restart is needed: comparing the direction
# of the optimization with the gradient to see if the direction is a descent
# direction (gradient-based validation), or by comparing function evaluation
# before and after the step (function-based validation). Normally, the gradient
# based validation is cheaper, because the gradient has to be calculated
# anyway, but if using the momentum-version of NAG, this isn't available,
# because the gradient is calculated at the wrong position. If using a
# line search that calculates the function, then you probably get the function
# based validation for free.
#
# opt An optimizer
# validation_type - one of "fn" or "gr" for function- or gradient-based
# validation, respectively.
# wait - number of iterations after a restart to wait before validating again.
adaptive_restart <- function(opt, validation_type, wait = 1) {
  stage_names <- names(opt$stages)
  if (!("momentum" %in% stage_names)) {
    stop("Adaptive restart is only applicable to optimizers which use momentum")
  }
  if (stage_names[[2]] != "momentum" && validation_type == "gr") {
    stop("Can't use gradient-based adaptive restart with Nesterov momentum")
  }
  opt$restart_wait <- wait
  append_depends(opt, "momentum", "direction",
                 c('adaptive_restart', paste0('validate_', validation_type)))
}

# Function-based adaptive restart
adaptive_restart_fn <- function(opt) {
  adaptive_restart(opt, "fn")
}

# Gradient-based adaptive restart
adaptive_restart_gr <- function(opt) {
  adaptive_restart(opt, "gr")
}


# Replace the usual momentum after step event with one which restarts
# the momentum if validation failed
require_adaptive_restart <- function(opt, par, fg, iter, par0, update) {
  if (!opt$ok && can_restart(opt, iter)) {
    opt <- life_cycle_hook("momentum", "init", opt, par, fg, iter)
    opt$restart_at <- iter
  }
  else {
    opt$cache$update_old <- update
  }
  opt
}
attr(require_adaptive_restart, 'event') <- 'after step'
# Should have the same name as normal update old: we want to replace that hook
attr(require_adaptive_restart, 'name') <- 'update_old'
attr(require_adaptive_restart, 'depends') <- 'update_old_init'

# Add a depend function to one of opt, a stage or sub stage
append_depends <- function(opt, stage_type = NULL, sub_stage_type = NULL,
                           new_depends) {
  if (!is.null(sub_stage_type)) {
    if (is.null(stage_type)) {
      stop("Must provide stage for sub_stage '", sub_stage_type, "'")
    }
    stage <- opt$stages[[stage_type]]
    if (is.null(stage)) {
      stop("No stage '", stage_type, "' exists for this optimizer")
    }
    depends <- stage[[sub_stage_type]]$depends
  }
  else if (!is.null(stage)) {
    stage <- opt$stages[[stage_type]]
    if (is.null(stage)) {
      stop("No stage '", stage_type, "' exists for this optimizer")
    }
    depends <- stage$depends
  }
  else {
    depends <- opt$depends
  }
  if (is.null(depends)) {
    depends <- c()
  }

  depends <- c(depends, new_depends)

  if (!is.null(sub_stage_type)) {
    opt$stages[[stage_type]][[sub_stage_type]]$depends <- depends
  }
  else if (!is.null(stage)) {
    opt$stages[[stage_type]]$depends <- depends
  }
  else {
    opt$depends <- depends
  }

  opt
}

# True if we aren't currently waiting between restarts
can_restart <- function(opt, iter) {
  is.null(opt$restart_at) || iter - opt$restart_at > opt$restart_wait
}

# Returns a termination list if step falls below step_tol
# A zero step is allowed if this is a restart step
check_step_conv <- function(opt, iter, step = NULL, step_tol = NULL) {
  if (is.null(step) || is.null(step_tol) || is_restart_iter(opt, iter) ||
      step >= step_tol) {
    return()
  }
  list(what = "step_tol", val = step)
}

# Return a termination list if maximum number of function and/or gradient
# calls has been exceeded
check_counts <- function(opt, max_fn, max_gr, max_fg) {
  terminate <- NULL
  if (opt$counts$fn >= max_fn) {
    terminate <- list(
      what = "max_fn",
      val = opt$counts$fn
    )
  }
  else if (opt$counts$gr >= max_gr) {
    terminate <- list(
      what = "max_gr",
      val = opt$counts$gr
    )
  }
  else if (opt$counts$fn + opt$counts$gr >= max_fg) {
    terminate <- list(
      what = "max_fg",
      val = opt$counts$fn + opt$counts$gr
    )
  }
  terminate
}

# Return a termination list if the gradient 2 norm tolerance (grad_tol) or
# infinity norm tolerance is reached. Termination is also indicated if
# any element of the gradient vector is not finite. Requires the gradient
# have already been calculated - this routine does NOT calculate it if it's
# not present
check_gr_conv <- function(opt, grad_tol, ginf_tol) {
  if (is.null(opt$cache$gr_curr)) {
    return()
  }

  if (any(!is.finite(opt$cache$gr_curr))) {
    return(list(what = "gr_inf", val = Inf))
  }

  if (!is.null(grad_tol)) {
    gtol <- norm2(opt$cache$gr_curr)
    if (gtol <= grad_tol) {
      return(list(what = "grad_tol", val = gtol))
    }
  }

  if (!is.null(ginf_tol)) {
    gitol <- norm_inf(opt$cache$gr_curr)
    if (gitol <= ginf_tol) {
      return(list(what = "ginf_tol", val = gitol))
    }
  }
}

# Return a termination list if the absolute or relative tolerance is reached
# for the difference between fn_old and fn_new. Termination is also indicated
# if fn_new is non-finite. Tolerance is not checked if this is a restart
# iteration.
check_fn_conv <- function(opt, iter, fn_old, fn_new, abs_tol, rel_tol) {
  if (!is.finite(fn_new)) {
    return(list(what = "fn_inf", val = fn_new))
  }

  if (is.null(fn_old)) {
    return()
  }

  if (is_restart_iter(opt, iter)) {
    return()
  }

  if (!is.null(abs_tol)) {
    atol <- abs(fn_old - fn_new)
    if (atol < abs_tol) {
      return(list(what = "abs_tol", val = atol))
    }
  }

  if (!is.null(rel_tol)) {
    rtol <- abs(fn_old - fn_new) / min(abs(fn_new), abs(fn_old))
    if (rtol < rel_tol) {
      return(list(what = "rel_tol", val = rtol))
    }
  }
}

# True if this iteration was marked as a restart
# Zero step size and function difference is allowed under these circumstances.
is_restart_iter <- function(opt, iter) {
  !is.null(opt$restart_at) && opt$restart_at == iter
}

# Various Gradient-based optimization routines: steepest descent, conjugate
# gradient, BFGS etc.

# Gradient Direction -----------------------------------------------------------

# Creates a direction sub stage
make_direction <- function(sub_stage) {
  make_sub_stage(sub_stage, 'direction')
}

# Steepest Descent
#
# normalize - If TRUE, then the returned direction vector is normalized to unit
# length. This can be useful for some adaptive line search methods, so that the
# total step length is purely determined by the line search value, rather than
# the product of the line search value and the magnitude of the direction.
sd_direction <- function(normalize = FALSE) {

  make_direction(list(
    calculate = function(opt, stage, sub_stage, par, fg, iter) {
      sub_stage$value <- -opt$cache$gr_curr

      if (sub_stage$normalize) {
        sub_stage$value <- normalize(sub_stage$value)
      }

      list(sub_stage = sub_stage)
    },
    normalize = normalize
  ))
}


# Conjugate Gradient ------------------------------------------------------

# Conjugate gradient
#
# ortho_check - If TRUE, check successive direction are sufficiently orthogonal.
#   If the orthogonality check is failed, then the next step is steepest descent.
# nu - the orthogonality threshold. Used only if ortho_check is TRUE. Compared
#   with g_old . g_new / g_new . g_new
# cg_update - Function to generate the next direction using a method of e.g.
#   Fletcher-Reeves or Polak-Ribiere. Pass one of the cg_update functions
#   below, e.g. pr_plus_update
cg_direction <- function(ortho_check = FALSE, nu = 0.1,
                         cg_update = pr_plus_update,
                         eps = .Machine$double.eps) {
  make_direction(list(
    ortho_check = ortho_check,
    nu = nu,
    cg_update = cg_update,
    eps = eps,
    init = function(opt, stage, sub_stage, par, fg, iter) {
      sub_stage$value <- rep(0, length(par))
      sub_stage$pm_old <- rep(0, length(par))
      list(sub_stage = sub_stage)
    },
    calculate = function(opt, stage, sub_stage, par, fg, iter) {
      gm <- opt$cache$gr_curr
      gm_old <- opt$cache$gr_old
      pm_old <- sub_stage$pm_old

      # direction is initially steepest descent
      pm <- -gm

      if (!is.null(gm_old)
          && (!sub_stage$ortho_check
              || !sub_stage$cg_restart(gm, gm_old, sub_stage$nu))) {
        beta <- sub_stage$cg_update(gm, gm_old, pm_old, sub_stage$eps)
        pm <- pm + (beta * pm_old)
        descent <- dot(gm, pm)
        if (descent >= 0) {
          #message("Next CG direction is not a descent direction, resetting to SD")
          pm <- -gm
        }
      }

      sub_stage$value <- pm

      list(sub_stage = sub_stage)
    }
   , after_step = function(opt, stage, sub_stage, par, fg, iter, par0,
                         update) {
     sub_stage$pm_old <- sub_stage$value
     list(sub_stage = sub_stage)
   }
    , depends = c("gradient_old")
  ))
}


# CG update formulae, grouped according to their numerators similar to the
# discussion in Hager and Zhang's survey paper
# The FR, CD and DY updates are all susceptible to "jamming": they can end up
# with very small step sizes and make little progress.
# The Fletcher-Reeves update.
fr_update <- function(gm, gm_old, pm_old, eps = .Machine$double.eps) {
  dot(gm, gm) / (dot(gm_old, gm_old) + eps)
}

# Conjugate Descent update due to Fletcher
cd_update <- function(gm, gm_old, pm_old, eps = .Machine$double.eps) {
  dot(gm, gm) / (dot(pm_old, (gm - gm_old)) + eps)
}

# The Dai-Yuan update.
dy_update <- function(gm, gm_old, pm_old, eps = .Machine$double.eps) {
  -cd_update(gm, gm_old, pm_old, eps)
}

# HS, PR and LS share a numerator. According to Hager and Zhang, they
# perform better in practice than the FR, CD and DY updates, despite less
# being known about their provable global convergence properties.

# The Hestenes-Stiefel update.
hs_update <- function(gm, gm_old, pm_old, eps = .Machine$double.eps) {
  -(dot(gm, gm_old) - dot(gm, gm_old)) / (dot(pm_old, (gm - gm_old)) + eps)
}

# An "HS+" modification of Hestenes-Stiefel, in analogy to the "PR+" variant of
# Polak-Ribiere suggested by Powell. As far as I can tell, Hager and Zhang
# suggested this modification.
# Hager, W. W., & Zhang, H. (2006).
# A survey of nonlinear conjugate gradient methods.
# \emph{Pacific journal of Optimization}, \emph{2}(1), 35-58.
hs_plus_update <- function(gm, gm_old, pm_old, eps = .Machine$double.eps) {
  beta <- hs_update(gm, gm_old, pm_old, eps)
  max(0, beta)
}

# The Polak-Ribiere method for updating the CG direction. Also known as
# Polak-Ribiere-Polyak (PRP)
pr_update <- function(gm, gm_old, pm_old, eps = .Machine$double.eps) {
  dot(gm, gm - gm_old) / (dot(gm_old, gm_old) + eps)
}

# The "PR+" update due to Powell. Polak-Ribiere update, but if negative,
# restarts the CG from steepest descent. Prevents a possible lack of
# convergence when using a Wolfe line search.
pr_plus_update <- function(gm, gm_old, pm_old, eps = .Machine$double.eps) {
  beta <- pr_update(gm, gm_old, pm_old, eps)
  max(0, beta)
}

# Liu-Storey update
ls_update <- function(gm, gm_old, pm_old, eps = .Machine$double.eps) {
  -hs_update(gm, gm_old, pm_old, eps)
}

# Hager-Zhang update as used in CG_DESCENT
hz_update <- function(gm, gm_old, pm_old, eps = .Machine$double.eps) {
  ym <- gm - gm_old
  py <- dot(pm_old, ym)
  dot(ym - 2 * pm_old * (dot(ym, ym) / (py + eps)), (gm / (py + eps)))
}

# "Restricted" Hager-Zhang update as used in CG_DESCENT to ensure
# convergence. Analogous to the PR+ and HS+ updates, but dynamically adjusts
# the lower bound as convergence occurs. Choice of eta is from the CG_DESCENT
# paper
hz_plus_update <- function(gm, gm_old, pm_old, eps = .Machine$double.eps) {
  beta <- hz_update(gm, gm_old, pm_old, eps)
  eta <- 0.01
  eta_k <- -1 / (dot(pm_old, pm_old) * min(eta, dot(gm_old, gm_old)))
  max(eta_k, beta)
}


# Restart criteria due to Powell
# Checks that successive gradient vectors are sufficiently orthogonal
# g_new . g_old / g_new . g_new  must be greater than or equal to nu.
cg_restart <- function(g_new, g_old, nu = 0.1) {
  # could only happen on first iteration
  if (is.null(g_old)) {
    return(TRUE)
  }
  ortho_test <- abs(dot(g_new, g_old)) / dot(g_new, g_new)
  should_restart <- ortho_test >= nu
  should_restart
}


# BFGS --------------------------------------------------------------------

# The Broyden Fletcher Goldfarb Shanno method
# scale_inverse - if TRUE, scale the inverse Hessian approximation on the first
#   step.
bfgs_direction <- function(eps =  .Machine$double.eps,
                           scale_inverse = FALSE) {
  make_direction(list(
    eps = eps,
    init = function(opt, stage, sub_stage, par, fg, iter) {
      n <- length(par)
      sub_stage$value <- rep(0, n)
      sub_stage$hm <- diag(1, n)
      list(sub_stage = sub_stage)
    },
    calculate = function(opt, stage, sub_stage, par, fg, iter) {
      gm <- opt$cache$gr_curr
      gm_old <- opt$cache$gr_old
      if (is.null(gm_old)) {
        pm <- -gm
      }
      else {
        sm <- opt$cache$update_old
        hm <- sub_stage$hm

        ym <- gm - gm_old

        if (iter == 2 && scale_inverse) {
          # Nocedal suggests this heuristic for scaling the first
          # approximation in Chapter 6 Section "Implementation"
          # Also used in the definition of L-BFGS
          gamma <- dot(sm, ym) / dot(ym, ym)
          hm <- gamma * hm
        }

        rho <- 1 / (dot(ym, sm) + sub_stage$eps)
        im <- diag(1, nrow(hm))

        rss <- rho * outer(sm, sm)
        irsy <- im - rho * outer(sm, ym)
        irys <- im - rho * outer(ym, sm)

        sub_stage$hm <- (irsy %*% (hm %*% irys)) + rss

        pm <- as.vector(-sub_stage$hm %*% gm)

        descent <- dot(gm, pm)
        if (descent >= 0) {
          pm <- -gm
        }
      }
      sub_stage$value <- pm
      list(sub_stage = sub_stage)
    }
    , depends = c("gradient_old", "update_old")
  ))
}


# L-BFGS ------------------------------------------------------------------

# The Limited Memory BFGS method
#
# memory - The number of previous updates to store.
# scale_inverse - if TRUE, scale the inverse Hessian approximation at each step.
lbfgs_direction <- function(memory = 5, scale_inverse = FALSE,
                            eps = .Machine$double.eps) {
  if (memory < 1) {
    stop("memory must be > 0")
  }
  make_direction(list(
    memory = memory,
    k = 0,
    eps = eps,
    init = function(opt, stage, sub_stage, par, fg, iter) {
      n <- length(par)
      sub_stage$value <- rep(0, n)

      sub_stage$rhos <- c()
      sub_stage$sms <- c()
      sub_stage$yms <- c()

      list(sub_stage = sub_stage)

    },
    calculate = function(opt, stage, sub_stage, par, fg, iter) {
      gm <- opt$cache$gr_curr
      gm_old <- opt$cache$gr_old

      if (is.null(gm_old)) {
        pm <- -gm
      }
      else {
        rhos <- sub_stage$rhos
        sms <- sub_stage$sms
        yms <- sub_stage$yms

        # discard oldest values if we've reached memory limit
        if (length(sms) == sub_stage$memory) {
          sms <- sms[2:length(sms)]
          yms <- yms[2:length(yms)]
          rhos <- rhos[2:length(rhos)]
        }

        # y_{k-1}, s_{k-1}, rho_{k-1} using notation in Nocedal
        ym <- gm - gm_old
        sm <- opt$cache$update_old
        rho <- 1 / (dot(ym, sm) + sub_stage$eps)

        # append latest values to memory
        sms <- c(sms, list(sm))
        yms <- c(yms, list(ym))
        rhos <- c(rhos, list(rho))

        qm <- gm
        alphas <- rep(0, length(rhos))
        # loop backwards latest values first
        for (i in length(rhos):1) {
          alphas[i] <- rhos[[i]] * dot(sms[[i]], qm)
          qm <- qm - alphas[[i]] * yms[[i]]
        }

        if (scale_inverse) {
          gamma <- dot(sm, ym) / (dot(ym, ym) + sub_stage$eps)
        }
        else {
          gamma <- 1
        }

        hm <- rep(gamma, length(par))
        pm <- hm * qm
        # loop forwards
        for (i in 1:length(rhos)) {
          beta <- rhos[[i]] * dot(yms[[i]], pm)
          pm <- pm + sms[[i]] * (alphas[[i]] - beta)
        }
        pm <- -pm

        descent <- dot(gm, pm)
        if (descent >= 0) {
          pm <- -gm
        }

        sub_stage$value <- pm
        sub_stage$sms <- sms
        sub_stage$yms <- yms
        sub_stage$rhos <- rhos

      }
      sub_stage$value <- pm
      list(sub_stage = sub_stage)
    }
    , depends = c("gradient_old", "update_old")
  ))
}


# Newton Method -----------------------------------------------------------

# Newton method. Requires the Hessian to be calculated, via a function hs in fg.
newton_direction <- function() {
  make_direction(list(
    calculate = function(opt, stage, sub_stage, par, fg, iter) {
      gm <- opt$cache$gr_curr
      if (is.null(fg$hs)) {
        stop("No Hessian function 'hs', defined for fg")
      }
      hm <- fg$hs(par)

      chol_result <- try({
        # O(N^3)
          rm <- chol(hm)
        },
        silent = TRUE)
      if (class(chol_result) == "try-error") {
        # Suggested by https://www.r-bloggers.com/fixing-non-positive-definite-correlation-matrices-using-r-2/
        # Refs:
        # FP Brissette, M Khalili, R Leconte, Journal of Hydrology, 2007,
        # Efficient stochastic generation of multi-site synthetic precipitation data
        # https://www.etsmtl.ca/getattachment/Unites-de-recherche/Drame/Publications/Brissette_al07---JH.pdf
        # Rebonato, R., & JÃ¤ckel, P. (2011).
        # The most general methodology to create a valid correlation matrix for risk management and option pricing purposes.
        # doi 10.21314/JOR.2000.023
        # Also O(N^3)
        eig <- eigen(hm)
        eig$values[eig$values < 0] <- 1e-10
        hm <- eig$vectors %*% (eig$values * diag(nrow(hm))) %*% t(eig$vectors)
        chol_result <- try({
          rm <- chol(hm)
        }, silent = TRUE)
      }
      if (class(chol_result) == "try-error") {
        # we gave it a good go, but let's just do steepest descent this time
        #message("Hessian is not positive-definite, resetting to SD")
        pm <- -gm
      }
      else {
        # Forward and back solving is "only" O(N^2)
        pm <- hessian_solve(rm, gm)

        descent <- dot(gm, pm)
        if (descent >= 0) {
          pm <- -gm
        }
      }
      sub_stage$value <- pm
      list(sub_stage = sub_stage)
    }
  ))
}

# A Partial Hessian approach: calculates the Cholesky decomposition of the
# Hessian (or some approximation) on the first iteration only. Future steps
# solve using this Hessian and the current gradient.
partial_hessian_direction <- function() {
  make_direction(list(
    init = function(opt, stage, sub_stage, par, fg, iter) {
      hm <- fg$hs(par)
      sub_stage$rm <- chol(hm)
      list(sub_stage = sub_stage)
    },
    calculate = function(opt, stage, sub_stage, par, fg, iter) {
      gm <- opt$cache$gr_curr
      rm <- sub_stage$rm
      pm <- hessian_solve(rm, gm)

      descent <- dot(gm, pm)
      if (descent >= 0) {
        pm <- -gm
      }
      sub_stage$value <- pm
      list(sub_stage = sub_stage)
    }
  ))
}

# Solves U'Ux = b
# U is upper triangular; need to solve U'(Ux) = b
# Step 1: Solve U'y = b which is a forwardsolve
# Step 2: Solve Ub = y which is a backsolve
# Can avoid explicit transpose in Step 1
# by passing the transpose argument to backsolve
upper_solve <- function(um, bm) {
  backsolve(um, backsolve(um, bm, transpose = TRUE))
}

# Given the upper triangular cholesky decomposition of the hessian (or an
# approximation), U, and the gradient vector, g, solve Up = -g
hessian_solve <- function(um, gm) {
  nucol <- ncol(um)
  ngcol <- length(gm) / nucol
  dim(gm) <- c(nucol, ngcol)
  pm <- upper_solve(um, -gm)
  dim(pm) <- NULL
  pm
}

# Gradient Dependencies ------------------------------------------------------------

# Calculate the gradient at par.
require_gradient <- function(opt, stage, par, fg, iter) {
  if (!has_gr_curr(opt, iter)) {
    opt <- calc_gr_curr(opt, par, fg$gr, iter)

    if (any(!is.finite(opt$cache$gr_curr))) {
      opt$terminate <- list(
        what = "gr_inf",
        val = Inf
      )
      opt$is_terminated <- TRUE
    }
  }

  list(opt = opt)
}
attr(require_gradient, 'event') <- 'before gradient_descent'
attr(require_gradient, 'name') <- 'gradient'

# Caches the gradient at the current step.
require_gradient_old <- function(opt, par, fg, iter, par0, update) {
  opt$cache$gr_old <- opt$cache$gr_curr
  opt
}
attr(require_gradient_old, 'event') <- 'after step'
attr(require_gradient_old, 'name') <- 'gradient_old'
# Line Search as described by Hager and Zhang in:
#
# Hager, W. W., & Zhang, H. (2005).
# A new conjugate gradient method with guaranteed descent and an efficient line
# search.
# SIAM Journal on Optimization, 16(1), 170-192.
#
# Hager, W. W., & Zhang, H. (2006).
# Algorithm 851: CG_DESCENT, a conjugate gradient method with guaranteed
# descent.
# ACM Transactions on Mathematical Software (TOMS), 32(1), 113-137.
#
# I have tried to indicate in the comments which parts of which routines
# match the notation in the above papers.


# Adapter -----------------------------------------------------------------

hager_zhang <- function(c1 = c2 / 2, c2 = 0.1, max_fn = Inf,
                        strong_curvature = TRUE,
                        approx_armijo = TRUE) {
  if (c2 < c1) {
    stop("hager-zhang line search: c2 < c1")
  }
  function(phi, step0, alpha,
           total_max_fn = Inf, total_max_gr = Inf, total_max_fg = Inf,
           pm) {
    maxfev <- min(max_fn, total_max_fn, total_max_gr, floor(total_max_fg / 2))
    if (maxfev <= 0) {
      return(list(step = step0, nfn = 0, ngr = 0))
    }
    res <- line_search_hz(alpha, step0, phi, c1 = c1, c2 = c2,
                               eps = 1e-6, theta = 0.5, rho = 5,
                               gamma = 0.66,
                               max_fn = maxfev,
                               xtol = 1e-6,
                               strong_curvature = strong_curvature,
                               always_check_convergence = TRUE,
                               approx_armijo = approx_armijo,
                               verbose = FALSE)

    res$ngr = res$nfn
    res
  }
}


# Line Search -------------------------------------------------------------

# Routine 'Line Search Algorithm' L0-3

# alpha initial guess
# step0
# phi
# c1
# c2
# eps - determines when to apply approximate Wolfe condition. Ignored if
# approx_armijo is FALSE.
# theta - bisection weight during bracket update
# rho - factor to increase step size by during bracket phase
# gamma - bisection weight if secant2 step fails
# max_fn - maximum number of function evaluations allowed
# xtol - stop if bracket size ever falls below alpha * xtol
# strong_curvature - use Strong curvature condition
# always_check_convergence - if FALSE, only check for satisfaction of the
#   Wolfe conditions when a step size is produced by interpolation, not
#   bisection or the bracketing phase. May produce a result closer to a
#   minimizer.
# approx_armijo - if TRUE, use the approximate version of the armijo condition
# when testing the Wolfe conditions.
line_search_hz <- function(alpha, step0, phi, c1 = 0.1, c2 = 0.9,
                           eps = 1e-6, theta = 0.5, rho = 5,
                           gamma = 0.66,
                           max_fn = Inf,
                           xtol = 1e-6,
                           strong_curvature = FALSE,
                           always_check_convergence = TRUE,
                           approx_armijo = TRUE,
                           verbose = FALSE) {
  ls_max_fn <- max_fn
  if (max_fn == 0) {
    return(list(step = step0, nfn = 0))
  }
  # For approximate Wolfe error parameters implicitly set delta = 0
  # no C or Q parameters to update
  eps_k <- eps * abs(step0$f)

  # Bisect alpha if needed in case initial alpha guess gives a mad or bad value
  nfn <- 0
  result <- find_finite(phi, alpha, max_fn, min_alpha = 0)
  nfn <- nfn + result$nfn
  if (!result$ok) {
    if (verbose) {
      message("Unable to create finite initial guess")
    }
    return(list(step = step0, nfn = nfn))
  }
  step_c <- result$step

  if (always_check_convergence && hz_ok_step(step_c, step0, c1, c2, eps_k,
                                             strong_curvature = strong_curvature,
                                             approx_armijo = approx_armijo)) {
    if (verbose) {
      message("initial step OK = ", formatC(step_c$alpha))
    }
    return(list(step = step_c, nfn = nfn))
  }
  ls_max_fn <- max_fn - nfn
  if (ls_max_fn <= 0) {
    if (verbose) {
      message("max fn reached after initial step")
    }
    return(list(step = step_c, nfn = nfn))
  }

  # L0
  br_res <- bracket_hz(step_c, step0, phi, eps_k, ls_max_fn, theta, rho,
                       xtol = xtol, verbose = verbose)
  bracket <- br_res$bracket
  nfn <- nfn + br_res$nfn
  if (!br_res$ok) {
    LOpos <- which.min(bracket_props(bracket, 'f'))
    if (verbose) {
      message("Failed to create bracket, aborting, alpha = ",
              formatC(bracket[[LOpos]]$alpha))
    }
    return(list(step = bracket[[LOpos]], nfn = nfn))
  }

  # Check for T1/T2
  if (always_check_convergence) {
    LOpos <- hz_ok_bracket_pos(bracket, step0, c1, c2, eps_k,
                               strong_curvature = strong_curvature,
                               approx_armijo = approx_armijo)
    if (LOpos > 0) {
      if (verbose) {
        message("Bracket OK after bracket phase alpha = ",
                formatC(bracket[[LOpos]]$alpha))
      }
      return(list(step = bracket[[LOpos]], nfn = nfn))
    }
  }
  ls_max_fn <- max_fn - nfn
  if (ls_max_fn <= 0) {
    if (verbose) {
      message("max fn reached after bracket phase")
    }
    LOpos <- which.min(bracket_props(bracket, 'f'))
    return(list(step = bracket[[LOpos]], nfn = nfn))
  }

  old_bracket <- bracket
  # Zoom step
  while (TRUE) {
    if (verbose) {
      message("BRACKET: ", format_bracket(old_bracket))
    }
    # L1

    # do some of S1 here in case it's already an ok step
    alpha_c <- secant_hz(old_bracket[[1]], old_bracket[[2]])

    if (!is_finite_numeric(alpha_c)) {
      # probably only an issue when tolerances are very low and approx armijo
      # is off: can get NaN when bracket size approaches zero
      LOpos <- which.min(bracket_props(bracket, 'f'))
      if (verbose) {
        message("bad secant alpha, aborting line search")
      }
      return(list(step = bracket[[LOpos]], nfn = nfn))
    }
    fsec_res <- find_finite(phi, alpha_c, ls_max_fn,
                            min_alpha = bracket_min_alpha(old_bracket))
    nfn <- nfn + fsec_res$nfn
    ls_max_fn <- max_fn - nfn
    if (!fsec_res$ok) {
      if (verbose) {
        message("No finite alpha during secant bisection, aborting line search")
      }
      break
    }
    step_c <- fsec_res$step

    if (verbose) {
      message("S1: secant step_c alpha = ", formatC(alpha_c))
    }
    if (hz_ok_step(step_c, step0, c1, c2, eps_k,
                   strong_curvature = strong_curvature,
                   approx_armijo = approx_armijo)) {
      if (verbose) {
        message("step OK after secant alpha = ", formatC(step_c$alpha))
      }
      return(list(step = step_c, nfn = nfn))
    }
    ls_max_fn <- max_fn - nfn
    if (ls_max_fn <= 0) {
      if (verbose) {
        message("max fn reached after secant")
      }
      bracket[[3]] <- step_c
      LOpos <- which.min(bracket_props(bracket, 'f'))
      return(list(step = bracket[[LOpos]], nfn = nfn))
    }

    sec2_res <- secant2_hz(old_bracket, step_c, step0, phi, eps_k, ls_max_fn,
                           theta, verbose = verbose)
    bracket <- sec2_res$bracket
    nfn <- nfn + sec2_res$nfn
    if (!sec2_res$ok) {
      break
    }

    if (verbose) {
      message("new bracket: ", format_bracket(bracket))
    }
    # Check for T1/T2
    LOpos <- hz_ok_bracket_pos(bracket, step0, c1, c2, eps_k,
                               strong_curvature = strong_curvature,
                               approx_armijo = approx_armijo)
    if (LOpos > 0) {
      # if we only allow interpolated steps to count as converged,
      # check that any satisfactory result of secant2 came from a secant step
      # (i.e. wasn't already part of the bracket)
      if (always_check_convergence ||
          (bracket[[LOpos]]$alpha != old_bracket[[1]]$alpha &&
           bracket[[LOpos]]$alpha != old_bracket[[2]]$alpha)) {
        if (verbose) {
          message("Bracket OK after secant2 alpha = ", formatC(bracket[[LOpos]]$alpha))
        }
        return(list(step = bracket[[LOpos]], nfn = nfn))
      }
    }
    ls_max_fn <- max_fn - nfn
    if (ls_max_fn <= 0) {
      if (verbose) {
        message("max fn reached after secant2")
      }
      break
    }

    # L2
    # Ensure the bracket size decreased by a factor of gamma
    old_range <- bracket_props(old_bracket, 'alpha')
    old_diff <- abs(old_range[2] - old_range[1])
    new_range <- bracket_props(bracket, 'alpha')
    new_diff <- abs(new_range[2] - new_range[1])

    if (verbose) {
      message("Bracket size = ", formatC(new_diff), " old bracket size = ",
              formatC(old_diff))
    }

    if (new_diff < xtol * max(new_range)) {
      if (verbose) {
        message("Bracket size reduced below tolerance")
      }

      break
    }

    if (new_diff > old_diff * gamma) {
      if (verbose) {
        message("Bracket size did not decrease sufficiently: bisecting")
      }
      # bracket size wasn't decreased sufficiently: bisection step
      alpha_c <- mean(new_range)
      bisec_res <- find_finite(phi, alpha_c, ls_max_fn,
                               min_alpha = bracket_min_alpha(bracket))
      nfn <- nfn + bisec_res$nfn
      ls_max_fn <- max_fn - nfn
      if (!bisec_res$ok) {
        if (verbose) {
          message("No finite alpha during bisection, aborting line search")
        }
        break
      }
      step_c <- bisec_res$step

      if (ls_max_fn <= 0) {
        if (verbose) {
          message("Reached max_fn, returning")
        }

        bracket[[3]] <- step_c
        LOpos <- which.min(bracket_props(bracket, 'f'))
        return(list(step = bracket[[LOpos]], nfn = nfn))
      }

      up_res <- update_bracket_hz(bracket, step_c, step0, phi, eps_k, ls_max_fn,
                                  xtol = xtol, theta)
      bracket <- up_res$bracket
      nfn <- nfn + up_res$nfn
      if (!up_res$ok) {
        break
      }
      # Check for T1/T2
      if (always_check_convergence) {
        LOpos <- hz_ok_bracket_pos(bracket, step0, c1, c2, eps_k,
                                   strong_curvature = strong_curvature,
                                   approx_armijo = approx_armijo)
        if (LOpos > 0) {
          if (verbose) {
            message("Bracket OK after bisection alpha = ", formatC(bracket[[LOpos]]$alpha))
          }
          return(list(step = bracket[[LOpos]], nfn = nfn))
        }
      }
      ls_max_fn <- max_fn - nfn
      if (ls_max_fn <= 0) {
        if (verbose) {
          message("max fn reached after bisection")
        }
        break
      }
    }

    # L3
    old_bracket <- bracket
  }

  LOpos <- which.min(bracket_props(bracket, 'f'))
  list(step = bracket[[LOpos]], nfn = nfn)
}

# Bracket -----------------------------------------------------------------

# Routine 'bracket' B1-3
# Generates an initial bracket satisfying the opposite slope condition
# or if max_fn is reached or a non-finite f/g value generated, returns the best
# two values it can find.
bracket_hz <- function(step_c, step0, phi, eps, max_fn, theta = 0.5, rho = 5,
                       xtol = .Machine$double.eps, verbose = FALSE) {
  step_c_old <- step0
  # used only if bracket step fails (hit max_fn or non-finite f/g)
  step_c_old_old <- step0

  ls_max_fn <- max_fn
  nfn <- 0

  # step_c is the latest attempt at a bracketed point
  # step_c_old is the previous step_c
  while (TRUE) {
    if (verbose) {
      message("Bracketing: step = ", formatC(step_c$alpha))
    }
    if (step_c$d > 0) {
      # B1 slope is +ve: bracketing successful: return [step_c_old, step_c]
      if (verbose) {
        message("B1: slope +ve")
      }
      return(list(bracket = list(step_c_old, step_c), nfn = nfn, ok = TRUE))
    }
    if (step_c$f > step0$f + eps) {
      # B2 slope is -ve but f is higher than starting point
      # we must have stepped too far beyond the minimum and the +ve slope
      # and its maximum
      # find the minimum by weighted bisection
      if (verbose) {
        message("B2: f > phi0 + eps")
      }

      if (ls_max_fn <= 0) {
        # avoid returning step_c in this case, which might be outside the
        # current minimizer "basin"
        return(list(bracket = list(step_c_old_old, step_c_old), nfn = nfn,
                    ok = FALSE))
      }

      # Probably could use step_c_old as LHS of bracket
      # but HZ paper specifies step0
      bracket_sub = list(step0, step_c)
      bisect_res <- update_bracket_bisect_hz(bracket_sub, step0, phi, eps,
                                             ls_max_fn, theta,
                                             xtol = xtol,
                                             verbose = verbose)
      bisect_res$nfn <- bisect_res$nfn + nfn
      # return bisection result: may have failed
      return(bisect_res)
    }

    # B3: slope is -ve and f < f0, so we haven't passed the minimum yet
    if (ls_max_fn <= 0) {
      return(list(bracket = list(step_c_old, step_c), nfn = nfn, ok = FALSE))
    }

    # extrapolate: increase the step size
    step_c_old_old <- step_c_old
    step_c_old <- step_c
    alpha <- step_c$alpha * rho

    fext_res <- find_finite(phi, alpha, ls_max_fn, min_alpha = step_c$alpha)
    nfn <- nfn + fext_res$nfn
    ls_max_fn <- max_fn - nfn
    if (!fext_res$ok) {
      if (verbose) {
        message("No finite alpha during extrapolation bisection, aborting line search")
      }
      return(list(bracket = list(step_c_old_old, step_c_old), nfn = nfn,
                  ok = FALSE))
    }
    step_c <- fext_res$step
  }
}

# Update ------------------------------------------------------------------

# routine 'update' U0-U3
# Given a bracket, create a new bracket with end points which are inside
# the original bracket.
# Returns with ok = TRUE if any of U0-U3 succeed (i.e. U0 is not a failure).
# Returns with ok = FALSE if U3 fails (i.e. exceed max_fn or non-finite
# function/gradient is calculated before bisection succeeds)
update_bracket_hz <- function(bracket, step_c, step0, phi, eps, max_fn,
                              theta = 0.5, xtol = .Machine$double.eps,
                              verbose = FALSE) {
  if (verbose) {
    message("U: alpha = ", formatC(step_c$alpha),
            " bracket = ", format_bracket(bracket))
  }
  nfn <- 0
  ok <- TRUE

  if (!is_in_bracket(bracket, step_c$alpha)) {
    # U0: c is not inside, reject it
    new_bracket <- bracket
    if (verbose) {
      message("U0: step not in bracket, reject")
    }
  }
  else if (step_c$d >= 0) {
    # U1: c is on the +ve slope, make it the new hi
    new_bracket <- list(bracket[[1]], step_c)
    if (verbose) {
      message("U1: step has +ve slope, new hi")
    }
  }
  else if (step_c$f <= step0$f + eps) {
    # U2: c is on the -ve slope and closer to minimum than a
    # make it the new lo
    new_bracket <- list(step_c, bracket[[2]])
    if (verbose) {
      message("U2: step has -ve slope and closer to minimum, new lo")
    }
  }
  else {
    # U3
    # c is on the -ve slope but larger than f0: must have missed the minimum
    # and the +ve slope and the maximum
    # find new hi by weighted bisection
    if (verbose) {
      message("U3: step has -ve slope but not closer to minimum, bisect")
    }
    sub_bracket <- list(bracket[[1]], step_c)
    sub_res <- update_bracket_bisect_hz(sub_bracket, step0, phi, eps, max_fn,
                                        theta, xtol = xtol,
                                        verbose = verbose)
    new_bracket <- sub_res$bracket
    nfn <- sub_res$nfn
    ok <- sub_res$ok
  }

  list(bracket = new_bracket, nfn = nfn, ok = ok)
}

# U3a-c from routine 'update'
#
# Use weighted bisection of the current bracket so that bracket[2] contains a
# step size with a +ve slope. bracket[1] will also be updated if a point
# with -ve slope closer to the minimizer is found.
#
# Also used during the bracket step if the step size gets too large.
# Called when step size leads to a -ve slope but f is > f0, implying that
# step size was so large it missed the local minimum, the +ve slope and the
# local maximum and we are now going downhill to some other minimum.
# Use weighted bisection until the hi of the bracket has a +ve slope
# lo of bracket will also be updated if we find a suitable point during
# bisection.
#
# If bisection succeeds, then this function returns with ok = TRUE.
# If the number of bisections exceeds max_fn, or if any step size contains a
# non-finite slope or function value, the most recent finite-valued bracket is
# returned with ok = FALSE
update_bracket_bisect_hz <- function(bracket, step0, phi, eps, max_fn,
                                     theta = 0.5,
                                     xtol = .Machine$double.eps,
                                     verbose = FALSE) {
  res <- bracket
  nfn <- 0
  ls_max_fn <- max_fn
  ok <- FALSE
  while (TRUE) {
    if (verbose) {
      message("U3: Bracket: ", format_bracket(res), " width = ",
              bracket_width(res))
    }
    if (bracket_width(res) <= xtol * res[[2]]$alpha) {
      if (verbose) {
        message("Relative bracket width reduced below tolerance, aborting")
      }
      break
    }

    ls_max_fn <- max_fn - nfn
    if (ls_max_fn <= 0) {
      if (verbose) {
        message("max_fn reached, aborting bisection bracket update")
      }
      break
    }

    # U3a new point is (weighted) bisection of current bracket
    alpha <- (1 - theta) * res[[1]]$alpha + theta * res[[2]]$alpha
    fwbi_res <- find_finite(phi, alpha, ls_max_fn,
                            min_alpha = bracket_min_alpha(res))
    nfn <- nfn + fwbi_res$nfn
    ls_max_fn <- max_fn - nfn
    if (!fwbi_res$ok) {
      if (verbose) {
        message("No finite alpha during weighted bisection, aborting line search")
      }
      break
    }
    step_d <- fwbi_res$step

    if (step_d$d >= 0) {
      # d is on +ve slope, make it the new hi and return
      res[[2]] <- step_d
      ok <- TRUE
      break
    }
    if (step_d$f <= step0$f + eps) {
      if (verbose) {
        message("U3b: alpha ", formatC(step_d$alpha),
                " f = ", formatC(step_d$f),
                " d  = ", formatC(step_d$d),
                " closer to minimizer: new lo")
      }
      # U3b: d is on -ve slope but closer to minimizer, make it new lo and loop
      res[[1]] <- step_d
    } else {
      # U3c: d has -ve slope but still > f0 so still too large a step,
      # make it the new hi and loop
      if (verbose) {
        message("U3b: alpha ", formatC(step_d$alpha),
                " f = ", formatC(step_d$f),
                " d  = ", formatC(step_d$d),
                " -ve slope but > f0: new hi")
      }
      res[[2]] <- step_d
    }
  }

  list(bracket = res, nfn = nfn, ok = ok)
}

# Secant ------------------------------------------------------------------

# Routine 'secant2'
# Do the secant step to generate c for step S1 outside of this routine because
# it may be an acceptable step without having to update any brackets
secant2_hz <- function(bracket, step_c, step0, phi, eps, max_fn,
                       theta = 0.5, xtol = .Machine$double.eps,
                       verbose = FALSE) {
  nfn <- 0
  ls_max_fn <- max_fn

  if (ls_max_fn <= 0) {
    return(list(bracket = bracket, nfn = nfn, ok = FALSE))
  }

  if (verbose) {
    message("S1: Creating AB")
  }
  bracket_AB_res <- update_bracket_hz(bracket, step_c, step0, phi, eps, theta,
                                      xtol = xtol, verbose = verbose)
  if (!bracket_AB_res$ok) {
    return(list(bracket = bracket, nfn = nfn, ok = FALSE))
  }
  bracket_AB <- bracket_AB_res$bracket

  if (verbose) {
    message("S1: secant alpha = ", formatC(step_c$alpha),
            " ab = ", format_bracket(bracket),
            " AB = ", format_bracket(bracket_AB))
  }

  ok <- TRUE
  # following two if blocks rely on exact floating point comparison
  if (step_c$alpha == bracket_AB[[2]]$alpha) {
    # S2 c == B
    alpha_cbar <- secant_hz(bracket[[2]], bracket_AB[[2]])

    if (verbose) {
      message("S2 c = B: cbar = secant(b, B) = (",
              formatC(bracket[[2]]$alpha), ", ",
              formatC(bracket_AB[[2]]$alpha), ") = ",
              formatC(alpha_cbar))
    }
    # update routine would also check that c_bar is in [A,B] but do it manually
    # here to avoid calculating phi(c_bar) if we don't need to
    if (is.finite(alpha_cbar) && is_in_bracket(bracket_AB, alpha_cbar)
        && max_fn > 0) {
      step_cbar <- phi(alpha_cbar)
      nfn <- nfn + 1
      max_fn <- ls_max_fn - nfn

      res <- update_bracket_hz(bracket_AB, step_cbar, step0, phi, eps, max_fn,
                               theta, xtol = xtol, verbose = verbose)
      new_bracket <- res$bracket
      nfn <- nfn + res$nfn
      max_fn <- ls_max_fn - nfn
      ok <- res$ok
    }
    else {
      new_bracket <- bracket_AB
    }
    if (verbose) {
      message("S2 bracket: ", format_bracket(new_bracket))
    }
  }
  else if (step_c$alpha == bracket_AB[[1]]$alpha) {
    # S3 c == A
    alpha_cbar <- secant_hz(bracket[[1]], bracket_AB[[1]])
    if (verbose) {
      message("S3 c = A: cbar = secant(a, A) = (",
              formatC(bracket[[1]]$alpha), ", ",
              formatC(bracket_AB[[1]]$alpha), ") = ",
              formatC(alpha_cbar))
    }
    if (is.finite(alpha_cbar) && is_in_bracket(bracket_AB, alpha_cbar)
        && max_fn > 0) {

      step_cbar <- phi(alpha_cbar)
      nfn <- nfn + 1
      max_fn <- ls_max_fn - nfn

      res <- update_bracket_hz(bracket_AB, step_cbar, step0, phi, eps, max_fn,
                               theta, xtol = xtol, verbose = verbose)
      new_bracket <- res$bracket
      nfn <- nfn + res$nfn
      max_fn <- ls_max_fn - nfn
      ok <- res$ok
    }
    else {
      new_bracket <- bracket_AB
    }
    if (verbose) {
      message("S3 bracket: ", format_bracket(new_bracket))
    }
  }
  else {
    # S4
    new_bracket <- bracket_AB
    if (verbose) {
      message("S4 bracket: ", format_bracket(new_bracket))
    }
  }

  list(bracket = new_bracket, nfn = nfn, ok = ok)
}

secant_hz <- function(step_a, step_b) {
  (step_a$alpha * step_b$d - step_b$alpha * step_a$d) / (step_b$d - step_a$d)
}


# Termination Conditions --------------------------------------------------

hz_ok_step <- function(step, step0, c1, c2, eps, strong_curvature = FALSE,
                       approx_armijo = TRUE) {
  if (strong_curvature) {
    ok <- strong_curvature_ok_step(step0, step, c2)
  }
  else {
    ok <- curvature_ok_step(step0, step, c2)
  }
  if (!ok) {
    return(ok)
  }

  if (armijo_ok_step(step0, step, c1)) {
    return(ok)
  }

  approx_armijo && (step$f <= step0$f + eps) &&
    approx_armijo_ok_step(step0, step, c1)
}

hz_ok_bracket_pos <- function(bracket, step0, c1, c2, eps,
                          strong_curvature = FALSE,
                          approx_armijo = TRUE) {
  ok_pos <- 0
  if (hz_ok_step(bracket[[1]], step0, c1, c2, eps,
                 strong_curvature = strong_curvature,
                 approx_armijo = approx_armijo)) {
    ok_pos <- 1
  }
  if (hz_ok_step(bracket[[2]], step0, c1, c2, eps,
                 strong_curvature = strong_curvature,
                 approx_armijo = approx_armijo)) {
    # if somehow we've reached a situation where both sides of the bracket
    # meet the conditions, choose the one with the lower function value
    if (ok_pos == 0 || bracket[[2]]$f < bracket[[1]]$f) {
      ok_pos <- 2
    }
  }
  ok_pos
}

# Adaptive algorithms, normally used in the neural network/deep learning
# communities, and normally associated with stochastic gradient descent.

# Implementation-wise treated as step size methods, but they actually modify
# both the direction and step size simultaneously to allow each parameter to
# be updated at a different rate. The direction method should always be
# steepest descent.

# Delta-Bar-Delta ----------------------------------------------------------
# A modification of Jacobs' Delta Bar Delta method is used to optimize
# the objective function in t-SNE.
#
# The heuristic is to look at successive directions of the parameter: if the
# direction is the same as the previous iteration, the minimum has yet to
# be encountered, so increase the step in that direction. If the direction
# has changed, then the minimum has been skipped, so reduce the step size.
#
# The t-SNE version differs from the version in the paper by using the update
# vector stored for use in momentum rather than storing a separate vector of
# exponentially weighted gradients.
#
# Default arguments are similar to bold driver, in that the learning rate values
# are multiplied, but for the authentic DBD experience (and also as used in
# t-SNE), you can specify kappa_fun to be `+` to add kappa to the learning rate
# when increasing the learning rate.
#
# Idea behind combining momentum with adaptive step size: slide 25 of
# http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
# notes that
# "Use the agreement in sign between the current gradient for a weight and the
# velocity for that weight (Jacobs, 1989)."
#
# kappa amount to increase the learning rate by.
# kappa_fun - operator to apply to kappa and the current learning rate when
#  increasing the learning rate. Set it to `+` to add to the current learning
#  rate rather than scaling up proportionally to the current value.
delta_bar_delta <- function(kappa = 1.1, kappa_fun = `*`,
                            phi = 0.5, epsilon = 1,
                            min_eps = 0,
                            theta = 0.1,
                            use_momentum = FALSE) {

  if (kappa <= 0) {
    stop("kappa must be positive")
  }
  if (!is_in_range(phi, 0, 1)) {
    stop("phi must be between 0 and 1")
  }
  if (!is_in_range(theta, 0, 1)) {
    stop("theta must be between 0 and 1")
  }

  make_step_size(list(
    name = "delta_bar_delta",
    kappa = kappa,
    kappa_fun = kappa_fun,
    phi = phi,
    min_eps = min_eps,
    theta = theta,
    epsilon = epsilon,
    use_momentum = use_momentum,
    init = function(opt, stage, sub_stage, par, fg, iter) {
      sub_stage$delta_bar_old <- rep(0, length(par))
      sub_stage$gamma_old <- rep(1, length(par))
      sub_stage$value <- rep(sub_stage$init_eps, length(par))
      list(sub_stage = sub_stage)
    },
    calculate = function(opt, stage, sub_stage, par, fg, iter) {
      delta <- opt$cache$gr_curr

      if (!is.numeric(sub_stage$epsilon)) {
        d0 <- dot(delta, stage$direction$value)
        sub_stage$epsilon <- guess_alpha0(sub_stage$epsilon,
                                          x0 = NULL,
                                          f0 = NULL,
                                          gr0 = delta,
                                          d0 = d0,
                                          try_newton_step = FALSE)
      }

      if (use_momentum && !is.null(opt$cache$update_old)) {
        # previous update includes -eps*grad_old, so reverse sign
        delta_bar_old <- -opt$cache$update_old
      }
      else {
        delta_bar_old <- sub_stage$delta_bar_old
      }
      # technically delta_bar_delta = delta_bar * delta
      # but only its sign matters, so just compare signs of delta_bar and delta
      # Force step size increase on first stage to be like the t-SNE
      # implementation
      if (all(delta_bar_old == 0)) {
        delta_bar_delta <- TRUE
      }
      else {
        delta_bar_delta <- sign(delta_bar_old) == sign(delta)
      }
      kappa <- sub_stage$kappa
      phi <- sub_stage$phi
      gamma_old <- sub_stage$gamma_old

      # signs of delta_bar and delta are the same, increase step size
      # if they're not, decrease.
      gamma <-
        (kappa_fun(gamma_old,kappa)) * abs(delta_bar_delta) +
        (gamma_old * phi) * abs(!delta_bar_delta)

      sub_stage$value <- clamp(sub_stage$epsilon * gamma,
                               min_val = sub_stage$min_eps)
      if (!use_momentum || is.null(opt$cache$update_old)) {
        theta <- sub_stage$theta
        sub_stage$delta_bar_old <- ((1 - theta) * delta) + (theta * delta_bar_old)
      }
      sub_stage$gamma_old <- gamma
      list(opt = opt, sub_stage = sub_stage)
    },
    depends = c("gradient")
  ))
}
# Life Cycle ------------------------------------------------------
# Various internals for registering functions that should fire during certain
# points of the optimization. You don't want to look to closely at any of this.

# Calls all hooks registered with the phase firing this event
life_cycle_hook <- function(phase, advice_type, opt, par, fg, iter, ...) {
  handler <- life_cycle_handler(phase, advice_type, opt)
  if (is.null(handler)) {
    opt <- default_handler(phase, advice_type, opt, par, fg, iter, ...)
  }
  else {
    opt <- handler(opt, par, fg, iter, ...)
  }
  opt
}

# Look for a handler function that will deal with this event
life_cycle_handler <- function(phase, advice_type, opt) {
  handlers <- opt$handlers
  if (is.null(handlers)) {
    return(NULL)
  }
  handlers <- handlers[[phase]]
  if (is.null(handlers)) {
    return(NULL)
  }
  handlers[[advice_type]]
}

# A default function to handle an event by just iterating over ever function
# that was registered and invoke them in turn
default_handler <- function(phase, advice_type, opt, par, fg, iter, ...) {
  hooks <- opt$hooks
  if (is.null(hooks)) {
    return(opt)
  }
  hooks <- hooks[[phase]]
  if (is.null(hooks)) {
    return(opt)
  }
  hooks <- hooks[[advice_type]]
  if (is.null(hooks)) {
    return(opt)
  }

  for (name in names(hooks)) {
    hook <- hooks[[name]]
    opt <- hook(opt, par, fg, iter, ...)
  }
  opt
}

# registers all hook functions:
register_hooks <- function(opt) {
  # Optimizer hook

  for (name in names(opt)) {
    if (!is.null(attr(opt[[name]], 'event'))) {
      opt <- register_hook(opt, opt[[name]])
    }
  }
  if (!is.null(opt$depends)) {
    opt <- depends_to_hooks(opt, opt)
  }

  # Stage hook
  for (i in 1:length(opt$stages)) {
    stage <- opt$stages[[i]]
    for (stage_prop_name in names(stage)) {
      stage_prop <- stage[[stage_prop_name]]
      if (!is.null(attr(stage_prop, 'event'))) {
        opt <- register_hook(opt, stage_prop, stage$type)
      }
    }

    if (!is.null(stage$depends)) {
      opt <- depends_to_hooks(opt, stage)
    }

    # Sub stage hooks
    for (sub_stage_type in c("direction", "step_size")) {
      sub_stage <- stage[[sub_stage_type]]
      for (sub_stage_prop_name in names(sub_stage)) {
        sub_stage_prop <- sub_stage[[sub_stage_prop_name]]
        if (!is.null(attr(sub_stage_prop, 'event'))) {
          opt <- register_hook(opt, sub_stage[[sub_stage_prop_name]],
                               stage$type, sub_stage_type)
        }
      }

      if (!is.null(sub_stage$depends)) {
        opt <- depends_to_hooks(opt, sub_stage)
      }
    }
  }
  opt
}

# An event is something like "before step"
# Borrowing terminology from Aspect Oriented Programming:
# The first part is the "advice type": before, after.
# The second part is the "join point": the life cycle function that is going
#  to fire.
# Event can consist of the just the join point, e.g. "init" and the advice
#   type is then implicitly assumed to be "during".
register_hook <- function(opt, hook,
                          stage_type = NULL,
                          sub_stage_type = NULL) {
  name <- attr(hook, "name")
  if (is.null(name)) {
    stop("No 'name' attribute for function")
  }
  if (!is.null(sub_stage_type) && name != "handler") {
    # Deambiguate the specific sub stage function by adding stage type
    # e.g. could choose bold driver for both gradient descent and momentum
    name <- paste0(stage_type, " ", name)
  }

  event <- attr(hook, "event")
  if (is.null(event)) {
    stop("hook function ", name, " no 'event' attribute")
  }

  event_tok <- strsplit(event, "\\s+")[[1]]

  advice_type <- event_tok[1]
  join_point <- event_tok[2]
  if (join_point == "stage"
      || join_point == "gradient_descent"
      || join_point == "momentum"
      && is.null(stage_type)) {
    stage_type <- join_point
  }

  # Functions defined outside of the stage/sub stage constructors can
  # define events completely e.g. "init momentum step_size"
  # but functions defined inside a sub stage don't know the stage
  # that they are defind for and only define e.g. "init step_size"
  # For the free functions, temporarily redefine the join point to just be e.g.
  # "step_size"
  if (length(event_tok) == 3) {
    stage_type <- event_tok[2]
    sub_stage_type <- event_tok[3]
    join_point <- sub_stage_type
  }

  if (!is.null(sub_stage_type)) {
    if (is.null(stage_type)) {
      stop("sub stage type '", sub_stage_type, "' but stage type is NULL")
    }
    if (join_point == "direction" || join_point == "step_size") {
      join_point <- paste0(stage_type, " ", join_point)
    }
    hook <- wrap_sub_stage_hook(hook, stage_type, sub_stage_type)
  }
  else if (!is.null(stage_type)) {
    hook <- wrap_stage_hook(hook, stage_type)
  }

  if (name == "handler") {
    opt <- store_handler(opt, join_point, advice_type, hook)
  }
  else {
    # store the hook
    opt <- store_hook(opt, join_point, advice_type, name, hook)
  }

  depends <- attr(hook, "depends")
  if (!is.null(depends)) {
    depends <- strsplit(depends, "\\s+")[[1]]
    for (depend in depends) {
      # eventually recursively calls this function
      opt <- depend_to_hook(opt, depend)
    }
  }

  opt
}

# Puts hook in the correct sub list
store_hook <- function(opt, join_point, advice_type, name, hook) {
  if (is.null(opt$hooks[[join_point]])) {
    opt$hooks[[join_point]] <- list()
  }
  join_point_hooks <- opt$hooks[[join_point]]

  if (is.null(join_point_hooks[[advice_type]])) {
    join_point_hooks[[advice_type]] <- list()
  }
  advice <- join_point_hooks[[advice_type]]

  advice[[name]] <- hook
  join_point_hooks[[advice_type]] <- advice
  opt$hooks[[join_point]] <- join_point_hooks

  opt
}

# Puts the handler in the correct sub list
store_handler <- function(opt, join_point, advice_type, handler) {
  if (is.null(opt$handlers[[join_point]])) {
    opt$handlers[[join_point]] <- list()
  }
  join_point_handlers <- opt$handlers[[join_point]]

  if (is.null(join_point_handlers[[advice_type]])) {
    join_point_handlers[[advice_type]] <- list()
  }

  join_point_handlers[[advice_type]] <- handler
  opt$handlers[[join_point]] <- join_point_handlers

  opt
}

# Wraps a hook that should be fired for a specific stage
# stage_type can be "gradient_descent", "momentum" etc., but also "stage"
# if it should be fired for every stage.
wrap_stage_hook <- function(stage_hook, stage_type) {
  callback <- stage_hook
  function(opt, par, fg, iter, ...) {
    if (stage_type == "stage") {
      stage <- opt$stages[[opt$stage_i]]
    }
    else {
      stage <- opt$stages[[stage_type]]
    }

    res <- callback(opt, stage, par, fg, iter, ...)

    if (!is.null(res$opt)) {
      opt <- res$opt
    }
    if (!is.null(res$stage)) {
      if (stage_type == "stage") {
        opt$stages[[opt$stage_i]] <- res$stage
      }
      else {
        opt$stages[[stage_type]] <- res$stage
      }
    }
    opt
  }
}

# Wraps a hook that should be fired for a specific sub stage.
# stage_type can be "gradient_descent", "momentum" etc., but also "stage"
# if it should be fired for every stage.
# sub_stage should be one of "direction" or "step_size"
wrap_sub_stage_hook <- function(sub_stage_hook, stage_type, sub_stage_type) {
  callback <- sub_stage_hook
  function(opt, par, fg, iter, ...) {
    if (stage_type == "stage") {
      stage <- opt$stages[[opt$stage_i]]
    }
    else {
      stage <- opt$stages[[stage_type]]
    }

    sub_stage <- stage[[sub_stage_type]]
    res <- callback(opt, stage, sub_stage, par, fg, iter, ...)
    if (!is.null(res$opt)) {
      opt <- res$opt
    }

    if (!is.null(res$stage)) {
      stage <- res$stage
    }

    if (!is.null(res$sub_stage)) {
      sub_stage <- res$sub_stage
    }

    stage[[sub_stage_type]] <- sub_stage

    if (stage_type == "stage") {
      opt$stages[[opt$stage_i]] <- stage
    }
    else {
      opt$stages[[stage_type]] <- stage
    }

    opt
  }
}

# Convert all functions named in the depends vector of a phase into a hook
depends_to_hooks <- function(opt, phase, stage_type = NULL,
                             sub_stage_type = NULL) {
  if (is.null(phase$depends)) {
    return(opt)
  }

  for (name in (phase$depends)) {
    opt <- depend_to_hook(opt, name, stage_type, sub_stage_type)
  }

  opt
}

# Convert a specific named function (in depend) into a hook
depend_to_hook <- function(opt, depend, stage_type = NULL,
                           sub_stage_type = NULL) {
  f_name <- paste0("require_", depend)
  f <- get0(f_name)
  if (!is.null(f)) {
    opt <- register_hook(opt, f, stage_type, sub_stage_type)
  }

  opt
}

# Lists all functions and the phases/events they should fire for.
list_hooks <- function(opt) {
  message("handlers")
  if (!is.null(opt$handlers)) {
    handlers <- opt$handlers
    for (phase in names(handlers)) {
      phandlers <- handlers[[phase]]
      for (advice in names(phandlers)) {
        message(advice, " ", phase)
      }
    }
  }

  message("hooks")
  if (!is.null(opt$hooks)) {
    hooks <- opt$hooks
    for (phase in names(hooks)) {
      phooks <- hooks[[phase]]
      for (advice in names(phooks)) {
        aphooks <- phooks[[advice]]
        for (name in names(aphooks)) {
          message(advice, " ", phase, ": ", name)
        }
      }
    }
  }
}
# Numerical Optimization
#
# Numerical optimization including conjugate gradient,
# Broyden-Fletcher-Goldfarb-Shanno (BFGS), and the limited memory BFGS.
#
# The function to be optimized should be passed as a list to the \code{fg}
# parameter. This should consist of:
# \itemize{
# \item{\code{fn}}. The function to be optimized. Takes a vector of parameters
#   and returns a scalar.
# \item{\code{gr}}. The gradient of the function. Takes a vector of parameters
# and returns a vector with the same length as the input parameter vector.
# \item{\code{fg}}. Optional function which calculates the function and
# gradient in thesame routine. Takes a vector of parameters and returns a list
# containing the function result as \code{fn} and the gradient result as
# \code{gr}.
# }
#
# The \code{fg} function is optional, but for some methods (e.g. line search
# methods based on the Wolfe criteria), both the function and gradient values
# are needed for the same parameter value. Calculating them in the same
# function can save time if there is a lot of shared work.
#
# @section Optimization Methods:
# The \code{method} specifies the optimization method:
#
# \itemize{
# \item \code{"SD"} is plain steepest descent. Not very effective on its own,
# but can be combined with various momentum approaches.
# \item \code{"BFGS"} is the Broyden-Fletcher-Goldfarb-Shanno quasi-Newton
# method. This stores an approximation to the inverse of the Hessian of the
# function being minimized, which requires storage proportional to the
# square of the length of \code{par}, so is unsuitable for large problems.
# \item \code{"L-BFGS"} is the Limited memory Broyden-Fletcher-Goldfarb-Shanno
# quasi-Newton method. This does not store the inverse Hessian approximation
# directly and so can scale to larger-sized problems than \code{"BFGS"}. The
# amount of memory used can be controlled with the \code{memory} parameter.
# \item \code{"CG"} is the conjugate gradient method. The \code{cg_update}
# parameter allows for different methods for choosing the next direction:
#   \itemize{
#     \item \code{"FR"} The method of Fletcher and Reeves.
#     \item \code{"PR"} The method of Polak and Ribiere.
#     \item \code{"PR+"} The method of Polak and Ribiere with a restart to
#     steepest descent if conjugacy is lost. The default.
#     \item \code{"HS"} The method of Hestenes and Stiefel.
#     \item \code{"DY"} The method of Dai and Yuan.
#     \item \code{"HZ"} The method of Hager and Zhang.
#     \item \code{"HZ+"} The method of Hager and Zhang with restart, as used
#     in CG_DESCENT.
#   }
# The \code{"PR+"} and \code{"HZ+"} are likely to be most robust in practice.
# Other updates are available more for curiosity purposes.
# \item \code{"NAG"} is the Nesterov Accelerated Gradient method. The exact
# form of the momentum update in this method can be controlled with the
# following parameters:
#   \itemize{
#   \item{\code{nest_q}} Strong convexity parameter. Must take a value
#   between 0 (strongly convex) and 1 (zero momentum). Ignored if
#   \code{nest_convex_approx} is \code{TRUE}.
#   \item{\code{nest_convex_approx}} If \code{TRUE}, then use an approximation
#   due to Sutskever for calculating the momentum parameter.
#   \item{\code{nest_burn_in}} Number of iterations to wait before using a
#   non-zero momentum.
#   }
# \item \code{"DBD"} is the Delta-Bar-Delta method of Jacobs.
# \item \code{"Momentum"} is steepest descent with momentum. See below for
# momentum options.
# }
#
# For more details on gradient-based optimization in general, and the BFGS,
# L-BFGS and CG methods, see Nocedal and Wright.
#
# @section Line Search:
# The parameter \code{line_search} determines the line search to be carried
# out:
#
# \itemize{
#   \item \code{"Rasmussen"} carries out a line search using the strong Wolfe
#   conditions as implemented by Carl Edward Rasmussen's minimize.m routines.
#   \item \code{"More-Thuente"} carries out a line search using the strong Wolfe
#   conditions and the method of More-Thuente. Can be abbreviated to
#   \code{"MT"}.
#   \item \code{"Schmidt"} carries out a line search using the strong Wolfe
#   conditions as implemented in Mark Schmidt's minFunc routines.
#   \item \code{"Backtracking"} carries out a back tracking line search using
#   the sufficient decrease (Armijo) condition. By default, cubic interpolation
#   is used to find an acceptable step size. A constant step size reduction
#   can be used by specifying a value for \code{step_down} between 0 and 1
#   (e.g. step size will be halved if \code{step_down} is set to \code{0.5}).
#   \item \code{"Bold Driver"} carries out a back tracking line search until a
#   reduction in the function value is found.
#   \item \code{"Constant"} uses a constant line search, the value of which
#   should be provided with \code{step0}. Note that this value will be
#   multiplied by the magnitude of the direction vector used in the gradient
#   descent method. For method \code{"SD"} only, setting the
#   \code{norm_direction} parameter to \code{TRUE} will scale the direction
#   vector so it has unit length.
# }
#
# If using one of the methods: \code{"BFGS"}, \code{"L-BFGS"}, \code{"CG"} or
# \code{"NAG"}, one of the Wolfe line searches: \code{"Rasmussen"} or
# \code{"More-Thuente"}, \code{"Schmidt"} or \code{"Hager-Zhang"} should be
# used, otherwise very poor performance is likely to be encountered. The
# following parameters can be used to control the line search:
#
#  \itemize{
#    \item{\code{c1}} The sufficient decrease condition. Normally left at its
#    default value of 1e-4.
#    \item{\code{c2}} The sufficient curvature condition. Defaults to 0.9 if
#    using the methods \code{"BFGS"} and \code{"L-BFGS"}, and to 0.1 for
#    every other method, more or less in line with the recommendations given
#    by Nocedal and Wright. The smaller the value of \code{c2}, the stricter
#    the line search, but it should not be set to smaller than \code{c1}.
#    \item{\code{step0}} Initial value for the line search on the first step.
#    If a positive numeric value is passed as an argument, that value is used
#    as-is. Otherwise, by passing a string as an argument, a guess is made
#    based on values of the gradient, function or parameters, at the starting
#    point:
#    \itemize{
#      \item{\code{"rasmussen"}} As used by Rasmussen in \code{minimize.m}:
#      \deqn{\frac{1}{1+\left|g\right|^2}}{1 / 1 + (|g|^2)}
#      \item{\code{"scipy"}} As used in scipy's \code{optimize.py}
#      \deqn{\frac{1}{\left|g\right|}}{1 / |g|}
#      \item{\code{"schmidt"}} As used by Schmidt in \code{minFunc.m}
#      (the reciprocal of the l1 norm of g)
#      \deqn{\frac{1}{\left|g\right|_1}}{1 / |g|1}
#      \item{\code{"hz"}} The method suggested by Hager and Zhang (2006) for
#      the CG_DESCENT software.
#    }
#    These arguments can be abbreviated.
#    \item{\code{step_next_init}} How to initialize subsequent line searches
#    after the first, using results from the previous line search:
#    \itemize{
#      \item{\code{"slope ratio"}} Slope ratio method.
#      \item{\code{"quadratic"}} Quadratic interpolation method.
#      \item{\code{"hz"}} The QuadStep method of Hager and Zhang (2006) for
#      the CG_DESCENT software.
#    }
#    These arguments can be abbreviated. Details on the first two methods
#    are provided by Nocedal and Wright.
#    \item{\code{try_newton_step}} For quasi-Newton methods (\code{"BFGS"} and
#    \code{"L-BFGS"}), setting this to \code{TRUE} will try the "natural" step
#    size of 1, whenever the \code{step_next_init} method suggests an initial
#    step size larger than that. On by default for BFGS and L-BFGS, off for
#    everything else.
#    \item{\code{strong_curvature}} If \code{TRUE}, then the strong curvature
#    condition will be used to check termination in Wolfe line search methods.
#    If \code{FALSE}, then the standard curvature condition will be used. The
#    default is \code{NULL} which lets the different Wolfe line searches choose
#    whichever is their default behavior. This option is ignored if not using
#    a Wolfe line search method.
#    \item{\code{approx_armijo}} If \code{TRUE}, then the approximate Armijo
#    sufficient decrease condition (Hager and Zhang, 2005) will be used to
#    check termination in Wolfe line search methods. If \code{FALSE}, then the
#    exact curvature condition will be used. The default is \code{NULL} which
#    lets the different Wolfe line searches choose whichever is their default
#    behavior. This option is ignored if not using a Wolfe line search method.
# }
#
# For the Wolfe line searches, the methods of \code{"Rasmussen"},
# \code{"Schmidt"} and \code{"More-Thuente"} default to using the strong
# curvature condition and the exact Armijo condition to terminate the line
# search (i.e. Strong Wolfe conditions). The default step size initialization
# methods use the Rasmussen method for the first iteration and quadratic
# interpolation for subsequent iterations.
#
# The \code{"Hager-Zhang"} Wolfe line search method defaults to the standard
# curvature condition and the approximate Armijo condition (i.e. approximate
# Wolfe conditions). The default step size initialization methods are those
# used by Hager and Zhang (2006) in the description of CG_DESCENT.
#
# If the \code{"DBD"} is used for the optimization \code{"method"}, then the
# \code{line_search} parameter is ignored, because this method controls both
# the direction of the search and the step size simultaneously. The following
# parameters can be used to control the step size:
#
# \itemize{
#   \item{\code{step_up}} The amount by which to increase the step size in a
#   direction where the current step size is deemed to be too short. This
#   should be a positive scalar.
#   \item{\code{step_down}} The amount by which to decrease the step size in a
#   direction where the currents step size is deemed to be too long. This
#   should be a positive scalar smaller than 1. Default is 0.5.
#   \item{\code{step_up_fun}} How to increase the step size: either the method of
#   Jacobs (addition of \code{step_up}) or Janet and co-workers (multiplication
#   by \code{step_up}). Note that the step size decrease \code{step_down} is always
#   a multiplication.
# }
#
# The \code{"bold driver"} line search also uses the \code{step_up} and
# \code{step_down} parameters with similar meanings to their use with the
# \code{"DBD"} method: the backtracking portion reduces the step size by a
# factor of \code{step_down}. Once a satisfactory step size has been found, the
# line search for the next iteration is initialized by multiplying the
# previously found step size by \code{step_up}.
#
# @section Momentum:
# For \code{method} \code{"Momentum"}, momentum schemes can be accessed
# through the momentum arguments:
#
# \itemize{
# \item{\code{mom_type}} Momentum type, either \code{"classical"} or
#   \code{"nesterov"} (case insensitive, can be abbreviated). Using
#   \code{"nesterov"} applies the momentum step before the
#   gradient descent as suggested by Sutskever, emulating the behavior of the
#   Nesterov Accelerated Gradient method.
# \item{\code{mom_schedule}} How the momentum changes over the course of the
#   optimization:
#   \itemize{
#   \item{If a numerical scalar is provided, a constant momentum will be
#     applied throughout.}
#   \item{\code{"nsconvex"}} Use the momentum schedule from the Nesterov
#   Accelerated Gradient method suggested for non-strongly convex functions.
#   Parameters which control the NAG momentum
#   can also be used in combination with this option.
#   \item{\code{"switch"}} Switch from one momentum value (specified via
#   \code{mom_init}) to another (\code{mom_final}) at a
#   a specified iteration (\code{mom_switch_iter}).
#   \item{\code{"ramp"}} Linearly increase from one momentum value
#   (\code{mom_init}) to another (\code{mom_final}).
#   \item{If a function is provided, this will be invoked to provide a momentum
#   value. It must take one argument (the current iteration number) and return
#   a scalar.}
#   }
#   String arguments are case insensitive and can be abbreviated.
# }
#
# The \code{restart} parameter provides a way to restart the momentum if the
# optimization appears to be not be making progress, inspired by the method
# of O'Donoghue and Candes (2013) and Su and co-workers (2014). There are three
# strategies:
# \itemize{
#   \item{\code{"fn"}} A restart is applied if the function does not decrease
#   on consecutive iterations.
#   \item{\code{"gr"}} A restart is applied if the direction of the
#   optimization is not a descent direction.
#   \item{\code{"speed"}} A restart is applied if the update vector is not
#   longer (as measured by Euclidean 2-norm) in consecutive iterations.
# }
#
# The effect of the restart is to "forget" any previous momentum update vector,
# and, for those momentum schemes that change with iteration number, to
# effectively reset the iteration number back to zero. If the \code{mom_type}
# is \code{"nesterov"}, the gradient-based restart is not available. The
# \code{restart_wait} parameter controls how many iterations to wait after a
# restart, before allowing another restart. Must be a positive integer. Default
# is 10, as used by Su and co-workers (2014). Setting this too low could
# cause premature convergence. These methods were developed specifically
# for the NAG method, but can be employed with any momentum type and schedule.
#
# If \code{method} type \code{"momentum"} is specified with no other values,
# the momentum scheme will default to a constant value of \code{0.9}.
#
# @section Convergence:
#
# There are several ways for the optimization to terminate. The type of
# termination is communicated by a two-item list \code{terminate} in the return
# value, consisting of \code{what}, a short string describing what caused the
# termination, and \code{val}, the value of the termination criterion that
# caused termination.
#
# The following parameters control various stopping criteria:
#
# \itemize{
#   \item{\code{max_iter}} Maximum number of iterations to calculate. Reaching
#   this limit is indicated by \code{terminate$what} being \code{"max_iter"}.
#   \item{\code{max_fn}} Maximum number of function evaluations allowed.
#   Indicated by \code{terminate$what} being \code{"max_fn"}.
#   \item{\code{max_gr}} Maximum number of gradient evaluations allowed.
#   Indicated by \code{terminate$what} being \code{"max_gr"}.
#   \item{\code{max_fg}} Maximum number of gradient evaluations allowed.
#   Indicated by \code{terminate$what} being \code{"max_fg"}.
#   \item{\code{abs_tol}} Absolute tolerance of the function value. If the
#   absolute value of the function falls below this threshold,
#   \code{terminate$what} will be \code{"abs_tol"}. Will only be triggered if
#   the objective function has a minimum value of zero.
#   \item{\code{rel_tol}} Relative tolerance of the function value, comparing
#   consecutive function evaluation results. Indicated by \code{terminate$what}
#   being \code{"rel_tol"}.
#   \item{\code{grad_tol}} Absolute tolerance of the l2 (Euclidean) norm of
#   the gradient. Indicated by \code{terminate$what} being \code{"grad_tol"}.
#   Note that the gradient norm is not a very reliable stopping criterion
#   (see Nocedal and co-workers 2002), but is quite commonly used, so this
#   might be useful for comparison with results from other optimizers.
#   \item{\code{ginf_tol}} Absolute tolerance of the infinity norm (maximum
#   absolute component) of the gradient. Indicated by \code{terminate$what}
#   being \code{"ginf_tol"}.
#   \item{\code{step_tol}} Absolute tolerance of the step size, i.e. the
#   Euclidean distance between values of \code{par} fell below the specified
#   value. Indicated by \code{terminate$what} being \code{"step_tol"}.
#   For those optimization methods which allow for abandoning the result of an
#   iteration and restarting using the previous iteration's value of
#   \code{par} an iteration, \code{step_tol} will not be triggered.
# }
#
# Convergence is checked between specific interations. How often is determined
# by the \code{check_conv_every} parameter, which specifies the number of
# iterations between each check. By default, this is set for every iteration.
#
# Be aware that if \code{abs_tol} or \code{rel_tol} are non-\code{NULL}, this
# requires the function to have been evaluated at the current position at the
# end of each iteration. If the function at that  position hasn't been
# calculated, it will be calculated and will contribute to the total reported
# in the \code{counts} list in the return value. The calculated function value
# is cached for use by the optimizer in the next iteration, so if the optimizer
# would have needed to calculate the function anyway (e.g. use of the strong
# Wolfe line search methods), there is no significant cost accrued by
# calculating it earlier for convergence calculations. However, for methods
# that don't use the function value at that location, this could represent a
# lot of extra function evaluations. On the other hand, not checking
# convergence could result in a lot of extra unnecessary iterations.
# Similarly, if \code{grad_tol} or \code{ginf_tol} is non-\code{NULL}, then
# the gradient will be calculated if needed.
#
# If extra function or gradient evaluations is an issue, set
# \code{check_conv_every} to a higher value, but be aware that this can cause
# convergence limits to be exceeded by a greater amount.
#
# Note also that if the \code{verbose} parameter is \code{TRUE}, then a summary
# of the results so far will be logged to the console whenever a convergence
# check is carried out. If the \code{store_progress} parameter is \code{TRUE},
# then the same information will be returned as a data frame in the return
# value. For a long optimization this could be a lot of data, so by default it
# is not stored.
#
# Other ways for the optimization to terminate is if an iteration generates a
# non-finite (i.e. \code{Inf} or \code{NaN}) gradient or function value.
# Some, but not all, line-searches will try to recover from the latter, by
# reducing the step size, but a non-finite gradient calculation during the
# gradient descent portion of opimization is considered catastrosphic by mizer,
# and it will give up. Termination under non-finite gradient or function
# conditions will result in \code{terminate$what} being \code{"gr_inf"} or
# \code{"fn_inf"} respectively. Unlike the convergence criteria, the
# optimization will detect these error conditions and terminate even if a
# convergence check would not be carried out for this iteration.
#
# The value of \code{par} in the return value should be the parameters which
# correspond to the lowest value of the function that has been calculated
# during the optimization. As discussed above however, determining which set
# of parameters requires a function evaluation at the end of each iteration,
# which only happens if either the optimization method calculates it as part
# of its own operation or if a convergence check is being carried out during
# this iteration. Therefore, if your method doesn't carry out function
# evaluations and \code{check_conv_every} is set to be so large that no
# convergence calculation is carried out before \code{max_iter} is reached,
# then the returned value of \code{par} is the last value encountered.
#
# @param par Initial values for the function to be optimized over.
# @param fg Function and gradient list. See 'Details'.
# @param method Optimization method. See 'Details'.
# @param norm_direction If \code{TRUE}, then the steepest descent direction
# is normalized to unit length. Useful for adaptive step size methods where
# the previous step size is used to initialize the next iteration.
# @param scale_hess if \code{TRUE}, the approximation to the inverse Hessian
# is scaled according to the method described by Nocedal and Wright
# (approximating an eigenvalue). Applies only to the methods \code{BFGS}
# (where the scaling is applied only during the first step) and \code{L-BFGS}
# (where the scaling is applied during every iteration). Ignored otherwise.
# @param memory The number of updates to store if using the \code{L-BFGS}
# method. Ignored otherwise. Must be a positive integer.
# @param cg_update Type of update to use for the \code{CG} method. Can be
# one of \code{"FR"} (Fletcher-Reeves), \code{"PR"} (Polak-Ribiere),
# \code{"PR+"} (Polak-Ribiere with a reset to steepest descent), \code{"HS"}
# (Hestenes-Stiefel), or \code{"DY"} (Dai-Yuan). Ignored if \code{method} is
# not \code{"CG"}.
# @param nest_q Strong convexity parameter for the NAG
# momentum term. Must take a value between 0 (strongly convex) and 1
# (zero momentum). Only applies using the NAG method or a momentum method with
# Nesterov momentum schedule. Also does nothing if \code{nest_convex_approx}
# is \code{TRUE}.
# @param nest_convex_approx If \code{TRUE}, then use an approximation due to
# Sutskever for calculating the momentum parameter in the NAG method. Only
# applies using the NAG method or a momentum method with Nesterov momentum
# schedule.
# @param nest_burn_in Number of iterations to wait before using a non-zero
# momentum. Only applies using the NAG method or a momentum method with
# Nesterov momentum schedule.
# @param step_up Value by which to increase the step size for the \code{"bold"}
# step size method or the \code{"DBD"} method.
# @param step_up_fun Operator to use when combining the current step size with
# \code{step_up}. Can be one of \code{"*"} (to multiply the current step size
# with \code{step_up}) or \code{"+"} (to add).
# @param step_down Multiplier to reduce the step size by if using the \code{"DBD"}
# method or the \code{"bold"} line search method. Should be a positive value
# less than 1. Also optional for use with the \code{"back"} line search method.
# @param dbd_weight Weighting parameter used by the \code{"DBD"} method only, and
# only if no momentum scheme is provided. Must be an integer between 0 and 1.
# @param line_search Type of line search to use. See 'Details'.
# @param c1 Sufficient decrease parameter for Wolfe-type line searches. Should
# be a value between 0 and 1.
# @param c2 Sufficient curvature parameter for line search for Wolfe-type line
# searches. Should be a value between \code{c1} and 1.
# @param step0 Initial value for the line search on the first step. See
# 'Details'.
# @param step_next_init For Wolfe-type line searches only, how to initialize
# the line search on iterations after the first. See 'Details'.
# @param try_newton_step For Wolfe-type line searches only, try the
# line step value of 1 as the initial step size whenever \code{step_next_init}
# suggests a step size > 1. Defaults to \code{TRUE} for quasi-Newton methods
# such as BFGS and L-BFGS, \code{FALSE} otherwise.
# @param ls_max_fn Maximum number of function evaluations allowed during a
# line search.
# @param ls_max_gr Maximum number of gradient evaluations allowed during a
# line search.
# @param ls_max_fg Maximum number of function or gradient evaluations allowed
# during a line search.
# @param ls_max_alpha_mult Maximum multiplier for alpha between iterations.
# Only applies for Wolfe-type line searches and if \code{step_next_init} is
# set to \code{"slope"}
# @param strong_curvature (Optional). If \code{TRUE} use the strong
# curvature condition in Wolfe line search. See the 'Line Search' section
# for details.
# @param approx_armijo (Optional). If \code{TRUE} use the approximate Armijo
# condition in Wolfe line search. See the 'Line Search' section for details.
# @param mom_type Momentum type, either \code{"classical"} or
# \code{"nesterov"}. See 'Details'.
# @param mom_schedule Momentum schedule. See 'Details'.
# @param mom_init Initial momentum value.
# @param mom_final Final momentum value.
# @param mom_switch_iter For \code{mom_schedule} \code{"switch"} only, the
# iteration when \code{mom_init} is changed to \code{mom_final}.
# @param use_init_mom If \code{TRUE}, then the momentum coefficient on
# the first iteration is non-zero. Otherwise, it's zero. Only applies if
# using a momentum schedule.
# @param mom_linear_weight If \code{TRUE}, the gradient contribution to the
# update is weighted using momentum contribution.
# @param restart Momentum restart type. Can be one of "fn", "gr" or "speed".
# See Details'. Ignored if no momentum scheme is being used.
# @param restart_wait Number of iterations to wait between restarts. Ignored
# if \code{restart} is \code{NULL}.
# @param max_iter Maximum number of iterations to optimize for. Defaults to
# 100. See the 'Convergence' section for details.
# @param max_fn Maximum number of function evaluations. See the 'Convergence'
# section for details.
# @param max_gr Maximum number of gradient evaluations. See the 'Convergence'
# section for details.
# @param max_fg Maximum number of function or gradient evaluations. See the
# 'Convergence' section for details.
# @param abs_tol Absolute tolerance for comparing two function evaluations.
# See the 'Convergence' section for details.
# @param rel_tol Relative tolerance for comparing two function evaluations.
# See the 'Convergence' section for details.
# @param grad_tol Absolute tolerance for the length (l2-norm) of the gradient
# vector. See the 'Convergence' section for details.
# @param ginf_tol Absolute tolerance for the infinity norm (maximum absolute
# component) of the gradient vector. See the 'Convergence' section for details.
# @param step_tol Absolute tolerance for the size of the parameter update.
# See the 'Convergence' section for details.
# @param check_conv_every Positive integer indicating how often to check
# convergence. Default is 1, i.e. every iteration. See the 'Convergence'
# section for details.
# @param log_every Positive integer indicating how often to log convergence
# results to the console. Ignored if \code{verbose} is \code{FALSE}.
# If not an integer multiple of \code{check_conv_every}, it will be set to
# \code{check_conv_every}.
# @param verbose If \code{TRUE}, log information about the progress of the
# optimization to the console.
# @param store_progress If \code{TRUE} store information about the progress
# of the optimization in a data frame, and include it as part of the return
# value.
# @return A list with components:
#\itemize{
#  \item{\code{par}} Optimized parameters. Normally, this is the best set of
#  parameters seen during optimization, i.e. the set that produced the minimum
#  function value. This requires that convergence checking with is carried out,
#  including function evaluation where necessary. See the 'Convergence'
#  section for details.
#  \item{\code{nf}} Total number of function evaluations carried out. This
#  includes any extra evaluations required for convergence calculations. Also,
#  a function evaluation may be required to calculate the value of \code{f}
#  returned in this list (see below). Additionally, if the \code{verbose}
#  parameter is \code{TRUE}, then function and gradient information for the
#  initial value of \code{par} will be logged to the console. These values
#  are cached for subsequent use by the optimizer.
#  \item{\code{ng}} Total number of gradient evaluations carried out. This
#  includes any extra evaluations required for convergence calculations using
#  \code{grad_tol}. As with \code{nf}, additional gradient calculations beyond
#  what you're expecting may have been needed for logging, convergence and
#  calculating the value of \code{g2} or \code{ginf} (see below).
#  \item{\code{f}} Value of the function, evaluated at the returned
#  value of \code{par}.
#  \item{\code{g2}} Optional: the length (Euclidean or l2-norm) of the
#  gradient vector, evaluated at the returned value of \code{par}. Calculated
#  only if \code{grad_tol} is non-null.
#  \item{\code{ginf}} Optional: the infinity norm (maximum absolute component)
#  of the gradient vector, evaluated at the returned value of \code{par}.
#  Calculated only if \code{ginf_tol} is non-null.
#  \item{\code{iter}} The number of iterations the optimization was carried
#  out for.
#  \item{\code{terminate}} List containing items: \code{what}, indicating what
#  convergence criterion was met, and \code{val} specifying the value at
#  convergence. See the 'Convergence' section for more details.
#  \item{\code{progress}} Optional data frame containing information on the
#  value of the function, gradient, momentum, and step sizes evaluated at each
#  iteration where convergence is checked. Only present if
#  \code{store_progress} is set to \code{TRUE}. Could get quite large if the
#  optimization is long and the convergence is checked regularly.
#}
# @references
#
# Hager, W. W., & Zhang, H. (2005).
# A new conjugate gradient method with guaranteed descent and an efficient line search.
# \emph{SIAM Journal on Optimization}, \emph{16}(1), 170-192.
#
# Hager, W. W., & Zhang, H. (2006).
# Algorithm 851: CG_DESCENT, a conjugate gradient method with guaranteed descent.
# \emph{ACM Transactions on Mathematical Software (TOMS)}, \emph{32}(1), 113-137.
#
# Jacobs, R. A. (1988).
# Increased rates of convergence through learning rate adaptation.
# \emph{Neural networks}, \emph{1}(4), 295-307.
#
# Janet, J. A., Scoggins, S. M., Schultz, S. M., Snyder, W. E., White, M. W.,
# & Sutton, J. C. (1998, May).
# Shocking: An approach to stabilize backprop training with greedy adaptive
# learning rates.
# In \emph{1998 IEEE International Joint Conference on Neural Networks Proceedings.}
# (Vol. 3, pp. 2218-2223). IEEE.
#
# More', J. J., & Thuente, D. J. (1994).
# Line search algorithms with guaranteed sufficient decrease.
# \emph{ACM Transactions on Mathematical Software (TOMS)}, \emph{20}(3), 286-307.
#
# Nocedal, J., Sartenaer, A., & Zhu, C. (2002).
# On the behavior of the gradient norm in the steepest descent method.
# \emph{Computational Optimization and Applications}, \emph{22}(1), 5-35.
#
# Nocedal, J., & Wright, S. (2006).
# Numerical optimization.
# Springer Science & Business Media.
#
# O'Donoghue, B., & Candes, E. (2013).
# Adaptive restart for accelerated gradient schemes.
# \emph{Foundations of computational mathematics}, \emph{15}(3), 715-732.
#
# Schmidt, M. (2005).
# minFunc: unconstrained differentiable multivariate optimization in Matlab.
# \url{http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html}
#
# Su, W., Boyd, S., & Candes, E. (2014).
# A differential equation for modeling Nesterov's accelerated gradient method: theory and insights.
# In \emph{Advances in Neural Information Processing Systems} (pp. 2510-2518).
#
# Sutskever, I. (2013).
# \emph{Training recurrent neural networks}
# (Doctoral dissertation, University of Toronto).
#
# Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013).
# On the importance of initialization and momentum in deep learning.
# In \emph{Proceedings of the 30th international conference on machine learning (ICML-13)}
# (pp. 1139-1147).
# @examples
# # Function to optimize and starting point defined after creating optimizer
# rosenbrock_fg <- list(
#   fn = function(x) { 100 * (x[2] - x[1] * x[1]) ^ 2 + (1 - x[1]) ^ 2  },
#   gr = function(x) { c( -400 * x[1] * (x[2] - x[1] * x[1]) - 2 * (1 - x[1]),
#                          200 *        (x[2] - x[1] * x[1])) })
# rb0 <- c(-1.2, 1)
#
# # Minimize using L-BFGS
# res <- mize(rb0, rosenbrock_fg, method = "L-BFGS")
#
# # Conjugate gradient with Fletcher-Reeves update, tight Wolfe line search
# res <- mize(rb0, rosenbrock_fg, method = "CG", cg_update = "FR", c2 = 0.1)
#
# # Steepest decent with constant momentum = 0.9
# res <- mize(rb0, rosenbrock_fg, method = "MOM", mom_schedule = 0.9)
#
# # Steepest descent with constant momentum in the Nesterov style as described
# # in papers by Sutskever and Bengio
# res <- mize(rb0, rosenbrock_fg, method = "MOM", mom_type = "nesterov",
#              mom_schedule = 0.9)
#
# # Nesterov momentum with adaptive restart comparing function values
# res <- mize(rb0, rosenbrock_fg, method = "MOM", mom_type = "nesterov",
#              mom_schedule = 0.9, restart = "fn")
# @export
mize <- function(par, fg,
                 method = "L-BFGS",
                 norm_direction = FALSE,
                 # L-BFGS
                 memory = 5,
                 scale_hess = TRUE,
                 # CG
                 cg_update = "PR+",
                 # NAG
                 nest_q = 0, # 1 - SD,
                 nest_convex_approx = FALSE,
                 nest_burn_in = 0,
                 # DBD
                 step_up = 1.1,
                 step_up_fun = "*",
                 step_down = NULL,
                 dbd_weight = 0.1,
                 # Line Search configuration
                 line_search = "More-Thuente",
                 c1 = 1e-4,
                 c2 = NULL,
                 step0 = NULL,
                 step_next_init = NULL,
                 try_newton_step = NULL,
                 ls_max_fn = 20,
                 ls_max_gr = Inf,
                 ls_max_fg = Inf,
                 ls_max_alpha_mult = Inf,
                 strong_curvature = NULL,
                 approx_armijo = NULL,
                 # Momentum
                 mom_type = NULL,
                 mom_schedule = NULL,
                 mom_init = NULL,
                 mom_final = NULL,
                 mom_switch_iter = NULL,
                 mom_linear_weight = FALSE,
                 use_init_mom = FALSE,
                 # Adaptive Restart
                 restart = NULL,
                 restart_wait = 10,
                 # Termination criterion
                 max_iter = 100,
                 max_fn = Inf,
                 max_gr = Inf,
                 max_fg = Inf,
                 abs_tol = sqrt(.Machine$double.eps),
                 rel_tol = abs_tol,
                 grad_tol = NULL,
                 ginf_tol = NULL,
                 step_tol = sqrt(.Machine$double.eps),
                 check_conv_every = 1,
                 log_every = check_conv_every,
                 verbose = FALSE,
                 store_progress = FALSE) {

  opt <- make_mize(method = method,
                   norm_direction = norm_direction,
                   scale_hess = scale_hess,
                   memory = memory,
                   cg_update = cg_update,
                   nest_q = nest_q, nest_convex_approx = nest_convex_approx,
                   nest_burn_in = nest_burn_in,
                   use_init_mom = use_init_mom,
                   step_up = step_up,
                   step_up_fun = step_up_fun,
                   step_down = step_down,
                   dbd_weight = dbd_weight,
                   line_search = line_search, step0 = step0, c1 = c1, c2 = c2,
                   step_next_init = step_next_init,
                   try_newton_step = try_newton_step,
                   ls_max_fn = ls_max_fn, ls_max_gr = ls_max_gr,
                   ls_max_fg = ls_max_fg,
                   ls_max_alpha_mult = ls_max_alpha_mult,
                   strong_curvature = strong_curvature,
                   approx_armijo = approx_armijo,
                   mom_type = mom_type,
                   mom_schedule = mom_schedule,
                   mom_init = mom_init,
                   mom_final = mom_final,
                   mom_switch_iter = mom_switch_iter,
                   mom_linear_weight = mom_linear_weight,
                   max_iter = max_iter,
                   restart = restart,
                   restart_wait = restart_wait)
  if (max_iter < 0) {
    stop("max_iter must be non-negative")
  }
  if (max_fn < 0) {
    stop("max_fn must be non-negative")
  }
  if (max_gr < 0) {
    stop("max_gr must be non-negative")
  }
  if (max_fg < 0) {
    stop("max_fg must be non-negative")
  }
  if (store_progress && is.null(check_conv_every)) {
    stop("check_conv_every must be non-NULL if store_progress is TRUE")
  }

  res <- opt_loop(opt, par, fg,
          max_iter = max_iter,
          max_fn = max_fn, max_gr = max_gr, max_fg = max_fg,
          abs_tol = abs_tol, rel_tol = rel_tol,
          grad_tol = grad_tol, ginf_tol = ginf_tol,
          step_tol = step_tol,
          check_conv_every = check_conv_every,
          log_every = log_every,
          store_progress = store_progress,
          verbose = verbose)

  Filter(Negate(is.null),
         res[c("f", "g2n", "ginfn", "nf", "ng", "par", "iter", "terminate",
               "progress")])
}

# Create an Optimizer
#
# Factory function for creating a (possibly uninitialized) optimizer.
#
# If the function to be optimized and starting point are not present at
# creation time, then the optimizer should be initialized using
# \code{\link{mize_init}} before being used with \code{\link{mize_step}}.
#
# See the documentation to \code{\link{mize}} for an explanation of all the
# parameters.
#
# Details of the \code{fg} list containing the function to be optimized and its
# gradient can be found in the 'Details' section of \code{\link{mize}}. It is
# optional for this function, but if it is passed to this function, along with
# the vector of initial values, \code{par}, the optimizer will be returned
# already initialized for this function. Otherwise, \code{\link{mize_init}}
# must be called before optimization begins.
#
# Additionally, optional convergence parameters may also be passed here, for
# use with \code{\link{check_mize_convergence}}. They are optional here if you
# plan to call \code{\link{mize_init}} later, or if you want to do your own
# convergence checking.
#
# @param method Optimization method. See 'Details' of \code{\link{mize}}.
# @param norm_direction If \code{TRUE}, then the steepest descent direction is
#   normalized to unit length. Useful for adaptive step size methods where the
#   previous step size is used to initialize the next iteration.
# @param scale_hess if \code{TRUE}, the approximation to the inverse Hessian is
#   scaled according to the method described by Nocedal and Wright
#   (approximating an eigenvalue). Applies only to the methods \code{BFGS}
#   (where the scaling is applied only during the first step) and \code{L-BFGS}
#   (where the scaling is applied during every iteration). Ignored otherwise.
# @param memory The number of updates to store if using the \code{L-BFGS}
#   method. Ignored otherwise. Must be a positive integer.
# @param cg_update Type of update to use for the \code{CG} method. Can be one
#   of \code{"FR"} (Fletcher-Reeves), \code{"PR"} (Polak-Ribiere), \code{"PR+"}
#   (Polak-Ribiere with a reset to steepest descent), \code{"HS"}
#   (Hestenes-Stiefel), or \code{"DY"} (Dai-Yuan). Ignored if \code{method} is
#   not \code{"CG"}.
# @param nest_q Strong convexity parameter for the \code{"NAG"} method's
#   momentum term. Must take a value between 0 (strongly convex) and 1 (results
#   in steepest descent).Ignored unless the \code{method} is \code{"NAG"} and
#   \code{nest_convex_approx} is \code{FALSE}.
# @param nest_convex_approx If \code{TRUE}, then use an approximation due to
#   Sutskever for calculating the momentum parameter in the NAG method. Only
#   applies if \code{method} is \code{"NAG"}.
# @param nest_burn_in Number of iterations to wait before using a non-zero
#   momentum. Only applies if using the \code{"NAG"} method or setting the
#   \code{momentum_type} to "Nesterov".
# @param step_up Value by which to increase the step size for the \code{"bold"}
#   step size method or the \code{"DBD"} method.
# @param step_up_fun Operator to use when combining the current step size with
#   \code{step_up}. Can be one of \code{"*"} (to multiply the current step size
#   with \code{step_up}) or \code{"+"} (to add).
# @param step_down Multiplier to reduce the step size by if using the
#   \code{"DBD"} method or the \code{"bold"}. Can also be used with the
#   \code{"back"} line search method, but is optional. Should be a positive
#   value less than 1.
# @param dbd_weight Weighting parameter used by the \code{"DBD"} method only,
#   and only if no momentum scheme is provided. Must be an integer between 0
#   and 1.
# @param line_search Type of line search to use. See 'Details' of
#   \code{\link{mize}}.
# @param c1 Sufficient decrease parameter for Wolfe-type line searches. Should
#   be a value between 0 and 1.
# @param c2 Sufficient curvature parameter for line search for Wolfe-type line
#   searches. Should be a value between \code{c1} and 1.
# @param step0 Initial value for the line search on the first step. See
#   'Details' of \code{\link{mize}}.
# @param step_next_init For Wolfe-type line searches only, how to initialize
#   the line search on iterations after the first. See 'Details' of
#   \code{\link{mize}}.
# @param try_newton_step For Wolfe-type line searches only, try the line step
#   value of 1 as the initial step size whenever \code{step_next_init} suggests
#   a step size > 1. Defaults to \code{TRUE} for quasi-Newton methods such as
#   BFGS and L-BFGS, \code{FALSE} otherwise.
# @param ls_max_fn Maximum number of function evaluations allowed during a line
#   search.
# @param ls_max_gr Maximum number of gradient evaluations allowed during a line
#   search.
# @param ls_max_fg Maximum number of function or gradient evaluations allowed
#   during a line search.
# @param ls_max_alpha_mult Maximum multiplier for alpha between iterations.
#   Only applies for Wolfe-type line searches and if \code{step_next_init} is
#   set to \code{"slope"}
# @param strong_curvature (Optional). If \code{TRUE} use the strong
#   curvature condition in Wolfe line search. See the 'Line Search' section of
#   \code{\link{mize}} for details.
# @param approx_armijo (Optional). If \code{TRUE} use the approximate Armijo
#   condition in Wolfe line search. See the 'Line Search' section of
#   \code{\link{mize}} for details.
# @param mom_type Momentum type, either \code{"classical"} or
#   \code{"nesterov"}.
# @param mom_schedule Momentum schedule. See 'Details' of \code{\link{mize}}.
# @param mom_init Initial momentum value.
# @param mom_final Final momentum value.
# @param mom_switch_iter For \code{mom_schedule} \code{"switch"} only, the
#   iteration when \code{mom_init} is changed to \code{mom_final}.
# @param mom_linear_weight If \code{TRUE}, the gradient contribution to the
#   update is weighted using momentum contribution.
# @param use_init_mom If \code{TRUE}, then the momentum coefficient on the
#   first iteration is non-zero. Otherwise, it's zero. Only applies if using a
#   momentum schedule.
# @param restart Momentum restart type. Can be one of "fn" or "gr". See
#   'Details' of \code{\link{mize}}.
# @param restart_wait Number of iterations to wait between restarts. Ignored if
#   \code{restart} is \code{NULL}.
# @param par (Optional) Initial values for the function to be optimized over.
# @param fg (Optional). Function and gradient list. See 'Details' of
#   \code{\link{mize}}.
# @param max_iter (Optional). Maximum number of iterations. See the
#   'Convergence' section of \code{\link{mize}} for details.
# @param max_fn (Optional). Maximum number of function evaluations. See the
#   'Convergence' section of \code{\link{mize}} for details.
# @param max_gr (Optional). Maximum number of gradient evaluations. See the
#   'Convergence' section of \code{\link{mize}} for details.
# @param max_fg (Optional). Maximum number of function or gradient evaluations.
#   See the 'Convergence' section of \code{\link{mize}} for details.
# @param abs_tol (Optional). Absolute tolerance for comparing two function
#   evaluations. See the 'Convergence' section of \code{\link{mize}} for
#   details.
# @param rel_tol (Optional). Relative tolerance for comparing two function
#   evaluations. See the 'Convergence' section of \code{\link{mize}} for
#   details.
# @param grad_tol (Optional). Absolute tolerance for the length (l2-norm) of
#   the gradient vector. See the 'Convergence' section of \code{\link{mize}}
#   for details.
# @param ginf_tol (Optional). Absolute tolerance for the infinity norm (maximum
#   absolute component) of the gradient vector. See the 'Convergence' section
#   of \code{\link{mize}} for details.
# @param step_tol (Optional). Absolute tolerance for the size of the parameter
#   update. See the 'Convergence' section of \code{\link{mize}} for details.
# @export
# @examples
# # Function to optimize and starting point
# rosenbrock_fg <- list(
#   fn = function(x) { 100 * (x[2] - x[1] * x[1]) ^ 2 + (1 - x[1]) ^ 2  },
#   gr = function(x) { c( -400 * x[1] * (x[2] - x[1] * x[1]) - 2 * (1 - x[1]),
#                          200 *        (x[2] - x[1] * x[1])) })
# rb0 <- c(-1.2, 1)
#
# # Create an optimizer and initialize it for use with the Rosenbrock function
# opt <- make_mize(method = "L-BFGS", par = rb0, fg = rosenbrock_fg)
#
# # Create optimizer without initialization
# opt <- make_mize(method = "L-BFGS")
#
# # Need to call mizer_init separately:
# opt <- mize_init(opt, rb0, rosenbrock_fg)
make_mize <- function(method = "L-BFGS",
                      norm_direction = FALSE,
                      # BFGS
                      scale_hess = TRUE,
                      memory = 5,
                      # CG
                      cg_update = "PR+",
                      # NAG
                      nest_q = 0,
                      nest_convex_approx = FALSE,
                      nest_burn_in = 0,
                      # DBD
                      step_up = 1.1,
                      step_up_fun = c("*", "+"),
                      step_down = NULL,
                      dbd_weight = 0.1,
                      # Line Search
                      line_search = "More-Thuente",
                      c1 = 1e-4, c2 = NULL,
                      step0 = NULL,
                      step_next_init = NULL,
                      try_newton_step = NULL,
                      ls_max_fn = 20,
                      ls_max_gr = Inf,
                      ls_max_fg = Inf,
                      ls_max_alpha_mult = Inf,
                      strong_curvature = NULL,
                      approx_armijo = NULL,
                      # Momentum
                      mom_type = NULL,
                      mom_schedule = NULL,
                      mom_init = NULL,
                      mom_final = NULL,
                      mom_switch_iter = NULL,
                      mom_linear_weight = FALSE,
                      use_init_mom = FALSE,
                      restart = NULL,
                      restart_wait = 10,
                      par = NULL,
                      fg = NULL,
                      max_iter = 100,
                      max_fn = Inf, max_gr = Inf, max_fg = Inf,
                      abs_tol = NULL,
                      rel_tol = abs_tol, grad_tol = NULL, ginf_tol = NULL,
                      step_tol = NULL) {

  if (memory < 1) {
    stop("memory must be > 0")
  }
  if (!is_in_range(nest_q, 0, 1)) {
    stop("nest_q must be between 0 and 1")
  }
  if (nest_burn_in < 0) {
    stop("nest_burn_in must be non-negative")
  }
  if (step_up <= 0) {
    stop("step_up must be positive")
  }
  step_up_fun <- match.arg(step_up_fun)
  if (!is.null(step_down) && !is_in_range(step_down, 0, 1)) {
    stop("step_down must be between 0 and 1")
  }
  if (!is_in_range(dbd_weight, 0, 1)) {
    stop("dbd_weight must be between 0 and 1")
  }
  if (!is_in_range(c1, 0, 1, lopen = FALSE, ropen = FALSE)) {
    stop("c1 must be between 0 and 1")
  }
  if (!is.null(c2) && !is_in_range(c2, c1, 1, lopen = FALSE, ropen = FALSE)) {
    stop("c2 must be between c1 and 1")
  }
  if (ls_max_fn < 0) {
    stop("ls_max_fn must be non-negative")
  }
  if (ls_max_gr < 0) {
    stop("ls_max_gr must be non-negative")
  }
  if (ls_max_fg < 0) {
    stop("ls_max_fg must be non-negative")
  }
  if (ls_max_alpha_mult <= 0) {
    stop("ls_max_alpha_mult must be positive")
  }
  if (restart_wait < 1) {
    stop("restart_wait must be a positive integer")
  }

  # Gradient Descent Direction configuration
  dir_type <- NULL
  method <- match.arg(tolower(method), c("sd", "newton", "phess", "cg", "bfgs",
                                "l-bfgs", "nag", "momentum", "dbd"))
  switch(method,
    sd = {
      dir_type <- sd_direction(normalize = norm_direction)
    },
    newton = {
      dir_type <- newton_direction()
      if (is.null(try_newton_step)) {
        try_newton_step <- TRUE
      }
    },
    phess = {
      dir_type <- partial_hessian_direction()
      if (is.null(try_newton_step)) {
        try_newton_step <- TRUE
      }
    },
    cg = {
      cg_update <- match.arg(tolower(cg_update),
                             c("fr", "cd", "dy",
                               "hs", "hs+", "pr", "pr+", "ls", "hz", "hz+"))
      cg_update_fn <- switch(cg_update,
        fr = fr_update,
        cd = cd_update,
        dy = dy_update,
        hs = hs_update,
        "hs+" = hs_plus_update,
        pr = pr_update,
        "pr+" = pr_plus_update,
        ls = ls_update,
        hz = hz_update,
        "hz+" = hz_plus_update
      )
      dir_type <- cg_direction(cg_update = cg_update_fn)
    },
    bfgs = {
      dir_type <- bfgs_direction(scale_inverse = scale_hess)
      if (is.null(try_newton_step)) {
        try_newton_step <- TRUE
      }
    },
    "l-bfgs" = {
      dir_type <- lbfgs_direction(memory = memory, scale_inverse = scale_hess)
      if (is.null(try_newton_step)) {
        try_newton_step <- TRUE
      }
    },
    nag = {
      dir_type <- sd_direction(normalize = norm_direction)
    },
    momentum = {
      dir_type <- sd_direction(normalize = norm_direction)
    },
    dbd = {
      dir_type <- sd_direction(normalize = norm_direction)
    },
    stop("Unknown method: '", method, "'")
  )

  # If it's not already been turned on, turn off the Newton step option
  if (is.null(try_newton_step)) {
    try_newton_step <- FALSE
  }

  # Line Search configuration
  step_type <- NULL
  line_search <- tolower(line_search)
  if (method == "dbd") {
    if (is.character(step0) || is.numeric(step0)) {
      eps_init <- step0
    }
    else {
      eps_init <- "rasmussen"
    }
    if (step_up_fun == "*") {
      step_up_fun <- `*`
    }
    else if (step_up_fun == "+") {
      step_up_fun <- `+`
    }
    else {
      stop("Unknown delta-bar-delta step_up function '", step_up_fun, "'")
    }
    if (is.null(step_down)) {
      step_down <- 0.5
    }
    step_type <- delta_bar_delta(epsilon = eps_init,
                                 kappa = step_up, kappa_fun = step_up_fun,
                                 phi = step_down, theta = dbd_weight,
                                 use_momentum = is.null(mom_schedule))
  }
  else {
    if (method %in% c("newton", "phess", "bfgs", "l-bfgs")) {
      if (is.null(c2)) {
        c2 <- 0.9
      }
      if (is.null(try_newton_step)) {
        try_newton_step <- TRUE
      }
    }
    else {
      if (is.null(c2)) {
        c2 <- 0.1
      }
      if (is.null(try_newton_step)) {
        try_newton_step <- FALSE
      }
    }

    line_search <- match.arg(tolower(line_search),
                             c("more-thuente", "mt", "rasmussen",
                               "bold driver",
                               "backtracking", "constant",
                               "schmidt", "minfunc", "armijo",
                               "hager-zhang", "hz"))
    if (line_search == "hager-zhang") {
      line_search <- "hz"
    }
    if (line_search == "more-thuente") {
      line_search <- "mt"
    }
    if (line_search == "minfunc") {
      line_search <- "schmidt"
    }

    if (line_search == "bold driver") {
      if (is.null(step_down)) {
        step_down <- 0.5
      }
    }

    # Set Wolfe line search termination defaults
    # Most Wolfe Line Searches use the standard Strong Wolfe conditions
    if (line_search %in% c("more-thuente", "mt", "rasmussen", "schmidt",
                           "minfunc")) {
      if (is.null(strong_curvature)) {
        strong_curvature <- TRUE
      }
      if (is.null(approx_armijo)) {
        approx_armijo <- FALSE
      }
    }

    # Hager-Zhang uses weak Wolfe condtions with an approximation to the
    # Armijo condition. Also use the step initialization methods used in
    # CG_DESCENT by default
    if (line_search == "hz") {
      if (is.null(strong_curvature)) {
        strong_curvature <- FALSE
      }
      if (is.null(approx_armijo)) {
        approx_armijo <- TRUE
      }
      if (is.null(step_next_init)) {
        step_next_init <- "hz"
      }
      if (is.null(step0)) {
        step0 <- "hz"
      }
    }
    else {
      if (is.null(step0)) {
        step0 <- "rasmussen"
      }
      if (is.null(step_next_init)) {
        step_next_init <- "quad"
      }
    }

    step_type <- switch(line_search,
      mt = more_thuente_ls(c1 = c1, c2 = c2,
                           initializer = tolower(step_next_init),
                           initial_step_length = step0,
                           try_newton_step = try_newton_step,
                           max_fn = ls_max_fn,
                           max_gr = ls_max_gr,
                           max_fg = ls_max_fg,
                           max_alpha_mult = ls_max_alpha_mult,
                           strong_curvature = strong_curvature,
                           approx_armijo = approx_armijo),
      rasmussen = rasmussen_ls(c1 = c1, c2 = c2,
                              initializer = tolower(step_next_init),
                              initial_step_length = step0,
                              try_newton_step = try_newton_step,
                              max_fn = ls_max_fn,
                              max_gr = ls_max_gr,
                              max_fg = ls_max_fg,
                              max_alpha_mult = ls_max_alpha_mult,
                              strong_curvature = strong_curvature,
                              approx_armijo = approx_armijo),
      "bold driver" = bold_driver(inc_mult = step_up, dec_mult = step_down,
                                  max_fn = ls_max_fn),
      constant = constant_step_size(value = step0),
      schmidt = schmidt_ls(c1 = c1, c2 = c2,
                           initializer = tolower(step_next_init),
                           initial_step_length = step0,
                           try_newton_step = try_newton_step,
                           max_fn = ls_max_fn,
                           max_gr = ls_max_gr,
                           max_fg = ls_max_fg,
                           max_alpha_mult = ls_max_alpha_mult,
                           strong_curvature = strong_curvature,
                           approx_armijo = approx_armijo),
      backtracking = schmidt_armijo_ls(c1 = c1,
                          initializer = tolower(step_next_init),
                          initial_step_length = step0,
                          try_newton_step = try_newton_step,
                          step_down = step_down,
                          max_fn = ls_max_fn,
                          max_gr = ls_max_gr,
                          max_fg = ls_max_fg,
                          max_alpha_mult = ls_max_alpha_mult),
      hz =  hager_zhang_ls(c1 = c1, c2 = c2,
                           initializer = tolower(step_next_init),
                           initial_step_length = step0,
                           try_newton_step = try_newton_step,
                           max_fn = ls_max_fn,
                           max_gr = ls_max_gr,
                           max_fg = ls_max_fg,
                           max_alpha_mult = ls_max_alpha_mult,
                           strong_curvature = strong_curvature,
                           approx_armijo = approx_armijo)
    )
  }

  # Create Gradient Descent stage
  opt <- make_opt(
    make_stages(
      gradient_stage(
        direction = dir_type,
        step_size = step_type)))

  # Momentum Configuration
  if (is.null(mom_type)) {
    mom_type <- "classical"
  }
  mom_type <- match.arg(tolower(mom_type), c("classical", "nesterov"))

  mom_direction <- momentum_direction()

  if (method == "nag") {
    # Nesterov Accelerated Gradient
    mom_type <- "classical"
    if (is.null(mom_schedule)) {
      mom_schedule <- "nsconvex"
    }
    mom_direction <- nesterov_momentum_direction()
  }
  else if (method == "momentum") {
    # Default momentum values
    if (mom_type == "nesterov") {
      mom_direction <- nesterov_momentum_direction()
    }
    if (is.null(mom_schedule)) {
      mom_schedule <- 0.9
    }
  }

  # Momentum configuration
  if (!is.null(mom_schedule)) {
    if (is.numeric(mom_schedule)) {
      mom_step <- make_momentum_step(
        mu_fn = make_constant(value = mom_schedule),
        use_init_mom = use_init_mom)
    }
    else if (is.function(mom_schedule)) {
      mom_step <- make_momentum_step(mu_fn = mom_schedule,
                                     use_init_mom = use_init_mom)
    }
    else {
      mom_schedule <- match.arg(tolower(mom_schedule),
                                c("ramp", "switch", "nsconvex"))

      mom_step <- switch(mom_schedule,
        ramp = make_momentum_step(
          make_ramp(init_value = mom_init,
                    final_value = mom_final,
                    wait = ifelse(use_init_mom, 0, 1)),
          use_init_mom = use_init_mom),
        "switch" = make_momentum_step(
          make_switch(
            init_value = mom_init,
            final_value = mom_final,
            switch_iter = mom_switch_iter),
          use_init_mom = use_init_mom),
        nsconvex = nesterov_step(burn_in = nest_burn_in, q = nest_q,
                                 use_approx = nest_convex_approx,
                                 use_init_mu = use_init_mom)
        )
    }

    mom_stage <- momentum_stage(
      direction = mom_direction,
      step_size = mom_step)

    opt <- append_stage(opt, mom_stage)

    if (mom_linear_weight) {
      opt <- append_stage(opt, momentum_correction_stage())
    }
  }

  # Adaptive Restart
  if (!is.null(restart)) {
    restart <- match.arg(tolower(restart), c("none", "fn", "gr", "speed"))
    if (restart != "none") {
      opt <- adaptive_restart(opt, restart, wait = restart_wait)
    }
  }

  # Initialize for specific dataset if par and fg are provided
  if (!is.null(par) && !is.null(fg)) {
    opt <- mize_init(opt, par, fg, max_iter = max_iter,
                     max_fn = max_fn, max_gr = max_gr, max_fg = max_fg,
                     abs_tol = abs_tol, rel_tol = rel_tol,
                     grad_tol = grad_tol, ginf_tol = ginf_tol,
                     step_tol = step_tol)
  }

  opt
}

#One Step of Optimization
#
#Performs one iteration of optimization using a specified optimizer.
#
#This function returns both the (hopefully) optimized vector of parameters, and
#an updated version of the optimizer itself. This is intended to be used when
#you want more control over the optimization process compared to the more black
#box approach of the \code{\link{mize}} function. In return for having to
#manually call this function every time you want the next iteration of
#optimization, you gain the ability to do your own checks for convergence,
#logging and so on, as well as take other action between iterations, e.g.
#visualization.
#
#Normally calling this function should return a more optimized vector of
#parameters than the input, or at  least leave the parameters unchanged if no
#improvement was found, although this is determined by how the optimizer was
#configured by \code{\link{make_mize}}. It is very possible to create an
#optimizer that can cause a solution to diverge. It is the responsibility of
#the caller to check that the result of the optimization step has actually
#reduced the value returned from function being optimized.
#
#Details of the \code{fg} list can be found in the 'Details' section of
#\code{\link{mize}}.
#
#@param opt Optimizer, created by \code{\link{make_mize}}.
#@param par Vector of initial values for the function to be optimized over.
#@param fg Function and gradient list. See the documentaion of
#  \code{\link{mize}}.
#@return Result of the current optimization step, a list with components:
#  \itemize{
#
#  \item{\code{opt}}. Updated version of the optimizer passed to the \code{opt}
#  argument Should be passed as the \code{opt} argument in the next iteration.
#
#  \item{\code{par}}. Updated version of the parameters passed to the
#  \code{par} argument. Should be passed as the \code{par} argument in the next
#  iteration.
#
#  \item{\code{nf}}. Running total number of function evaluations carried out
#  since iteration 1.
#
#  \item{\code{ng}}. Running total number of gradient evaluations carried out
#  since iteration 1.
#
#  \item{\code{f}}. Optional. The new value of the function, evaluated at the
#  returned value of \code{par}. Only present if calculated as part of the
#  optimization step (e.g. during a line search calculation).
#
#  \item{\code{g}}. Optional. The gradient vector, evaluated at the returned
#  value of \code{par}. Only present if the gradient was calculated as part of
#  the optimization step (e.g. during a line search calculation.)}
#
#@seealso \code{\link{make_mize}} to create a value to pass to \code{opt},
#  \code{\link{mize_init}} to initialize \code{opt} before passing it to this
#  function for the first time. \code{\link{mize}} creates an optimizer and
#  carries out a full optimization with it.
# @examples
# rosenbrock_fg <- list(
#   fn = function(x) {
#     100 * (x[2] - x[1] * x[1]) ^ 2 + (1 - x[1]) ^ 2
#   },
#   gr = function(x) {
#     c(
#      -400 * x[1] * (x[2] - x[1] * x[1]) - 2 * (1 - x[1]),
#       200 *        (x[2] - x[1] * x[1]))
#  })
#  rb0 <- c(-1.2, 1)
#
#  opt <- make_mize(method = "SD", line_search = "const", step0 = 0.0001,
#                   par = rb0, fg = rosenbrock_fg)
#  par <- rb0
#  for (iter in 1:3) {
#    res <- mize_step(opt, par, rosenbrock_fg)
#    par <- res$par
#    opt <- res$opt
#  }
#@export
mize_step <- function(opt, par, fg) {
  opt$iter <- opt$iter + 1
  iter <- opt$iter
  opt <- life_cycle_hook("step", "before", opt, par, fg, iter)

  par0 <- par
  step_result <- NULL

  for (i in 1:length(opt$stages)) {
    opt$stage_i <- i
    stage <- opt$stages[[i]]
    opt <- life_cycle_hook(stage$type, "before", opt, par, fg, iter)
    if (!is.null(opt$terminate)) {
      break
    }
    opt <- life_cycle_hook(stage$type, "during", opt, par, fg, iter)
    if (!is.null(opt$terminate)) {
      break
    }
    opt <- life_cycle_hook(stage$type, "after", opt, par, fg, iter)
    if (!is.null(opt$terminate)) {
      break
    }

    stage <- opt$stages[[i]]

    if (is.null(step_result)) {
      step_result <- stage$result
    }
    else {
      step_result <- step_result + stage$result
    }

    if (opt$eager_update) {
      par <- par + stage$result
    }

    opt <- life_cycle_hook("stage", "after", opt, par, fg, iter)
    if (!is.null(opt$terminate)) {
      break
    }
  }

  if (is.null(opt$terminate)) {
    opt$ok <- TRUE
    if (!opt$eager_update) {
      par <- par + step_result
    }

    # intercept whether we want to accept the new solution
    opt <- life_cycle_hook("validation", "before", opt, par, fg, iter,
                           par0, step_result)
    opt <- life_cycle_hook("validation", "during", opt, par, fg, iter,
                           par0, step_result)
  }
  # If the this solution was vetoed or the catastrophe happened,
  # roll back to the previous one.
  if (!is.null(opt$terminate) || !opt$ok) {
    par <- par0
  }

  if (is.null(opt$terminate)) {
    opt <- life_cycle_hook("step", "after", opt, par, fg, iter, par0,
                         step_result)
  }

  res <- list(opt = opt, par = par, nf = opt$counts$fn, ng = opt$counts$gr)
  if (has_fn_curr(opt, iter + 1)) {
    res$f <- opt$cache$fn_curr
  }
  if (has_gr_curr(opt, iter + 1)) {
    res$g <- opt$cache$gr_curr
  }
  res
}

# Initialize the Optimizer.
#
# Prepares the optimizer for use with a specific function and starting point.
#
# Should be called after creating an optimizer with \code{\link{make_mize}} and
# before beginning any optimization with \code{\link{mize_step}}. Note that if
# \code{fg} and \code{par} are available at the time \code{\link{mize_step}} is
# called, they can be passed to that function and initialization will be
# carried out automatically, avoiding the need to call \code{mize_init}.
#
# Optional convergence parameters may also be passed here, for use with
# \code{\link{check_mize_convergence}}. They are optional if you do your own
# convergence checking.
#
# Details of the \code{fg} list can be found in the 'Details' section of
# \code{\link{mize}}.
#
# @param opt Optimizer, created by \code{\link{make_mize}}.
# @param par Vector of initial values for the function to be optimized over.
# @param fg Function and gradient list. See the documentaion of
#   \code{\link{mize}}.
# @param max_iter (Optional). Maximum number of iterations. See the
#   'Convergence' section of \code{\link{mize}} for details.
# @param max_fn (Optional). Maximum number of function evaluations. See the
#   'Convergence' section of \code{\link{mize}} for details.
# @param max_gr (Optional). Maximum number of gradient evaluations. See the
#   'Convergence' section of \code{\link{mize}} for details.
# @param max_fg (Optional). Maximum number of function or gradient evaluations.
#   See the 'Convergence' section of \code{\link{mize}} for details.
# @param abs_tol (Optional). Absolute tolerance for comparing two function
#   evaluations. See the 'Convergence' section of \code{\link{mize}} for
#   details.
# @param rel_tol (Optional). Relative tolerance for comparing two function
#   evaluations. See the 'Convergence' section of \code{\link{mize}} for
#   details.
# @param grad_tol (Optional). Absolute tolerance for the length (l2-norm) of
#   the gradient vector. See the 'Convergence' section of \code{\link{mize}}
#   for details.
# @param ginf_tol (Optional). Absolute tolerance for the infinity norm (maximum
#   absolute component) of the gradient vector. See the 'Convergence' section
#   of \code{\link{mize}} for details.
# @param step_tol (Optional). Absolute tolerance for the size of the parameter
#   update. See the 'Convergence' section of \code{\link{mize}} for details.
# @return Initialized optimizer.
# @export
# @examples
#
# # Create an optimizer
# opt <- make_mize(method = "L-BFGS")
#
# # Function to optimize and starting point defined after creating optimizer
# rosenbrock_fg <- list(
#   fn = function(x) { 100 * (x[2] - x[1] * x[1]) ^ 2 + (1 - x[1]) ^ 2  },
#   gr = function(x) { c( -400 * x[1] * (x[2] - x[1] * x[1]) - 2 * (1 - x[1]),
#                          200 *        (x[2] - x[1] * x[1])) })
# rb0 <- c(-1.2, 1)
#
# # Initialize with function and starting point before commencing optimization
# opt <- mize_init(opt, rb0, rosebrock_fg)
#
# # Finally, can commence the optimization loop
# par <- rb0
# for (iter in 1:3) {
#   res <- mize_step(opt, par, rosenbrock_fg)
#   par <- res$par
#   opt <- res$opt
# }
#
mize_init <- function(opt, par, fg,
                      max_iter = Inf,
                      max_fn = Inf, max_gr = Inf, max_fg = Inf,
                      abs_tol = NULL,
                      rel_tol = abs_tol, grad_tol = NULL, ginf_tol = NULL,
                      step_tol = NULL) {
  opt <- register_hooks(opt)
  opt$iter <- 0
  opt <- life_cycle_hook("opt", "init", opt, par, fg, opt$iter)
  opt$convergence <- list(
    max_iter = max_iter,
    max_fn = max_fn,
    max_gr = max_gr,
    max_fg = max_fg,
    abs_tol = abs_tol,
    rel_tol = rel_tol,
    grad_tol = grad_tol,
    ginf_tol = ginf_tol,
    step_tol = step_tol
  )
  opt$is_initialized <- TRUE
  opt
}

#Mize Step Summary
#
#Produces a result summary for an optimization iteration. Information such as
#function value, gradient norm and step size may be returned.
#
#By default, convergence tolerance parameters will be used to determine what
#function and gradient data is returned. The function value will be returned if
#it was already calculated and cached in the optimization iteration. Otherwise,
#it will be calculated only if a non-null absolute or relative tolerance value
#was asked for. A gradient norm will be returned only if a non-null gradient
#tolerance was specified, even if the gradient is available.
#
#Note that if a function tolerance was specified, but was not calculated for
#the relevant value of \code{par}, they will be calculated here and the
#calculation does contribute to the total function count (and will be cached
#for potential use in the next iteration). The same applies for gradient
#tolerances and gradient calculation. Function and gradient calculation can
#also be forced here by setting the \code{calc_fn} and \code{calc_gr}
#(respectively) parameters to \code{TRUE}.
#
#@param opt Optimizer to generate summary for, from return value of
#  \code{\link{mize_step}}.
#@param par Vector of parameters at the end of the iteration, from return value
#  of \code{\link{mize_step}}.
#@param fg Function and gradient list. See the documentaion of
#  \code{\link{mize}}.
#@param par_old (Optional). Vector of parameters at the end of the previous
#  iteration. Used to calculate step size.
#@param calc_fn (Optional). If \code{TRUE}, force calculation of function if
#  not already cached in \code{opt}, even if it wouldn't be needed for
#  convergence checking.
#@return A list with the following items: \itemize{
#
#  \item \code{opt} Optimizer with updated state (e.g. function and gradient
#  counts).
#
#  \item \code{iter} Iteration number.
#
#  \item \code{f} Function value at \code{par}.
#
#  \item \code{g2n} 2-norm of the gradient at \code{par}.
#
#  \item \code{ginfn} Infinity-norm of the gradient at \code{par}.
#
#  \item \code{nf} Number of function evaluations so far.
#
#  \item \code{ng} Number of gradient evaluations so far.
#
#  \item \code{step} Size of the step between \code{par_old} and \code{par},
#  if \code{par_old} is provided.
#
#  \item \code{alpha} Step length of the gradient descent part of the step.
#
#  \item \code{mu} Momentum coefficient for this iteration}
#@export
#@examples
# rb_fg <- list(
#   fn = function(x) { 100 * (x[2] - x[1] * x[1]) ^ 2 + (1 - x[1]) ^ 2  },
#   gr = function(x) { c( -400 * x[1] * (x[2] - x[1] * x[1]) - 2 * (1 - x[1]),
#                          200 *        (x[2] - x[1] * x[1])) })
# rb0 <- c(-1.2, 1)
#
# opt <- make_mize(method = "BFGS", par = rb0, fg = rb_fg, max_iter = 30)
# mize_res <- mize_step(opt = opt, par = rb0, fg = rb_fg)
# # Get info about first step, use rb0 to compare new par with initial value
# step_info <- mize_step_summary(mize_res$opt, mize_res$par, rb_fg, rb0)
mize_step_summary <- function(opt, par, fg, par_old = NULL, calc_fn = NULL) {

  iter <- opt$iter
  # An internal flag useful for unit tests: if FALSE, doesn't count any
  # fn/gr calculations towards their counts. Can still get back fn/gr values
  # without confusing the issue of the expected number of fn/gr evaluations
  if (!is.null(opt$count_res_fg)) {
    count_fg <- opt$count_res_fg
  }
  else {
    count_fg <- TRUE
  }

  # Whether and what convergence info to calculate if fn/gr calculation not
  # explicitly asked for
  if (is.null(calc_fn)) {
    calc_fn <- is_finite_numeric(opt$convergence$abs_tol) ||
      is_finite_numeric(opt$convergence$rel_tol)
  }

  gr_norms <- c()
  if (is_finite_numeric(opt$convergence$grad_tol)) {
    gr_norms <- c(gr_norms, 2)
  }
  if (is_finite_numeric(opt$convergence$ginf_tol)) {
    gr_norms <- c(gr_norms, Inf)
  }
  calc_gr <- length(gr_norms) > 0

  f <- NULL
  if (calc_fn || has_fn_curr(opt, iter + 1)) {
    if (!has_fn_curr(opt, iter + 1)) {
      f <- fg$fn(par)
      if (count_fg) {
        opt <- set_fn_curr(opt, f, iter + 1)
        opt$counts$fn <- opt$counts$fn + 1
      }
    }
    else {
      f <- opt$cache$fn_curr
    }
  }

  g2n <- NULL
  ginfn <- NULL
  if (calc_gr || has_gr_curr(opt, iter + 1)) {
    if (!has_gr_curr(opt, iter + 1)) {
      g <- fg$gr(par)
      if (grad_is_first_stage(opt) && count_fg) {
        opt <- set_gr_curr(opt, g, iter + 1)
        opt$counts$gr <- opt$counts$gr + 1
      }
    }
    else {
      g <- opt$cache$gr_curr
    }
    if (2 %in% gr_norms) {
      g2n <- norm2(g)
    }
    if (Inf %in% gr_norms) {
      ginfn <- norm_inf(g)
    }
  }

  if (!is.null(par_old)) {
    step_size <- norm2(par - par_old)
  }
  else {
    step_size <- 0
  }

  alpha <- 0
  if (!is.null(opt$stages[["gradient_descent"]]$step_size$value)) {
    alpha <- norm2(opt$stages[["gradient_descent"]]$step_size$value)
    if (is.null(alpha)) {
      alpha <- 0
    }
  }

  res <- list(
    opt = opt,
    f = f,
    g2n = g2n,
    ginfn = ginfn,
    nf = opt$counts$fn,
    ng = opt$counts$gr,
    step = step_size,
    alpha = alpha,
    iter = iter
  )

  if ("momentum" %in% names(opt$stages)) {
    res$mu <- opt$stages[["momentum"]]$step_size$value
    if (is.null(res$mu)) {
      res$mu <- 0
    }
  }

  Filter(Negate(is.null), res)
}


# Check Optimization Convergence
#
# Updates the optimizer with information about convergence or termination,
# signalling if the optimization process should stop.
#
# On returning from this function, the updated value of \code{opt} will
# contain: \itemize{
#
# \item A boolean value \code{is_terminated} which is \code{TRUE} if
# termination has been indicated, and \code{FALSE} otherwise.
#
# \item A list \code{terminate} if \code{is_terminated} is \code{TRUE}. This
# contains two items: \code{what}, a short string describing what caused the
# termination, and \code{val}, the value of the termination criterion that
# caused termination. This list will not be present if \code{is_terminated} is
# \code{FALSE}.}
#
# Convergence criteria are only checked here. To set these criteria, use
# \code{\link{make_mize}} or \code{\link{mize_init}}.
#
# @param mize_step_info Step info for this iteration, created by
#   \code{\link{mize_step_summary}}
# @return \code{opt} updated with convergence and termination data. See
#   'Details'.
# @export
#@examples
# rb_fg <- list(
#   fn = function(x) { 100 * (x[2] - x[1] * x[1]) ^ 2 + (1 - x[1]) ^ 2  },
#   gr = function(x) { c( -400 * x[1] * (x[2] - x[1] * x[1]) - 2 * (1 - x[1]),
#                          200 *        (x[2] - x[1] * x[1])) })
# rb0 <- c(-1.2, 1)
#
# opt <- make_mize(method = "BFGS", par = rb0, fg = rb_fg, max_iter = 30)
# mize_res <- mize_step(opt = opt, par = rb0, fg = rb_fg)
# step_info <- mize_step_summary(mize_res$opt, mize_res$par, rb_fg, rb0)
# # check convergence by looking at opt$is_terminated
# opt <- check_mize_convergence(step_info)
check_mize_convergence <- function(mize_step_info) {

  opt <- mize_step_info$opt

  convergence <- opt$convergence

  terminate <- check_counts(opt, convergence$max_fn, convergence$max_gr,
                            convergence$max_fg)
  if (!is.null(terminate)) {
    opt$terminate <- terminate
    opt$is_terminated <- TRUE
    return(opt)
  }

  terminate <- check_step_conv(opt, opt$iter, mize_step_info$step,
                               convergence$step_tol)
  if (!is.null(terminate)) {
    opt$terminate <- terminate
    opt$is_terminated <- TRUE
    return(opt)
  }

  terminate <- check_gr_conv(opt, convergence$grad_tol, convergence$ginf_tol)
  if (!is.null(terminate)) {
    opt$terminate <- terminate
    opt$is_terminated <- TRUE
    return(opt)
  }

  if (!is.null(opt$cache$fn_curr)) {
    fn_new <- opt$cache$fn_curr
    fn_old <- convergence$fn_new
    convergence$fn_new <- fn_new
    opt$convergence <- convergence

    terminate <- check_fn_conv(opt, opt$iter, fn_old, fn_new,
                               convergence$abs_tol, convergence$rel_tol)
    if (!is.null(terminate)) {
      opt$is_terminated <- TRUE
      opt$terminate <- terminate
      return(opt)
    }
  }

  if (opt$iter == convergence$max_iter) {
    opt$is_terminated <- TRUE
    opt$terminate <- list(what = "max_iter", val = convergence$max_iter)
  }

  opt
}
# Momentum ----------------------------------------------------------------

# Create a direction sub stage for momentum
momentum_direction <- function(normalize = FALSE) {
  make_direction(list(
    name = "classical_momentum",
    calculate = function(opt, stage, sub_stage, par, fg, iter) {

      sub_stage$value <- opt$cache$update_old

      if (sub_stage$normalize) {
        sub_stage$value <- normalize(sub_stage$value)
      }
      list(sub_stage = sub_stage)
    },
    depends = c("update_old"),
    normalize = normalize
  ))
}

# Creates a step size sub stage for momentum
# mu_fn a function that takes an iteration number and returns the momentum.
#  Adaptive restart can restart the momentum in which case the function will
#  be passed an "effective" iteration number which may be smaller than the
#  actual iteration value.
# use_init_mom If TRUE, then always use the momentum coefficient specified by
#  mu_fn even when the effective iteration is 1 (first iteration or restart).
#  In some cases using non-standard momentum (e.g. Nesterov or linear-weighted),
#  this could result in the resulting step length being shorter or longer
#  than would be otherwise expected. If FALSE, then the momentum coefficient
#  is always zero.
make_momentum_step <- function(mu_fn,
                               min_momentum = 0,
                               max_momentum = 1,
                               use_init_mom = FALSE,
                               verbose = FALSE) {
  make_step_size(list(
    name = "momentum_step",
    init = function(opt, stage, sub_stage, par, fg, iter) {
      sub_stage$value <- 0
      sub_stage$t <- 1
      list(sub_stage = sub_stage)
    },
    calculate = function(opt, stage, sub_stage, par, fg, iter) {
      if (!use_init_mom && sub_stage$t <= 1) {
        sub_stage$value <- 0
      }
      else {
        sub_stage$value <-
          sclamp(sub_stage$mu_fn(sub_stage$t, opt$convergence$max_iter),
                 min = sub_stage$min_value,
                 max = sub_stage$max_value)
      }
      list(sub_stage = sub_stage)
    },
    after_step = function(opt, stage, sub_stage, par, fg, iter, par0,
                          update) {
      sub_stage$t <- sub_stage$t + 1

      list(sub_stage = sub_stage)
    },
    mu_fn = mu_fn,
    min_value = min_momentum,
    max_value = max_momentum,
    t = 0
  ))
}


# Function Factories ------------------------------------------------------

# A function that switches from one momentum value to another at the
# specified iteration.
make_switch <- function(init_value = 0.5, final_value = 0.8,
                        switch_iter = 250) {
  function(iter, max_iter) {
    if (iter >= switch_iter) {
      return(final_value)
    }
    else {
      return(init_value)
    }
  }
}

# A function that increases from init_value to final_value over
# max_iter iterations. Iter 0 will always return a value of zero, iter 1
# begins with init_value.
#
# wait - if set to a non-zero value, recalculates the values so that
# the init_value is used for 'wait' extra iterations, but with final_value
# still reached after max_iter iterations. Set to 1 for momentum calculations
# where in most cases the momentum on the first iteration would be either
# ignored or the value overridden and set to zero anyway. Stops a larger than
# expected jump on iteration 2.
make_ramp <- function(init_value = 0,
                      final_value = 0.9,
                      wait = 0) {

  function(iter, max_iter) {
    # actual number of iterations
    iters <- max_iter - 1 - wait
    # denominator of linear scaling
    n <- max(iters, 1)
    m <- (final_value - init_value) / n
    t <- iter - 1 - wait
    if (t < 0) {
      return(init_value)
    }

    (m * t) + init_value
  }
}

# A function that returns a constant momentum value
make_constant <- function(value) {
  function(iter, max_iter) {
    value
  }
}

# Momentum Correction -----------------------------------------------------

# Normally, momentum schemes are given as eps*grad + mu*old_update, but
# some momentum schemes define the update as: (1-mu)*eps*grad + mu*old_update
# which can easily be expanded as: eps*grad + mu*old_update - mu*eps*grad
# i.e. add an extra stage to substract a fraction (mu worth) of the gradient
# descent

# The momentum correction direction: the opposite direction the gradient
# descent.
momentum_correction_direction <- function() {
  make_direction(list(
    name = "momentum_correction_direction",
    calculate = function(opt, stage, sub_stage, par, fg, iter) {
      #message("Calculating momentum correction direction")

      grad_stage <- opt$stages[["gradient_descent"]]
      sub_stage$value <- -grad_stage$direction$value

      list(sub_stage = sub_stage)
    }
  ))
}

# The momentum correction step size: mu times the gradient descent step size.
momentum_correction_step <- function() {
  make_step_size(list(
    name = "momentum_correction_step",
    calculate = function(opt, stage, sub_stage, par, fg, iter) {

      grad_stage <- opt$stages[["gradient_descent"]]
      grad_step <- grad_stage$step_size$value

      mom_stage <- opt$stages[["momentum"]]
      mom_step <- mom_stage$step_size$value

      sub_stage$value <- grad_step * mom_step
      list(sub_stage = sub_stage)
    }
  ))
}

# Momentum Dependencies ------------------------------------------------------------

# Save this update for use in the next step
require_update_old <- function(opt, par, fg, iter, par0, update) {
  opt$cache$update_old <- update
  opt
}
attr(require_update_old, 'event') <- 'after step'
attr(require_update_old, 'name') <- 'update_old'
attr(require_update_old, 'depends') <- 'update_old_init'

# Initialize the old update vector
require_update_old_init <- function(opt, stage, sub_stage, par, fg, iter) {
  opt$cache$update_old <- rep(0, length(par))
  list(opt = opt)
}
attr(require_update_old_init, 'event') <- 'init momentum direction'
attr(require_update_old_init, 'name') <- 'update_old_init'


# More'-Thuente Line Search
#
# Combination of the cvsrch and cstep matlab files.
#
# Line Search Factory Function
#
# Returns a line search function using a variant of the More-Thuente
#  line search originally implemented in
#  \href{http://www.netlib.org/minpack/}{MINPACK}.
#
# @param c1 Constant used in sufficient decrease condition. Should take a value
#   between 0 and 1.
# @param c2 Constant used in curvature condition. Should take a value between
#   \code{c1} and 1.
# @param max_fn Maximum number of function evaluations allowed.
# @return Line search function.
# @references
# More, J. J., & Thuente, D. J. (1994).
# Line search algorithms with guaranteed sufficient decrease.
# \emph{ACM Transactions on Mathematical Software (TOMS)}, \emph{20}(3),
# 286-307.
# @seealso This code is based on a translation of the original MINPACK code
#  for Matlab by
#  \href{https://www.cs.umd.edu/users/oleary/software/}{Dianne O'Leary}.
more_thuente <- function(c1 = 1e-4, c2 = 0.1, max_fn = Inf, eps = 1e-6,
                         approx_armijo = FALSE,
                         strong_curvature = TRUE,
                         verbose = FALSE) {
  if (approx_armijo) {
    armijo_check_fn <- make_approx_armijo_ok_step(eps)
  }
  else {
    armijo_check_fn <- armijo_ok_step
  }

  wolfe_ok_step_fn <- make_wolfe_ok_step_fn(strong_curvature = strong_curvature,
                                            approx_armijo = approx_armijo,
                                            eps = eps)

  function(phi, step0, alpha,
           total_max_fn = Inf, total_max_gr = Inf, total_max_fg = Inf,
           pm = NULL) {
    maxfev <- min(max_fn, total_max_fn, total_max_gr, floor(total_max_fg / 2))
    if (maxfev <= 0) {
      return(list(step = step0, nfn = 0, ngr = 0))
    }
    res <- cvsrch(phi, step0, alpha = alpha, c1 = c1, c2 = c2,
                  maxfev = maxfev,
                  armijo_check_fn = armijo_check_fn,
                  wolfe_ok_step_fn = wolfe_ok_step_fn, verbose = verbose)
    list(step = res$step, nfn = res$nfn, ngr = res$nfn)
  }
}

# More'-Thuente Line Search
#
# This routine is a translation of Dianne O'Leary's Matlab code, which was
# itself a translation of the MINPACK original. Original comments to the Matlab
# code are at the end.
# @param phi Line function.
# @param step0 Line search values at starting point of line search.
# @param alpha Initial guess for step size.
# @param c1 Constant used in sufficient decrease condition. Should take a value
#   between 0 and 1.
# @param c2 Constant used in curvature condition. Should take a value between
#   c1 and 1.
# @param xtol Relative width tolerance: convergence is reached if width falls
#   below xtol * maximum step size.
# @param alpha_min Smallest acceptable value of the step size.
# @param alpha_max Largest acceptable value of the step size.
# @param maxfev Maximum number of function evaluations allowed.
# @param delta Value to force sufficient decrease of interval size on
#   successive iterations. Should be a positive value less than 1.
# @return List containing:
# \itemize{
#   \item \code{step} Best step found and associated line search info.
#   \item \code{info} Return code from convergence check.
#   \item \code{nfn}  Number of function evaluations.
# }
#   Translation of minpack subroutine cvsrch
#   Dianne O'Leary   July 1991
#     **********
#
#     Subroutine cvsrch
#
#     The purpose of cvsrch is to find a step which satisfies
#     a sufficient decrease condition and a curvature condition.
#     The user must provide a subroutine which calculates the
#     function and the gradient.
#
#     At each stage the subroutine updates an interval of
#     uncertainty with endpoints stx and sty. The interval of
#     uncertainty is initially chosen so that it contains a
#     minimizer of the modified function
#
#          f(x+alpha*pv) - f(x) - c1*alpha*(gradf(x)'pv).
#
#     If a step is obtained for which the modified function
#     has a nonpositive function value and nonnegative derivative,
#     then the interval of uncertainty is chosen so that it
#     contains a minimizer of f(x+alpha*pv).
#
#     The algorithm is designed to find a step which satisfies
#     the sufficient decrease condition
#
#           f(x+alpha*pv) <= f(x) + c1*alpha*(gradf(x)'pv),
#
#     and the curvature condition
#
#           abs(gradf(x+alpha*pv)'pv)) <= c2*abs(gradf(x)'pv).
#
#     If c1 is less than c2 and if, for example, the function
#     is bounded below, then there is always a step which satisfies
#     both conditions. If no step can be found which satisfies both
#     conditions, then the algorithm usually stops when rounding
#     errors prevent further progress. In this case alpha only
#     satisfies the sufficient decrease condition.
#
#     The subroutine statement is
#
#        subroutine cvsrch(fcn,n,x,f,g,pv,alpha,c1,c2,xtol,
#                          alpha_min,alpha_max,maxfev,info,nfev)
#     where
#
#	fcn is the name of the user-supplied subroutine which
#         calculates the function and the gradient.  fcn must
#      	  be declared in an external statement in the user
#         calling program, and should be written as follows.
#
#         function [f,g] = fcn(n,x) (Matlab)     (10/2010 change in documentation)
#	  (derived from Fortran subroutine fcn(n,x,f,g) )
#         integer n
#         f
#         x(n),g(n)
#	  ---
#         Calculate the function at x and
#         return this value in the variable f.
#         Calculate the gradient at x and
#         return this vector in g.
#	  ---
#	  return
#	  end
#
#       n is a positive integer input variable set to the number
#	  of variables.
#
#	x is an array of length n. On input it must contain the
#	  base point for the line search. On output it contains
#         x + alpha*pv.
#
#	f is a variable. On input it must contain the value of f
#         at x. On output it contains the value of f at x + alpha*pv.
#
#	g is an array of length n. On input it must contain the
#         gradient of f at x. On output it contains the gradient
#         of f at x + alpha*pv.
#
#	pv is an input array of length n which specifies the
#         search direction.
#
#	alpha is a nonnegative variable. On input alpha contains an
#         initial estimate of a satisfactory step. On output
#         alpha contains the final estimate.
#
#       c1 and c2 are nonnegative input variables. Termination
#         occurs when the sufficient decrease condition and the
#         directional derivative condition are satisfied.
#
#	xtol is a nonnegative input variable. Termination occurs
#         when the relative width of the interval of uncertainty
#	  is at most xtol.
#
#	alpha_min and alpha_max are nonnegative input variables which
#	  specify lower and upper bounds for the step.
#
#	maxfev is a positive integer input variable. Termination
#         occurs when the number of calls to fcn is at least
#         maxfev by the end of an iteration.
#
#	info is an integer output variable set as follows:
#
#	  info = 0  Improper input parameters.
#
#	  info = 1  The sufficient decrease condition and the
#                   directional derivative condition hold.
#
#	  info = 2  Relative width of the interval of uncertainty
#		    is at most xtol.
#
#	  info = 3  Number of calls to fcn has reached maxfev.
#
#	  info = 4  The step is at the lower bound alpha_min.
#
#	  info = 5  The step is at the upper bound alpha_max.
#
#	  info = 6  Rounding errors prevent further progress.
#                   There may not be a step which satisfies the
#                   sufficient decrease and curvature conditions.
#                   Tolerances may be too small.
#
#       nfev is an integer output variable set to the number of
#         calls to fcn.
#
#
#     Subprograms called
#
#	user-supplied......fcn
#
#	MINPACK-supplied...cstep
#
#	FORTRAN-supplied...abs,max,min
#
#     Argonne National Laboratory. MINPACK Project. June 1983
#     Jorge J. More', David J. Thuente
#
#     **********
cvsrch <- function(phi, step0, alpha = 1,
                   c1 = 1e-4, c2 = 0.1, xtol = .Machine$double.eps,
                   alpha_min = 0, alpha_max = Inf,
                   maxfev = Inf, delta = 0.66,
                   armijo_check_fn = armijo_ok_step,
                   wolfe_ok_step_fn = strong_wolfe_ok_step,
                   verbose = FALSE) {

  # increase width by this amount during zoom phase
  xtrapf <- 4
  infoc <- 1

  # Check the input parameters for errors.
  params_ok <- TRUE
  problems <- c()
  if (alpha <= 0.0) {
    params_ok <- FALSE
    problems <- c(problems, paste0("alpha <= 0.0: ", formatC(alpha)))
  }
  if (c1 < 0.0) {
    params_ok <- FALSE
    problems <- c(problems, paste0("c1 < 0.0: ", formatC(c1)))
  }
  if (c2 < 0.0) {
    params_ok <- FALSE
    problems <- c(problems, paste0("c2 < 0.0: ", formatC(c2)))
  }
  if (xtol < 0.0) {
    params_ok <- FALSE
    problems <- c(problems, paste0("xtol < 0.0: ", formatC(xtol)))
  }
  if (alpha_min < 0.0) {
    params_ok <- FALSE
    problems <- c(problems, paste0("alpha_min < 0.0: ", formatC(alpha_min)))
  }
  if (alpha_max < alpha_min) {
    params_ok <- FALSE
    problems <- c(problems, paste0("alpha_max ", formatC(alpha_max)
                         , " < alpha_min ", formatC(alpha_min)))
  }
  if (maxfev < 0) {
    params_ok <- FALSE
    problems <- c(problems, paste0("maxfev < 0: ", formatC(maxfev)))
  }
  if (!params_ok) {
    problems <- paste(problems, collapse = "; ")
    stop("Parameter errors detected: ", problems)
  }

  if (maxfev == 0) {
    return(list(step = step0, nfn = 0, info = 3))
  }

  # Check that pv is a descent direction: if not, return a zero step.
  if (step0$d >= 0.0) {
    return(list(step = step0, info = 6, nfn = 0))
  }
  dgtest <- c1 * step0$d

  # Initialize local variables.
  bracketed <- FALSE
  brackt <- FALSE
  stage1 <- TRUE
  nfev <- 0

  width <- alpha_max - alpha_min
  width_old <- 2 * width

  # The variables stx, fx, dgx contain the values of the step,
  # function, and directional derivative at the best step.
  # The variables sty, fy, dgy contain the value of the step,
  # function, and derivative at the other endpoint of
  # the interval of uncertainty.
  # The variables alpha, f, dg contain the values of the step,
  # function, and derivative at the current step.
  stepx <- step0
  stepy <- step0
  step <- list(alpha = alpha)

  #     Start of iteration.
  iter <- 0
  while (1) {
    iter <- iter + 1
    # Set the minimum and maximum steps to correspond
    # to the present interval of uncertainty.
    if (brackt) {
      stmin <- min(stepx$alpha, stepy$alpha)
      stmax <- max(stepx$alpha, stepy$alpha)
    } else {
      stmin <- stepx$alpha
      stmax <- step$alpha + xtrapf * (step$alpha - stepx$alpha)
    }

    # Force the step to be within the bounds alpha_max and alpha_min.
    step$alpha <- max(step$alpha, alpha_min)
    step$alpha <- min(step$alpha, alpha_max)

    if (verbose) {
    message("Bracket: [", formatC(stmin), ", ", formatC(stmax),
            "] alpha = ", formatC(step$alpha))
    }

    # Evaluate the function and gradient at alpha
    # and compute the directional derivative.
    # Additional check: bisect (if needed) until a finite value is found
    # (most important for first iteration)
    ffres <- find_finite(phi, step$alpha, maxfev - nfev, min_alpha = stmin)
    nfev <- nfev + ffres$nfn
    if (!ffres$ok) {
      if (verbose) {
        message("Unable to create finite alpha")
      }

      return(list(step = step0, nfn = nfev, info = 7))
    }
    step <- ffres$step

    # Test for convergence.
    info <- check_convergence(step0, step, brackt, infoc, stmin, stmax,
                              alpha_min, alpha_max, c1, c2, nfev,
                              maxfev, xtol,
                              armijo_check_fn = armijo_check_fn,
                              wolfe_ok_step_fn = wolfe_ok_step_fn,
                              verbose = verbose)

    # Check for termination.
    if (info != 0) {
      # If an unusual termination is to occur, then set step to the best step
      # found
      if (info == 2 || info == 3 || info == 6) {
        step <- stepx
      }
      if (verbose) {
        message("alpha = ", formatC(step$alpha))
      }
      return(list(step = step, info = info, nfn = nfev))
    }

    # In the first stage we seek a step for which the modified
    # function has a nonpositive value and nonnegative derivative.

    # In the original MINPACK the following test is:
    # if (stage1 .and. f .le. ftest1 .and.
    #    *       dg .ge. min(ftol,gtol)*dginit) stage1
    # which translates to: step$f <= f0 + alpha * c1 * d0 &&
    #            step$df >= min(c1, c2) * alpha * d0
    # The second test is the armijo condition and the third is the
    # curvature condition but using the smaller of c1 and
    # c2. This is nearly the standard Wolfe conditions, but because c1 is
    # always <= c2 for a convergent line search, this means
    # we would always use c1 for the curvature condition.
    # I have translated this faithfully, but it seems odd. Using c2 has no
    # effect on the test function from the More'-Thuente paper
    if (stage1 && wolfe_ok_step(step0, step, c1, min(c1, c2))) {
      stage1 <- FALSE
    }

    # A modified function is used to predict the step only if
    # we have not obtained a step for which the modified
    # function has a nonpositive function value and nonnegative
    # derivative, and if a lower function value has been
    # obtained but the decrease is not sufficient.
    if (stage1 && step$f <= stepx$f && !armijo_check_fn(step0, step, c1)) {
      # Define the modified function and derivative values.
      stepxm <- modify_step(stepx, dgtest)
      stepym <- modify_step(stepy, dgtest)
      stepm <- modify_step(step, dgtest)

      step_result <- cstep(stepxm, stepym, stepm, brackt, stmin, stmax,
                           verbose = verbose)

      brackt <- step_result$brackt
      infoc <- step_result$info
      stepxm <- step_result$stepx
      stepym <- step_result$stepy
      stepm <- step_result$step

      # Reset the function and gradient values for f.
      stepx <- unmodify_step(stepxm, dgtest)
      stepy <- unmodify_step(stepym, dgtest)
      step$alpha <- stepm$alpha
    } else {
      # Call cstep to update the interval of uncertainty
      # and to compute the new step.
      step_result <- cstep(stepx, stepy, step, brackt, stmin, stmax,
                           verbose = verbose)
      brackt <- step_result$brackt
      infoc <- step_result$info
      stepx <- step_result$stepx
      stepy <- step_result$stepy
      step <- step_result$step
    }

    if (!bracketed && brackt) {
      bracketed <- TRUE
      if (verbose) {
        message("Bracketed")
      }
    }

    # Force a sufficient decrease in the size of the interval of uncertainty.
    if (brackt) {
      # if the length of I does not decrease by a factor of delta < 1
      # then use a bisection step for the next trial alpha
      width_new <- abs(stepy$alpha - stepx$alpha)
      if (width_new >= delta * width_old) {
        if (verbose) {
          message("Interval did not decrease sufficiently: bisecting")
        }
        step$alpha <- stepx$alpha + 0.5 * (stepy$alpha - stepx$alpha)
      }
      width_old <- width
      width <- width_new
    }
  }
}


# Modify Line Search Values
#
# Modifies a line search function and directional derivative value.
# Used by MINPACK version of More'-Thuente line search algorithm.
#
# @param step Line search information.
# @param dgtest Product of the initial line search directional derivative and
#   the sufficent decrease condition constant.
# @return Modified step size.
modify_step <- function(step, dgtest) {
  stepm <- step
  stepm$f <- step$f - step$alpha * dgtest
  stepm$d <- step$d - dgtest
  stepm
}

# Un-modify Line Search Values
#
# Un-modifies a line search function and directional derivative value that was
# modified by the modify_step function. Used by MINPACK version of More'-Thuente
# line search algorithm.
#
# @param stepm Modified line search information.
# @param dgtest Product of the initial line search directional derivative and
#   the sufficent decrease condition constant.
# @return Unmodified step size.
unmodify_step <- function(stepm, dgtest) {
  stepm$f <- stepm$f + stepm$alpha * dgtest
  stepm$d <- stepm$d + dgtest
  stepm
}

# Check Convergence of More'-Thuente Line Search
#
# @param step0 Line search values at starting point.
# @param step Line search value at a step along the line.
# @param brackt TRUE if the step has been bracketed.
# @param infoc Return code of the last step size update.
# @param stmin Smallest value of the step size interval.
# @param stmax Largest value of the step size interval.
# @param alpha_min Smallest acceptable value of the step size.
# @param alpha_max Largest acceptable value of the step size.
# @param c1 Constant used in sufficient decrease condition. Should take a value
#   between 0 and 1.
# @param c2 Constant used in curvature condition. Should take a value between
#   c1 and 1.
# @param dgtest Product of the initial line search directional derivative and
#   the sufficent decrease condition constant.
# @param nfev Current number of function evaluations.
# @param maxfev Maximum number of function evaluations allowed.
# @param xtol Relative width tolerance: convergence is reached if width falls
#   below xtol * stmax.
# @return Integer code indicating convergence state:
#  \itemize{
#   \item \code{0} No convergence.
#   \item \code{1} The sufficient decrease condition and the directional
#     derivative condition hold.
#	  \item \code{2} Relative width of the interval of uncertainty
#		    is at most xtol.
#	  \item \code{3} Number of calls to fcn has reached maxfev.
#	  \item \code{4} The step is at the lower bound alpha_min.
#	  \item \code{5} The step is at the upper bound alpha_max.
#	  \item \code{6} Rounding errors prevent further progress.
# }
# NB dgtest was originally used in testing for min/max alpha test (code 4 and 5)
# but has been replaced with a call to the curvature test using c1 instead of c2
# so dgtest is no longer used in the body of the function.
check_convergence <- function(step0, step, brackt, infoc, stmin, stmax,
                              alpha_min, alpha_max, c1, c2, nfev,
                              maxfev, xtol, armijo_check_fn = armijo_ok_step,
                              wolfe_ok_step_fn = strong_wolfe_ok_step,
                              verbose = FALSE) {
  info <- 0
  if ((brackt && (step$alpha <= stmin || step$alpha >= stmax)) || infoc == 0) {
    if (verbose) {
      message("MT: Rounding errors prevent further progress: stmin = ",
            formatC(stmin), " stmax = ", formatC(stmax))
    }
    # rounding errors prevent further progress
    info <- 6
  }
  # use of c1 in curvature check is on purpose (it's in the MINPACK code)
  if (step$alpha == alpha_max && armijo_check_fn(step0, step, c1) &&
      !curvature_ok_step(step0, step, c1)) {
    # reached alpha_max
    info <- 5
    if (verbose) {
      message("MT: Reached alpha max")
    }

  }
  # use of c1 in curvature check here is also in MINPACK code
  if (step$alpha == alpha_min && (!armijo_check_fn(step0, step, c1) ||
                                  curvature_ok_step(step0, step, c1))) {
    # reached alpha_min
    info <- 4
    if (verbose) {
      message("MT: Reached alpha min")
    }

  }
  if (nfev >= maxfev) {
    # maximum number of function evaluations reached
    info <- 3
    if (verbose) {
      message("MT: exceeded fev")
    }
  }
  if (brackt && stmax - stmin <= xtol * stmax) {
    # interval width is below xtol
    info <- 2
    if (verbose) {
      message("MT: interval width is <= xtol: ", formatC(xtol * stmax))
    }

  }
  if (wolfe_ok_step_fn(step0, step, c1, c2)) {
    # success!
    info <- 1
    if (verbose) {
      message("Success!")
    }
  }
  info
}


# Part of the More'-Thuente line search.
#
# Updates the interval of uncertainty of the current step size and updates the
# current best step size.
#
# This routine is a translation of Dianne O'Leary's Matlab code, which was
# itself a translation of the MINPACK original. Original comments to the Matlab
# code are at the end.
#
# @param stepx One side of the updated step interval, and the associated
#     line search values.
# @param stepy Other side of the updated step interval, and the
#     associated line search values.
# @param step Optimal step size and associated line search
#     value.
# @param brackt TRUE if the interval has been bracketed.
# @param stpmin Minimum allowed interval length.
# @param stpmax Maximum allowed interval length.
# @return List containing:
# \itemize{
#   \item \code{stepx} One side of the updated step interval, and the associated
#     line search values.
#   \item \code{stepy} Other side of the updated step interval, and the
#     associated line search values.
#   \item \code{step} Updated optimal step size and associated line search
#     value.
#   \item \code{brackt} TRUE if the interval has been bracketed.
#   \item \code{info} Integer return code.
# }
# The possible integer return codes refer to the cases 1-4 enumerated in the
# original More'-Thuente paper that correspond to different line search values
# at the ends of the interval and the current best step size (and therefore
# the type of cubic or quadratic interpolation). An integer value of 0 indicates
# that the input parameters are invalid.
#
#
#   Translation of minpack subroutine cstep
#   Dianne O'Leary   July 1991
#     **********
#
#     Subroutine cstep
#
#     The purpose of cstep is to compute a safeguarded step for
#     a linesearch and to update an interval of uncertainty for
#     a minimizer of the function.
#
#     The parameter stx contains the step with the least function
#     value. The parameter stp contains the current step. It is
#     assumed that the derivative at stx is negative in the
#     direction of the step. If brackt is set true then a
#     minimizer has been bracketed in an interval of uncertainty
#     with end points stx and sty.
#
#     The subroutine statement is
#
#       subroutine cstep(stx,fx,dx,sty,fy,dy,stp,fp,dp,brackt,
#                        stpmin,stpmax,info)
#
#     where
#
#       stx, fx, and dx are variables which specify the step,
#         the function, and the derivative at the best step obtained
#         so far. The derivative must be negative in the direction
#         of the step, that is, dx and stp-stx must have opposite
#         signs. On output these parameters are updated appropriately.
#
#       sty, fy, and dy are variables which specify the step,
#         the function, and the derivative at the other endpoint of
#         the interval of uncertainty. On output these parameters are
#         updated appropriately.
#
#       stp, fp, and dp are variables which specify the step,
#         the function, and the derivative at the current step.
#         If brackt is set true then on input stp must be
#         between stx and sty. On output stp is set to the new step.
#
#       brackt is a logical variable which specifies if a minimizer
#         has been bracketed. If the minimizer has not been bracketed
#         then on input brackt must be set false. If the minimizer
#         is bracketed then on output brackt is set true.
#
#       stpmin and stpmax are input variables which specify lower
#         and upper bounds for the step.
#
#       info is an integer output variable set as follows:
#         If info <- 1,2,3,4,5, then the step has been computed
#         according to one of the five cases below. Otherwise
#         info <- 0, and this indicates improper input parameters.
#
#     Subprograms called
#
#       FORTRAN-supplied ... abs,max,min,sqrt
#                        ... dble
#
#     Argonne National Laboratory. MINPACK Project. June 1983
#     Jorge J. More', David J. Thuente
#
#     **********
cstep <-  function(stepx, stepy, step, brackt, stpmin, stpmax,
                   verbose = FALSE) {

  stx <- stepx$alpha
  fx <- stepx$f
  dx <- stepx$d
  dfx <- stepx$df

  sty <- stepy$alpha
  fy <- stepy$f
  dy <- stepy$d
  dfy <- stepy$df

  stp <- step$alpha
  fp <- step$f
  dp <- step$d
  dfp <- step$df

  delta <- 0.66
  info <- 0
  # Check the input parameters for errors.
  if ((brackt && (stp <= min(stx, sty) ||
                  stp >= max(stx, sty))) ||
      dx * (stp - stx) >= 0.0 || stpmax < stpmin) {
    list(
      stepx = stepx,
      stepy = stepy,
      step = step,
      brackt = brackt, info = info)
  }
  # Determine if the derivatives have opposite sign.
  sgnd <- dp * (dx / abs(dx))

  # First case. Trial function value is larger, so choose a step which is
  # closer to stx.
  # The minimum is bracketed.
  # If the cubic step is closer to stx than the quadratic step, the cubic step
  # is taken else the average of the cubic and quadratic steps is taken.
  if (fp > fx) {
    info <- 1
    bound <- TRUE

    stpc <- cubic_interpolate(stx, fx, dx, stp, fp, dp, ignoreWarnings = TRUE)
    stpq <- quadratic_interpolate(stx, fx, dx, stp, fp)

    if (is.nan(stpc)) {
      stpf <- stpq
    }
    else {
      if (abs(stpc - stx) < abs(stpq - stx)) {
        stpf <- stpc
      } else {
        stpf <- stpc + (stpq - stpc) / 2
      }
    }
    brackt <- TRUE

    # Second case. A lower function value and derivatives of
    # opposite sign. The minimum is bracketed. If the cubic
    # step is closer to stx than the quadratic (secant) step,
    # the cubic step is taken, else the quadratic step is taken.
  } else if (sgnd < 0.0) {
    info <- 2
    bound <- FALSE

    stpc <- cubic_interpolate(stx, fx, dx, stp, fp, dp, ignoreWarnings = TRUE)
    stpq <- quadratic_interpolateg(stp, dp, stx, dx)
    if (is.nan(stpc)) {
      stpf <- stpq
    }
    else {
      if (abs(stpc - stp) > abs(stpq - stp)) {
        stpf <- stpc
      } else {
        stpf <- stpq
      }
    }

    brackt <- TRUE

    # Third case. A lower function value, derivatives of the
    # same sign, and the magnitude of the derivative decreases.
    # The next trial step exists outside the interval so is an extrapolation.
    # The cubic may not have a minimizer. If it does, it may be in the
    # wrong direction, e.g. stc < stx < stp
    # The cubic step is only used if the cubic tends to infinity
    # in the direction of the step and if the minimum of the cubic
    # is beyond stp. Otherwise the cubic step is defined to be
    # either stpmin or stpmax. The quadratic (secant) step is also
    # computed and if the minimum is bracketed then the the step
    # closest to stx is taken, else the step farthest away is taken.
  } else if (abs(dp) < abs(dx)) {
    info <- 3
    bound <- TRUE
    theta <- 3 * (fx - fp) / (stp - stx) + dx + dp
    s <- norm(rbind(theta, dx, dp), "i")
    # The case gamma = 0 only arises if the cubic does not tend
    # to infinity in the direction of the step.
    gamma <- s * sqrt(max(0.,(theta / s) ^ 2 - (dx / s) * (dp / s)))
    if (stp > stx) {
      gamma <- -gamma
    }
    p <- (gamma - dp) + theta
    q <- (gamma + (dx - dp)) + gamma
    r <- p / q

    if (r < 0.0 && gamma != 0.0) {
      stpc <- stp + r * (stx - stp)
    } else if (stp > stx) {
      stpc <- stpmax
    } else {
      stpc <- stpmin
    }

    stpq <- quadratic_interpolateg(stp, dp, stx, dx)

    if (brackt) {
      if (abs(stp - stpc) < abs(stp - stpq)) {
        stpf <- stpc
      } else {
        stpf <- stpq
      }
    } else {
      if (abs(stp - stpc) > abs(stp - stpq)) {
        stpf <- stpc
      } else {
        stpf <- stpq
      }
    }
    # Fourth case. A lower function value, derivatives of the
    # same sign, and the magnitude of the derivative does
    # not decrease. If the minimum is not bracketed, the step
    # is either stpmin or stpmax, else the cubic step is taken.
  } else {
    info <- 4
    bound <- FALSE
    if (brackt) {
      stpc <- cubic_interpolate(sty, fy, dy, stp, fp, dp, ignoreWarnings = TRUE)
      if (is.nan(stpc)) {
        stpc <- (sty + stp) / 2
      }
      stpf <- stpc
    } else if (stp > stx) {
      stpf <- stpmax
    } else {
      stpf <- stpmin
    }
  }

  # Update the interval of uncertainty. This update does not
  # depend on the new step or the case analysis above.
  if (fp > fx) {
    sty <- stp
    fy <- fp
    dy <- dp
    dfy <- dfp
  } else {
    if (sgnd < 0.0) {
      sty <- stx
      fy <- fx
      dy <- dx
      dfy <- dfx
    }
    stx <- stp
    fx <- fp
    dx <- dp
    dfx <- dfp
  }

  # Compute the new step and safeguard it.
  stpf <- min(stpmax, stpf)
  stpf <- max(stpmin, stpf)
  stp <- stpf
  if (brackt && bound) {
    # if the new step is too close to an end point
    # replace with a (weighted) bisection (delta = 0.66 in the paper)
    if (verbose) {
      message("Step too close to end point, weighted bisection")
    }
    stb <- stx + delta * (sty - stx)
    if (sty > stx) {
      stp <- min(stb, stp)
    } else {
      stp <- max(stb, stp)
    }
  }
  list(
    stepx = list(alpha = stx, f = fx, d = dx, df = dfx),
    stepy = list(alpha = sty, f = fy, d = dy, df = dfy),
    step = list(alpha = stp, f = fp, d = dp, df = dfp),
    brackt = brackt, info = info)
}
# Nesterov Accelerated Gradient ------------------------------------------------

# This is the actual Nesterov Accelerated Gradient scheme, rather than
# the version discussed by Sutskever and popular in the deep learning community,
# although that is also available in Mizer.
#
# NAG can be considered to be an optimization consisting of:
# 1. A steepest descent step;
# 2. A pseudo-momentum step, using a specific schedule, depending on how
#    convex the function being optimized is.
# The pseudo-momentum step is:
# mu * [v + (v_grad - v_grad_old)]
# where v is the update vector, and v_grad and v_grad_old are the
# gradient components of the current and previous update, respectively.
# Overall, it replaces the gradient component of the previous velocity with the
# gradient velocity of the current iteration.
nesterov_momentum_direction <- function() {
  make_direction(list(
    name = "nesterov",
    init = function(opt, stage, sub_stage, par, fg, iter) {
      sub_stage$value <- rep(0, length(par))
      sub_stage$update <- rep(0, length(par))
      list(sub_stage = sub_stage)
    },
    calculate = function(opt, stage, sub_stage, par, fg, iter) {
      grad_update <- opt$stages[["gradient_descent"]]$result
      sub_stage$value <- grad_update + sub_stage$update
      list(sub_stage = sub_stage)
    },
    after_step = function(opt, stage, sub_stage, par, fg, iter, par0,
                          update) {
      sub_stage$update <- update - opt$stages[["gradient_descent"]]$result
      list(sub_stage = sub_stage)
    }
  ))
}

# Encapsulates all ways to create a Nesterov schedule
# burn_in Lags the calculation by this number of iterations. By setting this
#   to 2, you get the same "pattern" of results as if you were using the
#   Sutskever Nesterov Momentum approach (i.e. applying a classical momentum
#   step before a steepest descent step)
# q is inversely proportional to how strongly convex the function is
#   0 gives the highest momentum, 1 gives zero momentum. Often, q is assumed to
#   be zero. Ignored if use_approx is TRUE.
# use_approx Use the approximation to the momentum schedule given by
#   Sutskever and co-workers.
# use_init_mu If TRUE, then the momentum calculated on the first iteration uses
#   the calculated non-zero value, otherwise use zero. Because velocity is
#   normally zero initially, this rarely has an effect, unless linear weighting
#   of the momentum is being used. Ignored if use_approx is FALSE.
nesterov_step <- function(burn_in = 0, q = 0, use_approx = FALSE,
                          use_init_mu = FALSE) {
  if (!is_in_range(q, 0, 1)) {
    stop("q must be between 0 and 1")
  }
  if (burn_in < 0) {
    stop("burn_in must be non-negative")
  }

  if (use_approx) {
    nesterov_convex_approx_step(burn_in = burn_in,
                                use_init_mu = use_init_mu)
  }
  else {
    nesterov_convex_step(q = q, burn_in = burn_in)
  }
}

# Approximate Nesterov Convex Momentum Function Factory
#
# Instead of using the exact momentum schedule specified for NAG, use the
# approximation given by Sutskever. NB: this produces much larger momentums
# than the exact result for the first few iterations.
#
# burn_in Lags the calculation by this number of iterations. By setting this
#  to 2, you get the same "pattern" of results as if you were using the
#  Sutskever Nesterov Momentum approach (i.e. applying a classical momentum
#  step before a steepest descent step).
# use_init_mu If TRUE, then return a non-zero momentum on the first iteration.
#  Otherwise use zero. Although a velocity of zero normally enforces
#  steepest descent on the first iteration, for some methods (e.g.
#  NAG or linearly weighted classical momentum), this can have an effect.
#  Set this to TRUE to always get steepest decent.
make_nesterov_convex_approx <- function(burn_in = 0, use_init_mu = FALSE) {
  function(iter, max_iter) {
    # if we haven't waited long enough or we always use zero on the first
    # iteration, return 0
    if (iter < burn_in || (iter == burn_in && !use_init_mu)) {
      return(0)
    }

    1 - (3 / ((iter - burn_in) + 5))
  }
}

# Create a momentum step size sub stage using
# Sutskever's approximation to the NAG pseudo-momentum schedule.
# burn_in Lags the calculation by this number of iterations. By setting this
#  to 2, you get the same "pattern" of results as if you were using the
#  Sutskever Nesterov Momentum approach (i.e. applying a classical momentum
#  step before a steepest descent step).
# use_init_mu if TRUE, then on the first iteration, the momentum
#  uses the equation, which produces a momentum of 0.4. Otherwise, use a
#  momentum coefficient of zero. From reading various papers, a coefficient
#  of zero is probably the intended behavior.
#  Normally, the velocity vector is also zero on the first iteration, so this
#  makes no difference, but if you have linearly weighted the momentum,
#  you will get only 60% of the gradient step you might have been expecting
#  on the first step, and you will get a 60% longer step size if using NAG.
nesterov_convex_approx_step <- function(burn_in = 0, use_init_mu = FALSE) {
  make_momentum_step(mu_fn =
                       make_nesterov_convex_approx(burn_in = burn_in,
                                                   use_init_mu = use_init_mu),
                     min_momentum = 0,
                     max_momentum = 1,
                     use_init_mom = use_init_mu)
}

# The NAG pseudo-momentum schedule.
# q is inversely proportional to how strongly convex the function is
# 0 gives the highest momentum, 1 gives zero momentum. Often, q is assumed to be
# zero.
# burn_in Lags the calculation by this number of iterations. By setting this
#  to 2, you get the same "pattern" of results as if you were using the
#  Sutskever Nesterov Momentum approach (i.e. applying a classical momentum
#  step before a steepest descent step)
nesterov_convex_step <- function(burn_in = 0, q = 0) {
  if (q == 0) {
    # Use the expression for momentum from the Sutskever paper appendix
    nesterov_strong_convex_step(burn_in = burn_in)
  }
  else {
    # Use the expression for momentum from Candes paper which includes q term
    nesterov_convex_step_q(q = q, burn_in = burn_in)
  }
}

# The NAG pseudo-momentum schedule for strongly convex functions.
# This expression is missing the parameter "q" that measures how strongly convex
# the function is. It is implicitly zero, which gives the largest momentum
# values. This uses an expression in the appendix of the Sutskever paper, which
# is a bit simpler to calculate than the version given by O'Donoghue and Candes.
nesterov_strong_convex_step <- function(burn_in) {
  make_step_size(list(
    burn_in = burn_in,
    name = "nesterov_convex",
    init = function(opt, stage, sub_stage, par, fg, iter) {
      sub_stage$a_old <- 1
      list(sub_stage = sub_stage)
    },
    calculate = function(opt, stage, sub_stage, par, fg, iter) {
      if (iter < burn_in) {
        sub_stage$value <- 0
        sub_stage$a <- 1
      }
      else {
        a_old <- sub_stage$a_old
        a <- (1 + sqrt(4 * a_old * a_old + 1)) / 2
        sub_stage$value <- (a_old - 1) / a
        sub_stage$a <- a
      }
      list(sub_stage = sub_stage)
    },
    after_step = function(opt, stage, sub_stage, par, fg, iter, par0,
                          update) {
      if (!opt$ok) {
        sub_stage$a_old <- 1
      }
      else {
        sub_stage$a_old <- sub_stage$a
      }
      list(sub_stage = sub_stage)
    }
  ))
}

# The NAG pseudo-momentum schedule. This expression includes the parameter "q"
# that measures how strongly convex This uses the algorithm given by
# O'Donoghue and Candes, which is a bit more complex than the one in the
# appendix of the Sutskever paper (but that one assumes q = 0).
# See https://arxiv.org/abs/1204.3982 for more.
nesterov_convex_step_q <- function(q, burn_in = 0) {
  make_step_size(list(
    burn_in = burn_in,
    name = "nesterov_convex",
    init = function(opt, stage, sub_stage, par, fg, iter) {
      sub_stage$theta_old <- 1
      list(sub_stage = sub_stage)
    },
    calculate = function(opt, stage, sub_stage, par, fg, iter) {
      if (iter < burn_in) {
        sub_stage$value <- 0
        sub_stage$theta_old <- 1
      }
      else {
        theta_old <- sub_stage$theta_old
        thetas <- solve_theta(theta_old, q)
        theta <- max(thetas)
        # Step 4 of algorithm 1 in https://arxiv.org/abs/1204.3982
        # Calculates beta, effectively the momentum
        # q (taking a value between 0 and 1) is related to the strong convexity
        # parameter (which has the symbol mu in the paper, it's not the momentum!).
        # A q of 1 causes momentum to be zero. A q of 0 gives the same results as
        # the Sutskever momentum (not the approximation he gives, but the actual
        # expression given in the appendix/thesis).
        sub_stage$value <- theta_old * (1 - theta_old) / (theta_old * theta_old + theta)
        sub_stage$theta_old <- theta
        #message("Nesterov momentum = ", formatC(sub_stage$value))
      }
      list(sub_stage = sub_stage)
    },
    after_step = function(opt, stage, sub_stage, par, fg, iter, par0,
                          update) {
      #message("nesterov_convex: after step")
      if (!opt$ok) {
        sub_stage$theta_old <- 1
      }
      list(sub_stage = sub_stage)
    }
  ))
}

# solves quadratic equation. Returns either two roots (even if coincident)
# or NULL if there's no solution
solve_quad <- function(a, b, c) {
  disc <- b * b - 4 * a * c
  res <- c()
  if (disc > 0) {
    root_pos = (-b + sqrt(disc)) / (2 * a)
    root_neg = (-b - sqrt(disc)) / (2 * a)
    res <- c(root_pos, root_neg)
  }
  res
}

# Step 3 of algorithm 1 in https://arxiv.org/abs/1204.3982
solve_theta <- function(theta_old, q = 0) {
  theta2 <- theta_old * theta_old
  solve_quad(1, theta2 - q, -theta2)
}
# Optimizer ---------------------------------------------------------------

# Repeatedly minimizes par using opt until one of the termination conditions
# is met
opt_loop <- function(opt, par, fg, max_iter = 10, verbose = FALSE,
                     store_progress = FALSE, invalidate_cache = FALSE,
                     max_fn = Inf, max_gr = Inf, max_fg = Inf,
                     abs_tol = sqrt(.Machine$double.eps),
                     rel_tol = abs_tol, grad_tol = NULL, ginf_tol = NULL,
                     step_tol = .Machine$double.eps,
                     check_conv_every = 1, log_every = check_conv_every,
                     ret_opt = FALSE) {

  # log_every must be an integer multiple of check_conv_every
  if (!is.null(check_conv_every) && log_every %% check_conv_every != 0) {
    log_every <- check_conv_every
  }

  if (!opt$is_initialized) {
    opt <- mize_init(opt, par, fg, max_iter = max_iter,
                     max_fn = max_fn, max_gr = max_gr, max_fg = max_fg,
                     abs_tol = abs_tol, rel_tol = rel_tol,
                     grad_tol = grad_tol, ginf_tol = ginf_tol,
                     step_tol = step_tol)
  }

  progress <- data.frame()
  step_info <- NULL

  if (verbose || store_progress) {
    step_info <- mize_step_summary(opt, par, fg)
    opt <- step_info$opt
    if (store_progress) {
      progress <- update_progress(step_info, progress)
    }
    if (verbose) {
      opt_report(step_info, print_time = TRUE, print_par = FALSE)
    }
  }

  best_crit <- NULL
  best_fn <- Inf
  best_grn <- Inf
  best_par <- NULL
  if (!is.null(opt$cache$fn_curr)) {
    best_crit <- "fn"
    best_fn <- opt$cache$fn_curr
    best_par <- par
  }
  else if (!is.null(opt$cache$gr_curr)) {
    best_crit <- "gr"
    best_grn <- norm_inf(opt$cache$gr_curr)
    best_par <- par
  }

  iter <- 0
  par0 <- par
  if (max_iter > 0) {
    for (iter in 1:max_iter) {

      if (invalidate_cache) {
        opt <- opt_clear_cache(opt)
      }

      par0 <- par

      # We're going to use this below to guess whether our optimization
      # requires function evaluations (this is only useful if max_fn or max_fg
      # is specified, but not really time consuming)
      if (iter == 1) {
        fn_count_before <- opt$counts$fn
      }
      step_res <- mize_step(opt, par, fg)
      opt <- step_res$opt
      par <- step_res$par

      if (!is.null(opt$terminate)) {
        break
      }

      # After the first iteration, if we don't have the function available for
      # the current value of par, we probably won't have it at future iterations
      # So if we are limiting the number of function evaluations, we need to keep
      # one spare to evaluate fn after the loop finishes for when we return par
      if (iter == 1) {
        if (!has_fn_curr(opt, iter + 1)) {
          if (fn_count_before != opt$counts$fn) {
            opt$convergence$max_fn <- opt$convergence$max_fn - 1
          }
          opt$convergence$max_fg <- opt$convergence$max_fg - 1
        }
      }

      # Check termination conditions
      if (!is.null(check_conv_every) && iter %% check_conv_every == 0) {
        step_info <- mize_step_summary(opt, par, fg, par0)
        opt <- check_mize_convergence(step_info)

        if (store_progress && iter %% log_every == 0) {
          progress <- update_progress(step_info, progress)
        }
        if (verbose && iter %% log_every == 0) {
          opt_report(step_info, print_time = TRUE, print_par = FALSE)
        }

      }

      # might not have worked out which criterion to use on iteration 0
      if (has_fn_curr(opt, iter + 1)) {
        if (is.null(best_crit)) {
          best_crit <- "fn"
        }
        if (best_crit == "fn" && opt$cache$fn_curr < best_fn) {
          best_fn <- opt$cache$fn_curr
          best_par <- par
        }
      }
      else if (has_gr_curr(opt, iter + 1)) {
        if (is.null(best_crit)) {
          best_crit <- "gr"
        }
        if (best_crit == "gr" && norm_inf(opt$cache$gr_curr) < best_grn) {
          best_grn <- norm_inf(opt$cache$gr_curr)
          best_par <- par
        }
      }

      if (!is.null(opt$terminate)) {
        break
      }
    }
  }

  # If we were keeping track of the best result and that's not currently par:
  if (!is.null(best_par)
      && ((best_crit == "fn" && best_fn != opt$cache$fn_curr) ||
          (best_crit == "gr" && best_grn != norm_inf(opt$cache$gr_curr)))) {
    par <- best_par
    opt <- opt_clear_cache(opt)
    opt <- set_fn_curr(opt, best_fn, iter + 1)
    # recalculate result for this iteration
    step_info <- mize_step_summary(opt, par, fg, par0)
    if (verbose) {
      message("Returning best result found")
    }
  }

  if (is.null(step_info) || step_info$iter != iter || is.null(step_info$f)) {
    # Always calculate function value before return
    step_info <- mize_step_summary(opt, par, fg, par0, calc_fn = TRUE)
    opt <- step_info$opt
  }
  if (verbose && iter %% log_every != 0) {
    opt_report(step_info, print_time = TRUE, print_par = FALSE)
  }
  if (store_progress && iter %% log_every != 0) {
    progress <- update_progress(step_info, progress)
  }

  if (store_progress) {
    step_info$progress <- progress
  }
  if (!ret_opt) {
    step_info["opt"] <- NULL
  }

  if (is.null(opt$terminate)) {
    opt$terminate <- list(what = "max_iter", val = opt$convergence$max_iter)
  }
  step_info$terminate <- opt$terminate
  step_info$par <- par
  Filter(Negate(is.null), step_info)
}

# Clears the cache. Results should be identical whether a cache is used or not.
opt_clear_cache <- function(opt) {
  for (name in names(opt$cache)) {
    iter_name <- paste0(name, "_iter")
    if (!is.null(opt$cache[[iter_name]])) {
      opt$cache[[iter_name]] <- "invalid"
    }
  }
  opt
}

# Prints information about the current optimization result
opt_report <- function(step_info, print_time = FALSE, print_par = FALSE,
                       par = NULL) {

  fmsg <- ""
  if (!is.null(step_info$f)) {
    fmsg <- paste0(fmsg, " f = ", formatC(step_info$f))
  }
  if (!is.null(step_info$g2n)) {
    fmsg <- paste0(fmsg, " g2 = ", formatC(step_info$g2n))
  }
  if (!is.null(step_info$ginfn)) {
    fmsg <- paste0(fmsg, " ginf = ", formatC(step_info$ginfn))
  }

  msg <- paste0("iter ", step_info$iter
                , fmsg
                , " nf = ", step_info$nf
                , " ng = ", step_info$ng
                , " step = ", formatC(step_info$step)
  )

  if (print_time) {
    msg <- paste(format(Sys.time(), "%H:%M:%S"), msg, collapse = " ")
  }

  if (print_par) {
    msg <- paste0(msg, " par = ", vec_formatC(par))
  }

  message(msg)
}

# Transfers data from the result object to the progress data frame
update_progress <- function(step_info, progress) {
  res_names <- c("f", "g2n", "ginf", "nf", "ng", "step", "alpha", "mu")
  res_names <- Filter(function(x) { !is.null(step_info[[x]]) }, res_names)

  progress <- rbind(progress, step_info[res_names])

  # Probably not a major performance issue to regenerate column names each time
  colnames(progress) <- res_names
  rownames(progress)[nrow(progress)] <- step_info$iter
  progress
}

# Constructor -------------------------------------------------------------

# Creates an optimizer
make_opt <- function(stages,
                     verbose = FALSE) {
  opt <- list(
    init = function(opt, par, fg, iter) {
      opt <- default_handler("opt", "init", opt, par, fg, iter)
      for (i in 1:length(opt$stages)) {
        opt$stage_i <- i
        opt <- life_cycle_hook(opt$stages[[i]]$type, "init", opt, par, fg, iter)
      }
      opt
    },
    cache = list(),
    stages = stages,
    counts = make_counts(),
    hooks = list(),
    handlers = list(),
    eager_update = FALSE,
    is_terminated = FALSE,
    is_initialized = FALSE,
    verbose = verbose
  )

  if (!is.null(opt$init)) {
    attr(opt$init, 'event') <- 'init opt'
    attr(opt$init, 'name') <- 'handler'
  }
  opt
}

# Creates a stage of the optimizer: a gradient_descent or momentum stage
# normally
make_stage <- function(type, direction, step_size, depends = NULL) {

  stage <- list(
    type = type,
    direction = direction,
    step_size = step_size,
    init = function(opt, stage, par, fg, iter) {
      for (sub_stage_name in c("direction", "step_size")) {
        phase <- paste0(stage$type, " ", sub_stage_name)
        opt <- life_cycle_hook(phase, "init", opt, par, fg, iter)
      }

      list(opt = opt)
    },
    calculate = function(opt, stage, par, fg, iter) {
      for (sub_stage_name in c("direction", "step_size")) {
        phase <- paste0(stage$type, " ", sub_stage_name)
        opt <- life_cycle_hook(phase, "during", opt, par, fg, iter)
      }

      list(opt = opt)
    },
    after_stage = function(opt, stage, par, fg, iter) {
      for (sub_stage_name in c("direction", "step_size")) {
        phase <- paste0(stage$type, " ", sub_stage_name)
        opt <- life_cycle_hook(phase, "after", opt, par, fg, iter)
      }
      stage$result <- stage$direction$value * stage$step_size$value
      list(stage = stage)
    },
    counts = make_counts()
  )

  if (!is.null(depends)) {
    stage$depends <- depends
  }

  if (!is.null(stage$init)) {
    attr(stage$init, 'event') <- paste0('init ', type)
    attr(stage$init, 'name') <- 'handler'
  }
  if (!is.null(stage$calculate)) {
    attr(stage$calculate, 'event') <- paste0('during ', type)
    attr(stage$calculate, 'name') <- 'handler'
  }
  if (!is.null(stage$after_stage)) {
    attr(stage$after_stage, 'event') <- paste0('after ', type)
    attr(stage$after_stage, 'name') <- 'handler'
  }
  if (!is.null(stage$after_step)) {
    attr(stage$after_step, 'event') <- 'after step'
    attr(stage$after_step, 'name') <- paste0(type, ' after step')
  }

  res <- list()
  res[[type]] <- stage
  res
}

# Creates a sub stage: a direction or a step size
make_sub_stage <- function(sub_stage, type) {
  sub_stage$type <- type
  if (!is.null(sub_stage$init)) {
    attr(sub_stage$init, 'event') <- paste0('init ', sub_stage$type)
    attr(sub_stage$init, 'name') <- 'handler'
  }
  if (!is.null(sub_stage$calculate)) {
    attr(sub_stage$calculate, 'event') <- paste0('during ', sub_stage$type)
    attr(sub_stage$calculate, 'name') <- 'handler'
  }
  if (!is.null(sub_stage$after_step)) {
    attr(sub_stage$after_step, 'event') <- 'after step'
    attr(sub_stage$after_step, 'name') <-  paste0(sub_stage$type, ' after step')
  }
  sub_stage
}

# Creates a gradient_descent stage
gradient_stage <- function(direction, step_size) {
  make_stage(type = "gradient_descent", direction, step_size,
             depends = c('gradient'))
}

# Creates a momentum stage
momentum_stage <- function(direction = momentum_direction(normalize = FALSE),
                           step_size) {
  make_stage(type = "momentum", direction, step_size)
}

# Creates a momentum "correction" stage. If linear weighting is asked for, then
# mu * the gradient direction is substracted from the result.
momentum_correction_stage <- function(
  direction = momentum_correction_direction(),
  step_size = momentum_correction_step()) {
  make_stage(type = "momentum_correction", direction, step_size)
}

# Creates stages from a passed list. Should contain lists created from calling
# a specific stage function like momentum_stage or gradient_stage
make_stages <- function(...) {
  stages <- list()
  varargs <- list(...)
  for (arg in varargs) {
    for (i in names(arg)) {
      stages[[i]] <- arg[[i]]
    }
  }
  stages
}

# Add a stage to the end of an optimizer stage list
append_stage <- function(opt, stage) {
  opt$stages <- c(opt$stages, stage)
  opt
}

# Add a stage to the beginning of an optimizer stage list
prepend_stage <- function(opt, stage) {
  opt$stages <- c(stage, opt$stages)
  opt
}

# Initialize a list to store the number of times the function and gradient
# is called.
make_counts <- function() {
  list(
    fn = 0,
    gr = 0
  )
}

# Function / Gradient ----------------------------------------------------------------

# Uncached function evaluation for arbitrary values of par
calc_fn <- function(opt, par, fn) {
  opt$fn <- fn(par)
  opt$counts$fn <- opt$counts$fn + 1
  opt
}

# Cached function evaluation for par value after finding a step size
# (possibly re-usable)
calc_fn_new <- function(opt, par, fn, iter) {
  if (is.null(opt$cache$fn_new_iter) || opt$cache$fn_new_iter != iter) {
    opt <- set_fn_new(opt, fn(par), iter)
    opt$counts$fn <- opt$counts$fn + 1
  }
  opt
}

# Store val as fn_new for the specified iteration
set_fn_new <- function(opt, val, iter) {
  opt$cache$fn_new <- val
  opt$cache$fn_new_iter <- iter
  opt
}

# Cached function evaluation for par at starting point
# (possibly re-usable)
calc_fn_curr <- function(opt, par, fn, iter) {
  if (is.null(opt$cache$fn_curr_iter) || opt$cache$fn_curr_iter != iter) {
    opt <- set_fn_curr(opt, fn(par), iter)
    opt$counts$fn <- opt$counts$fn + 1
  }
  opt
}

# Store val as fn_curr for the specified iteration
set_fn_curr <- function(opt, val, iter) {
  opt$cache$fn_curr <- val
  opt$cache$fn_curr_iter <- iter
  opt
}

# Cached gradient evaluation for par value at start of iteration
# (possibly re-usable)
calc_gr_curr <- function(opt, par, gr, iter) {
  if (is.null(opt$cache$gr_curr_iter) || opt$cache$gr_curr_iter != iter) {
    opt <- set_gr_curr(opt, gr(par), iter)
    opt$counts$gr <- opt$counts$gr + 1
  }
  opt
}

# Store val as gr_curr for the specified iteration
set_gr_curr <- function(opt, val, iter) {
  opt$cache$gr_curr <- val
  opt$cache$gr_curr_iter <- iter
  opt
}

# Uncached gradient evaluation for arbitrary values of par
calc_gr <- function(opt, par, gr) {
  opt$gr <- gr(par)
  opt$counts$gr <- opt$counts$gr + 1
  opt
}

# Predicates --------------------------------------------------------------

# Does the optimizer only have one stage (e.g. a gradient-only approach like
# BFGS)
is_single_stage <- function(opt) {
  length(opt$stages) == 1
}

# Is stage the first stage in the optimizers list of stages
is_first_stage <- function(opt, stage) {
  stage$type == opt$stages[[1]]$type
}

# Is stage the last stage in the optimizers list of stages
is_last_stage <- function(opt, stage) {
  stage$type == opt$stages[[length(opt$stages)]]$type
}

# Is the first stage of optimization gradient descent
# (i.e. not nesterov momentum)
grad_is_first_stage <- function(opt) {
  is_first_stage(opt, opt$stages[["gradient_descent"]])
}

# Has fn_new already been calculated for the specified iteration
has_fn_new <- function(opt, iter) {
  (!is.null(opt$cache$fn_new)
   && !is.null(opt$cache$fn_new_iter)
   && opt$cache$fn_new_iter == iter)
}

# Has fn_curr already been calculated for the specified iteration
has_fn_curr <- function(opt, iter) {
  (!is.null(opt$cache$fn_curr)
   && !is.null(opt$cache$fn_curr_iter)
   && opt$cache$fn_curr_iter == iter)
}

# Has gr_curr already been calculated for the specified iteration
has_gr_curr <- function(opt, iter) {
  (!is.null(opt$cache$gr_curr)
   && !is.null(opt$cache$gr_curr_iter)
   && opt$cache$gr_curr_iter == iter)
}
# Interpolation and extrapolation functions.

# Estimate Minimum By Cubic Extrapolation
#
# Carries out cubic extrapolation based on the x, f(x), and f'(x) values
# at two points to find minimum value of x.
#
# @param x1 x value at first point.
# @param f1 f(x) value at first point.
# @param g1 f'(x) value at first point.
# @param x2 x value at second point.
# @param f2 f(x) value at second point.
# @param g2 f'(x) value at second point.
# @param ignoreWarnings If TRUE, don't warn if the extrapolation creates a
#   non-finite value.
# @return Cubic extrapolated estimate of minimum value of x.
cubic_extrapolate <- function(x1, f1, g1, x2, f2, g2, ignoreWarnings = FALSE) {
  A <- 6 * (f1 - f2) + 3 * (g2 + g1) * (x2 - x1)
  B <- 3 * (f2 - f1) - (2 * g1 + g2) * (x2 - x1)
  if (ignoreWarnings) {
    suppressWarnings(
      x1 - g1 * (x2 - x1) ^ 2 / (B + sqrt(B * B - A * g1 * (x2 - x1)))
    )
  }
}

# Estimate Step Size Minimum By Cubic Extrapolation
#
# Estimates step size corresponding to minimum of the line function using
# cubic extrapolation from two line search evaluations with both function
# and directional derivatives calculated.
#
# @param step1 Line search information for first step value.
# @param step2 Line search information for second step value.
# @return Cubic extrapolated estimate of step size which minimizes the line
#   function.
cubic_extrapolate_step <- function(step1, step2) {
  cubic_extrapolate(step1$alpha, step1$f, step1$d, step2$alpha, step2$f,
                    step2$d, ignoreWarnings = TRUE)
}

# Estimate Minimum By Cubic Interpolation
#
# Carries out cubic interpolation based on the x, f(x), and f'(x) values
# at two points to find minimum value of x.
#
# @param x1 x value at first point.
# @param f1 f(x) value at first point.
# @param g1 f'(x) value at first point.
# @param x2 x value at second point.
# @param f2 f(x) value at second point.
# @param g2 f'(x) value at second point.
# @param ignoreWarnings If TRUE, don't warn if the interpolation creates a
#   non-finite value.
# @return Cubic interpolated estimate of minimum value of x.
cubic_interpolate <- function(x1, f1, g1, x2, f2, g2, ignoreWarnings = FALSE) {
  # nwc(x1, f1, g1, x2, f2, g2)
  # A <- 6 * (f1 - f2) / (x2 - x1) + 3 * (g2 + g1)
  # B <- 3 * (f2 - f1) - (2 * g1 + g2) * (x2 - x1)
  # # num. error possible, ok!
  # suppressWarnings(
  #   x1 + (sqrt(B * B - A * g1 * (x2 - x1) ^ 2) -  B) / A
  # )
#  A <- 6 * (f1 - f2) + 3 * (g2 + g1) * (x2 - x1)
#  B <- 3 * (f2 - f1) - (2 * g1 + g2) * (x2 - x1)
#  x1 - g1 * (x2 - x1) ^ 2 / (B + sqrt(B * B - A * g1 * (x2 - x1)))
  d1 <- g1 + g2 - 3 * ((f1 - f2) / (x1 - x2))

  if (ignoreWarnings) {
    suppressWarnings(
      d2 <- sign(x2 - x1) * sqrt(d1 * d1 - g1 * g2)
    )
  }
  else {
    d2 <- sign(x2 - x1) * sqrt(d1 * d1 - g1 * g2)
  }
  x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
}

# Estimate Step Size Minimum By Cubic Interpolation
#
# Estimates step size corresponding to minimum of the line function using
# cubic interpolation from two line search evaluations with both function
# and directional derivatives calculated.
#
# @param step1 Line search information for first step value.
# @param step2 Line search information for second step value.
# @return Cubic interpolated estimate of step size which minimizes the line
#   function.
cubic_interpolate_step <- function(step1, step2) {
  cubic_interpolate(step1$alpha, step1$f, step1$d,
                    step2$alpha, step2$f, step2$d)
}

# Estimate Minimum By Quadratic Interpolation With One Gradient
#
# Carries out quadratic interpolation based on the x and f(x) values at two
# points, and the f'(x) value at the first point, to find minimum value of x.
#
# @param x1 x value at first point.
# @param f1 f(x) value at first point.
# @param g1 f'(x) value at first point.
# @param x2 x value at second point.
# @param f2 f(x) value at second point.
# @return Quadratic interpolated estimate of minimum value of x.
quadratic_interpolate <- function(x1, f1, g1, x2, f2) {
  x1 - (0.5 * g1 * (x2 - x1) ^ 2) / (f2 - f1 - g1 * (x2 - x1))
}


# Estimate Step Size Minimum By Quadratic Interpolation
#
# Estimates step size corresponding to minimum of the line function using
# quadratic interpolation from two line search evaluations. The function must
# have been evaluated at both points, but only the directional derivative at
# the first point is used.
#
# @param step1 Line search information for first step value.
# @param step2 Line search information for second step value.
# @return Quadratic interpolated estimate of step size which minimizes the line
#   function.
quadratic_interpolate_step <- function(step1, step2) {
  quadratic_interpolate(step1$alpha, step1$f, step1$d,
                        step2$alpha, step2$f)
}

# Estimate Minimum By Quadratic Interpolation With Two Gradients
#
# Carries out quadratic interpolation based on the x and f'(x) values at two
# points. Note that this does not use the function values at either of the
# points.
#
# @param x1 x value at first point.
# @param g1 f'(x) value at first point.
# @param x2 x value at second point.
# @param g2 f'(x) value at second point.
# @return Quadratic interpolated estimate of minimum value of x.
quadratic_interpolateg <- function(x1, g1, x2, g2) {
  x2 + (x1 - x2) * g2 / (g2 - g1)
}

# Tweak Extrapolated Point
#
# Prevents the extrapolated point from being too far away from or to close to
# the points used in the extrapolation.
#
# @param xnew 1D position of the new point.
# @param x1 1D position of the first points used in the extrapolation.
# @param x2 1D position of the second point used in the extrapolation.
# @param ext Maximum multiple of \code{x2} that \code{xnew} is allowed to be
#  extrapolated to.
# @param int Given the distance between \code{x1} and \code{x2}, specified what
#  multiple of that distance is the minimum allowed distance for \code{xnew}
#  from \code{x2}.
# @return A value of \code{xnew} that obeys the minimum and maximum distance
#  constraints from \code{x2}.
tweak_extrapolation <- function(xnew, x1, x2, ext, int) {
  # num prob | wrong sign?
  if (!is.double(xnew) || is.nan(xnew) || is.infinite(xnew) || xnew < 0) {
    # extrapolate maximum amount
    xnew <- x2 * ext
  } else if (xnew > (x2 * ext)) {
    # new point beyond extrapolation limit?
    # extrapolate maximum amount
    xnew <- x2 * ext
  } else if (xnew < (x2 + int * (x2 - x1))) {
    # new point too close to previous point?
    xnew <- x2 + int * (x2 - x1)
  }
  xnew
}

# Tweak Interpolated Point
#
# Prevents interpolated point from getting too close to either of the
# points used for the interpolation. If the point is not a number or infinite,
# then it is set to the bisection of the position of the two interpolating
# points before the check for a too-close approach is carried out.
#
# @param xnew Position of the interpolated point.
# @param x1 Position of the first point used for interpolation.
# @param x2 Position of the second point used for interpolation.
# @param int Given the distance between \code{x1} and \code{x2}, specifies what
#  multiple of that distance is the minimum allowed distance for \code{xnew}
#  from \code{x1} or \code{x2}.
# @return Tweaked position of \code{xnew} such that it is not too close to
#  either \code{x1} or \code{x2}.
tweak_interpolation <- function(xnew, x1, x2, int) {
  if (is.nan(xnew) || is.infinite(xnew)) {
    # if we had a numerical problem then bisect
    xnew <- (x1 + x2) / 2
  }
  # don't accept too close
  max(min(xnew, x2 - int * (x2 - x1)), x1 + int * (x2 - x1))
}
# Rasmussen Line Search
#
# Line Search Factory Function
#
# Line search algorithm originally written by Carl Edward Rasmussen in his
# conjugate gradient routine. It consists of two main parts:
# \enumerate{
#  \item Using cubic extrapolation from an initial starting guess for the step
#    size until either the sufficient decrease condition is not met or the
#    curvature condition is met.
#  \item Interpolation (quadratic or cubic) between that point and the start
#    point of the search until either a step size is found which meets the
#    Strong Wolfe conditions or the maximum number of allowed function
#    evaluations is reached.
# }
#
# The extrapolation and interpolation steps are bounded at each stage to ensure
# they don't represent too large or small a change to the step size.
#
# @param c1 Constant used in sufficient decrease condition. Should take a value
#   between 0 and 1.
# @param c2 Constant used in curvature condition. Should take a value between
#   c1 and 1.
# @param ext Extrapolation constant. Prevents step size extrapolation being
#   too large.
# @param int Interpolation constant. Prevents step size being too small.
# @param max_fn Maximum number of function evaluations allowed per line search.
# @return Line search function.
# @seealso Line search based on Matlab code by
#  \href{http://learning.eng.cam.ac.uk/carl/code/minimize/}{Carl Edward Rasmussen}
#  and also part of the Matlab
#  \href{(http://www.gaussianprocess.org/gpml/code/matlab/doc/)}{GPML} package.
rasmussen <- function(c1 = c2 / 2, c2 = 0.1, int = 0.1, ext = 3.0,
                      max_fn = Inf, xtol = 1e-6, eps = 1e-6, approx_armijo = FALSE,
                      strong_curvature = TRUE, verbose = FALSE) {
  if (c2 < c1) {
    stop("rasmussen line search: c2 < c1")
  }

  if (approx_armijo) {
    armijo_check_fn <- make_approx_armijo_ok_step(eps)
  }
  else {
    armijo_check_fn <- armijo_ok_step
  }

  wolfe_ok_step_fn <- make_wolfe_ok_step_fn(strong_curvature = strong_curvature,
                                            approx_armijo = approx_armijo,
                                            eps = eps)

  function(phi, step0, alpha,
           total_max_fn = Inf, total_max_gr = Inf, total_max_fg = Inf,
           pm = NULL) {
    maxfev <- min(max_fn, total_max_fn, total_max_gr, floor(total_max_fg / 2))
    if (maxfev <= 0) {
      return(list(step = step0, nfn = 0, ngr = 0))
    }

    res <- ras_ls(phi, alpha, step0, c1 = c1, c2 = c2, ext = ext, int = int,
                  max_fn = maxfev, armijo_check_fn = armijo_check_fn,
                  wolfe_ok_step_fn = wolfe_ok_step_fn, verbose = verbose)
    list(step = res$step, nfn = res$nfn, ngr = res$nfn)
  }
}

# Rasmussen Line Search
#
# Line Search Method
#
# @param phi Line function.
# @param alpha Initial guess for step size.
# @param step0 Line search values at starting point of line search.
# @param c1 Constant used in sufficient decrease condition. Should take a value
#   between 0 and 1.
# @param c2 Constant used in curvature condition. Should take a value between
#   c1 and 1.
# @param ext Extrapolation constant. Prevents step size extrapolation being
#   too large.
# @param int Interpolation constant. Prevents step size being too small.
# @param max_fn Maximum number of function evaluations allowed.
# @return List containing:
# \itemize{
#   \item step Valid step size or the last step size evaluated.
#   \item nfn Number of function evaluations.
# }
ras_ls <- function(phi, alpha, step0, c1 = 0.1, c2 = 0.1 / 2, ext = 3.0,
                   int = 0.1, max_fn = Inf, xtol = 1e-6,
                   armijo_check_fn = armijo_ok_step,
                   wolfe_ok_step_fn = strong_wolfe_ok_step,
                   verbose = verbose) {
  if (c2 < c1) {
    stop("Rasmussen line search: c2 < c1")
  }
  # extrapolate from initial alpha until either curvature condition is met
  # or the armijo condition is NOT met
  if (verbose) {
    message("Bracketing with initial step size = ", formatC(alpha))
  }
  ex_result <- extrapolate_step_size(phi, alpha, step0, c1, c2, ext, int,
                                     max_fn, armijo_check_fn, verbose = verbose)

  step <- ex_result$step
  nfn <- ex_result$nfn
  max_fn <- max_fn - nfn
  if (max_fn <= 0) {
    return(ex_result)
  }

  if (!ex_result$ok) {
    if (verbose) {
      message("Bracket phase failed, returning best step")
    }
    return(list(step = best_bracket_step(list(step0, step))), nfn = nfn)
  }

  if (verbose) {
    message("Bracket: ", format_bracket(list(step0, step)), " fn = ", nfn)
  }

  # interpolate until the Strong Wolfe conditions are met
  int_result <- interpolate_step_size(phi, step0, step, c1, c2, int, max_fn,
                                      xtol = xtol,
                                      armijo_check_fn = armijo_check_fn,
                                      wolfe_ok_step_fn = wolfe_ok_step_fn,
                                      verbose = verbose)
  if (verbose) {
    message("alpha = ", formatC(int_result$step$alpha))
  }
  int_result$nfn <- int_result$nfn + nfn
  int_result
}

# Increase Step Size
#
# @param phi Line function.
# @param alpha Initial step size.
# @param step0 Line search value at the initial step size.
# @param c1 Constant used in sufficient decrease condition. Should take a value
#   between 0 and 1.
# @param c2 Constant used in curvature condition. Should take a value between
#   c1 and 1.
# @param ext Extrapolation constant. Prevents step size extrapolation being
#   too large.
# @param int Interpolation constant. Prevents step size extrapolation being
#   too small.
# @param max_fn Maximum number of function evaluations allowed.
# @return List containing:
# \itemize{
#   \item step Valid step size or the last step size evaluated.
#   \item nfn Number of function evaluations.
# }
extrapolate_step_size <- function(phi, alpha, step0, c1, c2, ext, int,
                                  max_fn = 20,
                                  armijo_check_fn = armijo_ok_step,
                                  verbose = FALSE) {
  # holds the largest finite-valued step
  finite_step <- step0
  ext_alpha <- alpha
  ok <- FALSE
  nfn <- 0
  while (TRUE) {
    result <- find_finite(phi, ext_alpha, max_fn, min_alpha = 0)
    nfn <- nfn + result$nfn
    max_fn <- max_fn - result$nfn
    if (!result$ok) {
      if (verbose) {
        message("Couldn't find a finite alpha during extrapolation")
      }
      break
    }

    finite_step <- result$step

    if (extrapolation_ok(step0, finite_step, c1, c2, armijo_check_fn)) {
      ok <- TRUE
      break
    }
    if (max_fn <= 0) {
      break
    }

    ext_alpha <- tweaked_extrapolation(step0, finite_step, ext, int)
  }

  list(step = finite_step, nfn = nfn, ok = ok)
}

# Extrapolation Check
#
# Checks that an extrapolated step size is sufficiently large: either by
# passing the curvature condition or failing the sufficient decrease condition.
#
# @param step0 Line search values at starting point of line search.
# @param step Line search value at candiate step size.
# @param c1 Constant used in sufficient decrease condition. Should take a value
#   between 0 and 1.
# @param c2 Constant used in curvature condition. Should take a value between
#   c1 and 1.
# @return TRUE if the extrapolated point is sufficiently large.
extrapolation_ok <- function(step0, step, c1, c2, armijo_check_fn) {
  curvature_ok_step(step0, step, c2) || !armijo_check_fn(step0, step, c1)
}

# Extrapolate and Tweak Step Size
#
# Carries out an extrapolation of the step size, tweaked to not be too small
# or large.
#
# @param step0 Line search values at starting point of line search.
# @param step Line search value at candiate step size.
# @param c1 Constant used in sufficient decrease condition. Should take a value
#   between 0 and 1.
# @param c2 Constant used in curvature condition. Should take a value between
#   c1 and 1.
# @return extrapolated step size.
tweaked_extrapolation <- function(step0, step, ext, int) {
  ext_alpha <- cubic_extrapolate_step(step0, step)
  tweak_extrapolation(ext_alpha, step0$alpha, step$alpha, ext, int)
}

# Interpolate Step Size to Meet Strong Wolfe Condition.
#
# @param phi Line function.
# @param step0 Line search values at starting point of line search.
# @param step Line search value at candiate step size.
# @param c1 Constant used in sufficient decrease condition. Should take a value
#   between 0 and 1.
# @param c2 Constant used in curvature condition. Should take a value between
#   c1 and 1.
# @param int Interpolation constant. Prevents step size being too small.
# @param max_fn Maximum number of function evaluations allowed.
# @return List containing:
# \itemize{
#   \item step Valid step size or the last step size evaluated.
#   \item nfn Number of function evaluations.
# }
interpolate_step_size <- function(phi, step0, step, c1, c2, int, max_fn = 20,
                                  xtol = 1e-6,
                                  armijo_check_fn = armijo_ok_step,
                                  wolfe_ok_step_fn = strong_wolfe_ok_step,
                                  verbose = FALSE) {
  step2 <- step0
  step3 <- step
  nfn <- 0
  if (verbose) {
    message("Interpolating")
  }

  while (!wolfe_ok_step_fn(step0, step3, c1, c2) && nfn < max_fn) {
    if (step3$d > 0 || !armijo_check_fn(step0, step3, c1)) {
      step4 <- step3
    } else {
      step2 <- step3
    }

    if (step4$f > step0$f) {
      step3$alpha <- quadratic_interpolate_step(step2, step4)
    } else {
      step3$alpha <- cubic_interpolate_step(step2, step4)
    }

    if (verbose) {
      message("Bracket: ", format_bracket(list(step2, step4)),
              " alpha: ", formatC(step3$alpha), " f: ", formatC(step3$f),
              " d: ", formatC(step3$d), " nfn: ", nfn, " max_fn: ", max_fn)
    }
    step3$alpha <- tweak_interpolation(step3$alpha, step2$alpha, step4$alpha,
                                       int)
    # Check interpolated step is finite, and bisect if not, as in extrapolation
    # stage
    result <- find_finite(phi, step3$alpha, max_fn - nfn,
                          min_alpha = bracket_min_alpha(list(step2, step4)))
    nfn <- nfn + result$nfn
    if (!result$ok) {
      if (verbose) {
        message("Couldn't find a finite alpha during interpolation, aborting")
      }
      step3 <- best_bracket_step(list(step2, step4))
      break
    }
    step3 <- result$step


    if (bracket_width(list(step2, step4)) < xtol * step3$alpha) {
      if (verbose) {
        message("Bracket width: ", formatC(bracket_width(list(step2, step4))),
                " reduced below tolerance ", formatC(xtol * step3$alpha))
      }
      break
    }

  }
  list(step = step3, nfn = nfn)
}
# Translation of Mark Schmidt's minFunc line search code for satisfying the
# Strong Wolfe conditions (and also the Armijo conditions)
# http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html, 2005.

# Adapters ----------------------------------

# Uses the default line search settings: cubic interpolation/extrapolation
# Falling back to Armijo backtracking (also using cubic interpolation) if
# a non-legal value is found
schmidt <- function(c1 = c2 / 2, c2 = 0.1, max_fn = Inf, eps = 1e-6,
                    strong_curvature = TRUE, approx_armijo = FALSE) {
  if (c2 < c1) {
    stop("schmidt line search: c2 < c1")
  }
  function(phi, step0, alpha,
           total_max_fn = Inf, total_max_gr = Inf, total_max_fg = Inf,
           pm) {
    maxfev <- min(max_fn, total_max_fn, total_max_gr, floor(total_max_fg / 2))
    if (maxfev <= 0) {
      return(list(step = step0, nfn = 0, ngr = 0))
    }

    if (approx_armijo) {
      armijo_check_fn <- make_approx_armijo_ok_step(eps)
    }
    else {
      armijo_check_fn <- armijo_ok_step
    }

    if (strong_curvature) {
      curvature_check_fn <- strong_curvature_ok_step
    }
    else {
      curvature_check_fn <- curvature_ok_step
    }

    res <- WolfeLineSearch(alpha = alpha, f = step0$f, g = step0$df,
                           gtd = step0$d,
                           c1 = c1, c2 = c2, LS_interp = 2, LS_multi = 0,
                           maxLS = maxfev,
                           funObj = phi, varargin = NULL,
                           pnorm_inf = max(abs(pm)),
                           progTol = 1e-9,
                           armijo_check_fn = armijo_check_fn,
                           curvature_check_fn = curvature_check_fn,
                           debug = FALSE)
    res$ngr = res$nfn
    res
  }
}

# step_down if non-NULL, multiply the step size by this value when backtracking
# Otherwise, use a cubic interpolation based on previous function and derivative
# values
schmidt_armijo_backtrack <- function(c1 = 0.05, step_down = NULL, max_fn = Inf) {
  function(phi, step0, alpha,
           total_max_fn = Inf, total_max_gr = Inf, total_max_fg = Inf,
           pm) {
    maxfev <- min(max_fn, total_max_fn, total_max_gr, floor(total_max_fg / 2))
    if (maxfev <= 0) {
      return(list(step = step0, nfn = 0, ngr = 0))
    }
    if (!is.null(step_down)) {
      # fixed-size step reduction by a factor of step_down
      LS_interp <- 0
    }
    else {
      # cubic interpolation
      LS_interp <- 2
    }

    res <- ArmijoBacktrack(step = alpha, f = step0$f, g = step0$df,
                           gtd = step0$d,
                           c1 = c1, LS_interp = LS_interp, LS_multi = 0,
                           maxLS = maxfev, step_down = step_down,
                           funObj = phi, varargin = NULL,
                           pnorm_inf = max(abs(pm)),
                           progTol = 1e-9, debug = FALSE)

    res$ngr <- res$nfn
    res
  }
}

# Translated minFunc routines ---------------------------------------------

# Bracketing Line Search to Satisfy Wolfe Conditions
#
# Inputs:
#   x: starting location
#   step: initial step size
#   d: descent direction
#   f: function value at starting location
#   g: gradient at starting location
#   gtd: directional derivative at starting location
#   c1: sufficient decrease parameter
#   c2: curvature parameter
#   debug: display debugging information
#   LS_interp: type of interpolation
#   maxLS: maximum number of funEvals (changed from matlab original)
#   progTol: minimum allowable step length
#   funObj: objective function
#   varargin: parameters of objective function
#
# For the Wolfe line-search, these interpolation strategies are available ('LS_interp'):
#   - 0 : Step Size Doubling and Bisection
#   - 1 : Cubic interpolation/extrapolation using new function and gradient values (default)
#   - 2 : Mixed quadratic/cubic interpolation/extrapolation
# Outputs:
#   step: step length
#   f_new: function value at x+step*d
#   g_new: gradient value at x+step*d
#   funEvals: number function evaluations performed by line search
#   H: Hessian at initial guess (only computed if requested
# @returns [step,f_new,g_new,funEvals,H]
WolfeLineSearch <-
  function(alpha, f, g, gtd,
           c1 = 1e-4, c2 = 0.1, LS_interp = 1, LS_multi = 0, maxLS = 25,
           funObj, varargin = NULL,
           pnorm_inf, progTol = 1e-9, armijo_check_fn = armijo_ok_step,
           curvature_check_fn = strong_curvature_ok_step,
           debug = FALSE) {
    # Bracket an Interval containing a point satisfying the
    # Wolfe criteria
    step0 <- list(alpha = 0, f = f, df = g, d = gtd)

    bracket_res <- schmidt_bracket(alpha, LS_interp, maxLS, funObj, step0,
                                   c1, c2, armijo_check_fn, curvature_check_fn,
                                   debug)
    bracket_step <- bracket_res$bracket
    funEvals <- bracket_res$funEvals
    done <- bracket_res$done

    if (!bracket_is_finite(bracket_step)) {
      if (debug) {
        message('Switching to Armijo line-search')
      }
      alpha <- mean(bracket_props(bracket_step, 'alpha'))

      # Do Armijo
      armijo_res <- ArmijoBacktrack(alpha, step0$f, step0$df, step0$d,
                                    c1 = c1, LS_interp = LS_interp,
                                    LS_multi = LS_multi,
                                    maxLS = maxLS - funEvals,
                                    funObj = funObj, varargin = NULL,
                                    pnorm_inf = pnorm_inf,
                                    progTol = progTol, debug = debug)

      armijo_res$nfn <- armijo_res$nfn + funEvals
      return(armijo_res)
    }

    ## Zoom Phase
    # We now either have a point satisfying the criteria, or a bracket
    # surrounding a point satisfying the criteria
    # Refine the bracket until we find a point satisfying the criteria
    if (!done) {
      maxLS <- maxLS - funEvals
      zoom_res <- schmidt_zoom(bracket_step, LS_interp, maxLS, funObj,
                               step0, c1, c2, pnorm_inf, progTol,
                               armijo_check_fn, curvature_check_fn,
                               debug)

      funEvals <- funEvals + zoom_res$funEvals
      bracket_step <- zoom_res$bracket
    }

    list(step = best_bracket_step(bracket_step), nfn = funEvals)
  }

# Change from original: maxLS refers to maximum allowed funEvals, not LS iters
schmidt_bracket <- function(alpha, LS_interp, maxLS, funObj, step0, c1, c2,
                            armijo_check_fn, curvature_check_fn, debug) {
  # did we find a bracket
  ok <- FALSE
  # did we find a step that already fulfils the line search
  done <- FALSE

  step_prev <- step0

  # Evaluate the Objective and Gradient at the Initial Step
  ff_res <- find_finite(funObj, alpha, maxLS, min_alpha = 0)
  funEvals <- ff_res$nfn
  step_new <- ff_res$step

  LSiter <- 0
  while (funEvals < maxLS) {

    if (!ff_res$ok) {
      if (debug) {
        message('Extrapolated into illegal region, returning')
      }
      bracket_step <- list(step_prev, step_new)

      return(list(bracket = bracket_step, done = done,
                  funEvals = funEvals, ok = FALSE))
    }

    # See if we have found the other side of the bracket
    if (!armijo_check_fn(step0, step_new, c1) ||
        (LSiter > 1 && step_new$f >= step_prev$f)) {
      bracket_step <- list(step_prev, step_new)
      ok <- TRUE
      if (debug) {
        message('Armijo failed or step_new$f >= step_prev$f: bracket is [prev new]')
      }
      break
    }
    else if (curvature_check_fn(step0, step_new, c2)) {
      bracket_step <- list(step_new)
      ok <- TRUE
      done <- TRUE

      if (debug) {
        message('Sufficient curvature found: bracket is [new]')
      }
      break
    }
    else if (step_new$d >= 0) {
      bracket_step <- list(step_prev, step_new)

      if (debug) {
        message('step_new$d >= 0: bracket is [prev, new]')
      }
      break
    }

    minStep <- step_new$alpha + 0.01 * (step_new$alpha - step_prev$alpha)
    maxStep <- step_new$alpha * 10

    if (LS_interp <= 1) {
      if (debug) {
        message('Extending Bracket')
      }
      alpha_new <- maxStep
    }
    else if (LS_interp == 2) {
      if (debug) {
        message('Cubic Extrapolation')
      }
      alpha_new <- polyinterp(point_matrix_step(step_prev, step_new),
                              minStep, maxStep)
    }
    else {
      # LS_interp == 3
      alpha_new <- mixedExtrap_step(step_prev, step_new, minStep, maxStep,
                                    debug)
    }

    step_prev <- step_new

    ff_res <- find_finite(funObj, alpha_new, maxLS - funEvals,
                          min_alpha = step_prev$alpha)
    funEvals <- funEvals + ff_res$nfn
    step_new <- ff_res$step

    LSiter <- LSiter + 1
  }

  # If we ran out of fun_evals, need to repeat finite check for last iteration
  if (!ok && !ff_res$ok) {
    if (debug) {
      message('Extrapolated into illegal region, returning')
    }
  }

  if (funEvals >= maxLS && !ok) {
    if (debug) {
      message("max_fn reached in bracket step")
    }
  }

  list(bracket = bracket_step, done = done, funEvals = funEvals, ok = ok)
}

# Change from original: maxLS refers to max allowed funEvals not LSiters
schmidt_zoom <- function(bracket_step, LS_interp, maxLS, funObj,
                         step0, c1, c2, pnorm_inf, progTol, armijo_check_fn,
                         curvature_check_fn,
                         debug) {
  insufProgress <- FALSE
  Tpos <- 2 # position in the bracket of the current best step
  # mixed interp only: if true, save point from previous bracket
  LOposRemoved <- FALSE

  funEvals <- 0

  done <- FALSE
  while (!done && funEvals < maxLS) {
    # Find High and Low Points in bracket
    LOpos <- which.min(bracket_props(bracket_step, 'f'))
    HIpos <- -LOpos + 3 # 1 or 2, whichever wasn't the LOpos

    # Compute new trial value
    if (LS_interp <= 1 || !bracket_is_finite(bracket_step)) {
      if (!bracket_is_finite(bracket_step)) {
        message("Bad f/g in bracket - bisecting")
      }
      alpha <- mean(bracket_props(bracket_step, 'alpha'))
      if (debug) {
        message('Bisecting: trial step = ', formatC(alpha))
      }
    }
    else if (LS_interp == 2) {
      alpha <- polyinterp(
        point_matrix_step(bracket_step[[1]], bracket_step[[2]]),
        debug = debug)
      if (debug) {
        message('Grad-Cubic Interpolation: trial step = ', formatC(alpha))
      }
    }
    else {
      # Mixed Case #
      nonTpos <- -Tpos + 3
      if (!LOposRemoved) {
        oldLO <- bracket_step[[nonTpos]]
      }
      alpha <- mixedInterp_step(bracket_step, Tpos, oldLO, debug)
      if (debug) {
        message('Mixed Interpolation: trial step = ', formatC(alpha))
      }
    }

    # Ensure that alpha is finite
    if (!is.finite(alpha)) {
      alpha <- mean(bracket_props(bracket_step, 'alpha'))
      if (debug) {
        message("Non-finite trial alpha, bisecting: alpha = ", formatC(alpha))
      }
    }

    # Test that we are making sufficient progress
    bracket_alphas <- bracket_props(bracket_step, 'alpha')
    alpha_max <- max(bracket_alphas)
    alpha_min <- min(bracket_alphas)
    alpha_range <- alpha_max - alpha_min
    if (alpha_range > 0) {
      if (min(alpha_max - alpha, alpha - alpha_min) / alpha_range < 0.1) {
        if (debug) {
          message('Interpolation close to boundary')
        }

        if (insufProgress || alpha >= alpha_max || alpha <= alpha_min) {
          if (debug) {
            message('Evaluating at 0.1 away from boundary')
          }
          if (abs(alpha - alpha_max) < abs(alpha - alpha_min)) {
            alpha <- alpha_max - 0.1 * alpha_range
          }
          else {
            alpha <- alpha_min + 0.1 * alpha_range
          }
          insufProgress <- FALSE
        }
        else {
          insufProgress <- TRUE
        }
      }
      else {
        insufProgress <- FALSE
      }
    }

    # Evaluate new point

    # code attempts to handle non-finite values but this is easier in Matlab
    # where NaN can safely be compared with finite values (returning 0 in all
    # comparisons), whereas R returns NA. Instead, let's attempt to find a
    # finite value by bisecting repeatedly. If we run out of evaluations or
    # hit the bracket, we give up.
    ff_res <- find_finite(funObj, alpha, maxLS - funEvals,
                          min_alpha = bracket_min_alpha(bracket_step))
    funEvals <- funEvals + ff_res$nfn
    if (!ff_res$ok) {
      if (debug) {
        message("Failed to find finite legal step size in zoom phase, aborting")
      }
      break
    }
    step_new <- ff_res$step

    # Update bracket
    if (!armijo_check_fn(step0, step_new, c1) ||
        step_new$f >= bracket_step[[LOpos]]$f) {
      if (debug) {
        message("New point becomes new HI")
      }
      # Armijo condition not satisfied or not lower than lowest point
      bracket_step[[HIpos]] <- step_new
      Tpos <- HIpos
      # [LO, new]
    }
    else {
      if (curvature_check_fn(step0, step_new, c2)) {
        # Wolfe conditions satisfied
        done <- TRUE
        # [new, HI]
      }
      else if (step_new$d * (bracket_step[[HIpos]]$alpha - bracket_step[[LOpos]]$alpha) >= 0) {
        if (debug) {
          message("Old LO becomes new HI")
        }
        # Old HI becomes new LO
        bracket_step[[HIpos]] <- bracket_step[[LOpos]]

        if (LS_interp == 3) {
          if (debug) {
            message('LO Pos is being removed!')
          }
          LOposRemoved <- TRUE
          oldLO <- bracket_step[[LOpos]]
        }
        # [new, LO]
      }
      # else [new, HI]

      if (debug) {
        message("New point becomes new LO")
      }
      # New point becomes new LO
      bracket_step[[LOpos]] <- step_new
      Tpos <- LOpos
    }

    if (!done && bracket_width(bracket_step) * pnorm_inf < progTol) {
      if (debug) {
        message('Line-search bracket has been reduced below progTol')
      }
      break
    }
  } # end of while loop
  if (funEvals >= maxLS) {
    if (debug) {
      message('Line Search Exceeded Maximum Function Evaluations')
    }
  }
  list(bracket = bracket_step, funEvals = funEvals)
}

# Backtracking linesearch to satisfy Armijo condition
#
# Inputs:
#   x: starting location
#   t: initial step size
#   d: descent direction
#   f: function value at starting location
#   gtd: directional derivative at starting location
#   c1: sufficient decrease parameter
#   debug: display debugging information
#   LS_interp: type of interpolation
#   progTol: minimum allowable step length
#   doPlot: do a graphical display of interpolation
#   funObj: objective function
#   varargin: parameters of objective function
#
# For the Armijo line-search, several interpolation strategies are available
# ('LS_interp'):
#   - 0 : Step size halving
#   - 1 : Polynomial interpolation using new function values
#   - 2 : Polynomial interpolation using new function and gradient values (default)
#
# When (LS_interp = 1), the default setting of (LS_multi = 0) uses quadratic
# interpolation, while if (LS_multi = 1) it uses cubic interpolation if more
# than one point are available.
#
# When (LS_interp = 2), the default setting of (LS_multi = 0) uses cubic interpolation,
# while if (LS_multi = 1) it uses quartic or quintic interpolation if more than
# one point are available
#
# Outputs:
#   t: step length
#   f_new: function value at x+t*d
#   g_new: gradient value at x+t*d
#   funEvals: number function evaluations performed by line search
#
# recet change: LS changed to LS_interp and LS_multi

ArmijoBacktrack <-
  function(step, f, g, gtd,
           c1 = 1e-4,
           LS_interp = 2, LS_multi = 0, maxLS = Inf,
           step_down = 0.5,
           funObj,
           varargin = NULL,
           pnorm_inf,
           progTol = 1e-9, debug = FALSE)
  {
    # Evaluate the Objective and Gradient at the Initial Step

    f_prev <- NA
    t_prev <- NA
    g_prev <- NA
    gtd_prev <- NA

    fun_obj_res <- funObj(step)
    f_new <- fun_obj_res$f
    g_new <- fun_obj_res$df
    gtd_new <- fun_obj_res$d

    funEvals <- 1

    while (funEvals < maxLS && (f_new > f + c1 * step * gtd || !is.finite(f_new))) {
      temp <- step
      if (LS_interp == 0 || !is.finite(f_new)) {
        # Ignore value of new point
        if (debug) {
          message('Fixed BT')
        }
        step <- step_down * step
      }
      else if (LS_interp == 1 || !is.finite(g_new)) {
        # Use function value at new point, but not its derivative
        if (funEvals < 2 || LS_multi == 0 || !is.finite(f_prev)) {
          # Backtracking w/ quadratic interpolation based on two points
          if (debug) {
            message('Quad BT')
          }
          step <- polyinterp(point_matrix(c(0, step), c(f, f_new), c(gtd, NA)),
                             0, step)
        }
        else {
          # Backtracking w/ cubic interpolation based on three points
          if (debug) {
            message('Cubic BT')
          }
          step <-
            polyinterp(point_matrix(
              c(0, step, t_prev), c(f, f_new, f_prev), c(gtd, NA, NA)),
              0, step)
        }
      }
      else {
        # Use function value and derivative at new point
        if (funEvals < 2 || LS_multi == 0 || !is.finite(f_prev)) {
          # Backtracking w/ cubic interpolation w/ derivative
          if (debug) {
            message('Grad-Cubic BT')
          }
          step <- polyinterp(
            point_matrix(c(0, step), c(f, f_new), c(gtd, gtd_new)),
            0, step)
        }
        else if (!is.finite(g_prev)) {
          # Backtracking w/ quartic interpolation 3 points and derivative
          # of two
          if (debug) {
            message('Grad-Quartic BT')
          }

          step <- polyinterp(point_matrix(
            c(0, step, t_prev), c(f, f_new, f_prev), c(gtd, gtd_new, NA)),
            0, step)
        }
        else {
          # Backtracking w/ quintic interpolation of 3 points and derivative
          # of three
          if (debug) {
            message('Grad-Quintic BT')
          }

          step <- polyinterp(point_matrix(
            c(0, step, t_prev),
            c(f, f_new, f_prev),
            c(gtd, gtd_new, gtd_prev)),
            0, step)
        }
      }


      if (!is_finite_numeric(step)) {
        step <- temp * 0.6
      }
      # Adjust if change in step is too small/large
      if (step < temp * 1e-3) {
        if (debug) {
          message('Interpolated Value Too Small, Adjusting')
        }
        step <- temp * 1e-3

      } else if (step > temp * 0.6) {
        if (debug) {
          message('Interpolated Value Too Large, Adjusting')
        }
        step <- temp * 0.6
      }

      # Store old point if doing three-point interpolation
      if (LS_multi) {
        f_prev <- f_new
        t_prev <- temp

        if (LS_interp == 2) {
          g_prev <- g_new
          gtd_prev <- gtd_new
        }
      }

      fun_obj_res <- funObj(step)
      f_new <- fun_obj_res$f
      g_new <- fun_obj_res$df
      gtd_new <- fun_obj_res$d

      funEvals <- funEvals + 1

      # Check whether step size has become too small
      if (pnorm_inf * step <= progTol) {
        if (debug) {
          message('Backtracking Line Search Failed')
        }
        step <- 0
        f_new <- f
        g_new <- g
        gtd_new <- gtd
        break
      }
    }

    list(
      step = list(alpha = step, f = f_new, df = g_new, d = gtd_new),
      nfn = funEvals
    )
  }

mixedExtrap_step <- function(step0, step1, minStep, maxStep, debug) {
  mixedExtrap(step0$alpha, step0$f, step0$d, step1$alpha, step1$f, step1$d,
              minStep, maxStep, debug)
}

mixedExtrap <- function(x0, f0, g0, x1, f1, g1, minStep, maxStep, debug) {
  alpha_c <- polyinterp(point_matrix(c(x0, x1), c(f0, f1), c(g0, g1)),
                        minStep, maxStep, debug = debug)
  alpha_s <- polyinterp(point_matrix(c(x0, x1), c(f0, NA), c(g0, g1)),
                        minStep, maxStep, debug = debug)
  if (debug) {
    message("cubic ext = ", formatC(alpha_c), " secant ext = ", formatC(alpha_s),
            " minStep = ", formatC(minStep),
            " alpha_c > minStep ? ", alpha_c > minStep,
            " |ac - x1| = ", formatC(abs(alpha_c - x1)),
            " |as - x1| = ", formatC( abs(alpha_s - x1))
    )
  }
  if (alpha_c > minStep && abs(alpha_c - x1) < abs(alpha_s - x1)) {
    if (debug) {
      message('Cubic Extrapolation ', formatC(alpha_c))
    }
    res <- alpha_c
  }
  else {
    if (debug) {
      message('Secant Extrapolation ', formatC(alpha_s))
    }
    res <- alpha_s
  }
  res
}

mixedInterp_step <- function(bracket_step,
                        Tpos,
                        oldLO,
                        debug) {

  bracket <- c(bracket_step[[1]]$alpha, bracket_step[[2]]$alpha)
  bracketFval <- c(bracket_step[[1]]$f, bracket_step[[2]]$f)
  bracketDval <- c(bracket_step[[1]]$d, bracket_step[[2]]$d)

  mixedInterp(bracket, bracketFval, bracketDval, Tpos,
              oldLO$alpha, oldLO$f, oldLO$d, debug)
}

mixedInterp <- function(
  bracket, bracketFval, bracketDval,
  Tpos,
  oldLOval, oldLOFval, oldLODval,
  debug) {

  # Mixed Case
  nonTpos <- -Tpos + 3


    gtdT <- bracketDval[Tpos]
    gtdNonT <- bracketDval[nonTpos]
    oldLOgtd <- oldLODval
    if (bracketFval[Tpos] > oldLOFval) {
      alpha_c <- polyinterp(point_matrix(
        c(oldLOval, bracket[Tpos]),
        c(oldLOFval, bracketFval[Tpos]),
        c(oldLOgtd, gtdT)))
      alpha_q <- polyinterp(point_matrix(
        c(oldLOval, bracket[Tpos]),
        c(oldLOFval, bracketFval[Tpos]),
        c(oldLOgtd, NA)))
      if (abs(alpha_c - oldLOval) < abs(alpha_q - oldLOval)) {
        if (debug) {
          message('Cubic Interpolation')
        }
        res <- alpha_c
      }
      else {
        if (debug) {
          message('Mixed Quad/Cubic Interpolation')
        }
        res <- (alpha_q + alpha_c) / 2
      }
    }
    else if (dot(gtdT, oldLOgtd) < 0) {
      alpha_c <- polyinterp(point_matrix(
        c(oldLOval, bracket[Tpos]),
        c(oldLOFval, bracketFval[Tpos]),
        c(oldLOgtd, gtdT)))
      alpha_s <- polyinterp(point_matrix(
        c(oldLOval, bracket[Tpos]),
        c(oldLOFval, NA),
        c(oldLOgtd, gtdT)))
      if (abs(alpha_c - bracket[Tpos]) >= abs(alpha_s - bracket[Tpos])) {
        if (debug) {
          message('Cubic Interpolation')
        }
        res <- alpha_c
      }
      else {
        if (debug) {
          message('Quad Interpolation')
        }
        res <- alpha_s
      }
    }
    else if (abs(gtdT) <= abs(oldLOgtd)) {
      alpha_c <- polyinterp(point_matrix(
        c(oldLOval, bracket[Tpos]),
        c(oldLOFval, bracketFval[Tpos]),
        c(oldLOgtd, gtdT)), min(bracket), max(bracket))
      alpha_s <- polyinterp(point_matrix(
        c(oldLOval, bracket[Tpos]),
        c(NA, bracketFval[Tpos]),
        c(oldLOgtd, gtdT)), min(bracket), max(bracket))

      if (alpha_c > min(bracket) && alpha_c < max(bracket)) {
        if (abs(alpha_c - bracket[Tpos]) < abs(alpha_s - bracket[Tpos])) {
          if (debug) {
            message('Bounded Cubic Extrapolation')
          }
          res <- alpha_c
        }
        else {
          if (debug) {
            message('Bounded Secant Extrapolation')
          }
          res <- alpha_s
        }
      }
      else {
        if (debug) {
          message('Bounded Secant Extrapolation')
        }
        res <- alpha_s
      }

      if (bracket[Tpos] > oldLOval) {
        res <- min(bracket[Tpos] + 0.66 * (bracket[nonTpos] - bracket[Tpos]),
                   res)
      }
      else {
        res <- max(bracket[Tpos] + 0.66 * (bracket[nonTpos] - bracket[Tpos]),
                   res)
      }
    }
    else {
      res <- polyinterp(point_matrix(
        c(bracket[nonTpos], bracket[Tpos]),
        c(bracketFval[nonTpos], bracketFval[Tpos]),
        c(gtdNonT, gtdT)))
    }
    res
  }

# function [minPos] <- polyinterp(points,doPlot,xminBound,xmaxBound)
#
#   Minimum of interpolating polynomial based on function and derivative
#   values
#
#   It can also be used for extrapolation if {xmin,xmax} are outside
#   the domain of the points.
#
#   Input:
#       points(pointNum,[x f g])
#       xmin: min value that brackets minimum (default: min of points)
#       xmax: max value that brackets maximum (default: max of points)
#
#   set f or g to sqrt(-1) if they are not known
#   the order of the polynomial is the number of known f and g values minus 1
# points position, function and gradient values to interpolate.
# An n x 3 matrix where n is the number of points and each row contains
# x, f, g in columns 1-3 respectively.
# @return minPos
polyinterp <- function(points,
                       xminBound = range(points[, 1])[1],
                       xmaxBound = range(points[, 1])[2],
                       debug = FALSE) {

  # the number of known f and g values minus 1
  order <- sum(!is.na(points[, 2:3])) - 1

  # Code for most common case:
  #   - cubic interpolation of 2 points
  #       w/ function and derivative values for both
  if (nrow(points) == 2 && order == 3) {
    if (debug) {
      message("polyinterp common case")
    }
    # Solution in this case (where x2 is the farthest point):
    #    d1 <- g1 + g2 - 3*(f1-f2)/(x1-x2);
    #    d2 <- sqrt(d1^2 - g1*g2);
    #    minPos <- x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #    t_new <- min(max(minPos,x1),x2);
    minPos <- which.min(points[, 1])
    notMinPos <- -minPos + 3

    x1 <- points[minPos, 1]
    x2 <- points[notMinPos, 1]
    f1 <- points[minPos, 2]
    f2 <- points[notMinPos, 2]
    g1 <- points[minPos, 3]
    g2 <- points[notMinPos, 3]

    if (x1 - x2 == 0) {
      return(x1)
    }

    d1 <- g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2sq <- d1 ^ 2 - g1 * g2

    if (is_finite_numeric(d2sq) && d2sq >= 0) {
      d2 <- sqrt(d2sq)

      x <- x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
      if (debug) { message("d2 is real ", formatC(d2), " x = ", formatC(x)) }

      minPos <- min(max(x, xminBound), xmaxBound)
    }
    else {
      if (debug) { message("d2 is not real, bisecting") }

      minPos <- (xmaxBound + xminBound) / 2
    }

    return(minPos)
  }

  params <- polyfit(points)
  # If polynomial couldn't be found (due to singular matrix), bisect
  if (is.null(params)) {
    return((xminBound + xmaxBound) / 2)
  }

  # Compute Critical Points
  dParams <- rep(0, order)
  for (i in 1:order) {
    dParams[i] <- params[i + 1] * i
  }

  cp <- unique(c(xminBound, points[, 1], xmaxBound))

  # Remove mad, bad and dangerous to know critical points:
  # Must be finite, non-complex and not an extrapolation
  if (all(is.finite(dParams))) {
    cp <- c(cp,
            Re(Filter(
              function(x) {
                abs(Im(x)) < 1e-8 &&
                  Re(x) >= xminBound &&
                  Re(x) <= xmaxBound },
              polyroot(dParams))))
  }

  # Test Critical Points
  fcp <- polyval(cp, params)
  fminpos <- which.min(fcp)
  if (is.finite(fcp[fminpos])) {
    minpos <- cp[fminpos]
  }
  else {
    # Default to bisection if no critical points valid
    minpos <- (xminBound + xmaxBound) / 2
  }
  minpos
}

# Fits a polynomial to the known function and gradient values. The order of
# the polynomial is the number of known function and gradient values, minus one.
# points - an n x 3 matrix where n is the number of points and each row contains
#          x, f, g in columns 1-3 respectively.
# returns an array containing the coefficients of the polynomial in increasing
# order, e.g. c(1, 2, 3) is the polynomial 1 + 2x + 3x^2
# Returns NULL if the solution is singular
polyfit <- function(points) {
  nPoints <- nrow(points)
  # the number of known f and g values minus 1
  order <- sum(!is.na(points[, 2:3])) - 1

  # Constraints Based on available Function Values
  A <- NULL
  b <- NULL
  for (i in 1:nPoints) {
    if (!is.na(points[i, 2])) {
      constraint <- rep(0, order + 1)
      for (j in rev(0:order)) {
        constraint[order - j + 1] <- points[i, 1] ^ j
      }
      if (is.null(A)) {
        A <- constraint
      }
      else {
        A <- rbind(A, constraint)
      }
      if (is.null(b)) {
        b <- points[i, 2]
      }
      else {
        b <- c(b, points[i, 2])
      }
    }
  }

  # Constraints based on available Derivatives
  for (i in 1:nPoints) {
    if (!is.na(points[i, 3])) {
      constraint <- rep(0, order + 1)
      for (j in 1:order) {
        constraint[j] <- (order - j + 1) * points[i, 1] ^ (order - j)
      }
      if (is.null(A)) {
        A <- constraint
      }
      else {
        A <- rbind(A, constraint)
      }
      if (is.null(b)) {
        b <- points[i, 3]
      }
      else {
        b <- c(b, points[i, 3])
      }
    }
  }
  # Find interpolating polynomial
  params <- try(solve(A, b), silent = TRUE)
  if (class(params) == "numeric") {
    params <- rev(params)
  }
  else {
    params <- NULL
  }
}

# Evaluate 1D polynomial with coefs over the set of points x
# coefs - the coefficients for the terms of the polynomial ordered by
#   increasing degree, i.e. c(1, 2, 3, 4) represents the polynomial
#   4x^3 + 3x^2 + 2x + 1. This is the reverse of the ordering used by the Matlab
#   function, but is consistent with R functions like poly and polyroot
#   Also, the order of the arguments is reversed from the Matlab function
# Returns array of values of the evaluated polynomial
polyval <- function(x, coefs) {
  deg <- length(coefs) - 1
  # Sweep multiplies each column of the poly matrix by the coefficient
  rowSums(sweep(stats::poly(x, degree = deg, raw = TRUE),
                2, coefs[2:length(coefs)], `*`)) + coefs[1]
}

point_matrix <- function(xs, fs, gs) {
  matrix(c(xs, fs, gs), ncol = 3)
}

point_matrix_step <- function(step1, step2) {
  point_matrix(c(step1$alpha, step2$alpha), c(step1$f, step2$f),
               c(step1$d, step2$d))
}

# Step Size ---------------------------------------------------------------

1
# Constructor -------------------------------------------------------------

make_step_size <- function(sub_stage) {
  make_sub_stage(sub_stage, 'step_size')
}

# Constant ----------------------------------------------------------------

# A constant step size
constant_step_size <- function(value = 1) {
  make_step_size(list(
      name = "constant",
      calculate = function(opt, stage, sub_stage, par, fg, iter) {
        list(sub_stage = sub_stage)
      },
      value = value
    ))
}



# Bold Driver -------------------------------------------------------------
# Performs a back tracking line search, but rather than use the Armijo
# (sufficient decrease) condition, accepts the first step size that provides
# any reduction in the function. On the next iteration, the first candidate
# step size is a multiple of accepted step size at the previous iteration.
# inc_mult - the accepted step size at the previous time step will be multiplied
#   by this amount to generate the first candidate step size at the next
#   time step.
# dec_mult - the candidate step sizes will be multiplied by this value (and
#   hence should be a value between 0 and 1 exclusive) while looking for an
#   an acceptable step size.
# init_step_size - the initial candidate step size for the first line search.
bold_driver <- function(inc_mult = 1.1, dec_mult = 0.5,
                inc_fn = partial(`*`, inc_mult),
                dec_fn = partial(`*`, dec_mult),
                init_step_size = 1,
                min_step_size = sqrt(.Machine$double.eps),
                max_step_size = NULL,
                max_fn = Inf) {
  make_step_size(list(
    name = "bold_driver",
    init = function(opt, stage, sub_stage, par, fg, iter) {

      if (!is_first_stage(opt, stage)) {
        # Bold driver requires knowing f at the current location
        # If this step size is part of any stage other than the first
        # we have to turn eager updating
        opt$eager_update <- TRUE
      }
      sub_stage$value <- sub_stage$init_value
      list(opt = opt, sub_stage = sub_stage)
    },
    calculate = function(opt, stage, sub_stage, par, fg, iter) {
      pm <- stage$direction$value

      # Optionally use the gradient if it's available to give up early
      # if we're not going downhill
      if (stage == "gradient_descent"
          && has_gr_curr(opt, iter)
          && dot(opt$cache$gr_curr, pm) > 0) {
        sub_stage$value <- sub_stage$min_value
      }
      else {
        if (is_first_stage(opt, stage) && has_fn_curr(opt, iter)) {
          f0 <- opt$cache$fn_curr
        }
        else {
          opt <- calc_fn(opt, par, fg$fn)
          f0 <- opt$fn
        }

        max_fn <- max_fn_per_ls(opt, max_fn)
      alpha <- sub_stage$value
        para <- par + pm * alpha
        opt <- calc_fn(opt, para, fg$fn)
        num_steps <- 0
        while ((!is.finite(opt$fn) || opt$fn > f0)
               && alpha > sub_stage$min_value
               && num_steps < max_fn) {
          alpha <- sclamp(sub_stage$dec_fn(alpha),
                          min = sub_stage$min_value,
                          max = sub_stage$max_value)
          para <- par + pm * alpha
          opt <- calc_fn(opt, para, fg$fn)
          num_steps <- num_steps + 1
        }
        sub_stage$value <- alpha
        if (!is.finite(opt$fn)) {
          message(stage$type, " ", sub_stage$name,
                  " non finite cost found at iter ", iter)
          sub_stage$value <- sub_stage$min_value
          return(list(opt = opt, sub_stage = sub_stage))
        }

        if (is_last_stage(opt, stage)) {
          opt <- set_fn_new(opt, opt$fn, iter)
        }
      }
      list(opt = opt, sub_stage = sub_stage)
    },
    after_step = function(opt, stage, sub_stage, par, fg, iter, par0,
                          update) {
      alpha_old <- sub_stage$value
      # increase the step size for the next step
      if (opt$ok) {
        alpha_new <- sub_stage$inc_fn(alpha_old)
      }
      else {
        alpha_new <- alpha_old
      }

      sub_stage$value <- sclamp(alpha_new,
                                min = sub_stage$min_value,
                                max = sub_stage$max_value)


      if (opt$ok && is_last_stage(opt, stage) && has_fn_new(opt, iter)) {
        opt <- set_fn_curr(opt, opt$cache$fn_new, iter + 1)
      }

      list(opt = opt, sub_stage = sub_stage)
    },
    inc_fn = inc_fn,
    dec_fn = dec_fn,
    init_value = init_step_size,
    min_value = min_step_size,
    max_value = max_step_size
    ))
}


# Backtracking Line Search ------------------------------------------------

# At each stage, starts the line search at init_step_size, and then back tracks
# reducing, the step size by a factor of rho each time, until the Armijo
# sufficient decrease condition is satisfied.
backtracking <- function(rho = 0.5,
                        init_step_size = 1,
                        min_step_size = sqrt(.Machine$double.eps),
                        max_step_size = NULL,
                        c1 = 1e-4,
                        max_fn = Inf) {
  make_step_size(list(
    name = "backtracking",
    init = function(opt, stage, sub_stage, par, fg, iter) {

      if (!is_first_stage(opt, stage)) {
        # Requires knowing f at the current location
        # If this step size is part of any stage other than the first
        # we have to turn eager updating
        opt$eager_update <- TRUE
      }

      list(opt = opt, sub_stage = sub_stage)
    },
    calculate = function(opt, stage, sub_stage, par, fg, iter) {
      pm <- stage$direction$value
      sub_stage$value <- sub_stage$init_value

      # Optionally use the gradient if it's available to give up early
      # if we're not going downhill
      if (stage == "gradient_descent"
          && has_gr_curr(opt, iter)
          && dot(opt$cache$gr_curr, pm) > 0) {
        sub_stage$value <- sub_stage$min_value
      }
      else {

        if (is_first_stage(opt, stage) && has_fn_curr(opt, iter)) {
          f0 <- opt$cache$fn_curr
        }
        else {
          opt <- calc_fn(opt, par, fg$fn)
          f0 <- opt$fn
        }

        d0 = dot(opt$cache$gr_curr, pm)

        alpha <- sub_stage$value
        para <- par + pm * alpha
        opt <- calc_fn(opt, para, fg$fn)

        max_fn <- max_fn_per_ls(opt, max_fn)

        while ((!is.finite(opt$fn) || !armijo_ok(f0, d0, alpha, opt$fn, c1))
               && alpha > sub_stage$min_value
               && opt$counts$fn < max_fn) {
          alpha <- sclamp(alpha * rho,
                          min = sub_stage$min_value,
                          max = sub_stage$max_value)

          para <- par + pm * alpha
          opt <- calc_fn(opt, para, fg$fn)
        }
        sub_stage$value <- alpha
        if (!is.finite(opt$fn)) {
          message(stage$type, " ", sub_stage$name,
                  " non finite cost found at iter ", iter)
          sub_stage$value <- sub_stage$min_value
          return(list(opt = opt, sub_stage = sub_stage))
        }

        if (is_last_stage(opt, stage)) {
          opt <- set_fn_new(opt, opt$fn, iter)
        }
      }
      list(opt = opt, sub_stage = sub_stage)
    },
    after_step = function(opt, stage, sub_stage, par, fg, iter, par0,
                          update) {

      if (opt$ok && is_last_stage(opt, stage) && has_fn_new(opt, iter)) {
        opt <- set_fn_curr(opt, opt$cache$fn_new, iter + 1)
      }

      list(opt = opt, sub_stage = sub_stage)
    },
    init_value = init_step_size,
    min_value = min_step_size,
    max_value = max_step_size
  ))
}

max_fn_per_ls <- function(opt, max_ls_fn = Inf) {
  max_fn <- max_ls_fn
  if (!is.null(opt$convergence$max_fn)) {
    max_fn <- min(max_fn, opt$convergence$max_fn - opt$counts$fn)
  }
  if (!is.null(opt$convergence$max_fg)) {
    max_fn <- min(max_fn,
                  opt$convergence$max_fg - (opt$counts$fn + opt$counts$gr))
  }
  max_fn
}
# partial function application
partial <- function(f, ...) {
  args <- list(...)
  function(...) {
    do.call(f, c(args, list(...)))
  }
}

# Square of the Euclidean norm of a vector
sqnorm2 <- function(v) {
  dot(v, v)
}

# l1 norm of a vector
norm1 <- function(v) {
  sum(abs(v))
}

# l2 (Euclidean) norm of a vector
norm2 <- function(v) {
  sqrt(dot(v, v))
}

# Infinity norm of a vector
norm_inf <- function(v) {
  max(abs(v))
}

# normalize a vector to length 1
normalize <- function(v) {
  l <- norm2(v)
  if (l < .Machine$double.eps) {
    v
  }
  else {
    v / norm2(v)
  }
}

# dot product of a and b
dot <- function(a, b) {
  sum(a * b)
}

clamp <- function(x, min_val = .Machine$double.eps, max_val = NULL) {
  x[x < min_val] <- min_val
  if (!is.null(max_val)) {
    x[x > max_val] <- max_val
  }
  x
}

sclamp <- function(x, min, max) {
  base::min(base::max(x, min), max)
}

vec_formatC <- function(v) {
  paste(Map(function(x) { formatC(x) }, v), collapse = ", ")
}

# convert a list to a strng
format_list <- function(ll) {
  Reduce(function(acc, elem) {
    paste0(acc,
           ifelse(nchar(acc) == 0, "", " "),
           elem,
           " = ",
           ifelse(length(ll[[elem]]) == 1,
                  formatC(ll[[elem]]), vec_formatC(ll[[elem]])))
  },
  names(ll), "")
}

# returns TRUE if x is in the range (left, right). By default, this is
# an open range, i.e. x == left and x == right is in the range.
# lopen if FALSE then range is [left, right) i.e. x = left is not in the range
# ropen if FALSE then range is (left, right] i.e. x = right is not in the range
is_in_range <- function(x, left, right, lopen = TRUE, ropen = TRUE) {
  `%lop%` <- ifelse(lopen, `<=`, `<`)
  `%rop%` <- ifelse(ropen, `<=`, `<`)

  left %lop% x && x %rop% right
}

# Checks if nullable x is finite
is_finite_numeric <- function(x) {
  is.numeric(x) && is.finite(x)
}

# Logging Hooks -----------------------------------------------------------


require_log_vals <- function(opt, stage, par, fg, iter) {
  message(iter, " ", substr(stage$type, 1, 2)
          ," par = ", vec_formatC(par)
          ," p = ", vec_formatC(stage$direction$value)
          , " a = ", formatC(stage$step_size$value)
          , " ap = ", vec_formatC(stage$result)
          , " f = ", formatC(fg$fn(par)))
  list(opt = opt)
}
attr(require_log_vals, 'name') <- 'log_vals'
attr(require_log_vals, 'event') <- 'after stage'

require_keep_stage_fs <- function(opt, stage, par, fg, iter) {
  if (is.null(opt$all_fs)) { opt$all_fs <- c() }
  f <- fg$fn(par)
  opt$all_fs <- c(opt$all_fs, f)
  list(opt = opt)
}
attr(require_keep_stage_fs, 'name') <- 'require_keep_stage_fs'
attr(require_keep_stage_fs, 'event') <- 'after stage'

# Validate ----------------------------------------------------------------

# Checks that the function value has decreased over the step
require_validate_fn <- function(opt, par, fg, iter, par0, update) {
  if (can_restart(opt, iter)) {
    opt$ok <- opt$cache$fn_new < opt$cache$fn_curr
  }
  opt
}
attr(require_validate_fn, 'name') <- 'validate_fn'
attr(require_validate_fn, 'event') <- 'during validation'
attr(require_validate_fn, 'depends') <- 'fn_new fn_curr save_cache_on_failure'

# Checks that the gradient is a descent direction
# This relies on the gradient being calculated in the "classical" location
# i.e. not using the implementation of Nesterov Acceleration
require_validate_gr <- function(opt, par, fg, iter, par0, update) {
  if (can_restart(opt, iter)) {
    opt$ok <- dot(opt$cache$gr_curr, update) < 0
  }
  opt
}
attr(require_validate_gr, 'name') <- 'validate_gr'
attr(require_validate_gr, 'event') <- 'during validation'
attr(require_validate_gr, 'depends') <- 'gradient save_cache_on_failure'

# Checks that the update vector is getting larger
require_validate_speed <- function(opt, par, fg, iter, par0, update) {
  if (can_restart(opt, iter)) {
    opt$ok <- sqnorm2(update) > sqnorm2(opt$cache$update_old)
  }
  opt
}
attr(require_validate_speed, 'name') <- 'validate_speed'
attr(require_validate_speed, 'event') <- 'during validation'
attr(require_validate_speed, 'depends') <- 'save_cache_on_failure'

# Validate Dependencies ------------------------------------------------------------

# Caches the current fn value
require_fn_curr <- function(opt, par, fg, iter, par0, update) {
  if (!has_fn_curr(opt, iter)) {
    opt <- calc_fn_curr(opt, par, fg$fn, iter)
  }
  opt
}
attr(require_fn_curr, 'name') <- 'fn_curr'
attr(require_fn_curr, 'event') <- 'before step'
attr(require_fn_curr, 'depends') <- 'update_fn_cache'

# Caches the new fn value
require_fn_new <- function(opt, par, fg, iter, par0, update) {
  if (!has_fn_new(opt, iter)) {
    opt <- calc_fn_new(opt, par, fg$fn, iter)
  }
  opt
}
attr(require_fn_new, 'name') <- 'fn_new'
attr(require_fn_new, 'event') <- 'before validation'

# Caches the new fn value as the current value for the next iteration
require_update_fn_cache <- function(opt, par, fg, iter, par0, update) {
  if (opt$ok && has_fn_new(opt, iter)) {
    opt <- set_fn_curr(opt, opt$cache$fn_new, iter + 1)
  }
  opt
}
attr(require_update_fn_cache, 'name') <- 'update_fn_cache'
attr(require_update_fn_cache, 'event') <- 'after step'

# Keep the old cached values around in the event of failure
require_save_cache_on_failure <- function(opt, par, fg, iter, par0, update) {
  # not safe to re-use gr_curr and fn_curr unless gradient calc is the first
  # stage: Nesterov results in moving par via momentum before grad calc.
  # Different result will occur after restart
  if (!opt$ok && opt$stages[[1]]$type == "gradient_descent") {
    cache <- opt$cache
    for (name in names(cache)) {
      if (endsWith(name, "_curr")) {
        iter_name <- paste0(name, "_iter")
        cache_iter <- cache[[iter_name]]
        if (!is.null(cache_iter) && cache_iter == iter) {
          cache[[iter_name]] <- cache_iter + 1
        }
      }
    }
    opt$cache <- cache
  }
  opt
}
attr(require_save_cache_on_failure, 'name') <- 'save_cache_on_failure'
attr(require_save_cache_on_failure, 'event') <- 'after step'
# Functions for line searches

# p62 of Nocedal & Wright defines a "loose" line search as c1 = 1.e-4, c2 = 0.9
# But note that CG and SD methods are not considered suitable for loose line
# search because of the search directions are not well-scaled. c2 = 0.1 is
# suggested for CG on p34. With the Strong Wolfe conditions, reducing c2 makes
# the line search stricter (i.e. forces it closer to a minimum).

# More-Thuente ------------------------------------------------------------
more_thuente_ls <- function(c1 = c2 / 2, c2 = 0.1,
                            max_alpha_mult = Inf,
                            min_step_size = .Machine$double.eps,
                            initializer = "s",
                            initial_step_length = 1,
                            try_newton_step = FALSE,
                            stop_at_min = TRUE,
                            max_fn = Inf,
                            max_gr = Inf,
                            max_fg = Inf,
                            approx_armijo = FALSE,
                            strong_curvature = TRUE,
                            debug = FALSE) {
  if (!is_in_range(c1, 0, 1, lopen = FALSE, ropen = FALSE)) {
    stop("c1 must be between 0 and 1")
  }
  if (!is.null(c2) && !is_in_range(c2, c1, 1, lopen = FALSE, ropen = FALSE)) {
    stop("c2 must be between c1 and 1")
  }
  max_ls_fn <- min(max_fn, max_gr, floor(max_fg / 2))

  line_search(more_thuente(c1 = c1, c2 = c2,
                           max_fn = max_ls_fn,
                           strong_curvature = strong_curvature,
                           approx_armijo = approx_armijo),
              name = "more-thuente",
              max_alpha_mult = max_alpha_mult,
              min_step_size = min_step_size, stop_at_min = stop_at_min,
              initializer = initializer,
              initial_step_length = initial_step_length,
              try_newton_step = try_newton_step,
              debug = debug)
}


# Rasmussen ---------------------------------------------------------------

rasmussen_ls <- function(c1 = c2 / 2, c2 = 0.1, int = 0.1, ext = 3.0,
                         max_alpha_mult = Inf,
                         min_step_size = .Machine$double.eps,
                         initializer = "s",
                         initial_step_length = 1,
                         try_newton_step = FALSE,
                         stop_at_min = TRUE, eps = .Machine$double.eps,
                         max_fn = Inf,
                         max_gr = Inf,
                         max_fg = Inf,
                         strong_curvature = TRUE,
                         approx_armijo = FALSE,
                         debug = FALSE) {
  if (!is_in_range(c1, 0, 1, lopen = FALSE, ropen = FALSE)) {
    stop("c1 must be between 0 and 1")
  }
  if (!is.null(c2) && !is_in_range(c2, c1, 1, lopen = FALSE, ropen = FALSE)) {
    stop("c2 must be between c1 and 1")
  }

  max_ls_fn <- min(max_fn, max_gr, floor(max_fg / 2))

  line_search(rasmussen(c1 = c1, c2 = c2, int = int, ext = ext,
                        max_fn = max_ls_fn,
                        strong_curvature = strong_curvature,
                        approx_armijo = approx_armijo),
              name = "rasmussen",
              max_alpha_mult = max_alpha_mult,
              min_step_size = min_step_size, stop_at_min = stop_at_min,
              initializer = initializer,
              initial_step_length = initial_step_length,
              try_newton_step = try_newton_step,
              eps = eps,
              debug = debug)
}


# Schmidt (minfunc) -------------------------------------------------------

schmidt_ls <- function(c1 = c2 / 2, c2 = 0.1,
                         max_alpha_mult = Inf,
                         min_step_size = .Machine$double.eps,
                         initializer = "s",
                         initial_step_length = "schmidt",
                         try_newton_step = FALSE,
                         stop_at_min = TRUE, eps = .Machine$double.eps,
                         max_fn = Inf,
                         max_gr = Inf,
                         max_fg = Inf,
                         strong_curvature = TRUE,
                         approx_armijo = FALSE,
                         debug = FALSE) {
  if (!is_in_range(c1, 0, 1, lopen = FALSE, ropen = FALSE)) {
    stop("c1 must be between 0 and 1")
  }
  if (!is.null(c2) && !is_in_range(c2, c1, 1, lopen = FALSE, ropen = FALSE)) {
    stop("c2 must be between c1 and 1")
  }

  max_ls_fn <- min(max_fn, max_gr, floor(max_fg / 2))

  line_search(schmidt(c1 = c1, c2 = c2, max_fn = max_ls_fn,
                      strong_curvature = strong_curvature,
                      approx_armijo = approx_armijo),
              name = "schmidt",
              max_alpha_mult = max_alpha_mult,
              min_step_size = min_step_size, stop_at_min = stop_at_min,
              initializer = initializer,
              initial_step_length = initial_step_length,
              try_newton_step = try_newton_step,
              eps = eps,
              debug = debug)
}


schmidt_armijo_ls <- function(c1 = 0.005,
                       max_alpha_mult = Inf,
                       min_step_size = .Machine$double.eps,
                       initializer = "s",
                       initial_step_length = "schmidt",
                       try_newton_step = FALSE,
                       step_down = NULL,
                       stop_at_min = TRUE, eps = .Machine$double.eps,
                       max_fn = Inf,
                       max_gr = Inf,
                       max_fg = Inf,
                       debug = FALSE) {
  if (!is_in_range(c1, 0, 1, lopen = FALSE, ropen = FALSE)) {
    stop("c1 must be between 0 and 1")
  }

  max_ls_fn <- min(max_fn, max_gr, floor(max_fg / 2))

  line_search(schmidt_armijo_backtrack(c1 = c1, max_fn = max_ls_fn,
                                       step_down = step_down),
              name = "schmidt_armijo",
              max_alpha_mult = max_alpha_mult,
              min_step_size = min_step_size, stop_at_min = stop_at_min,
              initializer = initializer,
              initial_step_length = initial_step_length,
              try_newton_step = try_newton_step,
              eps = eps,
              debug = debug)
}


# Hager-Zhang -------------------------------------------------------------

hager_zhang_ls <- function(c1 = c2 / 2, c2 = 0.1,
                           max_alpha_mult = Inf,
                           min_step_size = .Machine$double.eps,
                           initializer = "hz",
                           initial_step_length = "hz",
                           try_newton_step = FALSE,
                           stop_at_min = TRUE, eps = .Machine$double.eps,
                           max_fn = Inf,
                           max_gr = Inf,
                           max_fg = Inf,
                           strong_curvature = FALSE,
                           approx_armijo = TRUE,
                           debug = FALSE) {
  if (!is_in_range(c1, 0, 1, lopen = FALSE, ropen = FALSE)) {
    stop("c1 must be between 0 and 1")
  }
  if (!is.null(c2) && !is_in_range(c2, c1, 1, lopen = FALSE, ropen = FALSE)) {
    stop("c2 must be between c1 and 1")
  }

  max_ls_fn <- min(max_fn, max_gr, floor(max_fg / 2))

  line_search(hager_zhang(c1 = c1, c2 = c2, max_fn = max_ls_fn,
                          strong_curvature = strong_curvature,
                          approx_armijo = approx_armijo),
              name = "hager-zhang",
              max_alpha_mult = max_alpha_mult,
              min_step_size = min_step_size, stop_at_min = stop_at_min,
              initializer = initializer,
              initial_step_length = initial_step_length,
              try_newton_step = try_newton_step,
              eps = eps,
              debug = debug)

}

# Line Search -------------------------------------------------------------

line_search <- function(ls_fn,
                        name,
                        initializer = "slope ratio",
                        try_newton_step = FALSE,
                        initial_step_length = 1,
                        max_alpha_mult = Inf,
                        min_step_size = .Machine$double.eps,
                        stop_at_min = TRUE,
                        debug = FALSE,
                        eps = .Machine$double.eps) {

  initializer <- match.arg(tolower(initializer),
                           c("slope ratio", "quadratic", "hz", "hager-zhang"))
  if (initializer == "hager-zhang") {
    initializer <- "hz"
  }

  if (!is.numeric(initial_step_length)) {
    initializer0 <- match.arg(tolower(initial_step_length),
                                     c("rasmussen", "scipy", "schmidt",
                                       "hz", "hager-zhang"))
    if (initializer0 == "hager-zhang") {
      initializer0 <- "hz"
    }
  }
  else {
    initializer0 <- initial_step_length
  }

  make_step_size(list(
    name = name,
    eps = eps,
    init = function(opt, stage, sub_stage, par, fg, iter) {
      #message("Initializing Wolfe line search for ", stage$type)

      if (!is_first_stage(opt, stage)) {
        # Requires knowing f at the current location
        # If this step size is part of any stage other than the first
        # we have to turn on eager updating
        #message("Wolfe: setting stage updating to eager")
        opt$eager_update <- TRUE
      }

      sub_stage$value <- NULL
      sub_stage$alpha0 <- NULL
      sub_stage$d0 <- NULL
      sub_stage$f0 <- NULL
      sub_stage$df <- NULL

      list(opt = opt, sub_stage = sub_stage)
    },
    calculate = function(opt, stage, sub_stage, par, fg, iter) {

      pm <- stage$direction$value
      if (norm2(pm) < .Machine$double.eps) {
        sub_stage$value <- 0
        if (is_last_stage(opt, stage)) {
          opt <- set_fn_new(opt, opt$cache$fn_curr, iter)
        }
        return(list(opt = opt, sub_stage = sub_stage))
      }

      if (is_first_stage(opt, stage) && has_fn_curr(opt, iter)) {
#        message(sub_stage$name, ": fetching fn_curr from cache ", formatC(opt$cache$fn_curr))
        f0 <- opt$cache$fn_curr
      }
      else {
        opt <- calc_fn(opt, par, fg$fn)
        f0 <- opt$fn
      }

      #message("gr = ", vec_formatC(opt$cache$gr_curr), " pm = ", vec_formatC(pm))
      step0 <- list(
        alpha = 0,
        f = f0,
        df = opt$cache$gr_curr,
        d = dot(opt$cache$gr_curr, pm)
      )

      alpha_prev <- sub_stage$value

      phi_alpha <- make_phi_alpha(par, fg, pm,
                                  calc_gradient_default = TRUE, debug = debug)


      # described on p59 of Nocedal and Wright
      if (initializer == "slope ratio" && !is.null(sub_stage$d0)) {
        sub_stage$value <- step_next_slope_ratio(alpha_prev, sub_stage$d0,
                                            step0, eps, max_alpha_mult)
      }
      else if (initializer == "quadratic" && !is.null(sub_stage$f0)) {
        # quadratic interpolation
        sub_stage$value <- step_next_quad_interp(sub_stage$f0, step0,
                                            try_newton_step = try_newton_step)
      }
      else if (initializer == "hz" && !is.null(alpha_prev)) {
        step_next_res <- step_next_hz(phi_alpha, alpha_prev, step0)
        sub_stage$value <- step_next_res$alpha
        opt$counts$fn <- opt$counts$fn + step_next_res$fn
      }

      if (is.null(sub_stage$value) || sub_stage$value <= 0) {
        sub_stage$value <- guess_alpha0(initializer0,
                                        par,
                                        step0$f,
                                        step0$df,
                                        step0$d,
                                        try_newton_step)
      }

      sub_stage$alpha0 <- sub_stage$value
      sub_stage$d0 <- step0$d
      sub_stage$f0 <- step0$f

      max_fn <- Inf
      max_gr <- Inf
      max_fg <- Inf
      if (!is.null(opt$convergence$max_fn) && is.finite(opt$convergence$max_fn)) {
        max_fn <- opt$convergence$max_fn - opt$counts$fn
      }
      if (!is.null(opt$convergence$max_gr) && is.finite(opt$convergence$max_gr)) {
        max_gr <- opt$convergence$max_gr - opt$counts$gr
      }
      if (!is.null(opt$convergence$max_fg) && is.finite(opt$convergence$max_fg)) {
        max_fg <- opt$convergence$max_fg - (opt$counts$fn + opt$counts$gr)
      }

      if (max_fn <= 0 || max_gr <= 0 || max_fg <= 0) {
        sub_stage$value <- 0
        if (is_last_stage(opt, stage)) {
          opt <- set_fn_new(opt, step0$f, iter)
          sub_stage$df <- step0$df
        }
      }
      else {
        ls_result <- ls_fn(phi_alpha, step0, sub_stage$value,
                           total_max_fn = max_fn, total_max_gr = max_gr,
                           total_max_fg = max_fg, pm = pm)
        sub_stage$value <- ls_result$step$alpha
        opt$counts$fn <- opt$counts$fn + ls_result$nfn
        opt$counts$gr <- opt$counts$gr + ls_result$ngr

        if (is_last_stage(opt, stage)) {
          opt <- set_fn_new(opt, ls_result$step$f, iter)
          if (is.null(ls_result$step$df)) {
            sub_stage$df <- rep(sub_stage$eps, length(par))
          }
          else {
            sub_stage$df <- ls_result$step$df
          }
        }
      }

      list(opt = opt, sub_stage = sub_stage)
    },
    after_step = function(opt, stage, sub_stage, par, fg, iter, par0,
                          update) {
      if (opt$ok && is_last_stage(opt, stage) && has_fn_new(opt, iter)) {
        opt <- set_fn_curr(opt, opt$cache$fn_new, iter + 1)
      }

      if (opt$ok && is_single_stage(opt)) {
        opt <- set_gr_curr(opt, sub_stage$df, iter + 1)
      }

      list(opt = opt)
    },
    depends = c("gradient")
  ))
}

make_phi_alpha <- function(par, fg, pm,
                           calc_gradient_default = FALSE, debug = FALSE) {
  # LS functions are responsible for updating fn and gr count
  function(alpha, calc_gradient = calc_gradient_default) {
    y_alpha <- par + (alpha * pm)

    if (!is.null(fg$fg) && calc_gradient) {
      fg_res <- fg$fg(y_alpha)
      f <- fg_res$fn
      g <- fg_res$gr

      step <- list(
        alpha = alpha,
        f = f,
        df = g,
        d = dot(g, pm)
      )
    }

    else {
      f <- fg$fn(y_alpha)
      step <- list(
        alpha = alpha,
        f = f
      )

      if (calc_gradient) {
        step$df <- fg$gr(y_alpha)
        step$d <- dot(step$df, pm)
      }
    }

    step$par <- y_alpha

    if (debug) {
      message(format_list(step))
    }
    step
  }
}

# Ensure Valid Step Size
#
# Given an initial step size, if either the function value or the directional
# derivative is non-finite (NaN or infinite), reduce the step size until
# finite values are found.
#
# @param phi Line function.
# @param alpha Initial step size.
# @param min_alpha Minimum step size.
# @param max_fn Maximum number of function evaluations allowed.
# @return List containing:
# \itemize{
#   \item step Valid step size or the last step size evaluated, or NULL if
#     max_fn == 0.
#   \item nfn Number of function evaluations.
#   \item ok If a valid step was found
# }
find_finite <- function(phi, alpha, min_alpha = 0, max_fn = 20) {
  nfn <- 0
  ok <- FALSE
  step <- NULL
  while (nfn < max_fn && alpha >= min_alpha) {
    step <- phi(alpha)
    nfn <- nfn + 1
    if (step_is_finite(step)) {
      ok <- TRUE
      break
    }
    alpha <- (min_alpha + alpha) / 2
  }
  list(step = step, nfn = nfn, ok = ok)
}

step_is_finite <- function(step) {
  is.finite(step$f) && is.finite(step$df)
}


# Initial Step Length -----------------------------------------------------

# Set the initial step length. If initial_step_length is a numeric scalar,
# then use that as-is. Otherwise, use one of several variations based around
# the only thing we know (the directional derivative)
guess_alpha0 <- function(guess_type, x0, f0, gr0, d0,
                           try_newton_step = FALSE) {
  if (is.numeric(guess_type)) {
    return(guess_type)
  }

  s <- switch(guess_type,
    rasmussen = step0_rasmussen(d0),
    scipy = step0_scipy(gr0, d0),
    schmidt = step0_schmidt(gr0),
    hz = step0_hz(x0, f0, gr0, psi0 = 0.01)
  )

  if (try_newton_step) {
    s <- min(1, 1.01 * s)
  }
  s
}

# From minimize.m
step0_rasmussen <- function(d0) {
  1 / (1 - d0)
}

# found in _minimize_bfgs in optimize.py with this comment:
# # Sets the initial step guess to dx ~ 1
# actually sets f_old to f0 + 0.5 * ||g||2 then uses f_old in the quadratic
# update formula. If you do the algebra,  you get -||g||2 / d
# Assuming steepest descent for step0, this can be simplified further to
# 1 / sqrt(-d0), but may as well not assume that
step0_scipy <- function(gr0, d0) {
  -norm2(gr0) / d0
}

# Mark Schmidt's minFunc.m uses reciprocal of the one-norm
step0_schmidt <- function(gr0) {
  1 / norm1(gr0)
}

# I0 in the 'initial' routine in CG_DESCENT paper
step0_hz <- function(x0, f0, gr0, psi0 = 0.01) {
  alpha <- 1
  if (is.null(x0)) {
    return(alpha)
  }
  ginf_norm <- norm_inf(gr0)
  if (ginf_norm != 0) {
    xinf_norm <- norm_inf(x0)
    if (xinf_norm != 0) {
      alpha <- psi0 * (xinf_norm / ginf_norm)
    }
    else if (is_finite_numeric(f0) && f0 != 0) {
      g2_norm2 <- sqnorm2(gr0)
      if (is_finite_numeric(g2_norm2) && g2_norm2 != 0) {
        alpha <- psi0 * (abs(f0) / ginf_norm ^ 2)
      }
    }
  }
  alpha
}

# Next Step Length --------------------------------------------------------

# described on p59 of Nocedal and Wright
# slope ratio method
step_next_slope_ratio <- function(alpha_prev, d0_prev, step0, eps,
                                  max_alpha_mult) {
  # NB the p vector must be a descent direction or the directional
  # derivative will be positive => a negative initial step size!
  slope_ratio <- d0_prev / (step0$d + eps)
  s <- alpha_prev * min(max_alpha_mult, slope_ratio)
  max(s, eps)
}

# quadratic interpolation
step_next_quad_interp <- function(f0_prev, step0, try_newton_step = FALSE) {
  s <- 2  * (step0$f - f0_prev) / step0$d
  if (try_newton_step) {
    s <- min(1, 1.01 * s)
  }
  s
  max(.Machine$double.eps, s)
}

# steps I1-2 in the routine 'initial' of the CG_DESCENT paper
step_next_hz <- function(phi, alpha_prev, step0, psi1 = 0.1, psi2 = 2) {
  if (alpha_prev < .Machine$double.eps) {
    return(list(alpha = .Machine$double.eps, fn = 0))
  }

  # I2: use if QuadStep fails
  alpha <- alpha_prev * psi2
  nfn <- 0
  # I1: QuadStep
  # If function is reduced at the initial guess, carry out a quadratic
  # interpolation. If the resulting quadratic is strongly convex, use the
  # minimizer of the quadratic. Otherwise, use I2.
  step_psi <- phi(alpha_prev * psi1, calc_gradient = FALSE)
  nfn <- 1
  if (step_psi$f <= step0$f) {
    alpha_q <- quadratic_interpolate_step(step0, step_psi)
    # A 1D quadratic Ax^2 + Bx + C is strongly convex if A > 0. Second clause in
    # if statement is A expressed in terms of the minimizer (this is easy to
    # derive by looking at Nocedal & Wright 2nd Edition, equations 3.57 and
    # 3.58)
    if (alpha_q > 0 && -step0$d / (2 * alpha_q) > 0) {
      alpha <- alpha_q
    }
  }
  alpha <- max(.Machine$double.eps, alpha)
  list(alpha = alpha, fn = nfn)
}

# Line Search Checks -------------------------------------------------------

# Armijo Rule (or Sufficient Decrease Condition)
#
# Line search test.
#
# The sufficient decrease condition is met if the line search step length yields
# a decrease in the function value that is sufficiently large (relative to the
# size of the step).
#
# This test prevents large step sizes that, while representing a function value
# decrease, don't reduce it by very much, which could indicate that the
# function minimum has been stepped over and you're now going back up the slope.
# Also, this condition can always be met by taking a sufficiently small step,
# so line searches involving this condition can always terminate. The downside
# is that you can end up taking very small steps, so it's usual to combine this
# condition with one that encourages larger step sizes.
#
# @param f0 Value of function at starting point of line search.
# @param d0 Directional derivative at starting point of line search.
# @param alpha the step length.
# @param fa Value of function at alpha.
# @param c1 the sufficient decrease constant. Should take a value between 0 and
#   1.
# @return \code{TRUE} if the step \code{alpha} represents a sufficient decrease.
armijo_ok <- function(f0, d0, alpha, fa, c1) {
  fa <= f0 + c1 * alpha * d0
}

# Armijo Rule (or Sufficient Decrease Condition)
#
# Line search test.
#
# The sufficient decrease condition is met if the line search step length yields
# a decrease in the function value that is sufficiently large (relative to the
# size of the step).
#
# This test prevents large step sizes that, while representing a function value
# decrease, don't reduce it by very much, which could indicate that the
# function minimum has been stepped over and you're now going back up the slope.
# Also, this condition can always be met by taking a sufficiently small step,
# so line searches involving this condition can always terminate. The downside
# is that you can end up taking very small steps, so it's usual to combine this
# condition with one that encourages larger step sizes.
#
# @param step0 Line search values at starting point.
# @param step Line search value at a step along the line.
# @param c1 the sufficient decrease constant. Should take a value between 0 and
#   1.
# @return \code{TRUE} if the step represents a sufficient decrease.
armijo_ok_step <- function(step0, step, c1) {
  armijo_ok(step0$f, step0$d, step$alpha, step$f, c1)
}


# Curvature Condition
#
# Line search test.
#
# Ensures that the directional derivative of the line search direction at a
# candidate step size is greater than a specified fraction of the slope of the
# line at the starting point of the search. This condition is used to stop step
# sizes being too small.
#
# In combination with the sufficient decrease condition \code{\link{armijo_ok}}
# these conditions make up the Wolfe conditions.
#
# @param d0 Directional derivative at starting point.
# @param da Directional derivative at step alpha.
# @param c2 Curvature condition constant. Should take a value between \code{c1}
#  (the constant used in the sufficient decrease condition check) and 1.
# @return \code{TRUE} if the curvature condition is met.
curvature_ok <- function(d0, da, c2) {
  da > c2 * d0
}

# Curvature Condition
#
# Line search test.
#
# Ensures that the directional derivative of the line search direction at a
# candidate step size is greater than a specified fraction of the slope of the
# line at the starting point of the search. This condition is used to stop step
# sizes being too small.
#
# In combination with the sufficient decrease condition \code{\link{armijo_ok}}
# these conditions make up the Wolfe conditions.
#
# @param step0 Line search values at starting point.
# @param step Line search value at a step along the line.
# @param c2 Curvature condition constant. Should take a value between \code{c1}
#  (the constant used in the sufficient decrease condition check) and 1.
# @return \code{TRUE} if the curvature condition is met.
curvature_ok_step <- function(step0, step, c2) {
  curvature_ok(step0$d, step$d, c2)
}

# Strong Curvature Condition
#
# Line search test.
#
# Ensures that the value of the directional derivative of the line search
# direction at a candidate step size is equal to or greater than a specified
# fraction of the slope of the line at the starting point of the search, while
# having the same direction. This condition is used to make the step size lie
# close to a stationary point. Unlike the normal curvature condition, a step
# size where the sign of the gradient changed (e.g. the minimum had been
# skipped) would not be acceptable for the strong curvature condition.
#
# In combination with the sufficient decrease condition \code{\link{armijo_ok}}
# these conditions make up the Strong Wolfe conditions.
#
# @param d0 Directional derivative at starting point.
# @param da Directrional derivative at step alpha.
# @param c2 Curvature condition constant. Should take a value between \code{c1}
#  (the constant used in the sufficient decrease condition check) and 1.
# @return \code{TRUE} if the strong curvature condition is met.
strong_curvature_ok <- function(d0, da, c2) {
  abs(da) <= -c2 * d0
}

# Strong Curvature Condition
#
# Line search test.
#
# Ensures that the value of the directional derivative of the line search
# direction at a candidate step size is equal to or greater than a specified
# fraction of the slope of the line at the starting point of the search, while
# having the same direction. This condition is used to make the step size lie
# close to a stationary point. Unlike the normal curvature condition, a step
# size where the sign of the gradient changed (e.g. the minimum had been
# skipped) would not be acceptable for the strong curvature condition.
#
# In combination with the sufficient decrease condition \code{\link{armijo_ok}}
# these conditions make up the Strong Wolfe conditions.
#
# @param step0 Line search values at starting point.
# @param step Line search value at a step along the line.
# @param c2 Curvature condition constant. Should take a value between \code{c1}
#  (the constant used in the sufficient decrease condition check) and 1.
# @return \code{TRUE} if the curvature condition is met.
strong_curvature_ok_step <- function(step0, step, c2) {
  strong_curvature_ok(step0$d, step$d, c2)
}

# Are the Strong Wolfe Conditions Met?
#
# Step size check.
#
# Returns true if the Strong Wolfe conditions are met, consisting of the
# sufficient decrease conditions and the strong curvature condition.
#
# @param f0 Function value at starting point.
# @param d0 Directional derivative value at starting point.
# @param alpha Step length.
# @param fa Function value at alpha.
# @param da Directional derivative at alpha.
# @param c1 Constant used in sufficient decrease condition. Should take a value
#   between 0 and 1.
# @param c2 Constant used in curvature condition. Should take a value between
#   c1 and 1.
# @return TRUE if the Strong Wolfe condition is met by the step size.
strong_wolfe_ok <- function(f0, d0, alpha, fa, da, c1, c2) {
  armijo_ok(f0, d0, alpha, fa, c1) &&
    strong_curvature_ok(d0, da, c2)
}

# Are the Strong Wolfe Conditions Met for the Given Step?
#
# Line search test.
#
# Returns true if the candidate step size meets the Strong Wolfe conditions,
# consisting of the sufficient decrease conditions and the strong curvature
# condition.
#
# @param step0 Line search values at starting point of line search.
# @param step Line search value at candiate step size.
# @param c1 Constant used in sufficient decrease condition. Should take a value
#   between 0 and 1.
# @param c2 Constant used in curvature condition. Should take a value between
#   c1 and 1.
# @return TRUE if the Strong Wolfe condition is met by the step size.
strong_wolfe_ok_step <- function(step0, step, c1, c2) {
  armijo_ok_step(step0, step, c1) && strong_curvature_ok_step(step0, step, c2)
}

# Are the Wolfe Conditions Met for the Given Step?
wolfe_ok_step <- function(step0, step, c1, c2) {
  armijo_ok_step(step0, step, c1) && curvature_ok_step(step0, step, c2)
}

# Create a Wolfe Conditions check function allowing for either approximate or
# exact Armijo condition and weak or strong curvature condition
make_wolfe_ok_step_fn <- function(approx_armijo = FALSE,
                                  strong_curvature = TRUE, eps = 1e-6) {

  approx_armijo_ok_fn <- make_approx_armijo_ok_step(eps)

  function(step0, step, c1, c2) {
    if (approx_armijo) {
      ok <- approx_armijo_ok_fn(step0, step, c1)
    }
    else {
      ok <- armijo_ok_step(step0, step, c1)
    }

    if (ok) {
      if (strong_curvature) {
        ok <- strong_curvature_ok_step(step0, step, c2)
      }
      else {
        ok <- curvature_ok_step(step0, step, c2)
      }
    }
    ok
  }
}

# Create Approximation Armijo check function:
# Checks approximate Armijo conditions only if exact Armijo check fails and
# if function value is sufficiently close to step0 value according to eps
make_approx_armijo_ok_step <- function(eps) {
  function(step0, step, c1) {
    eps_k <- eps * abs(step0$f)

    if (armijo_ok_step(step0, step, c1)) {
      return(TRUE)
    }
    (step$f <= step0$f + eps_k) && approx_armijo_ok_step(step0, step, c1)
  }
}

# Is the approximate Armijo condition met?
#
# Suggested by Hager and Zhang (2005) as part of the Approximate Wolfe
# Conditions. The second of these conditions is identical to the (weak)
# curvature condition.
#
# The first condition applies the armijo condition to a quadratic approximation
# to the function, which allows for higher precision in finding the minimizer.
#
# It is suggested that the approximate version of the Armijo condition be used
# when fa is 'close' to f0, e.g. fa <= f0 + eps * |f0| where eps = 1e-6
#
# c1 should be < 0.5
approx_armijo_ok <- function(d0, da, c1) {
  (2 * c1 - 1) * d0 >= da
}

# Is the approximate Armijo condition met for the given step?
approx_armijo_ok_step <- function(step0, step, c1) {
  approx_armijo_ok(step0$d, step$d, c1)
}


# Bracket -----------------------------------------------------------------

bracket_is_finite <- function(bracket) {
  all(is.finite(c(bracket_props(bracket, c('f', 'd')))))
}

# extracts all the properties (e.g. 'f', 'df', 'd' or 'alpha') from all members
# of the bracket. Works if there are one or two bracket members. Can get
# multiple properties at once, by providing an array of the properties,
# e.g. bracket_props(bracket, c('f', 'd'))
bracket_props <- function(bracket, prop) {
  unlist(sapply(bracket, `[`, prop))
}

bracket_width <- function(bracket) {
  bracket_range <- bracket_props(bracket, 'alpha')
  abs(bracket_range[2] - bracket_range[1])
}

bracket_min_alpha <- function(bracket) {
  min(bracket_props(bracket, 'alpha'))
}

best_bracket_step <- function(bracket) {
  LOpos <- which.min(bracket_props(bracket, 'f'))
  bracket[[LOpos]]
}


is_in_bracket <- function(bracket, alpha) {
  is_in_range(alpha, bracket[[1]]$alpha, bracket[[2]]$alpha)
}

format_bracket <- function(bracket) {
  paste0("[", formatC(bracket[[1]]$alpha), ", ", formatC(bracket[[2]]$alpha),
         "]")
}
list(
  mize = mize,
  opt_step = mize_step,
  opt_init = mize_init,
  make_mize = make_mize,
  opt_report = opt_report,
  mize_step_summary = mize_step_summary,
  opt_clear_cache = opt_clear_cache,
  check_mize_convergence = check_mize_convergence
)
}
