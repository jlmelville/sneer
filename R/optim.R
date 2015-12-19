# optim.R
############################# optimization Routines ###

make_opt <- function(grad_pos_fn = classical_grad_pos,
                     direction = steepest_descent(), step_size = bold_driver(),
                     update = no_momentum(), normalize_grads = TRUE,
                     mat_name = "ym", recenter = TRUE,
                     verbose = TRUE) {
  opt <- list()

  opt$mat_name <- mat_name
  opt$normalize_grads <- normalize_grads

  opt$grad_pos_fn <- grad_pos_fn
  opt$direction_method <- direction
  opt$step_size_method <- step_size
  opt$update_method <- update

  da_fn <- NULL
  if (!is.null(direction$after_step)) {
    da_fn <- direction$after_step
  }
  sa_fn <- NULL
  if (!is.null(step_size$after_step)) {
    sa_fn <- step_size$after_step
  }
  ua_fn <- NULL
  if (!is.null(update$after_step)) {
    ua_fn <- update$after_step
  }

  opt$after_step <- make_after_step(direction_after_step_fn = da_fn,
                                    step_size_after_step_fn = sa_fn,
                                    update_after_step_fn = ua_fn,
                                    recenter = recenter,
                                    verbose = verbose)

  dv_fn <- NULL
  if (!is.null(direction$validate)) {
    dv_fn <- direction$validate
  }
  sv_fn <- NULL
  if (!is.null(step_size$validate)) {
    sv_fn <- step_size$validate
  }
  uv_fn <- NULL
  if (!is.null(update$validate)) {
    uv_fn <- update$validate
  }
  opt$validate <- make_validate_solution(direction_validation_fn = dv_fn,
                                         step_size_validation_fn = sv_fn,
                                         update_validation_fn = uv_fn,
                                         verbose = verbose)

  opt$init <- function(opt, inp, out, method) {
    if (!is.null(opt$direction_method$init)) {
      opt <- opt$direction_method$init(opt, inp, out, method)
    }
    if (!is.null(opt$step_size_method$init)) {
      opt <- opt$step_size_method$init(opt, inp, out, method)
    }
    if (!is.null(opt$update_method$init)) {
      opt <- opt$update_method$init(opt, inp, out, method)
    }

    opt
  }

  opt
}


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

# update out with new solution (Y, W, qm etc)
update_solution <- function(opt, inp, out, method) {
  new_out <- out
  new_solution <- new_out[[opt$mat_name]] + opt$update_method$update
  new_out[[opt$mat_name]] <- new_solution
  update_out(inp, new_out, method, opt$mat_name)
}

make_after_step <- function(recenter = TRUE,
                            direction_after_step_fn = NULL,
                            step_size_after_step_fn = NULL,
                            update_after_step_fn = NULL, verbose = FALSE) {
  after_step <- list()

  if (!is.null(direction_after_step_fn)) {
    after_step$direction <- direction_after_step_fn
  }

  if (!is.null(step_size_after_step_fn)) {
    after_step$step_size <- step_size_after_step_fn
  }

  if (!is.null(update_after_step_fn)) {
    after_step$update <- update_after_step_fn
  }

  if (recenter) {
    after_step$recenter <- function(opt, inp, out, new_out, ok, iter) {
      vm <- new_out[[opt$mat_name]]
      vm <- sweep(vm, 2, colMeans(vm))  # subtract colMeans from each column
      new_out[[opt$mat_name]] <- vm

      list(new_out = new_out)
    }
  }

  function(opt, inp, out, new_out, ok, iter) {
    for (name in names(after_step)) {
      result <- after_step[[name]](opt, inp, out, new_out, ok, iter)
      if (!is.null(result$opt)) {
        opt <- result$opt
      }
      if (!is.null(result$inp)) {
        inp <- result$inp
      }
      if (!is.null(result$out)) {
        out <- result$out
      }
      if (!is.null(result$new_out)) {
        new_out <- result$new_out
      }
    }
    list(opt = opt, inp = inp, out = out, new_out = new_out)
  }
}

make_validate_solution <- function(direction_validation_fn = NULL,
                                   step_size_validation_fn = NULL,
                                   update_validation_fn = NULL, verbose = FALSE) {
  validate_solution <- list()

  if (!is.null(direction_validation_fn)) {
    validate_solution$direction_validation_func <- direction_validation_fn
  }

  if (!is.null(step_size_validation_fn)) {
    validate_solution$step_size_validation_func <- step_size_validation_fn
  }

  if (!is.null(update_validation_fn)) {
    validate_solution$update_validation_func <- update_validation_fn
  }

  function(opt, inp, out, new_out, method) {
    all_good <- TRUE

    for (name in names(validate_solution)) {
      result <- validate_solution[[name]](opt, inp, out, new_out, method)
      if (!is.null(result$opt)) {
        opt <- result$opt
      }
      if (!is.null(result$inp)) {
        inp <- result$inp
      }
      if (!is.null(result$out)) {
        out <- result$out
      }
      if (!is.null(result$new_out)) {
        new_out <- result$new_out
      }
      if (!is.null(result$method)) {
        method <- result$method
      }
      if (!is.null(result$ok)) {
        if (!result$ok) {
          all_good <- FALSE
        }
      }
    }
    list(ok = all_good, opt = opt, inp = inp, out = out, new_out = new_out,
         method = method)
  }
}

### Gradient Locations ###

classical_grad_pos <- function(opt, inp, out, method) {
  gradient(inp, out, method, opt$mat_name)
}

nesterov_grad_pos <- function(opt, inp, out, method) {
  prev_update <- opt$update_method$update
  mu <- opt$update_method$momentum

  opt$update_method$update <- mu * prev_update
  new_out <- update_solution(opt, inp, out, method)

  gradient(inp, new_out, method, opt$mat_name)
}


### Step Size ###
bold_driver <- function(increase_mult = 1.1, decrease_mult = 0.5,
                        increase_fn = partial(`*`, increase_mult),
                        decrease_fn = partial(`*`, decrease_mult),
                        initial_step_size = 1,
                        min_step_size = sqrt(.Machine$double.eps),
                        max_step_size = NULL, allow_uphill = FALSE) {
  list(
    inc_fn = increase_fn,
    dec_fn = decrease_fn,
    initial_step_size = initial_step_size,
    min_step_size = min_step_size,
    max_step_size = max_step_size,
    init = function(opt, inp, out, method) {
      opt$step_size_method$old_cost <- method$cost_fn(inp, out)
      opt$step_size_method$step_size <- opt$step_size_method$initial_step_size
      opt
    },
    get_step_size = function(opt, inp, out, method) {
      opt$step_size_method$step_size
    },
    validate = function(opt, inp, out, new_out, method) {
      cost <- method$cost_fn(inp, new_out)
      if (allow_uphill) {
        ok <- TRUE
      } else {
        ok <- cost < opt$step_size_method$old_cost
      }

      opt$step_size_method$cost <- cost
      list(ok = ok, opt = opt)
    },
    after_step = function(opt, inp, out, new_out, ok, iter) {
      if (allow_uphill) {
        ok <- opt$step_size_method$cost < opt$step_size_method$old_cost
      }
      s_old <- opt$step_size_method$step_size
      if (ok) {
        s_new <- opt$step_size_method$inc_fn(opt$step_size_method$step_size)
      } else {
        s_new <- opt$step_size_method$dec_fn(opt$step_size_method$step_size)
        opt$step_size_method$cost <- opt$step_size_method$old_cost
      }
      ds <- s_new - s_old

      opt$step_size_method$step_size <- s_old + ds
      opt$step_size_method$step_size <- clamp(opt$step_size_method$step_size,
                                              opt$step_size_method$min_step_size)
      opt$step_size_method$old_cost <- opt$step_size_method$cost

      list(opt = opt)
    }
  )
}


## Jacobs ##
jacobs <- function(inc_mult = 1.1, dec_mult = 0.5,
                   inc_fn = partial(`*`, inc_mult),
                   dec_fn = partial(`*`, dec_mult),
                   init_step_size = 1, min_step_size = .Machine$double.eps,
                   max_step_size = NULL) {
  list(
    inc_fn = inc_fn,
    dec_fn = dec_fn,
    initial_step_size = init_step_size,
    min_step_size = min_step_size,
    max_step_size = max_step_size,
    init = function(opt, inp, out, method) {
      v <- out[[opt$mat_name]]
      opt$step_size_method$step_size <-
        matrix(opt$step_size_method$initial_step_size, nrow(v), ncol(v))
      opt
    },
    get_step_size = function(opt, inp, out, method) {

      gm <- opt$gm
      old_step_size <- opt$step_size_method$step_size
      inc_fn <- opt$step_size_method$inc_fn
      dec_fn <- opt$step_size_method$dec_fn
      old_update <- opt$update_method$update
      min_step_size <- opt$step_size_method$min_step_size

      new_step_size <- jacobs_step_size(gm, old_step_size,
                                        old_update, inc_fn, dec_fn)

      # clamp to min_gain to avoid negative learning rate
      new_step_size <- clamp(new_step_size, min_step_size)
    }
  )
}

tsne_jacobs <- jacobs(inc_fn = partial(`+`, 0.2), dec_mult = 0.8,
                      min_step_size = 0.1)

#' Calculate a new step size matrix based on the sign of the gradient versus
#' that of the previous step.
#'
#' @details For each direction, the sign of the gradient is compared with that
#' of the update in the previous time step. If it's the same sign, increase the
#' step size; if the sign has changed, then it's assumed that the minimum was
#' missed and the current location brackets the minimum. In this case, the step
#' size is decreased.
#'
#' @param gm Gradient matrix.
#' @param step_size Step size for the previous iteration.
#' @param update update_method matrix for the previous iteration.
#' @param inc_fn Function to apply to \code{step_size} to increase its elements.
#' @param dec_fn Function to apply to \code{step_size} to decrease its elements.
#' @return the new step size.
jacobs_step_size <- function(gm, step_size, update, inc_fn, dec_fn) {
  # the step size also includes the sign accounting for the descent so if
  # the old step is the opposite sign of the current gradient that implies
  # the old gradient had the same sign
  inc_fn(step_size) * abs(sign(gm) != sign(update)) +
    dec_fn(step_size) * abs(sign(gm) == sign(update))
}

### update_method ###
## Momentum ##

step_momentum <- function(initial_momentum = 0.5, final_momentum = 0.8,
                          switch_iter = 250, verbose = TRUE) {
  list(
    initial_momentum = initial_momentum,
    final_momentum = final_momentum,
    mom_switch_iter = switch_iter,
    init = function(opt, inp, out, method) {
      opt$update_method$momentum <- opt$update_method$initial_momentum
      opt$update_method$update <- matrix(0, nrow(out[[opt$mat_name]]),
                                         ncol(out[[opt$mat_name]]))
      opt
    },
    get_update = momentum_update,
    after_step = function(opt, inp, out, new_out, ok, iter) {
      if (iter == opt$update_method$mom_switch_iter) {
        if (verbose) {
          message("Switching momentum to ", final_momentum, " at iter ", iter)
        }
        opt$update_method$momentum <- opt$update_method$final_momentum
      }

      list(opt = opt)
    }
  )
}

linear_momentum <- function(max_iter, initial_momentum = 0,
                            final_momentum = 0.9) {
  list(
    initial_momentum = initial_momentum,
    final_momentum = final_momentum,
    init = function(opt, inp, out, method) {
      opt$update_method$momentum <- opt$update_method$initial_momentum
      opt$update_method$update <- matrix(0, nrow(out[[opt$mat_name]]),
                                         ncol(out[[opt$mat_name]]))
      opt
    },
    get_update = momentum_update,
    after_step = function(opt, inp, out, new_out, ok, iter) {
      mu_i <- opt$update_method$initial_momentum
      mu_f <- opt$update_method$final_momentum
      mu <- (mu_f - mu_i) / max_iter
      opt$update_method$momentum <- (mu * iter) + mu_i

      list(opt = opt)
    }
  )
}


nesterov_non_convex_momentum <- function() {
  list(
    initial_momentum = 0.5,
    init = function(opt, inp, out, method) {
      opt$update_method$momentum <- opt$update_method$initial_momentum
      opt$update_method$update <- matrix(0, nrow(out[[opt$mat_name]]),
                                         ncol(out[[opt$mat_name]]))
      opt
    },
    get_update = momentum_update,
    after_step = function(opt, inp, out, new_out, ok, iter) {
      opt$update_method$momentum <- 1 - (3 / (iter + 5))
      list(opt = opt)
    }
  )
}


no_momentum <- function() {
  list(
    init = function(opt, inp, out, method) {
      opt$update_method$momentum <- 0
      opt$update_method$update <- matrix(0, nrow(out[[opt$mat_name]]),
                                         ncol(out[[opt$mat_name]]))
      opt
    },
    get_update = momentum_update
  )
}


momentum_update <- function(opt, inp, out, method) {
  direction <- opt$direction_method$direction
  step_size <- opt$step_size_method$step_size
  prev_update <- opt$update_method$update
  mu <- opt$update_method$momentum
  grad_update <- step_size * direction

  (mu * prev_update) + ((1 - mu) * grad_update)
}

## Steepest Descent ##
steepest_descent <- function() {
  list(
    get_direction = function(opt, inp, out, method, iter) {
      list(direction = -opt$gm)
    }
  )
}
