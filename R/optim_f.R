min_fn <- function(opt, par, fn, gr, maxit = 100, nreport = 10,
                   min_cost = 0, report_at = NULL, keep_costs = FALSE,
                   keep_steps = FALSE,
                   verbose = TRUE) {
  report_every <- max(1, round(maxit / nreport))

  opt$gradient$calculate <- function(opt, inp, out, method, iter) {
    pos <- opt$gradient$calculate_position(opt, inp, out, method, iter)
    gm <- par_to_out(gr(mat_to_par(pos$ym)), opt, inp, method, nrow = 1)$ym
    list(gm = gm)
  }

  cost0 <- fn(par)
  res <- list(
    par = par,
    cost = cost0
  )
  if (keep_costs) {
    res$costs <- cost0
  }
  if (keep_steps) {
    res$steps <- 0
  }
  if (verbose) {
    message("0: cost = ", formatC(cost0), " ",
            " par = ", format_vec(par))
  }
  if (maxit == 0) {
    return(res)
  }

  old_cost <- cost0
  #rtol <- .Machine$double.xmax
  for (iter in 1:maxit) {
    opt_res <- opt_step(opt, par, fn, gr, iter)
    par <- opt_res$par
    opt <- opt_res$opt
    cost <- fn(par)
    #rtol <- reltol(cost, old_cost)
    old_cost <- cost
    res$par <- par
    res$counts <- opt$counts
    res$cost <- cost
    if (keep_costs) {
      res$costs <- c(res$costs, cost)
    }
    if (keep_steps) {
      res$steps <- c(res$steps, opt_res$opt$step_size$value)
    }
    if (verbose && (iter %% report_every == 0 || iter %in% report_at)) {
      message(iter,": cost = ", formatC(cost),
              " par = ", paste(formatC(par), collapse = ' '),
              " nfn = ", opt$counts$fn, " ngr = ", opt$counts$gr)
    }
    if (cost < min_cost) {
      break
    }
  }

  res
}

opt_step <- function(opt, par, fn, gr, iter) {
  counts <- opt$counts
  if (is.null(counts$fn)) {
    counts$fn <- 0
  }
  if (is.null(counts$gr)) {
    counts$gr <- 0
  }

  inp <- NULL
  method <- list(
    update_out_fn = function(inp, new_out, method) { new_out },
    cost_fn = function(inp, out, method) {
      counts$fn <<- counts$fn + 1
      fn(mat_to_par(out$ym))
    },
    counts = list(fn = 0)
  )

  out <- par_to_out(par, opt, inp, method, nrow = 1)
  if (iter == 1) {
    opt <- opt$init(opt, inp, out, method)
  }

  opt$gm <- opt$gradient$calculate(opt, inp, out, method, iter)$gm
  counts$gr <- counts$gr + 1
#  message("grad calculated at: ", format_vec(pos$ym),
#          " gm: ", format_vec(opt$gm))

  if (any(is.nan(opt$gm))) {
    stop("NaN in grad. descent at iter ", iter)
  }

  opt$grad_length <- length_vec(opt$gm)

  direction_result <- opt$direction$calculate(opt, inp, out, method, iter)
  opt <- direction_result$opt

  if (opt$normalize_direction) {
    opt$direction$value <- normalize(opt$direction$value)
  }

  step_size_result <- opt$step_size$calculate(opt, inp, out, method, iter)
  opt <- step_size_result$opt

  update_result <- opt$update$calculate(opt, inp, out, method, iter)
  opt <- update_result$opt

  proposed_out <- update_solution(opt, inp, out, method)

  # intercept whether we want to accept the new solution e.g. bold driver
  ok <- TRUE
  if (!is.null(opt$validate)) {
    validation_result <- opt$validate(opt, inp, out, proposed_out, method, iter)
    opt <- validation_result$opt
    inp <- validation_result$inp
    out <- validation_result$out
    proposed_out <- validation_result$proposed_out
    method <- validation_result$method
    ok <- validation_result$ok
  }

  if (ok) {
    new_out <- proposed_out
  } else {
    new_out <- out
  }

  if (!is.null(opt$after_step)) {
    after_step_result <- opt$after_step(opt, inp, out, new_out, ok, iter)
    opt <- after_step_result$opt
    inp <- after_step_result$inp
    out <- after_step_result$outfr
    new_out <- after_step_result$new_out
  }

  par <- mat_to_par(new_out$ym)
  cost <- fn(par)

  if (!ok) {
    if (!is.null(opt$old_cost)) {
      opt$cost <- opt$old_cost
    }
  }
  if (!is.null(opt$cost)) {
    opt$old_cost <- opt$cost
  }

  opt$counts <- counts

  list(opt = opt, inp = inp, out = new_out, cost = cost, par = par)
}

fn_opt <- function(...) {
  make_opt(recenter = FALSE, ...)
}

rb_optim <- function(method = "CG", maxit = 10000, REPORT = 1000,
                     reltol = .Machine$double.eps, ...) {
  fr <- rosenbrock_banana$fr
  grr <- rosenbrock_banana$grr

  x0 <- c(-1.2, 1)

  control <- c(maxit = maxit, reltol = reltol, REPORT = REPORT, list(...))

  optim(par = x0, fn = fr, gr = grr, method = method,
        control = control)
}

format_vec <- function(vec) {
  paste(formatC(vec), collapse = ' ')
}

