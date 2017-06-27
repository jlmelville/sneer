library(sneer)
context("Per-point cost decomposition")

dxm <- matrix(
  c(
    0,         0.5212644, 1.0352926, 0.1874560,
    0.5212644, 0,         0.8208077, 0.3625277,
    1.0352926, 0.8208077, 0,         1.0030080,
    0.1874560, 0.3625277, 1.0030080, 0
  ),
  nrow = 4,
  byrow = TRUE
)

dym <- matrix(
  c(
    0        , 1.0536639, 1.2548910, 0.1241702,
    1.0536639, 0        , 1.2891483, 1.7605622,
    1.2548910, 1.2891483, 0        , 0.6984493,
    0.1241702, 1.7605622, 0.6984493, 0
  ),
  nrow = 4,
  byrow = TRUE
)

pm <- matrix(
  c(
    0.96778093, 0.4803334,  0.5778160,  0.16866689,
    0.04093099, 0.24378704, 0.6469862,  0.15534931,
    0.45747659, 0.1738215,  0.09109848, 0.03285557,
    0.01878910, 0.2071784,  0.9701524,  0.23717644
  ),
  nrow = 4
)

qm <- matrix(
  c(
    0.8027810,  0.4966343, 0.4535958, 0.03131244,
    0.55702956, 0.4198836, 0.4566622, 0.77890452,
    0.80129515, 0.8324861, 0.2220281, 0.35916868,
    0.09285781, 0.9816577, 0.3392806, 0.3763771
  ),
  nrow = 4
)

inp <- list(dm = dxm, pm = pm)
out <- list(dm = dym, qm = qm)

expect_point_sum <- function(cost, extra_label = "") {
  method = list(eps = .Machine$double.eps, cost = cost)
  if (!is.null(cost$after_init_fn)) {
    res <- cost$after_init_fn(inp, out, method)
    if (!is.null(res$method)) {
      method <- res$method
    }
    if (!is.null(res$inp)) {
      inp <- res$inp
    }
    if (!is.null(res$out)) {
      out <- res$out
    }
  }
  expect_equal(sum(cost$point(inp, out, method)),
               cost$fn(inp, out, method),
               label = paste0(cost$name, " point sum ", extra_label))
}

test_that("Sum of point costs equal distance cost function value", {
  expect_point_sum(metric_stress_fg())
  expect_point_sum(metric_sstress_fg())
  expect_point_sum(sammon_fg())
})

test_that("Sum of point costs equal probability cost function value", {
  expect_point_sum(kl_fg())
  expect_point_sum(reverse_kl_fg())
  expect_point_sum(nerv_fg())
  expect_point_sum(nerv_fg(lambda = 0.25), extra_label = "lambda = 0.25")
  expect_point_sum(jse_fg())
  expect_point_sum(jse_fg(kappa = 0.25), extra_label = "kappa = 0.25")
})


