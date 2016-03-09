library(sneer)
context("Backtracking")

## Rosenbrock Banana function
rosenbrock_banana <- list(
  fr = function(x) {
    x1 <- x[1]
    x2 <- x[2]
    100 * (x2 - x1 * x1) ^ 2 + (1 - x1) ^ 2
  },
  grr = function(x) { ## Gradient of 'fr'
    x1 <- x[1]
    x2 <- x[2]
    c(-400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1),
      200 *      (x2 - x1 * x1))
  }
)
x00 <- c(0, 0)


# No great insight here: just characterize how backtracking performs to detect
# when I break it
res00 <- min_fn(
  fn_opt(
    direction = steepest_descent(),
    step_size = backtracking(c1 = 1e-4),
    update = constant_momentum(0), normalize_direction = FALSE),
  x00, rosenbrock_banana$fr, rosenbrock_banana$grr, maxit = 20,
  min_cost = .Machine$double.eps, keep_costs = TRUE, keep_steps = TRUE,
  verbose = FALSE)
costsx00 <- c(1.0000000, 0.8292966, 0.7346546, 0.6750831, 0.6409418, 0.6149921,
              0.5957437, 0.5794900, 0.5665957, 0.5561917, 0.5502063, 0.5492324,
              0.5227925, 0.5035221, 0.4896256, 0.4777331, 0.4675673, 0.4583285,
              0.4501454, 0.4428688, 0.4370758)
expect_equal(res00$costs, costsx00, tolerance = 1e-7)
stepsx00 <- c(0.000000000, 0.107374182, 0.007452485, 0.007527009, 0.007602280,
              0.007678302, 0.007755085, 0.007832636, 0.007910963, 0.007990072,
              0.008069973, 0.008150673, 0.006585744, 0.006651601, 0.006718117,
              0.006785298, 0.006853151, 0.006921683, 0.006990899, 0.007060808,
              0.007131417)
expect_equal(res00$steps, stepsx00, tolerance = 1e-7)

