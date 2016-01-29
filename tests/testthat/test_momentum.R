library(sneer)
context("Momentum")

mu <- function(update) {
  opt <- make_opt(update = update)
  function(iter) {
    opt$update$t <- iter
    update$momentum$calculate(opt, NULL ,NULL, NULL, iter)
  }
}

constant_m <- mu(constant_momentum(0.5))
expect_equal(constant_m(1), 0.5)
expect_equal(constant_m(500), 0.5)
expect_equal(constant_m(1000), 0.5)

constant_m <- mu(constant_momentum(0.25))
expect_equal(constant_m(1), 0.25)
expect_equal(constant_m(500), 0.25)
expect_equal(constant_m(1000), 0.25)

constant_m <- mu(constant_momentum(0.95))
expect_equal(constant_m(1), 0.95)
expect_equal(constant_m(500), 0.95)
expect_equal(constant_m(1000), 0.95)

step_m <- mu(step_momentum(init_momentum = 0.2, final_momentum = 0.6,
                         switch_iter = 100))
expect_equal(step_m(1), 0.2)
expect_equal(step_m(99), 0.2)
expect_equal(step_m(100), 0.6)
expect_equal(step_m(101), 0.6)
expect_equal(step_m(1000), 0.6)

linear_m <- mu(linear_momentum(max_iter = 1000, init_momentum = 0.1,
                               final_momentum = 0.8))
expect_equal(linear_m(0), 0.1)
expect_equal(linear_m(500), 0.45)
expect_equal(linear_m(1000), 0.8)

nest_m <- mu(nesterov_nsc_momentum())
expect_equal(nest_m(1), 0.5)
expect_equal(nest_m(5), 0.7)
expect_equal(nest_m(10), 0.8)
expect_equal(nest_m(20), 0.88)
expect_equal(nest_m(50), 0.9455, tolerance = 0.0001)
expect_equal(nest_m(500), 0.9941, tolerance = 0.0001)
expect_equal(nest_m(1000), 0.9970, tolerance = 0.0001)

nest_max_m <- mu(nesterov_nsc_momentum(max_momentum = 0.9))
expect_equal(nest_max_m(1), 0.5)
expect_equal(nest_max_m(5), 0.7)
expect_equal(nest_max_m(10), 0.8)
expect_equal(nest_max_m(20), 0.88)
expect_equal(nest_max_m(50), 0.9)
expect_equal(nest_max_m(500), 0.9)
expect_equal(nest_max_m(1000), 0.9)
