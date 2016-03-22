library(sneer)
context("Similarity")

irisd2 <- dist(iris[, 1:4]) ^ 2

test_that("Heavy tail matches exponential or Student-t at alpha limits", {
expect_equal(heavy_tail_weight(irisd2, alpha = 1.5e-8), exp_weight(irisd2),
             label = "exp when alpha = 0")
expect_equal(heavy_tail_weight(irisd2, alpha = 1.5e-8, beta = 2),
             exp_weight(irisd2, beta = 2),
             label = "exp(2) with alpha = 0, beta = 2")
expect_equal(heavy_tail_weight(irisd2, alpha = 1), tdist_weight(irisd2),
             label = "t-dist when alpha = 1")
})

test_that("Asymmetric exponential weights rows differently", {
  nums <- c(1,2,3,4,5,6,7,8,9)
  m <- matrix(nums, nrow = 3, byrow = TRUE)
  m <- log(m)

  ek_symm <- exp_kernel()
  expect_equal(attr(ek_symm$fn, 'type'), 'symm')
  expect_equal(ek_symm$fn(ek_symm, m), matrix(1 / nums, nrow = 3, byrow = TRUE),
                                           label = "global weighting")

  ek_asymm <- exp_kernel(beta = c(1, 2, 3))
  expect_equal(attr(ek_asymm$fn, 'type'), 'asymm')

  expect_equal(ek_asymm$fn(ek_asymm, m),
               matrix(c(1 / nums[1:3], 1 / (nums[4:6]) ^ 2, 1/nums[7:9] ^ 3),
                      nrow = 3, byrow = TRUE), label = "per-row weighting")

})

test_that("Asymmetric heavy tail function weights rows differently", {
  nums <- c(1,2,3,4,5,6,7,8,9)
  m <- matrix(nums, nrow = 3, byrow = TRUE)

  # single a, single b
  htk_ss <- heavy_tail_kernel(alpha = 0.5, beta = 1)
  expect_equal(attr(htk_ss$fn, 'type'), 'symm')
  ssw <- matrix(c(
    0.44444444, 0.25000000, 0.16000000, # a = 0.5, b = 1
    0.11111111, 0.08163265, 0.06250000, # a = 0.5, b = 1
    0.04938272, 0.04000000, 0.03305785  # a = 0.5, b = 1
  ), nrow = 3, byrow = TRUE)
  expect_equal(htk_ss$fn(htk_ss, m), ssw,
               info = "HS weighting 1 alpha, 1 beta", tolerance = 1e-7)

  # multiple a, single b
  htk_ms <- heavy_tail_kernel(alpha = c(0.5, 1, 1.5), beta = 1)
  expect_equal(attr(htk_ms$fn, 'type'), 'asymm')
  # should have same row 1 as ssw
  msw <- matrix(c(
    0.44444444, 0.25000000, 0.16000000, # a = 0.5, b = 1
    0.2000000,  0.1666667,  0.1428571,  # a = 1,   b = 1
    0.1962764,  0.1808719,  0.1681724   # a = 1.5, b = 1
  ), nrow = 3, byrow = TRUE)
  expect_equal(htk_ms$fn(htk_ms, m), msw,
               info = "HS weighting multiple alpha, single beta",
               tolerance = 1e-7)

  # single a, multiple b
  htk_sm <- heavy_tail_kernel(alpha = 0.5, beta = c(0.5, 1, 1.5))
  expect_equal(attr(htk_sm$fn, 'type'), 'asymm')
  # should have same row 2 as ssw
  smw <- matrix(c(
    0.6400000,  0.44444444, 0.32653061, # a = 0.5, b = 0.5
    0.2000000,  0.1666667,  0.1428571,  # a = 0.5, b = 1
    0.0256000,  0.02040816, 0.01664932  # a = 0.5, b = 1.5
  ), nrow = 3, byrow = TRUE)
  expect_equal(htk_ms$fn(htk_ms, m), msw,
               info = "HS weighting single alpha, multiple beta",
               tolerance = 1e-7)

  # multiple a, multiple b
  htk_mm <- heavy_tail_kernel(alpha = c(0.5, 1, 1.5), beta = c(0.5, 1, 1.5))
  expect_equal(attr(htk_mm$fn, 'type'), 'asymm')
  # should have same row 1 as smw and same row 2 as msw
  mmw <- matrix(c(
    0.6400000,  0.44444444, 0.32653061, # a = 0.5, b = 0.5
    0.2000000,  0.1666667,  0.1428571,  # a = 1,   b = 1
    0.1527531,  0.1404422,  0.1303449   # a = 1.5, b = 1.5
  ), nrow = 3, byrow = TRUE)
  expect_equal(htk_mm$fn(htk_mm, m), mmw,
               info = "HS weighting multiple alpha, multiple beta",
               tolerance = 1e-7)

})


expect_fd <- function(kernel, m, label, diff = 1e-4, tolerance = 1e-6) {
  k_gan <- kernel$gr(kernel, m)
  k_gfd <- kernel_gr_fd(kernel, m, diff = diff)
  expect_equal(k_gan, k_gfd, label = label,
               expected.label = "finite difference gradient",
               tolerance = tolerance)
}

test_that("analytical gradient matches finite difference", {
  m <- matrix(c(-0.4899386, -0.3886341, -0.3704233, -0.4530830,
                -0.4076637, -0.3776508, -0.3170777, -0.3887747,
                -0.4680165, -0.5583384, -0.3639630, -0.6095128,
                -0.8849361, -0.7469212, -0.3758769, -0.7540928))

  expect_fd(exp_kernel(), m, "default exponential")
  expect_fd(exp_kernel(beta = 2), m, "exponential with non-default beta")
  expect_fd(tdist_kernel(), m, "Student T")
  expect_fd(heavy_tail_kernel(), m, "default heavy tail", diff = 1e-3,
            tolerance = 1e-5)
  expect_fd(heavy_tail_kernel(beta = 1, alpha = 1), m, "heavy tail b=1, a=1")
  expect_fd(heavy_tail_kernel(beta = 2, alpha = 1), m, "heavy tail b=2, a=1",
            diff = 1e-5)
  expect_fd(heavy_tail_kernel(beta = 0.5, alpha = 1), m,
            "heavy tail b=0.5, a=1")
  expect_fd(heavy_tail_kernel(beta = 1, alpha = 0.5), m,
            "heavy tail b=1, a=0.5")
  expect_fd(heavy_tail_kernel(beta = 2, alpha = 0.5), m,
            "heavy tail b=2, a=0.5", diff = 1e-5)
  expect_fd(heavy_tail_kernel(beta = 0.5, alpha = 0.5), m,
            "heavy tail b=0.5, a=1")
  expect_fd(heavy_tail_kernel(beta = 1, alpha = 1.5), m,
            "heavy tail b=1, a=0.5", diff = 1e-5)
  expect_fd(heavy_tail_kernel(beta = 2, alpha = 1.5), m,
            "heavy tail b=2, a=0.5", diff = 1e-5)
  expect_fd(heavy_tail_kernel(beta = 0.5, alpha = 1.5), m,
            "heavy tail b=0.5, a=1")

  betas <- c(0.4923117, 0.4146415, 0.6220991, 0.9000282)
  expect_fd(exp_kernel(beta = betas), m, "Asymmetric exponential")

  alphas <- c(0.8334264, 1.3868922, 1.4136361, 1.1317302)
  expect_fd(heavy_tail_kernel(beta = betas, alpha = 1), m,
            "heavy tail multiple b, single a")
  expect_fd(heavy_tail_kernel(beta = 1, alpha = alphas), m,
            "heavy tail single b, multiple a")
  expect_fd(heavy_tail_kernel(beta = betas, alpha = alphas), m,
            "heavy tail multiple b, multiple a")
})


