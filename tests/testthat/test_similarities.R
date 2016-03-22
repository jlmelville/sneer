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
  expect_fd(exp_asymm_kernel(beta = betas), m, "Asymmetric exponential")
})


