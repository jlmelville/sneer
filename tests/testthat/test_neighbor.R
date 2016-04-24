library(sneer)
context("Neighbor Retrieval")

set.seed(1)
n  <- 101
din <- matrix(runif(n * n), nrow = n)
dout <- matrix(runif(n * n), nrow = n)
random_crm <- coranking_matrix(din, dout)
perfect_crm <- coranking_matrix(din, din)

test_that("for random distances, QNX is ~ k / n - 1", {
  for (k in c(25, 50, 75)) {
    random_qnx <- qnx_crm(random_crm, k = k)
    expect_equal(random_qnx, k / (n - 1), tolerance = 1e-2, scale = 1,
                 info = paste0("random ", k, "-ary preservation is ",
                               formatC(k / (n - 1))))
  }
})

test_that("for perfect preservation, QNX is 1", {
  for (k in c(25, 50, 75)) {
    perfect_qnx <- qnx_crm(perfect_crm, k = k)
    expect_equal(perfect_qnx, 1, tolerance = 5e-2, scale = 1,
                 info = paste0("random ", k, "-ary preservation is 1"))
  }
})

test_that("for random distances, RNX is ~ 0", {
  for (k in c(25, 50, 75)) {
    random_rnx <- rnx_crm(random_crm, k = k)
    expect_equal(random_rnx, 0, tolerance = 5e-2, scale = 1,
                 info = paste0("random r", k, "-ary preservation is 0"))
  }
})

test_that("for perfect preservation, RNX is 1", {
  for (k in c(25, 50, 75)) {
    perfect_rnx <- rnx_crm(perfect_crm, k = k)
    expect_equal(perfect_rnx, 1, tolerance = 5e-2, scale = 1,
                 info = paste0("perfect r", k, "-ary preservation is 1"))
  }
})

test_that("for perfect preservation, RNX AUC is 1", {
  expect_equal(rnx_auc_crm(perfect_crm), 1)
})

test_that("for random preservation, RNX AUC is ~0", {
  expect_equal(rnx_auc_crm(random_crm), 0, tolerance = 5e-2)
})

