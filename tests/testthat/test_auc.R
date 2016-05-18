library(sneer)
context("ROC/PR AUC")

skip_if_not_installed("PRROC")

set.seed(1)
n <- 60
dm_rand <- matrix(runif(n * n), nrow = n)
labels <- c(rep("1", n / 2), rep("0", n / 2))

test_that("random AUC PR is ~0.5", {
  expect_equal(pr_auc_embed(dm_rand, labels)$av_auc, 0.5, tolerance = 1e-2)
})
test_that("random AUC ROC is ~0.5", {
  expect_equal(roc_auc_embed(dm_rand, labels)$av_auc, 0.5, tolerance = 0.1)
})

class_1 <- matrix(1, nrow = n / 2, ncol = 2)
class_0 <- matrix(0, nrow = n / 2, ncol = 2)
m <- rbind(class_1, class_0)
dm_perfect <- distance_matrix(m)

test_that("perfect AUC PR is 1", {
  expect_equal(pr_auc_embed(dm_perfect, labels)$av_auc, 1.0)
})
test_that("perfect AUC ROC is 1", {
  expect_equal(roc_auc_embed(dm_perfect, labels)$av_auc, 1.0)
})

dm_anti <- matrix(seq(length.out = n * n), nrow = n)
test_that("anti-enrichment AUC PR is non-zero", {
  expect_equal(pr_auc_embed(dm_anti, labels)$label_av$`0`, 0.299,
               tolerance = 1e-2)
})
test_that("anti-enrichment AUC ROC is 0", {
  expect_equal(roc_auc_embed(dm_anti, labels)$label_av$`0`, 0.0)
})

# swap three columns of the distance matrix
dm_swap <- dm_anti
dm_swap[, c(1,2,3,48,49,50)] <- dm_anti[, c(50,49,48,3,2,1)]
test_that("AUC PR for good and bad performance", {
  pr_auc <- pr_auc_embed(dm_swap, labels)
  expect_equal(pr_auc$label_av$`0`, 0.407, tolerance = 1e-2)
  expect_equal(pr_auc$label_av$`1`, 0.722, tolerance = 1e-2)
})
test_that("AUC ROC for good and bad performance, bigger delta than PR", {
  roc_auc <- roc_auc_embed(dm_swap, labels)
  expect_equal(roc_auc$label_av$`0`, 0.157, tolerance = 1e-2)
  expect_equal(roc_auc$label_av$`1`, 0.843, tolerance = 1e-2)
})

# repeat with imbalanced classes
labels <- c(rep("1", n / 4), rep("0", (3 * n) / 4))
test_that(
  "random AUC PR with imbalanced classes is the fraction of positives", {
  pr_auc <- pr_auc_embed(dm_rand, labels)
  expect_equal(pr_auc$label_av$`0`, 0.75, tolerance = 0.1, scale = 1)
  expect_equal(pr_auc$label_av$`1`, 0.25, tolerance = 0.1, scale = 1)
})
test_that("random AUC ROC imbalanced classes is still 0.5", {
  roc_auc <- roc_auc_embed(dm_rand, labels)
  expect_equal(roc_auc$label_av$`0`, 0.5, tolerance = 0.1, scale = 1)
  expect_equal(roc_auc$label_av$`1`, 0.5, tolerance = 0.1, scale = 1)
})

class_1 <- matrix(1, nrow = n / 4, ncol = 2)
class_0 <- matrix(0, nrow = (3 * n) / 4, ncol = 2)
m <- rbind(class_1, class_0)
dm_perfect <- distance_matrix(m)

test_that("perfect AUC PR is 1", {
  expect_equal(pr_auc_embed(dm_perfect, labels)$av_auc, 1.0)
})
test_that("perfect AUC ROC is 1", {
  expect_equal(roc_auc_embed(dm_perfect, labels)$av_auc, 1.0)
})

dm_anti <- matrix(seq(length.out = n * n), nrow = n)
test_that("anti-enrichment AUC PR is non-zero and larger for more imbalanced classes", {
  expect_equal(pr_auc_embed(dm_anti, labels)$label_av$`0`, 0.533,
               tolerance = 1e-2)
})
test_that("anti-enrichment AUC ROC is 0", {
  expect_equal(roc_auc_embed(dm_anti, labels)$label_av$`0`, 0.0)
})

dm_swap <- dm_anti
dm_swap[, c(1,2,3,48,49,50)] <- dm_anti[, c(50,49,48,3,2,1)]
test_that("AUC PR for good and bad performance unbalanced classes", {
  pr_auc <- pr_auc_embed(dm_swap, labels)
  expect_equal(pr_auc$label_av$`0`, 0.63, tolerance = 1e-2)
  expect_equal(pr_auc$label_av$`1`, 0.518, tolerance = 1e-2)
})
test_that("AUC ROC for good and bad performance unbalanced classes", {
  roc_auc <- roc_auc_embed(dm_swap, labels)
  expect_equal(roc_auc$label_av$`0`, 0.209, tolerance = 1e-2)
  expect_equal(roc_auc$label_av$`1`, 0.791, tolerance = 1e-2)
})
