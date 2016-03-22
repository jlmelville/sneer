library(sneer)
context("Cost Gradients")

# we're just using any old random positive numbers to test the gradients
# don't want them close to zero otherwise the backwards difference results
# in negative probabilities which mess with any logarithms.

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

inp <- list(pm = pm)
out <- list(qm = qm)

expect_cost_fd <- function(cost, inp, out, label, diff = 1e-5, tolerance = 1e-6) {
  method <- list(cost = cost, eps = .Machine$double.eps)
  k_gan <- cost$gr(inp, out, method)
  k_gfd <- cost_gradient_fd(inp, out, method, diff = diff)
  expect_equal(k_gan, k_gfd, label = label,
               expected.label = "finite difference gradient",
               tolerance = tolerance)
}

test_that("analytical gradient matches finite difference", {
expect_cost_fd(kl_fg(), inp, out, "Kullback Leibler", diff = 1e-7)
expect_cost_fd(reverse_kl_fg(), inp, out, "Reverse Kullback Leibler",
               diff = 1e-7)
expect_cost_fd(nerv_fg(), inp, out, "NeRV default", diff = 1e-7)
expect_cost_fd(nerv_fg(lambda = 0), inp, out, "NeRV lambda=0",
               diff = 1e-7)
expect_cost_fd(nerv_fg(lambda = 1), inp, out, "NeRV lambda=1",
               diff = 1e-7)
expect_cost_fd(jse_fg(), inp, out, "JSE default", diff = 1e-7)
# kappa can't actually be set to exactly 0 or 1
expect_cost_fd(jse_fg(kappa = 0.01), inp, out, "JSE kappa -> 0", diff = 1e-7)
expect_cost_fd(jse_fg(kappa = 0.99), inp, out, "JSE kappa -> 1", diff = 1e-7)
})
