# Gradient Calculation
#
# Calculate the gradient of the cost function for a specified position.
#
# @param inp Input data.
# @param out Output data containing the desired position.
# @param method Embedding method.
# @param mat_name Name of the matrix in the output data list that contains the
# embedded coordinates.
# @return List containing:
# \item{\code{km}}{Stiffness matrix.}
# \item{\code{gm}}{Gradient matrix.}
gradient <- function(inp, out, method, mat_name = "ym") {
  km <- method$stiffness_fn(method, inp, out)
  gm <- stiff_to_grads(out[[mat_name]], km)
  list(km = km, gm = gm)
}

# Finite Difference Gradient Calculation
#
# Calculate the gradient of the cost function for a specified position using
# a finite difference.
#
# Only intended for testing that analytical gradients have been calculated
# correctly.
#
# @param inp Input data.
# @param out Output data containing the desired position.
# @param method Embedding method.
# @param mat_name Name of the matrix in the output data list that contains the
# embedded coordinates.
# @param diff Step size to take in finite difference calculation.
# @return List containing:
# \item{\code{gm}}{Gradient matrix.}
gradient_fd <- function(inp, out, method, mat_name = "ym", diff = 1e-4) {
  ym <- out[[mat_name]]
  nr <- nrow(ym)
  nc <- ncol(ym)

  grad <- matrix(0, nrow = nr, ncol = nc)
  for (i in 1:nr) {
    for (j in 1:nc) {
      ymij_old <- ym[i, j]
      ym[i, j] <- ymij_old + diff
      out_fwd <- set_solution(inp, ym, method, mat_name, out)
      res <- set_solution(inp, ym, method, mat_name, out)
      out_fwd <- res$out
      inp_fwd <- res$inp
      cost_fwd <- calculate_cost(method, inp_fwd, out_fwd)

      ym[i, j] <- ymij_old - diff
      out_back <- set_solution(inp, ym, method, mat_name, out)
      res <- set_solution(inp, ym, method, mat_name, out)
      out_back <- res$out
      inp_back <- res$inp
      cost_back <- calculate_cost(method, inp_back, out_back)

      fd <- (cost_fwd - cost_back) / (2 * diff)
      grad[i, j] <- fd

      ym[i, j] <- ymij_old
      res <- set_solution(inp, ym, method, mat_name, out)
      out <- res$out
      inp <- res$inp
    }
  }

  list(gm = grad)
}

# Gradient Matrix from Stiffness Matrix
#
# Convert stiffness matrix to gradient matrix.
#
# @param ym Embedded coordinates.
# @param km Stiffness matrix.
# @return Gradient matrix.
stiff_to_grads <- function(ym, km) {
  gm <- matrix(0, nrow(ym), ncol(ym))
  for (i in 1:nrow(ym)) {
    disp <- sweep(-ym, 2, -ym[i, ]) #  matrix of y_ik - y_jk
    gm[i, ] <- apply(disp * km[, i], 2, sum) # row is sum_j (km_ji * disp)
  }
  gm
}
