# Gradient Calculation

stiffness <- function(method, inp, out) {
  if (!is.null(method[["stiffness"]])) {
    km <- method$stiffness$fn(method, inp, out)
  }
  else {
    km <- method$stiffness_fn(method, inp, out)
  }
  diag(km) <- 0
  km
}

# Calculates an embedding gradient where distances are not transformed before
# weighting (e.g. metric MDS, Sammon map)
# As luck would have it, this is identical to the squared-distance gradient,
# because in the distance-based case, the factor of two arises from
# kij = dc/dij = dc/dji = kji so kji + kij = 2kij and we've accounted for
# 1/dij in the stiffness expressions.
dist_gradient <- function() {
  list(
    fn = function(inp, out, method, mat_name = "ym") {
      km <- stiffness(method, inp, out)
      # multiply K by 2
      gm <- stiff_to_grads(out[[mat_name]], 2 * km)
      list(km = km, gm = gm)
    },
    name = "dist"
  )
}

# Calculates an embedding gradient where distances are squared (i.e. SNE-like
# methods)
dist2_gradient <- function() {
   list(
     fn = function(inp, out, method, mat_name = "ym") {
        km <- stiffness(method, inp, out)
        # multiply K by 2
        gm <- stiff_to_grads(out[[mat_name]], 2 * km)
        list(km = km, gm = gm)
    },
    name = "dist2"
   )
}
# Calculates an embedding gradient for methods with a defined transformation
# step
# NB Better to use a specific gradient function for e.g. squared distance
generic_gradient <- function() {
  list(
    fn = function(inp, out, method, mat_name = "ym") {
      if (!is.null(method$transform)) {
        df_dd <- method$transform$gr(inp, out, method) / (out$dm + method$eps)
      }
      else {
        df_dd <- 1
      }

      km <- stiffness(method, inp, out)
      gm <- stiff_to_grads(out[[mat_name]], km * df_dd)
      list(km = km, gm = gm)
    },
    name = "generic",
    keep = c("dm")
  )
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


