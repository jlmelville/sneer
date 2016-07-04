# Code that implements the Spectral Direction method
# @references
# Vladymyrov, M., & Carreira-Perpiñán, M. A. (2012).
# Partial-Hessian Strategies for Fast Learning of Nonlinear Embeddings.
# In \emph{Proceedings of the 29th International Conference on Machine Learning (ICML-12)}
# (pp. 345-352).
#
# The Carreira-Perpiñán group has a page dedicated to available software,
# which includes a superior Matlab implementation, including sparse matrix
# support (which is vital for the method's efficiency) in Matlab:
# http://faculty.ucmerced.edu/mcarreira-perpinan/software.html

# Spectral Directions Optimizer
#
# Function to create an optimizer that uses the Spectral Directions optimizer
# of Vladymyrov and Carreira-Perpiñán.
#
# This optimizer relies on the input probability matrix being symmetric.
# Thus it can only be used with embedding methods that use "joint" probability
# types. This rules out methods such as ASNE, NeRV and JSE, although the
# symmetric versions will work with it.
#
# Also, be aware that because sneer does not work with sparse matrices,
# spectral directions may not work well with large datasets: it requires a
# Cholesky decomposition of the input probability matrix, which is O(N^3) in
# complexity.
#
# Also, use of the More-Thuente or Rasmussen line search methods requires
# installing and loading the 'rcgmin' project from
# \url{https://github.com/jlmelville/rcgmin}. The bold driver algorithm works
# surprisingly well (surprising to me, anyway) with it, though.
#
# @seealso
# A Matlab impementation by the original authors:
# http://faculty.ucmerced.edu/mcarreira-perpinan/software.html
#
# @param line_search Type of line search to use: \code{"bold"} for Bold Driver,
#  \code{"back"} for backstepping, \code{"mt"} for the method of More-Thuente,
#  and \code{"r"} for that of Rasmussen. The last two require the rcgmin
#  package to be loaded.
# @param c1 Constant used in sufficient decrease condition. Should take a value
#   between 0 and 1.
# @param c2 Constant used in curvature condition. Should take a value between
#   \code{c1} and 1.
# @return Optimizer.
optim_spectral <- function(line_search = "mt", c1 = 1e-4, c2 = 0.1) {

  if (line_search == "mt") {
    if (!requireNamespace("rcgmin",
                          quietly = TRUE,
                          warn.conflicts = FALSE)) {
      stop("Using More-Thuente line search requires 'rcgmin' package")
    }
    step_size <- more_thuente_ls(c1 = c1, c2 = c2)
  }
  else if (line_search == "r") {
    if (!requireNamespace("rcgmin",
                          quietly = TRUE,
                          warn.conflicts = FALSE)) {
      stop("Using Rasmussen line search requires 'rcgmin' package")
    }
    step_size <- rasmussen_ls(c1 = c1, c2 = c2)
  }
  else if (line_search == "bold") {
    step_size <- bold_driver()
  }
  else if (line_search == "back") {
    step_size <- backtracking(c1 = c1)
  }
  make_opt(gradient = classical_gradient(),
           direction = spectral_direction(),
           step_size = step_size,
           update = no_momentum(),
           normalize_direction = FALSE)
}

# Spectral Direction
#
# Factory function for creating an optimizer direction method.
#
# Creates a spectral direction method for use by the optimizer.
#
# @seealso The return value of this function is intended for internal use of
# the sneer framework only. See \code{optimization_direction_interface}
# for details on the functions and values defined for this method.
#
# @return Spectral Direction method.
# @examples
# # Use as part of the make_opt function for configuring an optimizer's
# # direction method:
# make_opt(direction = spectral_direction())
# @family sneer optimization direction methods
spectral_direction <- function() {
  list(
    init = function(opt, inp, out, method) {

      dm <- diag(indegree_centrality(inp$pm))
      # Graph Laplacian , L+ = D+ - W+
      lm <- dm - inp$pm
      dlm <- diag(dm)
      # enforce positive definiteness by adding a small number
      mu <- min(dlm[dlm > 0]) * 1e-10
      # Cholesky decomposition of graph Laplacian
      opt$rm <- chol(4 * (lm + mu))
      opt
    },

    calculate = function(opt, inp, out, method, iter) {
      # Solve Bp = -g for p where B is the graph laplacian and g is the gradient
      # we already decomposed B into R'R by the Cholesky decomposition
      # Now we can solve by:
      # R'y = -g solve for y by forward solve
      # Rp = y solve for p by back solve
      y <- forwardsolve(t(opt$rm), -opt$gm)
      opt$direction$value <- backsolve(opt$rm, y)

      list(opt = opt)
    }
  )
}
