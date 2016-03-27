#' R Optimizer
#'
#' Function to create an optimizer, using the R general-purpose optimization
#' methods.
#'
#' This uses the optimization methods in \code{\link[stats]{optim}}, notably
#' the conjugate gradient method (used in the NeRV paper), and the low-memory
#' approximation to BFGS (used in the multiscale JSE paper).
#'
#' Their only downside is that you have much less control over their inner
#' workings, and they run synchronously until the specified number of
#' iterations, given by the \code{batch_iter} parameter, has been run
#' (or convergence causes the routine to exit early). During those iterations,
#' the rest of the embedding parameters (e.g. tricks and reports) won't get
#' a chance to run.
#'
#' However, each call to the R optimizer is only treated as one step of
#' optimization from the point of view of the embedding algorithm. As a result,
#' you should be aware that if you swap a home-grown optimizer
#' created by \code{\link{make_opt}} with one using \code{ropt} and
#' \code{batch_iter = 10}, if you don't change any of the other parameters,
#' (most pertinently, the value of \code{max_iter}), your optimizer may run for
#' much longer (maybe ten times longer) than you were expecting.
#'
#' This might be a bit annoying, but it's probably better than the alternative
#' which might seem more natural: increment the number of embedding iterations
#' by the size of \code{batch_iter} after each call of the R optimizer. The
#' problem with this approach is that from the point of view of all the other
#' components of the embedding algorithm, the iteration number jumped from e.g.
#' 10 to 20, without any value in between. This would be bad news if you had
#' set up a "trick" to reduce the input probability perplexity of an
#' embedding by reducing it at iterations 12, 14 and 16. The trick part of the
#' embedding never sees the iteration at these values, so they never run.
#' Similarly, if you set the report to run every 25 steps, you would fail
#' to run the report half the time.
#'
#' DO NOT just set \code{batch_iter} to 1 to try and get round this problem.
#' The R optimizer only retains the memory of the current optimization within
#' each batch. As both CG and L-BFGS methods rely on building up a "memory" of
#' previous descent directions, setting the \code{batch_iter} to too low a
#' value will reduce their behavior to steepest descent. A \code{batch_iter} of
#' 15-25 seems adequate.
#'
#' @param method The optimization method to be used.
#' @param batch_iter Number of steps of minimization to carry out per invocation
#'  of the optimizer.
#' @param ... Other options to pass to the \code{control} parameter of the
#'   underlying \code{\link[stats]{optim}} function.
#' @return Optimizer using R general-purpose optimization methods.
#' @seealso \code{\link[stats]{optim}} for more details on what options
#'  can be passed.
#' @examples
#' # Conjugate Gradient with the Polak-Ribiere method. 20 steps of optimization.
#' ropt(method = "CG", type = 2, batch_iter = 20)
#'
#' # low-memory version of BFGS. 15 steps of optimization per invocation.
#' ropt(method = "L-BFGS-B", batch_iter = 15)
#'
#' \dontrun{
#' # Should be passed to the opt argument of an embedding function.
#' # Total number of iterations is 100, so the R optimizer will be invoked
#' # 10 times, each time running up to 10 iterations. The reporter will
#' # report after every 20 iterations of the embedding function.
#'  embed_prob(opt = ropt(method = "L-BFGS-B", batch_iter = 10),
#'             reporter = make_reporter(..., report_every = 20),
#'             max_iter = 100)
#'
#' }
#' @family sneer optimization methods
#' @export
ropt <- function(method = "CG", batch_iter = 20, ...) {
  list(
    mat_name = "ym",
    optimize_step = ropt_step,
    method = method,
    control_params = c(maxit = batch_iter, ...),
    gradient = classical_gradient()
  )
}

#' One Round of Optimization
#'
#' @param opt Optimizer
#' @param method Embedding method.
#' @param inp Input data.
#' @param out Output data.
#' @param iter Iteration number.
#' @return List consisting of:
#'   \item{\code{opt}}{Updated optimizer.}
#'   \item{\code{inp}}{Updated input.}
#'   \item{\code{out}}{Updated output.}
ropt_step <- function(opt, method, inp, out, iter) {
  fr <- make_optim_f(opt, inp, method, iter)
  grr <- make_optim_g(opt, inp, method, iter)

  par <- mat_to_par(out$ym)

  result <- optim(par = par, fn = fr, gr = grr, method = opt$method,
                  control = opt$control_params)

  if (!is.null(inp$xm)) {
    nr <- nrow(inp$xm)
  }
  else {
    nr <- nrow(inp$dm)
  }
  out <- par_to_out(result$par, opt, inp, method, nr)

  list(opt = opt, inp = inp, out = out)
}

#' Convert Cost Function
#'
#' This function takes the current state of the embedding and returns a
#' cost function which can be used by \code{\link[stats]{optim}} as the
#' \code{fn} parameter.
#'
#' @param opt Optimizer
#' @param method Embedding method.
#' @param inp Input data.
#' @param iter Iteration number.
#' @return Cost function taking a vector of parameters and returning the scalar
#'  cost.
make_optim_f <- function(opt, inp, method, iter) {
  if (!is.null(inp$xm)) {
    nr <- nrow(inp$xm)
  }
  else {
    nr <- nrow(inp$dm)
  }
  function(par) {
    out <- par_to_out(par, opt, inp, method, nr)
    calculate_cost(method, inp, out)
  }
}

#' Convert Gradient Function
#'
#' This function takes the current state of the embedding and returns a
#' gradient function which can be used by \code{\link[stats]{optim}} as the
#' \code{gr} parameter.
#'
#' @param opt Optimizer
#' @param method Embedding method.
#' @param inp Input data.
#' @param iter Iteration number.
#' @return Gradient function taking a vector of parameters and returning
#'  a vector of gradient values.
make_optim_g <- function(opt, inp, method, iter) {
  if (!is.null(inp$xm)) {
    nr <- nrow(inp$xm)
  }
  else {
    nr <- nrow(inp$dm)
  }
  function(par) {
    out <- par_to_out(par, opt, inp, method, nr)
    gm <- opt$gradient$calculate(opt, inp, out, method, iter)$gm
    mat_to_par(gm)
  }
}

#' Convert 1D Parameter to Sneer Output
#'
#' This function converts the 1D parameter format expected by
#' \code{\link[stats]{optim}} into the matrix format used by sneer.
#'
#' @param par Vector of embedded coordinates.
#' @param opt Optimizer
#' @param inp Input data.
#' @param method Embedding method.
#' @param nrow Number of rows in the sneer output matrix.
#' @return Output data with coordinates converted from \code{par}.
par_to_out <- function(par, opt, inp, method, nrow) {
  ym <- matrix(par, nrow = nrow)
  set_solution(inp, ym, method, mat_name = opt$mat_name)
}

#' Convert Matrix to 1D Parameter Vector
#'
#' This function takes a matrix used by sneer internally (e.g. output
#' coordinates or gradient matrix) and converts into a 1D vector, as used
#' by \code{\link[stats]{optim}}. The matrix is converted columnwise.
#'
#' @param mat Matrix to convert.
#' @return Matrix in vector form.
mat_to_par <- function(mat) {
  as.vector(mat)
}
