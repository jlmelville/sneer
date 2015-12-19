# tricks.R
make_tricks <- function(early_exaggeration = TRUE, P_exaggeration = 4,
                        exaggeration_off_iter = 50, verbose = TRUE) {
  tricks <- list()
  if (early_exaggeration) {
    exaggeration_func <- function(inp, out, stiffness, opt, iter) {
      if (iter == 0) {
        inp$pm <- inp$pm * P_exaggeration
      }
      if (iter == exaggeration_off_iter) {
        if (verbose) {
          message("Exaggeration off at iter: ", iter)
        }
        inp$pm <- inp$pm / P_exaggeration
      }

      list(inp = inp)
    }
    tricks$exaggeration <- exaggeration_func
  }

  function(inp, out, stiffness, opt, iter) {
    for (name in names(tricks)) {
      result <- tricks[[name]](inp, out, stiffness, opt, iter)
      if (!is.null(result$inp)) {
        inp <- result$inp
      }
      if (!is.null(result$out)) {
        out <- result$out
      }
      if (!is.null(result$stiffness)) {
        stiffness <- result$stiffness
      }
      if (!is.null(result$opt)) {
        opt <- result$opt
      }
    }
    list(inp = inp, out = out, stiffness = stiffness, opt = opt)
  }
}
