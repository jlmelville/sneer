# Preprocessing functions for the input data.

#' Create preprocessing callback
#'
#' Creates a callback for use in the embedding routine. Preprocessing will be
#' applied to the input coordinates (if provided) before embedding-specific
#' input initialization occurs (e.g. calculation of distances and
#' probabilities). A variety of common preprocessing options are available.
#' The range scaling and auto scaling options are mutually exclusive, but can
#' be combined with whitening if required. No preprocessing will be carried
#' out if the input data is already a distance matrix.
#'
#' @param range_scale_matrix If \code{TRUE}, input coordinates will be scaled
#' so that elements are between (\code{rmin}, \code{rmax}).
#' @param range_scale If \code{TRUE}, input coordinates will be scaled
#' @param rmin Minimum value if using \code{range_scale_matrix} and
#' \code{range_scale}.
#' @param rmax Maximum value if using \code{range_scale_matrix} and
#' \code{range_scale}.
#' @param auto_scale If \code{TRUE}, input coordinates will be centered and
#' scaled so that each column has a mean of 0 and a standard deviation of 1.
#' @param whiten If \code{TRUE}, input coordinates will be whitened.
#' @param zwhiten If \code{TRUE}, input coordinates will be whitened using the
#' ZCA transform.
#' @param initial_dims Number of components to use in the whitening preprocess.
#' Only use if \code{whiten} or \code{zwhiten} is \code{TRUE}.
#' @param preprocess_fn Custom preprocessing function to be applied. Should have
#' the signature \code{xm} where \code{xm} is the input coordinate matrix, and
#' returns the processed coordinate matrix.
#' @param verbose If \code{TRUE}, log information about preprocessing.
#' @return Function with signature \code{fn(xm)} where \code{xm} is the input
#' coordinate matrix. Function returns the coordinate matrix after applying
#' the specificed preprocessing.
make_preprocess <- function(range_scale_matrix = FALSE, range_scale = FALSE,
                            rmin = 0, rmax = 1, auto_scale = FALSE,
                            whiten = FALSE, zwhiten = FALSE, initial_dims = 30,
                            preprocess_fn = NULL, verbose = TRUE) {
  preprocess <- list()

  preprocess$filter_zero_var_cols <- function(xm) {
    old_ncols <- ncol(xm)
    xm <- varfilter(xm)
    new_ncols <- ncol(xm)
    if (verbose) {
      message("Removed ", old_ncols - new_ncols, " columns, ",
              new_ncols, " remaining")
    }
    xm
  }

  if (auto_scale) {
    preprocess$auto_scale <- function(xm) {
      if (verbose) {
        message("Auto scaling")
      }
      scale(xm)
    }
  }

  if (range_scale_matrix) {
    preprocess$range_scale_matrix <- function(xm) {
      if (verbose) {
        message("Range scaling matrix in (", rmin, ", ", rmax, ")")
      }
      range_scale_matrix(xm, rmin = rmin, rmax = rmax)
    }
  }

  if (range_scale) {
    preprocess$range_scale <- function(xm) {
      if (verbose) {
        message("Range scaling columns in (", rmin, ", ", rmax, ")")
      }
      range_scale(xm, rmin = rmin, rmax = rmax)
    }
  }

  if (whiten) {
    preprocess$whiten <- function(xm) {
      initial_dims <- min(initial_dims, ncol(xm))
      if (verbose) {
        message("PCA whitening with ", initial_dims, " components")
      }
      whiten(xm, ncomp = initial_dims)
    }
  }

  if (zwhiten) {
    preprocess$zwhiten <- function(xm) {
      initial_dims <- min(initial_dims, ncol(xm))
      if (verbose) {
        message("ZCA Whitening with ", initial_dims, " components")
      }
      whiten(xm, ncomp = initial_dims, zca = TRUE)
    }
  }

  if (!is.null(preprocess_fn)) {
    preprocess$preprocess_fn <- preprocess_fn
  }

  function(xm) {
    if (class(xm) != "dist") {
      xm <- as.matrix(xm)
      for (name in names(preprocess)) {
        xm <- preprocess[[name]](xm)
      }
    }
    xm
  }
}


#' Range scales numeric matrix.
#'
#' Elements are scaled so that they are within (\code{rmin}, \code{rmax}).
#'
#' @param xm Matrix to range scale.
#' @param rmin Minimum value in the scaled matrix.
#' @param rmax Maximum value in the scaled matrix.
#' @return Range scaled data.
range_scale_matrix <- function(xm, rmin = 0, rmax = 1) {
  xmin <- min(xm)
  xmax <- max(xm)
  xrange <- xmax - xmin
  rrange <- rmax - rmin

  ((xm - xmin) * (rrange / xrange)) + rmin
}

#' Range scales columns of numeric matrix.
#'
#' Values are scaled so that for each column, values are within
#' (\code{min}, \code{max}).
#'
#' @param xm Matrix to range scale.
#' @param rmin Minimum value per column in the scaled matrix.
#' @param rmax Maximum value per column in the scaled matrix.
#' @return Range scaled matrix.
range_scale <- function(xm, rmin = 0, rmax = 1) {
  xmin <- apply(xm, 2, min)

  xmax <- apply(xm, 2, max)
  xrange <- xmax - xmin
  rrange <- rmax - rmin

  xm <- sweep(xm, 2, xmin)
  xm <- sweep(xm, 2, rrange / xrange, "*")
  sweep(xm, 2, rmin, "+")
}

#' Data whitening.

#' Whitens the data so that the covariance matrix is the identity matrix
#' (variances of the data are all one and the covariances are zero).
#'
#' Whitening consists of performing PCA on the centered (and optionally
#' scaled to unit standard deviation column) input data, optionally removing
#' eigenvectors corresponding to the smallest eigenvectors, and then rotating
#' the data to decorrelate it. Each component is additionally scaled by the
#' square root of the corresponding eigenvalue so that the variances of the
#' rotated solution are all equal to one.
#'
#' Because whitening transformations are not unique, a further transformation
#' may be carried out on the PCA whitened solution to produce the Zero-Phase
#' Component Analysis (ZCA) whitening, which uses the rotation that best
#' reproduces the input data in a least-squares sense, while maintaining the
#' whitened properties. This may be useful for some image datasets where a
#' whitening which preserves the original data structure as much as possible
#' may be desirable.
#'
#' If ZCA is used as the whitening transform then the returned whitened data
#' will have the same dimensionality as the input data xm, even if a value for
#' \code{ncomp} was also provided. This is simply due to the form of the
#' matrix multiplication that produces the ZCA transform. Additionally, ZCA
#' is most often applied to images where the whitened data is intended to be
#' visualized in the same way as the input data and so the dimensionality would
#' need to be identical anyway. Nonetheless, the dimensionality reduction has
#' still occurred, with the ZCA simply reconstructing the truncated data back
#' into the original (albeit whitened) space.
#'
#' @param xm Matrix to whiten.
#' @param scale If \code{TRUE}, then the centered columns of xm are divided by
#' their standard deviations.
#' @param zca If \code{TRUE}, then apply the Zero-Phase Component Analysis
#' (ZCA) whitening. If ZCA is used then the returned whitened data will have
#' as many columns as was in the input data, even if a value of \code{ncomp}
#' was specified.
#' @param ncomp Number of components to keep in a reduced dimension
#' representation of the data.
#' @param epsilon Regularization parameter to apply to eigenvalues, to avoid
#' numerical instabilities with eigenvalues close to zero. Values of 1.e-5 to
#' 0.1 seem common.
#' @param verbose If \code{TRUE}, then debug messages will be logged.
#' @return Whitened matrix.
whiten <- function(xm, scale = FALSE, zca = FALSE, ncomp = min(dim(xm)),
                   epsilon = 1.e-5, verbose = TRUE) {
  xm <- scale(xm, scale = scale)
  n <- nrow(xm)
  # This implementation does SVD directly on xm
  # Uses La.svd so that the loadings V are already transposed and we
  # only need to untranspose if ZCA is asked for.
  svdx <- La.svd(xm, nu = 0, nv = ncomp)
  dm <- diag(sqrt(n - 1) / (svdx$d[1:ncomp] + epsilon), nrow = ncomp)
  vm <- svdx$vt[1:ncomp, , drop = FALSE]
  wm <- dm %*% vm
  if (zca) {
    wm <- t(vm) %*% wm
  }
  xm %*% t(wm)
}

#' Filter low variance columns.
#'
#' Removes columns from the data with variance lower than a threshold. Some
#' techniques can't handle zero variance columns.
#'
#' @param xm Matrix to filter.
#' @param minvar Minimum variance allowed for a column.
#' @return Data with all columns with a variance lower than \code{minvar}
#' removed.
varfilter <- function(xm, minvar = 0.0) {
  vars <- apply(xm, 2, var)
  xm[, vars > minvar, drop = FALSE]
}


