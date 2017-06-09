# Miscellaneous small functions.

# Partial Function Application
#
# @param f Function to partially apply.
# @param ... params of \code{f} to apply.
# @return Partially applied version of \code{f}.
partial <- function(f, ...) {
  args <- list(...)
  function(...) {
    do.call(f, c(args, list(...)))
  }
}

# Clamp Numerical Values
#
# Values are truncated so that they lie within (\code{min_val, max_val}). In
# embedding this is used to prevent individual probabilities values getting
# too small and causing underflow or some other horrible explosion.
#
# @param x Matrix.
# @param min_val Minimum value allowed for any element in the matrix.
# @param max_val Maximum value allowed for any element in the matrix.
# @return Matrix with the clamped values.
clamp <- function(x, min_val = .Machine$double.eps, max_val = NULL) {
  x[x < min_val] <- min_val
  if (!is.null(max_val)) {
    x[x > max_val] <- max_val
  }
  x
}

# Clamp Scalar Numerical Value
#
# Value is truncated so that it lies within (\code{min, max}).
#
# @param x Value.
# @param min If \code{x} is smaller than this value, it will be truncated to
#  this value.
# @param max If \code{x} is larger than this value, it will be truncated to
#  this value.
# @return Clamped value.
sclamp <- function(x, min, max) {
  base::min(base::max(x, min), max)
}

# Length of a Vector (or Matrix)
#
# @param x Matrix.
# @return Length (2-norm) of the matrix.
length_vec <- function(x) {
  sqrt(sum(x ^ 2))
}

# Scale a Vector (or Matrix) to Length 1
#
# @param x Matrix.
# @return \code{x} with elements scaled such that its length equals 1.
normalize <- function(x) {
  x / (length_vec(x) + .Machine$double.eps)
}

# Relative Tolerance
#
# @param x real value.
# @param y real value.
# @return the relative tolerance between the two values.
reltol <- function(x, y) {
  abs(x - y) / min(abs(x), abs(y))
}

# Summarise Data Distribution
#
# @param vals Array or matrix of data.
# @param msg Label to identify the data summary.
summarize <- function(vals, msg = "") {
  if (methods::is(vals, "matrix")) {
    vals <- array(vals)
  }
  message(msg, ": ", paste(names(summary(vals)), ":",
                           summary(vals, digits = 4),
                           "|",
                           collapse = ""))
}

# Euclidean Distance Matrix
#
# Creates an Euclidean distance matrix with the type "\code{matrix}", rather
# than an  object of class "\code{dist}", which the \code{stats} function
# \code{dist} produces.
#
# @param xm Matrix of coordinates.
# @param min_dist Truncate any inter-point distances in \code{xm} less than
# this value.
# @return Distance matrix.
distance_matrix <- function(xm, min_dist = .Machine$double.eps) {
  if (!methods::is(xm, "dist")) {
    # a lot faster than as.matrix(dist(x)) - no effect on unit test results
    dm <- sqrt(clamp(coords_to_dist2(as.matrix(xm))))
  } else {
    dm <- as.matrix(xm)
  }
  dm <- clamp(dm, min_dist)
}

# Upper Triangle of a Matrix as a Vector
#
# Useful if you need to get all of the distances in a distance matrix, without
# including self-distances or double counting. However, all index information
# is lost.
#
# @param x Matrix.
# @return Vector of elements in the upper triangle of the matrix.
upper_tri <- function(x) {
  x[upper.tri(x)]
}

# Remove NULL Members from a List
#
# @param l List.
# @return List with NULL members removed.
# @examples
# \dontrun{
# mylist <- list(foo = "bar", cleesh = NULL, baz = "qux", nitfol = NULL)
# names(remove_nulls(mylist)) == c("foo", "baz")
# }
remove_nulls <- function(l) {
  l[!vapply(l, is.null, logical(1))]
}

# Dot Product
#
# Returns the dot product of two vectors. But will also treat two matrices
# of equal dimensions as if they were two vectors. For the purposes of an
# emedding, this allows the various matrices involved to keep their "natural"
# shape, i.e. the gradient can be calculated as an N x N matrix without having
# to convert it into a vector, as normally required in optimization methods.
#
# @param a A vector or matrix.
# @param b Another vector or matrix of the same dimensions as \code{a}.
# @return Dot product of \code{a} and \code{b}.
# @examples
# \dontrun{
#  a <- c(1, 2, 3, 4)
#  b <- c(5, 6, 7, 8)
#  dot(a, b) # 70
#  am <- matrix(a, nrow = 2) # turn them into matrices
#  bm <- matrix(b, nrow = 2)
#  dot(am, bm) # still 70
# }
dot <- function(a, b) {
  sum(a * b)
}

# Replace Members of a List
#
# Updates a list with a set of named arguments, adding to, or replacing the
# current contents.
#
# Given a list and some named arguments, the contents of the list will
# augmented by the named arguments. If the list already contains a value with
# a name in the named arguments, it will be replaced.
#
# @param l A list.
# @param ... Named arguments to add to the list.
# @return The list \code{l} with the contents of the rest of the function
# arguments added to it.
# @examples
# \dontrun{
#  nato <- list(b = "bravo", m = "moto", x = "xray")
#  lapd <- lreplace(nato,  b = "boy", m = "mary", s = "sam")
#  # lapd is list(b = "boy", m = "mary", x = "xray", s = "sam")
# }
lreplace <- function(l, ...) {
  varargs <- list(...)
  for (i in names(varargs)) {
    l[[i]] <- varargs[[i]]
  }
  l
}

# Looks at all the columns in a data frame, returning the name of the last
# column which is a factor or NULL if there are no factors present.
last_factor_column_name <- function(df) {
  factor_name <- NULL
  factor_names <- filter_column_names(df, is.factor)
  if (length(factor_names) > 0) {
    factor_name <- factor_names[length(factor_names)]
  }
  factor_name
}

# Looks at all the columns in a data frame, returning the name of the last
# column which contains colors or NULL if there are no factors present.
last_color_column_name <- function(df) {
  color_column_name <- NULL
  color_column_names <- filter_column_names(df, is_color_column)
  if (length(color_column_names) > 0) {
    color_column_name <- color_column_names[length(color_column_names)]
  }
  color_column_name
}

# returns TRUE if vector x consists of colors
is_color_column <- function(x) {
  !is.numeric(x) && all(is_color(x))
}

# Applies pred to each column in df and returns the names of each column that
# returns TRUE.
filter_column_names <- function(df, pred) {
  names(df)[(vapply(df, pred, logical(1)))]
}

# Given a vector of character types x, returns a vector of the same length,
# where each element is a boolean indicating if the element in x is a valid
# color.
# @note Taken from
# \url{http://stackoverflow.com/questions/13289009/check-if-character-string-is-a-valid-color-representation}
# @note numeric values are always seen as being valid colors!
# @examples
# is_color(c(NA, "black", "blackk", "1", "#00", "#000000", 1000))
#  <NA>   black  blackk       1     #00 #000000    1000
#  TRUE    TRUE   FALSE    TRUE   FALSE    TRUE    TRUE
is_color <- function(x) {
  vapply(x, function(X) {
    tryCatch(is.matrix(grDevices::col2rgb(X)),
             error = function(e) FALSE)
  }, logical(1))
}

# Initialize Embedding
#
# A convenience function which intializes an embedder with a small amount
# of data. Useful for interactive exploration of e.g. gradients.
#
# @param method Embedding method.
# @param xm Input data matrix.
# @param preprocess Preprocessing method.
# @param init_inp Input initializer.
# @param init_out Output initializer.
# @param opt Optimizer.
# @return A list containing:
#   \item{\code{inp}}{Initialized input}
#   \item{\code{out}}{Initialized output}
#   \item{\code{method}}{Initialized embedding method}
#   \item{\code{opt}}{Initialized optimizer}
#   \item{\code{report}}{Initialized report}
iembed <- function(method,
                   xm = datasets::iris[1:50, 1:4],
                   preprocess = make_preprocess(range_scale_matrix = TRUE,
                                   verbose = FALSE),
                   init_inp = inp_from_perp(perplexity = 20, verbose = FALSE),
                   init_out = out_from_PCA(verbose = FALSE),
                   opt = mize_grad_descent()
) {

  init_embed(xm, method, preprocess = preprocess,
             init_inp = init_inp,
             init_out = init_out,
             opt = opt)
}


vec_formatC <- function(v) {
  paste(Map(function(x) { formatC(x) }, v), collapse = ", ")
}
