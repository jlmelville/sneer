# util.R

clamp <- function(x, min_val = .Machine$double.eps, max_val = NULL) {
  x[x < min_val] <- min_val
  if (!is.null(max_val)) {
    x[x > max_val] <- max_val
  }
  x
}

# the length of a vector (or matrix)
length_vec <- function(x) {
  sqrt(sum(x ^ 2))
}

# scale a vector (or matrix) to length 1
normalize <- function(x) {
  x / length_vec(x)
}

reltol <- function(x, y) {
  abs(x - y) / min(abs(x), abs(y))
}


# logging
summarise <- function(vals, msg = "") {
  if (class(vals) == "matrix")
    vals <- array(vals)
  message(msg, ": ", paste(names(summary(vals)), ":", summary(vals), "|",
                           collapse = ""))
}

# return a vector of the distances of a matrix (or distance matrix)
distance_matrix <- function(xm, min_dist = .Machine$double.eps,
                            as_upper = FALSE) {
  if (class(xm) != "dist") {
    dm <- as.matrix(dist(xm))
  } else {
    dm <- as.matrix(xm)
  }
  dm <- clamp(dm, min_dist)
  if (as_upper) {
    dm <- upper_tri(dm)
  }
  dm
}

upper_tri <- function(x) {
  x[upper.tri(x)]
}


# only used in logging
shannon <- function(pm) {
  -sum((apply(pm * log(pm + .Machine$double.eps), 1, sum)) / sum(pm))
}
