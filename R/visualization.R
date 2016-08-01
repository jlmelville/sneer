# Functions to help with visualizing embedding during the optimization
# procedure.

#' Embedding Plot, Colored By Category
#'
#' Plots the embedded coordinates, with each point colored by a specified
#' color.
#'
#' @param coords Matrix of embedded coordinates, with as many rows as
#'  observations, and 2 columns.
#' @param categories Vector containing equivalent categories for each
#'  observation \code{coords}. Qualitative information which will
#'  be converted to a color. Each value will be applied to the equivalent
#'  row in \code{coords}. Can be a vector of levels or strings. If not provided,
#'  then all points are assumed to have the same category.
#' @param cex Size of the points.
#' @param palette Either the name of a Qualitative ColorBrewer palette to use
#'  for assigning colors to \code{categories}, \emph{or} the name of a palette
#'  function (e.g. \code{"rainbow"}). The palette function should be able to
#'  take a single numeric argument, the number of colors required. For some
#'  applicable functions, see the \code{Palettes} help page in the
#'  \code{grDevices} package (e.g. by running the \code{?rainbow} command).
#' @param as_text If \code{TRUE}, plot the category name rather than a point.
#'  Only useful if the category names are short, unique, and the data set has
#'  few observations. If no \code{categories} vector is provided, the category
#'  name is displayed as '+'.
#' @note Use of this function with ColorBrewer qualitative palette names
#' requires that the \code{RColorBrewer} package be installed.
#' @seealso
#' \link{embed_quant_plot}, to color the points using a numerical scale.
#'
#' More information on ColorBrewer is available at its website,
#'  \url{http://www.colorbrewer2.org}.
#' @export
#' @examples
#' \dontrun{
#' # Embed with PCA
#' pca_iris <- embed(iris, method = "pca", scale_type = "a", ret = c("dy"))
#' # Visualize the resulting embedding, colored by iris species, using the
#' # rainbow palette
#' embed_plot(pca_iris$coords, iris$Species, palette = "rainbow")
#'
#' # topo.colors palette
#' embed_plot(pca_iris$coords, iris$Species, palette = "topo.colors")
#'
#' # default palette needs RColorBrewer installed and loaded
#' library("RColorBrewer")
#' embed_plot(pca_iris$coords, iris$Species)
#'
#' # Use the "Dark2" ColorBrewer palette
#' embed_plot(pca_iris$coords, iris$Species, palette = "Dark2")
#'
#' # Can plot the category names instead of points, but looks bad if they're
#' # long (or the dataset is large)
#' embed_plot(pca_iris$coords, iris$Species, palette = "Dark2",
#'  cex = 0.5, as_text = TRUE)
#' }
embed_plot <- function(coords, categories = rep("+", nrow(coords)), cex = 1,
                       palette = "Set1", as_text = FALSE) {

  if (class(categories) != "factor") {
    categories <- as.factor(categories)
  }
  category_names <- unique(categories)
  ncolors <- length(category_names)

  if (palette %in% c("Set1", "Set2", "Set3", "Pastel1", "Pastel2", "Dark2")) {
    if (!requireNamespace("RColorBrewer", quietly = TRUE,
                          warn.conflicts = FALSE)) {
      stop("Palette '", palette, "' requires 'RColorBrewer' package")
    }
    colorPalette <- colorBrewerPalette(palette, ncolors)
  }
  else {
    palette_fn <- get(palette)
    if (is.null(palette_fn)) {
      stop("Unknown palette function '",palette,"'")
    }
    colorPalette <- palette_fn(ncolors)
  }

  if (as_text) {
    plot(coords, type = 'n')
  }
  else {
    plot(coords, pch = 20, cex = cex, col = colorPalette[categories])
  }

  if (as_text) {
    text(coords, labels = categories, cex = cex, col = colorPalette[categories])
  }
}

#' Embedding Plot, Numerical Color Scale
#'
#' Plots the embedded coordinates, using a ColorBrewer palette to color each
#' point according to a numerical vector.
#'
#' @note Use of this function requires that the \code{RColorBrewer} package be
#'  installed.
#' @param coords Matrix of embedded coordinates, with as many rows as
#'  observations, and 2 columns.
#' @param quant_vec Vector containing quantitative information which will
#'  be converted to a color. Each value will be applied to the equivalent
#'  row in \code{coords}.
#' @param name Name of the ColorBrewer palette to use for visualizing
#'  \code{quant_vec}.
#' @param num_colors Number of unique colors to map values in \code{quant_vec}
#'  to.
#' @param limits The range that the colors should map over. If not specified,
#'  then the range of \code{x}. This is useful if there is some external
#'  absolute scale that should be used.
#' @param cex Size of the points.
#' @param top If not \code{NULL}, only the specified number of points will be
#'  displayed, corresponding to those with the highest values in \code{vec},
#'  after sorting by decreasing order.
#' @seealso
#' More information on ColorBrewer is available at its website,
#' \url{http://www.colorbrewer2.org}.
#' @export
#' @examples
#' \dontrun{
#' tsne_iris <- embed(iris, method = "tsne", scale_type = "a", ret = c("dx", "dy", "deg"))
#' # how well is the 32 nearest neighborhood preserved for each point?
#' nbr_pres_32 <- nbr_pres(tsne_iris$dx, tsne_iris$dy, 32)
#' # visualize preservation, use absolute scale of 0-1 for colors.
#' embed_quant_plot(tsne_iris$coords, nbr_pres_32, limits = c(0, 1))
#'
#' # visualize 10 points with the hightest degree centrality
#' embed_quant_plot(tsne_iris$coords, tsne_iris$deg, top = 10)
#' }
embed_quant_plot <- function(coords, quant_vec, name = "Blues", num_colors = 15,
                             limits = NULL, cex = 0.5, top = NULL) {

  if (!requireNamespace("RColorBrewer", quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("embed_quant_plot function requires 'RColorBrewer' package")
  }

  if (!is.null(top)) {
    svec <- sort(quant_vec, decreasing = TRUE)
    quant_vec[quant_vec < svec[top]] <- 0
  }

  plot(coords, col = map2color(quant_vec, name = name, n = num_colors,
                               limits = limits),
       pch = 20, cex = 0.75)
}

# Map Numbers to Colors
#
# Maps a numeric vector to an equivalent set of colors based on the specified
# ColorBrewer palette. Use the diverging or sequential.
#
# Sequential palettes names:
#  Blues BuGn BuPu GnBu Greens Greys Oranges OrRd PuBu PuBuGn PuRd Purples
#  RdPu Reds YlGn YlGnBu YlOrBr YlOrRd
# Diverging palette names:
#  BrBG PiYG PRGn PuOr RdBu RdGy RdYlBu RdYlGn Spectral
#
# @note Use of this function requires that the \code{RColorBrewer} packages be
#  installed.
# @note This function is based off a Stack Overflow answer by user "Dave X":
#  \url{http://stackoverflow.com/a/18749392}
#
# @param x Numeric vector.
# @param name Name of the ColorBrewer palette.
# @param n Number of unique colors to map values in \code{x} to.
# @param limits The range that the colors should map over. If not specified,
#  then the range of \code{x}. This is useful if there is some external
#  absolute scale that should be used.
# @seealso
# More information on ColorBrewer is available at its website,
# \url{http://www.colorbrewer2.org}.
# @examples
# \dontrun{
# # Plot Iris dataset sepal width vs length, colored by petal length, using
# # 20 colors ranging from Purple to Green (PRGn):
# plot(iris[, c("Sepal.Length", "Sepal.Width")],
#  col = map2color(iris$Petal.Length, "PRGn", 20), pch = 20, cex = 1.5)
# }
map2color <- function(x, name = "Blues", n = 15, limits = NULL) {
  if (!requireNamespace("RColorBrewer", quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("map2color function requires 'RColorBrewer' package")
  }
  if (is.null(limits)) {
    limits <- range(x)
  }
  pal <- colorBrewerPalette(name, n)
  pal[findInterval(x, seq(limits[1], limits[2], length.out = length(pal) + 1),
                   all.inside = TRUE)]
}

# PCA Scores Plot
#
# Carries out PCA on a data frame (using the provided indexes) and then
# plots the first two scores, coloring the points with a user-specified
# label.
#
# @param df Data frame.
# @param pca_indexes Numeric vector containing the column indexes in the
#  data frame which should be used for the PCA calculation. Default is
#  \code{NULL}, in which case all numeric columns will be used.
# @param label_name name of the column that contains a value to be used
#  to color each point. Default is \code{NULL}, in which case the first
#  encountered factor column will be used.
# @param center If \code{TRUE}, mean-center the columns.
# @param scale If \code{TRUE}, scale the columns to have unit variance.
# @param cex Numeric \strong{c}haracter \strong{ex}pansion factor;
#  multiplied by \code{\link[graphics]{par}("cex")} yields the final
#  character size of the labels.
# @examples
# \dontrun{
#  # PCA on the scaled iris dataset, use the "Species" column to display
#  scores_plot(iris, 1:4, "Species", scale = TRUE)
#
#  # PCA on the s1k dataset, use all numeric values, and color by the only
#  # factor ("Label"). Change text size.
#  scores_plot(s1k, cex = 1)
# }
scores_plot <- function(df, pca_indexes = NULL, label_name = NULL,
                        center = TRUE, scale = FALSE,  cex = 0.5) {
  if (is.null(pca_indexes)) {
    pca_indexes <- sapply(df, is.numeric)
  }

  if (is.null(label_name)) {
    factor_names <- names(df)[(sapply(df, is.factor))]
    if (length(factor_names) == 0) {
      stop("Couldn't find a factor column in data frame to use for label")
    }
    else {
      label_name <- factor_names[1]
    }
  }

  if (is.null(df[[label_name]])) {
    stop("Data frame does not have a '",label_name,
         "' column for use as a label")
  }
  pca <- prcomp(df[, pca_indexes], retx = TRUE, center = center, scale. = scale)
  plot(pca$x[, 1:2], type = 'n')
  text(pca$x[, 1:2], labels = df[[label_name]], cex = cex,
       col = rainbow(length(levels(df[[label_name]])))[df[[label_name]]])
}

# PCA Scores Plot Using ggplot2 and ColorBrewer Palettes
#
# Carries out PCA on a dataset and then displays the first two scores in a
# scatterplot using ggplot2 and a color palette from RColorBrewer.
#
# @note Use of this function requires that the \code{ggplot2} and
# \code{RColorBrewer} packages be installed.
#
# @param df Data frame.
# @param pca_indexes Numeric vector containing the column indexes in the
#  data frame which should be used for the PCA calculation. Default is
#  \code{NULL}, in which case all numeric columns will be used.
# @param label_name name of the column that contains a value to be used
#  to color each point. Default is \code{NULL}, in which case the first
#  encountered factor column will be used.
# @param center If \code{TRUE}, mean-center the columns.
# @param scale If \code{TRUE}, scale the columns to have unit variance.
# @param size Size of the points.
# @param palette String giving the name of a ColorBrewer Palette. To see the
#  available palettes run the function
#  \code{RColorBrewer::display.brewer.all()}. Although subject to change, the
#  following palettes from the "qualitative" subset are suggested:
#  \itemize{
#    \item \code{Set1}
#    \item \code{Set2}
#    \item \code{Set3}
#    \item \code{Pastel1}
#    \item \code{Pastel2}
#    \item \code{Dark2}
#  }
# @param legend If \code{TRUE}, then the legend will be displayed. Set to
#  \code{FALSE} if there are a lot of separate categories that would appear
#  in the legend, which can result in the legend taking up more space than
#  the actual plot.
# @param legend_rows If non-null, then sets the number of rows to display
#  the legend items in. If the legend is taking up too much space, you may
#  want to experiment with setting the number of rows manually, rather than
#  just setting the \code{legend} parameter to \code{FALSE}.
# @seealso
# More information on ColorBrewer is available at its website,
# \url{http://www.colorbrewer2.org}.
# @examples
#  \dontrun{
#  # PCA on the scaled iris dataset, use the "Species" column to display
#  scores_qplot(iris, 1:4, "Species", scale = TRUE)
#
#  # PCA on s1k dataset, use all numeric indices and first factor (defaults)
#  # for coloring with the "Set3" palette, set point size to 2,
#  # and display three rows in the legend
#  scores_qplot(s1k, size = 2, palette = "Set3", legend_rows = 3)
# }
scores_qplot <- function(df, pca_indexes = NULL, label_name = NULL,
                         center = TRUE, scale = FALSE,
                         size = 1,
                         palette = "Set1",
                         legend = TRUE, legend_rows = NULL) {
  if (!requireNamespace("ggplot2", quietly = TRUE, warn.conflicts = FALSE)) {
    stop("scores_qplot function requires 'ggplot2' package")
  }
  if (!requireNamespace("RColorBrewer", quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("scores_qplot function requires 'RColorBrewer' package")
  }

  if (is.null(pca_indexes)) {
    pca_indexes <- sapply(df, is.numeric)
  }

  if (is.null(label_name)) {
    factor_names <- names(df)[(sapply(df, is.factor))]
    if (length(factor_names) == 0) {
      stop("Couldn't find a factor column in data frame to use for label")
    }
    else {
      label_name <- factor_names[1]
    }
  }

  if (is.null(df[[label_name]])) {
    stop("Data frame does not have a '",label_name,
         "' column for use as a label")
  }

  pca <- prcomp(df[, pca_indexes], retx = TRUE, center = center, scale. = scale)
  scatterqplot(df, x = pca$x[, 1], y = pca$x[, 2],
               label_name = label_name, size = size, palette = palette,
               x_label = "t1", y_label = "t2",
               legend = legend, legend_rows = legend_rows)
}

# Scatterplot Using ggplot2 and ColorBrewer Palettes
#
# Scatterplot using ggplot2 with coloring via categorical data and a palette
# from RColorBrewer.
#
# @note Use of this function requires that the \code{ggplot2} and
# \code{RColorBrewer} packages be installed.
#
# @param df Data frame.
# @param x Vector of x values to plot for the x-coordinate.
# @param y Vector of y values to plot for the y-coordinate.
# @param label_name name of the column that contains a value to be used
#  to color each point. Default is \code{NULL}, in which case the first
#  encountered factor column will be used.
# @param size Size of the points.
# @param palette String giving the name of a ColorBrewer Palette. To see the
#  available palettes run the function
#  \code{RColorBrewer::display.brewer.all()}. Although subject to change, the
#  following palettes from the "qualitative" subset are suggested:
#  \itemize{
#    \item \code{Set1}
#    \item \code{Set2}
#    \item \code{Set3}
#    \item \code{Pastel1}
#    \item \code{Pastel2}
#    \item \code{Dark2}
#  }
# @param x_label Label for the x-axis.
# @param y_label Label for the y-axis.
# @param legend If \code{TRUE}, then the legend will be displayed. Set to
#  \code{FALSE} if there are a lot of separate categories that would appear
#  in the legend, which can result in the legend taking up more space than
#  the actual plot.
# @param legend_rows If non-null, then sets the number of rows to display
#  the legend items in. If the legend is taking up too much space, you may
#  want to experiment with setting the number of rows manually, rather than
#  just setting the \code{legend} parameter to \code{FALSE}.
# @seealso
# More information on ColorBrewer is available at its website,
# \url{http://www.colorbrewer2.org}.
scatterqplot <- function(df, x, y, label_name = NULL, size = 1,
                         palette = "Set1", x_label = "x", y_label = "y",
                         legend = TRUE, legend_rows = NULL) {
  if (!requireNamespace("ggplot2", quietly = TRUE, warn.conflicts = FALSE)) {
    stop("scatterqplot function requires 'ggplot2' package")
  }
  if (!requireNamespace("RColorBrewer", quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("scatterqplot function requires 'RColorBrewer' package")
  }

  if (is.null(label_name)) {
    factor_names <- names(df)[(sapply(df, is.factor))]
    if (length(factor_names) == 0) {
      stop("Couldn't find a factor column in data frame to use for label")
    }
    else {
      label_name <- factor_names[1]
    }
  }

  if (is.null(df[[label_name]])) {
    stop("Data frame does not have a '",label_name,
         "' column for use as a label")
  }

  ncolors <- length(unique(df[[label_name]]))
  colorPalette <- colorBrewerPalette(palette, ncolors)

  score_plot <-
    ggplot2::qplot(x, y, data = df, colour = df[[label_name]], size = I(size)) +
    ggplot2::scale_color_manual(values = colorPalette, name = label_name) +
    ggplot2::theme(legend.position = "bottom") +
    ggplot2::labs(x = x_label, y = y_label)

  if (!is.null(legend_rows)) {
    score_plot <- score_plot +
      ggplot2::guides(color = ggplot2::guide_legend(nrow = legend_rows))
  }
  else if (!legend) {
    score_plot <- score_plot + ggplot2::theme(legend.position = "none")
  }
  print(score_plot)
}

# Embedding Plots
#
# Factory function for a plotting callback which can be used by the reporter
# function of an embedding to plot the current (two-dimensional) embedding.
#
# @param x Data frame containing label information for the embedded data.
# @param attr_name Name of the label column in \code{x}.
# @param label_fn Function with the signature \code{label_fn(labels)} where
# \code{labels} is a vector of labels for each point in the data set, and
# the function returns a vector of labels suitable for displaying in the plot
# (e.g. with each label truncated so that only the first three characters)
# will be used.
# @param mat_name The name of the matrix containing the embedded data in the
# output list \code{out} which will be passed to the plot function.
# @param cex Numeric \strong{c}haracter \strong{ex}pansion factor;
#   multiplied by \code{\link[graphics]{par}("cex")} yields the final
#   character size of the labels.
# @return Function which will take an output list, and produce a 2D plot of
# the embedding.
# @seealso \code{make_reporter} for how to use this function for
# configuring visualization of the progress of an embedding.
# @examples
# # For s1k dataset, plot 2D embedding with "Label" factor to identify each
# # point on the plot
# make_plot(s1k, "Label")
#
# # For iris dataset, plot 2D embedding with first two characters of the
# # "Species" factor to identify each point on the plot
# make_plot(iris, "Species", make_label(2))
#
# # Should be passed to the plot_fn argument of the reporter factory function:
# \dontrun{
#  embed_prob(reporter = make_reporter(report_every = 100,
#                                     normalize_cost = TRUE,
#                                     plot = make_plot(iris, "Species")),
#                                      ...)
# }
# @family sneer plot functions
make_plot <- function(x, attr_name,
                      label_fn = function(labels) {
                        labels
                      },
                      mat_name = "ym", cex = 1) {
  embedding_plot <- make_embedding_plot(x, attr_name, label_fn, cex = cex)
  function(out) {
    embedding_plot(out[[mat_name]])
  }
}

# Embedding Plot Using \code{graphics} Library
#
# Create a function which when invoked on a 2D matrix, plots the embedding
# with color-coded labels.
#
# @param x Data frame containing label information for the embedded data.
# @param attr_name Name of the label column in \code{x}.
# @param label_fn Function with the signature \code{label_fn(labels)} where
# \code{labels} is a vector of labels for each point in the data set. The
# function should return a vector of labels suitable for displaying in the
# plot.
# @param cex Numeric \strong{c}haracter \strong{ex}pansion factor;
#   multiplied by \code{\link[graphics]{par}("cex")} yields the final
#   character size of the labels.
# @return Function which will take an output list, and produce a 2D plot of
# the embedding.
# @seealso \code{make_label} for an example of a suitable
# argument to \code{label_fn}.
#
# @examples
# \dontrun{
# # Create a plot function for the Iris dataset
# iris_plot <- make_embedding_plot(iris, "Species")
#
# # PCA on iris
# pca_iris <- prcomp(iris[, 1:4], center = TRUE, retx = TRUE)
# # view the first two scores:
# iris_plot(pca_iris$x[, 1:2])
#
# # TSNE on iris
# tsne_iris <- embed_prob(iris[, 1:4])
# # view the TSNE embedding
# iris_plot(tsne_iris$ym)
#
# # Same plot, with smaller labels
# make_embedding_plot(iris, "Species", cex = 0.5)(tsne_iris$ym)
#}
make_embedding_plot <- function(x, attr_name, label_fn = NULL, cex = 1) {
  attr <- x[[attr_name]]
  uniq_attr <- sort(unique(attr))
  colors <- rainbow(length(uniq_attr))
  names(colors) <- uniq_attr
  if (!is.null(label_fn)) {
    labels <- label_fn(attr)
  } else {
    labels <- attr
  }
  function(ym) {
    plot(ym, type = "n", xlab = "D1", ylab = "D2")
    text(ym, labels = labels, col = colors[attr], cex = cex)
  }
}

# Labels for 2D Embedding Plot
#
# Even quite short labels can create a crowded looking embedding plot.
# Use this function to only print the first few characters of each label.
#
# @param num_label_chars The number of characters to plot from the label
# for each data point.
# @return a function which can be passed to the \code{label_fn} parameter
# of the \code{make_plot} function.
make_label <- function(num_label_chars = 1) {
  partial(substr, start = 0, stop = num_label_chars)
}

# Embedding Plot for Iris Dataset
#
# Creates a function which can be used to visualize embeddings of the iris
# dataset.
#
# Wrapper function for visualizing the iris data set. If embedding the iris
# data set, the result of invoking this can be passed to the \code{plot_fn}
# parameter of the \code{make_reporter} function.
#
# @param num_label_chars The number of characters to plot from the label
# for each data point.
# @return Function for plotting the embedded iris data set.
# @family sneer plot functions
make_iris_plot <- function(num_label_chars = 1) {
  make_plot(iris, "Species", make_label(num_label_chars))
}


# Embedding Plot Using \code{ggplot2} and \code{RColorBrewer} Library
#
# Creates a function which can be used to visualize embeddings from sneer
# output results for a particular dataset using ggplot2.
#
# @note Use of this function requires that the \code{ggplot2} and
# \code{RColorBrewer} packages be installed.
#
# @param df Data frame containing label information for the embedded data.
# @param attr_name Name of the label column in \code{df}.
# @param mat_name The name of the matrix containing the embedded data in the
#   output list \code{out} which will be passed to the plot function.
# @param size Size of the points.
# @param palette String giving the name of a ColorBrewer Palette. To see the
#  available palettes run the function
#  \code{RColorBrewer::display.brewer.all()}. Although subject to change, the
#  following palettes from the "qualitative" subset are suggested:
#  \itemize{
#    \item \code{Set1}
#    \item \code{Set2}
#    \item \code{Set3}
#    \item \code{Pastel1}
#    \item \code{Pastel2}
#    \item \code{Dark2}
#  }
# @param legend If \code{TRUE}, then the legend will be displayed. Set to
#  \code{FALSE} if there are a lot of separate categories that would appear
#  in the legend, which can result in the legend taking up more space than
#  the actual plot.
# @param legend_rows If non-null, then sets the number of rows to display
#  the legend items in. If the legend is taking up too much space, you may
#  want to experiment with setting the number of rows manually, rather than
#  just setting the \code{legend} parameter to \code{FALSE}.
# @return Function with signature \code{plot_fn(out)} where \code{out} is
# a return value from a sneer embedding function. On invocation, the
# data will be plotted.
# @seealso
# More information on ColorBrewer is available at its website,
# \url{http://www.colorbrewer2.org}.
#
# @examples
# \dontrun{
# mds_iris <- embed_dist(iris[, 1:4])
# iris_view <- make_qplot(iris, "Species")
# iris_view(mds_iris)
# }
make_qplot <- function(df, attr_name = "Label", mat_name = "ym", size = 1,
                       palette = "Set1",
                       legend = TRUE, legend_rows = NULL) {
  if (!requireNamespace("ggplot2", quietly = TRUE, warn.conflicts = FALSE)) {
    stop("make_qplot function requires 'ggplot2' package")
  }
  if (!requireNamespace("RColorBrewer", quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("make_qplot function requires 'RColorBrewer' package")
  }
  embedding_plot <- make_embedding_qplot(df, attr_name, size = size,
                                         palette = palette, legend = legend,
                                         legend_rows = legend_rows)

  function(out) {
      embedding_plot(out[[mat_name]])
  }
}

# Embedded Coordinates Plot Using \code{ggplot2} Library
#
# Creates a ggplot2 function for embeddings of a given dataset with coloring
# from RColorBrewer palettes.
#
# @note Use of this function requires that the \code{ggplot2} and
# \code{RColorBrewer} packages be installed.
#
# @param df Data frame containing label information for the embedded data.
# @param attr_name Name of the label column in \code{df}.
# @param size Size of the points.
# @param palette String giving the name of a ColorBrewer Palette. To see the
#  available palettes run the function
#  \code{RColorBrewer::display.brewer.all()}. Although subject to change, the
#  following palettes from the "qualitative" subset are suggested:
#  \itemize{
#    \item \code{Set1}
#    \item \code{Set2}
#    \item \code{Set3}
#    \item \code{Pastel1}
#    \item \code{Pastel2}
#    \item \code{Dark2}
#  }
# @param legend If \code{TRUE}, then the legend will be displayed. Set to
#  \code{FALSE} if there are a lot of separate categories that would appear
#  in the legend, which can result in the legend taking up more space than
#  the actual plot.
# @param legend_rows If non-null, then sets the number of rows to display
#  the legend items in. If the legend is taking up too much space, you may
#  want to experiment with setting the number of rows manually, rather than
#  just setting the \code{legend} parameter to \code{FALSE}.
# @return Function with signature \code{plot_fn(ym)} where \code{ym} is a
# 2D matrix of embedded coordinates of data set \code{x}. On invocation, the
# data will be plotted.
# @seealso
# More information on ColorBrewer is available at its website,
# \url{http://www.colorbrewer2.org}.
#
# @examples
# \dontrun{
# # make two different embeddings of the iris dataset
# prcomp_iris <- prcomp(iris[, 1:4], scale. = TRUE, retx = TRUE)
# mds_iris <- embed_dist(iris[, 1:4], method = mmds(eps = 1e-4),
#                        opt = bold_nag(),
#                        init_out = out_from_matrix(prcomp_iris$x[, 1:2]),
#                        max_iter = 40)
# iris_view <- make_embedding_qplot(iris, "Species")
# iris_view(prcomp_iris$x)
# iris_view(mds_iris$ym)
# }
make_embedding_qplot <- function(df, attr_name = "Label", size = 1,
                                 palette = "Set1",
                                 legend = TRUE, legend_rows = NULL) {
  if (!requireNamespace("ggplot2", quietly = TRUE, warn.conflicts = FALSE)) {
    stop("make_embedding_qplot function requires 'ggplot2' package")
  }
  if (!requireNamespace("RColorBrewer", quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("make_embedding_qplot function requires 'RColorBrewer' package")
  }

  function(ym) {
    colnames(ym) <- NULL
    scatterqplot(df, x = ym[, 1], y = ym[, 2], label_name = attr_name,
                 size = size, palette = palette,
                 x_label = "D1", y_label = "D2",
                 legend = legend, legend_rows = legend_rows)
  }
}

# Interpolated Color Brewer Palette
#
# Returns a vector of colors from the specified palette, interpolated if the
# number of requested colors is larger than the number of colors in the
# palette. Sequential and Diverging palettes are suitable for numerical scales.
# The Qualitiative palettes are intended for categorical values.
#
# Sequential palettes names:
#  Blues BuGn BuPu GnBu Greens Greys Oranges OrRd PuBu PuBuGn PuRd Purples
#  RdPu Reds YlGn YlGnBu YlOrBr YlOrRd
# Diverging palette names:
#  BrBG PiYG PRGn PuOr RdBu RdGy RdYlBu RdYlGn Spectral
# Qualitative:
#  Accent Dark2 Paired	Pastel1 Pastel2	Set1 Set2	Set3
# @param name Name of the palette.
# @param n Number of colors desired.
# @return Vector of \code{n} colors from the palette.
# @seealso
# More information on ColorBrewer is available at its website,
# \url{http://www.colorbrewer2.org}.
colorBrewerPalette <- function(name, n) {
  if (!requireNamespace("RColorBrewer", quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("colorBrewerPalette function requires 'RColorBrewer' package")
  }
  colorRampPalette(
    RColorBrewer::brewer.pal(
      RColorBrewer::brewer.pal.info[name,]$maxcolors, name))(n)
}

