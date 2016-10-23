# Functions to help with visualizing embedding during the optimization
# procedure.

#' Embedding Plot, Colored By Category
#'
#' Plots the embedded coordinates, with each point colored by a specified
#' color.
#'
#' The \code{x} argument can be used to provide a suitable vector of colors
#' from either a data frame or vector.
#'
#' If a data frame is provided, then a vector of colors will be looked for. If
#' it's present, it will be used as the \code{colors} argument directly.
#' Otherwise, a factor column will be looked for, and each level will be mapped
#' to a different color. Otherwise, one color will be used for each point. If
#' more than one column of a type is found in the data frame, the last one
#' encountered is used.
#'
#' If a vector is provided, a similar procedure to the data frame is used when
#' mapping from its content to a vector of colors. Additionally, a numeric vector
#' can be provided, which will be linearly mapped to a color scheme.
#'
#' @param coords Matrix of embedded coordinates, with as many rows as
#'  observations, and 2 columns.
#' @param colors Vector containing colors for each coordinate.
#' @param x Either a data frame or a column that can be used to derive a
#'  suitable vector of colors. Ignored if \code{colors} is provided.
#' @param palette Name of a ColorBrewer palette to use for assigning colors to
#'  \code{categories}. Probably should be one of the "Qualitative" set. Ignored
#'  if the \code{col_ramp} parameter is supplied.
#' @param col_ramp Color ramp function, accepting an integer n as an argument
#'  and returning n colors. For some applicable functions, see the
#'  \code{Palettes} help page in the \code{grDevices} package (e.g. by running
#'  the \code{?rainbow} command).
#' @param num_colors Number of unique colors to map to from \code{x}, if
#'  \code{x} is a numeric vector. Otherwise ignored.
#' @param limits The range that the colors should map over when mapping from a
#'  numeric vector. If not specified, then the range of \code{x}. This is useful
#'  if there is some external absolute scale that should be used. Ignored if
#'  \code{x} is not a numeric vector.
#' @param top If not \code{NULL}, only the specified number of points will be
#'  displayed, corresponding to those with the highest values in \code{vec},
#'  after sorting by decreasing order.
#'  num_colors = 15, limits = NULL, top = NULL,
#' @param cex Size of the points. Ignored if \code{as_text} is \code{TRUE}.
#' @param text Vector of label text to display instead of a point. If the labels
#'  are long or the data set is large, this is unlikely to be very legible, but
#'  is occasionally useful.
#' @note Use of this function with ColorBrewer qualitative palette names
#' requires that the \code{RColorBrewer} package be installed.
#'
#' More information on ColorBrewer is available at its website,
#'  \url{http://www.colorbrewer2.org}.
#' @export
#' @examples
#' \dontrun{
#' # Embed with PCA
#' pca_iris <- sneer(iris, method = "pca", scale_type = "a", ret = c("dy"))
#' # Visualize the resulting embedding, colored by iris species, using the
#' # rainbow palette
#' embed_plot(pca_iris$coords, x = iris$Species, col_ramp = rainbow)
#'
#' # topo.colors palette
#' embed_plot(pca_iris$coords, x = iris$Species, col_ramp = topo.colors)
#'
#' # Pass in data frame and it will use any factor column it finds
#' embed_plot(pca_iris$coords, x = iris)
#'
#'#' library("RColorBrewer")
#' # Use the "Dark2" ColorBrewer palette
#' embed_plot(pca_iris$coords, x = iris, palette = "Dark2")
#'
#' # Can plot the category names instead of points, but looks bad if they're
#' # long (or the dataset is large)
#' embed_plot(pca_iris$coords, x = iris$Species, cex = 0.5, as_text = TRUE)
#'
#' tsne_iris <- sneer(iris, method = "tsne", scale_type = "a",
#'                    ret = c("dx", "dy", "deg"))
#' # how well is the 32 nearest neighborhood preserved for each point?
#' nbr_pres_32 <- nbr_pres(tsne_iris$dx, tsne_iris$dy, 32)
#' # visualize preservation, use absolute scale of 0-1 for colors.
#' embed_plot(tsne_iris$coords, x = nbr_pres_32, limits = c(0, 1))
#'
#' # visualize 10 points with the hightest degree centrality
#' embed_plot(tsne_iris$coords, x = tsne_iris$deg, top = 10)
#' }
embed_plot <- function(coords, colors = NULL, x = NULL,
                       palette = NULL, col_ramp = grDevices::rainbow,
                       num_colors = 15, limits = NULL, top = NULL,
                       cex = 1, text = NULL) {

  if (is.null(colors)) {
    # Ensure col_ramp is ignored if a Color Brewer palette is used
    if (!is.null(palette)) {
      col_ramp <- NULL
    }

    if (!is.null(x)) {
      colors <- color_helper(x, palette = palette, col_ramp = col_ramp,
                             num_colors = num_colors, limits = limits,
                             top = top)
    }
    else {
      colors <- make_palette(ncolors = nrow(coords),
                             name = palette, col_ramp = col_ramp)
    }
  }

  if (!is.null(text)) {
    graphics::plot(coords, type = 'n')
    graphics::text(coords, labels = text, cex = cex, col = colors)
  }
  else {
    graphics::plot(coords, pch = 20, cex = cex, col = colors)
  }
}

color_helper <- function(x,
                        palette = NULL, col_ramp = grDevices::rainbow,
                        num_colors = 15, limits = NULL, top = NULL,
                        verbose = FALSE) {
  if (!is.null(palette)) {
    col_ramp <- NULL
  }
  if (class(x) == 'data.frame') {
    color_helper_df(x, palette = palette, col_ramp = col_ramp,
                    verbose = verbose)
  }
  else {
    color_helper_column(x,
                        palette = palette, col_ramp = col_ramp,
                        num_colors = num_colors, limits = limits, top = top,
                        verbose = verbose)
  }
}


# Try and find a meaningful vector of colors from a data frame.
# If the data frame contains at least one column of colors, use the last column
# of colors found.
# Otherwise, if the data frame contains at least one column of factors, map
# from the last factor column found to a list of colors.
# Otherwise, color each point as if it was its own factor level
# @note R considers numbers to be acceptable colors because \code{col2rgb}
# interprets them as indexes into a palette. Columns of numbers are NOT treated
# as colors by color_helper. Stick with color names (e.g. "goldenrod") or
# rgb strings (e.g. "#140000" or "#140000FF" if including alpha values).
color_helper_df <- function(df,
                         palette = NULL, col_ramp = grDevices::rainbow,
                         verbose = FALSE) {

  # Is there a color column?
  color_name <- last_color_column_name(df)
  if (!is.null(color_name)) {
    if (verbose) {
      message("Found color column ", color_name)
    }
    return(df[[color_name]])
  }

  # Is there a factor column?
  label_name <- last_factor_column_name(df)
  if (!is.null(label_name)) {
    if (verbose) {
      message("Found a factor ", label_name, " for mapping to colors")
    }
    return(factor_to_colors(df[[label_name]],
                            palette = palette, col_ramp = col_ramp))
  }

  # use one color per point
  make_palette(ncolors = nrow(df), name = palette, col_ramp = col_ramp)
}

color_helper_column <- function(x,
                             palette = NULL, col_ramp = grDevices::rainbow,
                             num_colors = 15, limits = NULL, top = NULL,
                             verbose = FALSE) {
  # Is this a color column - return as-is
  if (is_color_column(x)) {
    return(x)
  }

  # Is it numeric - map to palette (which should be sequential or diverging)
  if (is.numeric(x)) {
    colors <- numeric_to_colors(x, palette = palette, col_ramp = col_ramp,
                             n = num_colors, limits = limits)
    if (!is.null(top)) {
      svec <- sort(x, decreasing = TRUE)
      colors[x < svec[top]] <- NA
    }
    return(colors)
  }

  # Is it a factor - map to palette (which should be categorial)
  if (is.factor(x)) {
    return(factor_to_colors(x, palette = palette, col_ramp = col_ramp))
  }

  # Otherwise one color per point (doesn't really matter what the palette is!)
  make_palette(ncolors = length(x), name = palette, col_ramp = col_ramp)
}


# Map a vector of factor levels, x,  to a vector of colors taken from either
# an RColorBrewer palette name, or a color ramp function.
# @examples
# factor_to_colors(iris$Species, "Set3") # Color Brewer palette
# factor_to_colors(iris$Species, rainbow) # color ramp function
factor_to_colors <- function(x, palette = NULL, col_ramp = grDevices::rainbow) {
  # Ignore the col_ramp parameter if a palette name is provided
  if (!is.null(palette)) {
    col_ramp <- NULL
  }
  category_names <- unique(x)
  ncolors <- length(category_names)
  color_palette = make_palette(ncolors = ncolors,
                               name = palette, col_ramp = col_ramp)
  color_palette[x]
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
# plot(iris[, c("Sepal.Length", "Sepal.Width")], cex = 1.5, pch = 20,
#  col = numeric_to_colors(iris$Petal.Length, palette = "PRGn", n = 20))
#
# # Use the rainbow color ramp function
# plot(iris[, c("Sepal.Length", "Sepal.Width")], cex = 1.5, pch = 20,
#  col = numeric_to_colors(iris$Petal.Length, col_ramp = rainbow, n = 20))
# }
numeric_to_colors <- function(x, palette = "Blues", col_ramp = NULL, n = 15,
                              limits = NULL) {
  if (!requireNamespace("RColorBrewer", quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("numeric_to_colors function requires 'RColorBrewer' package")
  }
  if (is.null(limits)) {
    limits <- range(x)
  }
  pal <- make_palette(name = palette, ncolors = n, col_ramp = col_ramp)
  pal[findInterval(x, seq(limits[1], limits[2], length.out = length(pal) + 1),
                   all.inside = TRUE)]
}

# Color Palette with Specified Number of Colors
#
# Creates a color palette with the specified number of colors, interpolating
# ColorBrewer palettes by default.
#
# This function is designed to make it easy to use the ColorBrewer palettes,
# particularly with the qualitative sets, without having to worry about a plot
# not being displayed because the palette didn't have enough colors for the
# number of categories required. Admittedly, you probably shouldn't be using
# the palette in that case, but it's better to see the plot.
#
# Rather than specify a ColorBrewer palette by name, you can also pass in
# a color ramp function of any kind via the \code{col_ramp} parameter. For some
# applicable ramp functions, see the \code{Palettes} help page in the
# \code{grDevices} package (e.g. by running the \code{?rainbow} command).
#
# @param ncolors Number of colors desired for the palette.
# @param name The name of a ColorBrewer palette. Ignored if \code{col_ramp} is
#  specified.
# @param col_ramp Function accepting an integer n as an argument and returning
#  n colors. If specified, the \code{name} parameter will be ignored.
# @value A palette with the specified number of colors, interpolated if
#  necessary.
make_palette <- function(ncolors, name = "Set3", col_ramp = NULL) {
  if (!is.null(col_ramp)) {
    palette <- col_ramp(ncolors)
  }
  else {
    if (!requireNamespace("RColorBrewer", quietly = TRUE,
                          warn.conflicts = FALSE)) {
      stop("make_palette function requires 'RColorBrewer' package")
    }
    palette <- color_brewer_palette(name, ncolors)
  }
  palette
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
# @param center If \code{TRUE}, mean-center the columns.
# @param scale If \code{TRUE}, scale the columns to have unit variance.
# @param colors Vector of colors to apply to each point. If \code{NULL}, then
#  the data frame will be searched for a suitable vector of colors or a vector
#  of factor levels to map to the \code{grDevices::rainbow} color scheme.
# @param cex Numeric \strong{c}haracter \strong{ex}pansion factor;
#  multiplied by \code{\link[graphics]{par}("cex")} yields the final
#  character size of the labels.
# @param text Vector of label text to display instead of a point. If the labels
#  are long or the data set is large, this is unlikely to be very legible, but
#  is occasionally useful.
# @param verbose If \code{TRUE} log messages about any default behavior
# (e.g. number of columns used in PCA or category column)
# @examples
# \dontrun{
#  # PCA on the scaled iris dataset, use the "Species" column as labels.
#  # Change the text size to make it easier to read.
#  scores_plot(iris, 1:4, scale = TRUE, text = iris$Species, cex = 0.5)
#
#  # PCA on the s1k dataset, use all numeric values, and use default coloring
#  # (mapping factor levels to colors).
#  scores_plot(s1k)
# }
scores_plot <- function(df, pca_indexes = NULL,
                        center = TRUE, scale = FALSE,
                        colors = NULL,
                        cex = 1, text = NULL,
                        verbose = FALSE) {
  if (is.null(pca_indexes)) {
    pca_indexes <- which(vapply(df, is.numeric, logical(1)))
  }
  if (is.null(pca_indexes)) {
    stop("Couldn't find any numeric columns to carry PCA out on!")
  }
  else if (verbose) {
    message("Using ", length(pca_indexes), " columns for PCA")
  }

  pca <- stats::prcomp(df[, pca_indexes], retx = TRUE, center = center,
                       scale. = scale)

  embed_plot(pca$x[, 1:2],
             x = df,
             col_ramp = grDevices::rainbow,
             cex = cex,
             text = text)
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
    pca_indexes <- vapply(df, is.numeric, logical(1))
  }

  if (is.null(label_name)) {
    factor_names <- names(df)[(vapply(df, is.factor, logical(1)))]
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

  pca <- stats::prcomp(df[, pca_indexes], retx = TRUE, center = center,
                       scale. = scale)
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
scatterqplot <- function(df, x, y, label_name = NULL, labels = NULL, size = 1,
                         palette = "Set1", x_label = "x", y_label = "y",
                         legend = TRUE, legend_rows = NULL) {
  if (!requireNamespace("ggplot2", quietly = TRUE, warn.conflicts = FALSE)) {
    stop("scatterqplot function requires 'ggplot2' package")
  }
  if (!requireNamespace("RColorBrewer", quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("scatterqplot function requires 'RColorBrewer' package")
  }

  if (is.null(labels) && is.null(label_name)) {
    factor_names <- names(df)[(vapply(df, is.factor, logical(1)))]
    if (length(factor_names) == 0) {
      stop("Couldn't find a factor column in data frame to use for label")
    }
    else {
      label_name <- factor_names[1]
    }
  }
  if (is.null(labels)) {
    if (is.null(df[[label_name]])) {
      stop("Data frame does not have a '",label_name,
           "' column for use as a label")
    }
    labels <- df[[label_name]]
  }
  else {
    label_name <- "Labels"
  }

  ncolors <- length(unique(labels))
  colorPalette <- color_brewer_palette(palette, ncolors)

  score_plot <-
    ggplot2::qplot(x, y, colour = labels, size = I(size)) +
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
# @param x Data frame.
# @param attr_name Name of the label column in \code{x}. Ignored if
#  \code{labels} is provided.
# @param labels Vector of labels.
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
make_plot <- function(x, attr_name, labels = NULL,
                      label_fn = function(labels) {
                        labels
                      },
                      mat_name = "ym", cex = 1) {
  embedding_plot <- make_embedding_plot(x, attr_name, labels = labels,
                                        label_fn = label_fn, cex = cex)
  function(out) {
    embedding_plot(out[[mat_name]])
  }
}

# Embedding Plot Using \code{graphics} Library
#
# Create a function which when invoked on a 2D matrix, plots the embedding
# with color-coded labels.
#
# @param x Data frame.
# @param attr_name Name of the label column in \code{x}. Ignored if
#  \code{labels} is provided.
# @param labels Vector of labels for \code{x}.
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
make_embedding_plot <- function(x, attr_name, labels = NULL, label_fn = NULL,
                                cex = 1) {
  if (!is.null(labels)) {
    attr <- labels
  }
  else {
    attr <- x[[attr_name]]
  }
  uniq_attr <- sort(unique(attr))
  colors <- grDevices::rainbow(length(uniq_attr))
  names(colors) <- uniq_attr
  if (!is.null(label_fn)) {
    labels <- label_fn(attr)
  } else {
    labels <- attr
  }
  function(ym) {
    graphics::plot(ym, type = "n", xlab = "D1", ylab = "D2")
    graphics::text(ym, labels = labels, col = colors[attr], cex = cex)
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
  make_plot(datasets::iris, "Species", make_label(num_label_chars))
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
# @param attr_name Name of the label column in \code{df}. Ignored if
#  \code{labels} is provided.
# @param labels Vector of labels.
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
make_qplot <- function(df, attr_name = "Label", labels = NULL, mat_name = "ym",
                       size = 1,
                       palette = "Set1",
                       legend = TRUE, legend_rows = NULL) {
  if (!requireNamespace("ggplot2", quietly = TRUE, warn.conflicts = FALSE)) {
    stop("make_qplot function requires 'ggplot2' package")
  }
  if (!requireNamespace("RColorBrewer", quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("make_qplot function requires 'RColorBrewer' package")
  }
  embedding_plot <- make_embedding_qplot(df, attr_name, labels = labels,
                                         size = size,
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
make_embedding_qplot <- function(df, attr_name = "Label", labels = NULL,
                                 size = 1,
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
                 labels = labels,
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
#  Accent Dark2 Paired Pastel1 Pastel2	Set1 Set2	Set3
# @param name Name of the palette.
# @param ncolors Number of colors desired.
# @return Vector of \code{n} colors from the palette.
# @seealso
# More information on ColorBrewer is available at its website,
# \url{http://www.colorbrewer2.org}.
color_brewer_palette <- function(name, ncolors) {
  if (!requireNamespace("RColorBrewer", quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("color_brewer_palette function requires 'RColorBrewer' package")
  }
  make_color_brewer_ramp(name)(ncolors)
}

# Interpolated Color Brewer Ramp
#
# Creates a color ramp function using the ColorBrewer palettes, with
# interpolation if the requested number of colors exceeds the maximum number of
# colors in the palette. Sequential and Diverging palettes are suitable for
# numerical scales. The Qualitiative palettes are intended for categorical
# values.
#
# Sequential palettes names:
#  Blues BuGn BuPu GnBu Greens Greys Oranges OrRd PuBu PuBuGn PuRd Purples
#  RdPu Reds YlGn YlGnBu YlOrBr YlOrRd
# Diverging palette names:
#  BrBG PiYG PRGn PuOr RdBu RdGy RdYlBu RdYlGn Spectral
# Qualitative:
#  Accent Dark2 Paired Pastel1 Pastel2	Set1 Set2	Set3
#
# @param name Name of the palette.
# @return Function accepting an integer n as an argument and returning n colors.
#
# @note This function requires the RColorBrewer package to be installed and
#  loaded.
# @seealso
# More information on ColorBrewer is available at its website,
# \url{http://www.colorbrewer2.org}.
make_color_brewer_ramp <- function(name) {
  if (!requireNamespace("RColorBrewer", quietly = TRUE,
                        warn.conflicts = FALSE)) {
    stop("colorBrewerPalette function requires 'RColorBrewer' package")
  }
  if (!name %in% rownames(RColorBrewer::brewer.pal.info)) {
    stop("Unknown ColorBrewer name '", name, "', must be one of ",
         paste(rownames(RColorBrewer::brewer.pal.info), collapse = ", "))
  }

  function(n) {
    max_colors <- RColorBrewer::brewer.pal.info[name,]$maxcolors
    n <- max(n, 3)
    if (n <= max_colors) {
      RColorBrewer::brewer.pal(n, name)
    }
    else {
      grDevices::colorRampPalette(RColorBrewer::brewer.pal(max_colors, name))(n)
    }
  }
}
