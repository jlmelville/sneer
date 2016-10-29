### Examples

More examples than you can shake a stick at:

```R
# PCA on iris dataset and plot result using Species label name
res <- sneer(iris, indexes = 1:4, label_name = "Species", method = "pca")

# Same as above, but with sensible defaults (use all numeric columns, plot
# with first factor column found)
res <- sneer(iris, method = "pca")

# scale columns so each one has mean 0 and variance 1
res <- sneer(iris, method = "pca", scale_type = "a")

# full species name on plot is cluttered, so just use the first two
# letters and half size
res <- sneer(iris, method = "pca", scale_type = "a", label_chars = 2,
             label_size = 0.5)

# You need to install and load ggplot2 and RColorBrewer yourself
library(ggplot2)
library(RColorBrewer)
# Optionally use ggplot2 and RColorBrewer palettes for the plot
res <- sneer(iris, method = "pca", scale_type = "a", plot_type = "g")

# Use a different ColorBrewer palette, bigger points, and range scale each
# column
res <- sneer(iris, method = "pca", scale_type = "r", plot_type = "g",
             color_scheme = "Dark2", label_size = 2)

# metric MDS starting from the PCA
res <- sneer(iris, method = "mmds", scale_type = "a", init = "p")

# Sammon map starting from random distribution
res <- sneer(iris, method = "sammon", scale_type = "a", init = "r")

# TSNE with a perplexity of 32, initialize from PCA
res <- sneer(iris, method = "tsne", scale_type = "a", init = "p",
             perplexity = 32)
# default settings are to use TSNE with perplexity 32 and initialization
# from PCA so the following is the equivalent of the above
res <- sneer(iris, scale_type = "a")

# Use the standard tSNE optimization method (Jacobs step size method) with
# step momentum. Range scale the matrix and use an aggressive learning
# rate (epsilon).
res <- sneer(iris, scale_type = "m", perplexity = 25, opt = "tsne",
             epsilon = 500)

# NeRV method, starting at a more global perplexity and slowly stepping
# towards a value of 32 (might help avoid local optima)
res <- sneer(iris, scale_type = "a", method = "nerv", perp_scale = "step")

# NeRV method has a lambda parameter - closer to 1 it gets, the more it
# tries to avoid false positives (close points in the map that aren't close
# in the input space):
res <- sneer(iris, scale_type = "a", method = "nerv", perp_scale = "step",
             lambda = 1)

# Original NeRV paper transferred input exponential similarity kernel
# precisions to the output kernel, and initialized from a uniform random
# distribution
res <- sneer(iris, scale_type = "a", method = "nerv", perp_scale = "step",
             lambda = 1, prec_scale = "t", init = "u")

# Like NeRV, the JSE method also has a controllable parameter that goes
# between 0 and 1, called kappa. It gives similar results to NeRV at 0 and
# 1 but unfortunately the opposite way round! The following gives similar
# results to the NeRV embedding above:
res <- sneer(iris, scale_type = "a", method = "jse", perp_scale = "step",
             kappa = 0)

# Rather than step perplexities, use multiscaling to combine and average
# probabilities across multiple perplexities. Output kernel precisions
# can be scaled based on the perplexity value (compare to NeRV example
# which transferred the precision directly from the input kernel)
res <- sneer(iris, scale_type = "a", method = "jse", perp_scale = "multi",
             prec_scale = "s")

# HSSNE has a controllable parameter, alpha, that lets you control how
# much extra space to give points compared to the input distances.
# Setting it to 1 is equivalent to TSNE, so 1.1 is a bit of an extra push:
res <- sneer(iris, scale_type = "a", method = "hssne", alpha = 1.1)

# wTSNE treats the input probability like a graph where the probabilities
# are weighted edges and adds extra repulsion to nodes with higher degrees
res <- sneer(iris, scale_type = "a", method = "wtsne")

# can use a step-function input kernel to make input probability more like
# a k-nearest neighbor graph (but note that we don't take advantage of the
# sparsity for performance purposes, sadly)
res <- sneer(iris, scale_type = "a", method = "wtsne", perp_kernel_fun = "step")

# Some quality measures are available to quantify embeddings
# The area under the RNX curve measures whether neighbors in the input
# are still neighors in the output space
res <- sneer(iris, scale_type = "a", method = "wtsne",
             quality_measures =  c("n"))

# Create a 5D gaussian with its own column specifying colors to use
# for each point (in this case, random)
g5d <- data.frame(matrix(rnorm(100 * 5), ncol = 5),
                  color = rgb(runif(100), runif(100), runif(100)),
                  stringsAsFactors = FALSE)
# Specify the name of the color column and the plot will use it rather than
# trying to map factor levels to colors
res <- sneer(g5d, method = "pca", color_name = "color")

# If your dataset labels divide the data into natural classes, can calculate
# average area under the ROC and/or precision-recall curve too, but you need to
# have installed the PRROC package. And all these techniques can be slow (scale
# with the square of the number of observations).
library(PRROC)
res <- sneer(iris, scale_type = "a", method = "wtsne",
             quality_measures =  c("n", "r", "p"))

# export the distance matrices and do whatever quality measures we want at our
# leisure
res <- sneer(iris, scale_type = "a", method = "wtsne", ret = c("dx", "dy"))

# export degree centrality, input weight function precision parameters,
# and intrinsic dimensionality
res <- sneer(iris, scale_type = "a", method = "wtsne",
  ret = c("deg", "prec", "dim"))

# Plot the embedding as points colored by category, using the rainbow
# color ramp function:
embed_plot(res$coords, iris$Species, color_scheme = rainbow)

# Load the RColorBrewer Library
library(RColorBrewer)

# Use a ColorBrewer Qualitative palette
# NB: ColorBrewer as a string)
embed_plot(res$coords, iris$Species, color_scheme = "Dark2")

# Visualize embedding colored by various values:
# Degree centrality
embed_plot(res$coords, x = res$deg)
# Intrinsic Dimensionality using the PRGn palette
# (requires RColorBrewer package to be installed)
embed_plot(res$coords, x = res$dim, color_scheme = "PRGn")
# Input weight function precision parameter with the Spectral palette
embed_plot(res$coords, x = res$prec, color_scheme = "Spectral")
# Calculate the 32-nearest neighbor preservation for each observation
# 0 means no neighbors preserved, 1 means all of them
pres32 <- nbr_pres(res$dx, res$dy, 32)
embed_plot(res$coords, x = pres32, cex = 1.5)

```
