#' sneer: Stochastic Neighbor Embedding Experiments in R.
#'
#' A package for exploring probability-based embedding and related forms
#' of dimensionality reduction. Its main goal is to implement multiple
#' embedding methods within a single framework so comparison between them is
#' easier, without worrying about the effect of differences in preprocessing,
#' optimization and heuristics.
#'
#' @section Embedding:
#'
#' The \code{\link{sneer}} function provides a variety of methods for embedding,
#' including:
#'
#' \itemize{
#'  \item{Stochastic Neighbor Embedding and variants (ASNE, SSNE and TSNE)}
#'  \item{Metric MDS using the STRESS and SSTRESS functions}
#'  \item{Sammon Mapping}
#'  \item{Heavy-tailed Symmetric Stochastic Neighbour Embedding (HSSNE)}
#'  \item{Neigbor Retrieval Visualizer (NeRV)}
#'  \item{Jensen-Shannon Embedding (JSE)}
#' }
#'
#' See the documentation for the function for the exact list of methods
#' and variations.
#'
#' Optimization is carried out using the momentum interpretation
#' of the Nesterov Accelerated Gradient method (Sutskever et al 2013.) with
#' an adaptive restart (O'Donoghue and Candes 2013). This seems a bit more
#' robust compared to the usual t-SNE optimization method across the different
#' methods exposed by sneer.
#'
#' Some other optimization methods are also available:
#' \itemize{
#'  \item{The low-memory BFGS method from the \code{optim} function.}
#'  \item{A Polak-Ribiere Conjugate Gradient optimizer, if you install and load
#'  the rcgmin package from https://github.com/jlmelville/rcgmin}
#'  \item{If you install rcgmin, you can also use the two line search algorithms
#'  it implements with the NAG optimizer, instead of the bold driver approach.}
#'  \item{The Spectral Direction method of Vladymyrov and Carreira-Perpinan.
#'  This can only be applied to probability-based embedding methods that use a
#'  symmetric input probability matrix (not NeRV or JSE, but SSNE and t-SNE
#'  are fine), but works very well. However, it is intended to be used with
#'  sparse matrices, because it internally uses a Cholesky decomposition of the
#'  probability matrix, which has a complexity of O(N^3). Currently, lack of
#'  support for sparse matrices in sneer restricts its applicability to smaller
#'  datasets. This can be used with the bold driver line search method, but
#'  to be on the safe side, you might want to also consider the line search
#'  methods provided by the rcgmin package.}
#' }
#'
#' @section Visualization:
#'
#' The \code{\link{embed_plot}} function will take the output of the
#' \code{\link{sneer}} function and provide a visualization of the embedding.
#' If you install the \code{RColorBrewer} package installed, you can use the
#' ColorBrewer palettes by name.
#'
#' @section Quantifying embedding quality:
#'
#' Some functions are available for attempting to quantify embedding quality,
#' independent of the particular loss function used for an embedding method.
#' The \code{\link{nbr_pres}} function will measure how well the embedding
#' preserves a neighborhood of a given size around each observation. The
#' \code{\link{rnx_auc_embed}} function implements the Area Under the Curve
#' of the RNX curve (Lee et al. 2015), which generalizes the neighborhood
#' preservation to account for all neighborhood sizes, with a bias towards
#' smaller neighborhoods.
#'
#' If your observations have labels which could be used for a classification
#' task, then there are also functions which will use these labels to calculate
#' the Area Under the ROC or PR (Precision/Recall) Curve, using the embedded
#' distances to rank each observation: these are \code{\link{roc_auc_embed}}
#' and \code{\link{pr_auc_embed}} functions, respectively. Note that to use
#' these two functions, you must have the \code{PRROC} package installed.
#'
#' @section Synthetic Dataset:
#' There's a synthetic dataset in this package, called \code{s1k}. It consists
#' of a 1000 points representing a fuzzy 9D simplex. It's intended to
#' demonstrate the "crowding effect" and require the sort of
#' probability-based embedding methods provided in this package (PCA does a
#' horrible job of separated the 10 clusters in the data). See \code{s1k}
#' for more details.
#'
#' @examples
#' \dontrun{
#' # Do t-SNE on the iris dataset, scaling columns to zero mean and
#' # unit variance.
#' res <- sneer(iris, scale_type = "a")
#'
#' # Use the weighted TSNE variant and export the input and output distance
#' # matrices.
#' res <- sneer(iris, scale_type = "a", method = "wtsne", ret = c("dx", "dy"))
#'
#' # calculate the 32-nearest neighbor preservation for each observation
#' # 0 means no neighbors preserved, 1 means all of them
#' pres32 <- nbr_pres(res$dx, res$dy, 32)
#'
#' # Calculate the Area Under the RNX Curve
#' rnx_auc <- rnx_auc_embed(res$dx, res$dy)
#'
#' # Load the PRROC library
#' library(PRROC)
#'
#' # Calculate the Area Under the Precision Recall Curve for the embedding
#' pr <- pr_auc_embed(res$dy, iris$Species)
#'
#' # Similarly, for the ROC curve:
#' roc <- roc_auc_embed(res$dy, iris$Species)
#'
#' # Load the RColorBrewer library
#' library(RColorBrewer)
#' # Plot the embedding, with points colored by the neighborhood preservation
#' embed_plot(res$coords, x = pres32, color_scheme = "Blues")
#' }
#' @references
#'
#' t-SNE, SNE and ASNE
#' Van der Maaten, L., & Hinton, G. (2008).
#' Visualizing data using t-SNE.
#' \emph{Journal of Machine Learning Research}, \emph{9}(2579-2605).
#'
#' NeRV
#' Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010).
#' Information retrieval perspective to nonlinear dimensionality reduction for
#' data visualization.
#' \emph{Journal of Machine Learning Research}, \emph{11}, 451-490.
#'
#' JSE
#' Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013).
#' Type 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in
#' dimensionality reduction based on similarity preservation.
#' \emph{Neurocomputing}, \emph{112}, 92-108.
#'
#' Nesterov Accelerated Gradient:
#' Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013).
#' On the importance of initialization and momentum in deep learning.
#' In \emph{Proceedings of the 30th international conference on machine learning (ICML-13)}
#' (pp. 1139-1147).
#'
#' O'Donoghue, B., & Candes, E. (2013).
#' Adaptive restart for accelerated gradient schemes.
#' \emph{Foundations of computational mathematics}, \emph{15}(3), 715-732.
#'
#' Spectral Direction:
#' Vladymyrov, M., & Carreira-Perpinan, M. A. (2012).
#' Partial-Hessian Strategies for Fast Learning of Nonlinear Embeddings.
#' In \emph{Proceedings of the 29th International Conference on Machine Learning (ICML-12)}
#' (pp. 345-352).
#'
#' @docType package
#' @name sneer
NULL
