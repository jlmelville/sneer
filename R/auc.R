#' Average Area Under the Precision-Recall Curve
#'
#' Embedding quality measure.
#'
#' The PR curve plots precision (also known as positive predictive value, PPV)
#' against recall (also known as the true positive rate). The area under the
#' curve provides similar information compared to the area under the ROC curve,
#' but may be more appropriate when classes are highly imbalanced.
#'
#' This function calculates the PR curve N times, where N is the number of the
#' observations. The label of the Nth observation is set as the positive class
#' and then the other observations are ranked according to their distance from
#' the Nth observation in the output coordinates (lower distances being better).
#' Observations with the same label as the Nth observation count as positive
#' observations. The final reported result is the average over all observations.
#'
#' Perfect retrieval results in an AUC of 1. For random retrieval, the value
#' is the proportion of the positive class labels for that curve.
#'
#' @note Use of this function requires that the \code{PRROC} package be
#' installed.
#'
#' @param inp Input data. This should be storing the class labels as
#' \code{inp$labels}, as vector with the labels ordered in the same way as
#' the observations in the distance matrices.
#' @param out Output data. If the output distance matrix is not stored as
#' \code{out$dm}, it will be calculated.
#' @return Area Under the Precision-Recall curve, averaged over each
#' observation.
#' @references
#' Keilwagen, J., Grosse, I., & Grau, J. (2014).
#' Area under precision-recall curves for weighted and unweighted data.
#' \emph{PloS One}, \emph{9}(3), e92209.
#'
#' Davis, J., & Goadrich, M. (2006, June).
#' The relationship between Precision-Recall and ROC curves.
#' In \emph{Proceedings of the 23rd international conference on Machine
#' learning}
#' (pp. 233-240). ACM.
pr_auc <- function(inp, out) {
  if (!requireNamespace("PRROC", quietly = TRUE, warn.conflicts = FALSE)) {
    stop("pr_auc function requires 'PRROC' package")
  }
  if (is.null(out$dm)) {
    out$dm <- distance_matrix(out$ym)
  }
  list(name = "av_pr_auc",
       value = auc_mat(out$dm, inp$labels, pr_auc_row)$av_auc)
}

#' Average Area Under the ROC Curve
#'
#' Embedding quality measure.
#'
#' The ROC curve plots the true positive rate vs false positive rate.
#' This function calculates the curve N times, where N is the number of the
#' observations. The label of the Nth observation is set as the positive class
#' and then the other observations are ranked according to their distance from
#' the Nth observation in the output coordinates (lower distances being better).
#' Observations with the same label as the Nth observation count as positive
#' observations. The final reported result is the average over all observations.
#'
#' Perfect retrieval results in an AUC of 1. For random retrieval gives a value
#' of 0.5.
#'
#' @note Use of this function requires that the \code{PRROC} package be
#' installed.
#'
#' @param inp Input data. This should be storing the class labels as
#' \code{inp$labels}, as vector with the labels ordered in the same way as
#' the observations in the distance matrices.
#' @param out Output data. If the output distance matrix is not stored as
#' \code{out$dm}, it will be calculated.
#' @return Area Under the ROC curve, averaged over each observation.
roc_auc <- function(inp, out) {
  if (!requireNamespace("PRROC", quietly = TRUE, warn.conflicts = FALSE)) {
    stop("roc_auc function requires 'PRROC' package")
  }
  if (is.null(out$dm)) {
    out$dm <- distance_matrix(out$ym)
  }
  list(name = "av_roc_auc",
       value = auc_mat(out$dm, inp$labels, roc_auc_row)$av_auc)
}

#' Area Under the PR Curve of an Observation
#'
#' Embedding quality measure.
#'
#' The PR curve plots precision (also known as positive predictive value, PPV)
#' against recall (also known as the true positive rate). The area under the
#' curve provides similar information compared to the area under the ROC curve,
#' but may be more appropriate when classes are highly imbalanced.
#'
#' This function calculates the curve with the label of the specified
#' observation set as the positive class. The other observations are then
#' ranked according to their distance from the ith observation
#' (lower distances being better). Observations with the same label as the
#' specified observation count as the positive observations.
#'
#' Perfect retrieval results in an AUC of 1. Random retrieval gives a value
#' of the proportion of positive class with respect to the entire data set
#' (e.g. if there are 20 observations with the positive class label in a
#' dataset of 100, then the random AUC is 0.2).
#'
#' @note Use of this function requires that the \code{PRROC} package be
#' installed.
#'
#' @param dm Distance matrix.
#' @param labels Vector of labels, of the same size as the number of rows
#' (or columns) in the distance matrix.
#' @param i The row of the distance matrix to use in the PR calculation.
#' @return Area Under the curve.
#' @references
#'
#' Keilwagen, J., Grosse, I., & Grau, J. (2014).
#' Area under precision-recall curves for weighted and unweighted data.
#' \emph{PloS One}, \emph{9}(3), e92209.
#'
#' Davis, J., & Goadrich, M. (2006, June).
#' The relationship between Precision-Recall and ROC curves.
#' In \emph{Proceedings of the 23rd international conference on Machine
#' learning}
#' (pp. 233-240). ACM.
pr_auc_row <- function(dm, labels, i) {
  if (!requireNamespace("PRROC", quietly = TRUE, warn.conflicts = FALSE)) {
    stop("pr_auc_row function requires 'PRROC' package")
  }
  pos_ind <- which(labels == labels[i], arr.ind = TRUE)
  pos_ind <- pos_ind[pos_ind != i]
  pos_dist <- dm[i, pos_ind]

  neg_ind <- which(labels != labels[i], arr.ind = TRUE)
  neg_dist <- dm[i, neg_ind]

  as.numeric(PRROC::pr.curve(scores.class0 = -pos_dist,
                             scores.class1 = -neg_dist)$auc.davis.goadrich)
}

#' Area Under the ROC Curve of an Observation
#'
#' Embedding quality measure.
#'
#' The ROC curve plots the true positive rate vs false positive rate.
#' This function calculates the curve with the label of the specified
#' observation set as the positive class. The other observations are then
#' ranked according to their distance from the ith observation
#' (lower distances being better). Observations with the same label as the
#' specified observation count as the positive observations.
#'
#' Perfect retrieval results in an AUC of 1. For random retrieval gives a value
#' of 0.5.
#'
#' @note Use of this function requires that the \code{PRROC} package be
#' installed.
#'
#' @param dm Distance matrix.
#' @param labels Vector of labels, of the same size as the number of rows
#' (or columns) in the distance matrix.
#' @param i The row of the distance matrix to use in the ROC calculation.
#' @return Area Under the curve.
roc_auc_row <- function(dm, labels, i) {
  if (!requireNamespace("PRROC", quietly = TRUE, warn.conflicts = FALSE)) {
    stop("roc_auc_row function requires 'PRROC' package")
  }
  pos_ind <- which(labels == labels[i], arr.ind = TRUE)
  pos_ind <- pos_ind[pos_ind != i]
  pos_dist <- dm[i, pos_ind]

  neg_ind <- which(labels != labels[i], arr.ind = TRUE)
  neg_dist <- dm[i, neg_ind]

  as.numeric(
    PRROC::roc.curve(scores.class0 = -pos_dist, scores.class1 = -neg_dist)$auc)
}

#' Average Area Under a Curve
#'
#' Embedding quality measure.
#'
#' This function calculates a curve using the specified function, repeating the
#' procedure N times, where N is the number of the observations. Each time
#' a different row of the distance matrix is used. The label of the Nth
#' observation is set as the positive class and then the other observations are
#' ranked according to their distance from the Nth observation in the output
#' coordinates (lower distances being better). Observations with the same label
#' as the Nth observation count as positive observations. The final reported
#' result is the average over all observations.
#'
#' @note Use of this function requires that the \code{PRROC} package be
#' installed.
#'
#' @param dm Distance matrix.
#' @param labels Vector of labels, of the same size as the number of rows
#' (or columns) in the distance matrix.
#' @param auc_row_fn A function which can calculate the Area Under a Curve
#' for a particular quality measure. Should have the signature
#' \code{auc_row_fn(dm, labels, i)} where \code{i} is the ith row of the
#' distance matrix, and should return a scalar value giving the area under the
#' curve using the ith row of the distance matrix.
#' @return A list containing:
#' \item{av_auc}{Area Under the curve, averaged over each observation.}
#' The list also contains the average AUC per class label, with each average
#' being named after the class label.
auc_mat <- function(dm, labels, auc_row_fn) {
  av_auc <- 0
  n <- nrow(dm)
  ns <- list()
  result <- list()
  label_av <- list()
  for (i in 1:n) {
    auc <- auc_row_fn(dm, labels, i)
    if (!is.nan(auc)) {
      av_auc <- av_auc + auc
    }
    label <- as.character(labels[[i]])
    if (is.null(label_av[[label]])) {
      label_av[[label]] <- auc
      ns[[label]] <- 1
    }
    else {
      label_av[[label]] <- label_av[[label]] + auc
      ns[[label]] <- ns[[label]] + 1
    }
  }
  for (label in names(ns)) {
    if (ns[[label]] == 0) {
      label_av[[label]] <- 0
    }
    else {
      label_av[[label]] <- label_av[[label]] / ns[[label]]
    }
  }
  result$av_auc <- av_auc / n
  result$label_av <- label_av
  result
}
