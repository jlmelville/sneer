# The stiffness expression using the generic "plug in" derivation as given by
# Lee at al.

# Stiffness Functions -----------------------------------------------------

plugin_stiffness <- function() {
  list(
    fn = plugin_stiffness_fn,
    name = "Plugin",
    keep = c("qm", "wm", "d2m")
  )
}

# Plugin Stiffness
#
# Calculates the Stiffness matrix of an embedding method using the plugin
# gradient formulation.
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return Stiffness matrix.
plugin_stiffness_fn <- function(method, inp, out) {
  if (!is.null(method$out_prob_type)) {
    prob_type <- method$out_prob_type
  }
  else {
    prob_type <- method$prob_type
  }

  fn_name <- paste0('plugin_stiffness_', prob_type)
  stiffness_fn <- get(fn_name)
  if (is.null(stiffness_fn)) {
    stop("Unable to find plugin stiffness function for ", prob_type)
  }
  stiffness_fn(method, inp, out)
}

# Plugin Stiffness for Row Probabilities
#
# Calculates the stiffness matrix for row probability based embedding methods.
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return Stiffness matrix.
plugin_stiffness_row <- function(method, inp, out) {
  dc_dq <- method$cost$gr(inp, out, method)
  dw_df <- method$kernel$gr(method$kernel, out$d2m)

  wm_sum <-  rowSums(out$wm)
  km <- rowSums(dc_dq * out$qm)
  km <- sweep(dc_dq, 1, km) # subtract row sum from each row element
  km <- km * (dw_df / wm_sum)
  km + t(km)
}

# Plugin Stiffness for Conditional Probabilities
#
# Calculates the stiffness matrix for conditional probability based embedding
# methods.
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return Stiffness matrix.
plugin_stiffness_cond <- function(method, inp, out) {
  km <- plugin_stiffness_pair(method, inp, out)
  km + t(km)
}

# Plugin Stiffness for Joint Probabilities
#
# The stiffness matrix is identical for joint and conditional P matrices, because
# they both sum over all pairs, rather than all points.
#
# if dw/df is symmetric (i.e. kernel is symmetric) kij = kji
# and we could replace K + K' with 2K
# NB: not enough enforce symmetry of Q: we would have to symmetrize W.
plugin_stiffness_joint <- function(method, inp, out) {
  km <- plugin_stiffness_pair(method, inp, out)
  if (attr(method$kernel$fn, 'type') == "symm") {
    2 * km
  }
  else {
    km + t(km)
  }
}

# Plugin Stiffness Matrix K
#
# Calculates the stiffness matrix used by probability-based embedding methods
# employing pair-wise normalization.
#
# @param inp Input data.
# @param out Output data.
# @param method Embedding method.
# @return stiffness matrix.
plugin_stiffness_pair <- function(method, inp, out) {
  dc_dq <- method$cost$gr(inp, out, method)
  dw_df <- method$kernel$gr(method$kernel, out$d2m)
  wm_sum <- sum(out$wm)
  (dc_dq - sum(dc_dq * out$qm)) * (dw_df / wm_sum)
}

plugin_stiffness_un <- function(method, inp, out) {
  dc_dw <- method$cost$gr(inp, out, method)
  dw_df <- method$kernel$gr(method$kernel, out$d2m)
  km <- (dc_dw * dw_df)
  km + t(km)
}
