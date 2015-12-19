# Stiffness.R

# Stiffness
tsne_stiffness <- function() {
  f <- function(pm, qm, wm) {
    4 * (pm - qm) * wm
  }

  list(
    cost_fn = kl_cost,
    weight_fn = tdist_weight,
    stiffness_fn = function(stiffness, inp, out) {
      f(inp$pm, out$qm, out$wm)
    },
    prob_out_fn = weights_to_pcond,
    update_out_fn = function(inp, out, stiffness, wm) {
      out$wm <- wm
      out
    },
    after_init_fn = function(inp, out, stiffness) {
      inp$pm <- prow_to_pjoint(inp$pm)
      list(inp = inp)
    }
  )
}

ssne_stiffness <- function() {
  f <- function(pm, qm) {
    4 * (pm - qm)
  }

  list(
    cost_fn = kl_cost,
    weight_fn = exp_weight,
    stiffness_fn = function(stiffness, inp, out) {
      f(inp$pm, out$qm)
    },
    prob_out_fn = weights_to_pcond,
    update_out_fn = NULL,
    after_init_fn = function(inp, out, stiffness) {
      inp$pm <- prow_to_pjoint(inp$pm)
      list(inp = inp)
    }
  )
}

asne_stiffness <- function() {
  f <- function(pm, qm) {
    km <- 2 * (pm - qm)
    km + t(km)
  }

  list(
    cost_fn = kl_cost,
    weight_fn = exp_weight,
    stiffness_fn = function(stiffness, inp, out) {
      f(inp$pm, out$qm)
    },
    prob_out_fn = weights_to_prow,
    update_out_fn = NULL,
    after_init_fn = function(inp, out, stiffness) {
      inp$pm <- clamp(inp$pm)
      list(inp = inp)
    }
  )
}

tasne_stiffness <- function() {
  f <- function(pm, qm, wm) {
    km <- 2 * (pm - qm) * wm
    km + t(km)
  }

  list(
    cost_fn = kl_cost,
    weight_fn = tdist_weight,
    stiffness_fn = function(stiffness, inp, out) {
      f(inp$pm, out$qm, out$wm)
    },
    prob_out_fn = weights_to_prow,
    update_out_fn = function(inp, out, stiffness, wm) {
      out$wm <- wm
      out
    },
    after_init_fn = function(inp, out, stiffness) {
      inp$pm <- clamp(inp$pm)
      list(inp = inp)
    }
  )
}
