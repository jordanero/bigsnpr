################################################################################

#' ssctpr
#'
#' @inheritParams snp_ldpred2_grid
#' @param delta Vector of shrinkage parameters to try (L2-regularization).
#' @param lambda1 Vector of lasso parameters to try
#' @param lambda2 Vector of cross-trait penalty parameters to try
#' @param dfmax Maximum number of non-zero effects in the model.
#'   Default is `200e3`.
#' @param maxiter Maximum number of iterations before convergence.
#'   Default is `500`.
#' @param tol Tolerance parameter for assessing convergence.
#'   Default is `1e-5`.
#'
#' @return A matrix of effect sizes, one vector (column) for each row in
#'   `attr(<res>, "grid_param")`. Missing values are returned when strong
#'   divergence is detected.
#'
#' @export
#'
snp_ssctpr <- function(corr, df_beta,
                       delta = signif(seq_log(1e-3, 3, 6), 1),
                       lambda1 = c(.1, .5),
                       lambda2 = c(0, .1),
                       dfmax = 200e3, maxiter = 500, check_divergence = TRUE,
                       tol = 1e-5, ncores = 1, logfile = '', lasso_sum = FALSE) {

  grid_param <- expand.grid(lambda1 = lambda1, lambda2 = lambda2, delta = delta)

  ord <- with(grid_param, order(lambda1 * lambda2 * (1 + delta)))
  inv_ord <- match(seq_along(ord), ord)

  bigparallelr::register_parallel(ncores)

  res_grid <- foreach(ic = ord, .export = "ssctpr") %dopar% {
    
    time <- system.time(
      if (!lasso_sum) {
        res <- ssctpr(
          corr               = corr,
          beta_hat           = df_beta$cor1,
          secondary_beta_hat = df_beta$cor2,
          lambda1            = grid_param$lambda1[ic],
          lambda2            = grid_param$lambda2[ic],
          delta              = grid_param$delta[ic],
          dfmax              = dfmax,
          maxiter            = maxiter,
          tol                = tol,
          check_divergence   = check_divergence,
          logfile            = logfile
        )
      } else {
        res <- lassosum2(
          corr             = corr,
          beta_hat         = df_beta$cor1,
          lambda           = grid_param$lambda1[ic],
          delta            = grid_param$delta[ic],
          dfmax            = dfmax,
          maxiter          = maxiter,
          check_divergence = check_divergence,
          tol              = tol
        )
      }
    )

    res$time <- time[["elapsed"]]
    res
  }
  res_grid <- res_grid[inv_ord]

  grid_param$num_iter <- sapply(res_grid, function(.) .$num_iter)
  grid_param$time <- sapply(res_grid, function(.) .$time)
  beta_grid <- do.call("cbind", lapply(res_grid, function(.) .$beta_est))
  grid_param$sparsity <- colMeans(beta_grid == 0)

  list(beta_est = beta_grid, grid_param = grid_param)

}

################################################################################
