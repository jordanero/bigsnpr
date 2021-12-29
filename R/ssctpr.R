################################################################################

#' ssctpr
#'
#' @inheritParams snp_ldpred2_grid
#' @param delta Vector of shrinkage parameters to try (L2-regularization).
#' @param nlambda1 Number of different lambda1s to try (L1-regularization).
#'   Default is `20`.
#' @param lambda1.min.ratio Ratio between last and first lambda1 to try.
#'   Default is `0.01`.
#' @param lambda1_max Max lambda1 parameter to try.
#' @param nlambda2 Number of different lambda2s to try (L1-regularization).
#'   Default is `20`.
#' @param lambda2.min.ratio Ratio between last and first lambda2 to try.
#'   Default is `0.01`.
#' @param lambda2_max Max lambda2 parameter to try.
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
                       nlambda1 = 20, lambda1.min.ratio = 0.01, lambda1_max = NULL, include_lambda1_zero = TRUE,
                       nlambda2 = 20, lambda2.min.ratio = 0.01, lambda2_max = 1e-3, include_lambda2_zero = TRUE,
		       dfmax = 200e3, maxiter = 500, 
		       tol = 1e-5, ncores = 1, logfile = '') {
    

  if (is.null(lambda1_max)) {
    lambda1_max <- max(abs(beta_hat))
  }
  seq_lambda1 <- seq_log(lambda1_max, lambda1.min.ratio * lambda1_max, nlambda1)
  if (include_lambda1_zero) {
    seq_lambda1 <- c(seq_lambda1, 0)
  }
  seq_lambda2 <- seq_log(lambda2_max, lambda2.min.ratio * lambda2_max, nlambda2)
  if (include_lambda2_zero) {
    seq_lambda2 <- c(seq_lambda2, 0)
  }
  grid_param <- expand.grid(lambda1 = seq_lambda1, lambda2 = seq_lambda2, delta = delta)

  ord <- with(grid_param, order(lambda1 * lambda2 * (1 + delta)))
  inv_ord <- match(seq_along(ord), ord)

  bigparallelr::register_parallel(ncores)

  res_grid <- foreach(ic = ord, .export = "ssctpr") %dopar% {
    
    time <- system.time(
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
	logfile            = logfile
      )
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
