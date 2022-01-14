/******************************************************************************/

#include <bigstatsr/arma-strict-R-headers.h>
#include <bigstatsr/utils.h>
#include <bigsparser/SFBM.h>
#include <iostream>
#include <fstream>

/******************************************************************************/

inline double ssctpr_update(double u, double l1, double l2, double s, double one_plus_delta) {
  if (u > (l1 - l2 * s)) {
    return (u - l1 + l2 * s) / (one_plus_delta  + l2);

  } else if (u >= (-l1 - l2 * s)) {
    return 0;

  } else {
    return (u + l1 + l2 * s) / (one_plus_delta + l2);
  }
}

/******************************************************************************/

// [[Rcpp::export]]
List ssctpr(Environment corr,
               const arma::vec& beta_hat,
	       const arma::vec& secondary_beta_hat,
               double lambda1,
               double lambda2,
               double delta,
               double dfmax,
               int maxiter,
               double tol,
	       bool check_divergence,
               std::string logfile	
) {

  XPtr<SFBM> sfbm = corr["address"];

  int m = beta_hat.size();
  myassert_size(sfbm->nrow(), m);
  myassert_size(sfbm->ncol(), m);

  arma::vec curr_beta(m, arma::fill::zeros), dotprods(m, arma::fill::zeros);

  double one_plus_delta = 1 + delta;
  double gap0 = arma::dot(beta_hat, beta_hat);


  
  int k = 0;

  bool logging = !logfile.empty();
  std::ofstream myfile(logfile.c_str(), arma::ios::app);

  for (; k < maxiter; k++) {
    if (logging) {
      myfile << "k = " + std::to_string(k) + "   \n";
    }

    bool conv = true;
    double df = 0;
    double gap = 0;



    for (int j = 0; j < m; j++) {

      double resid = beta_hat[j] - dotprods[j];
      gap += resid * resid;
      double u_j = curr_beta[j] + resid;

      
      if (logging) {
        myfile << "j = " + std::to_string(j) + "   \n";
        myfile << "r_j = " + std::to_string(beta_hat[j]) + "   \n";
        myfile << "old_beta_j= " + std::to_string(curr_beta[j]) + "   \n";
        myfile << "u = " + std::to_string(u_j) + "   \n";
        myfile << "secondary_beta  = " + std::to_string(secondary_beta_hat[j]) + "   \n";
        myfile << "one_plus_delta = " + std::to_string(one_plus_delta) + "   \n";
      }

      double new_beta_j = ssctpr_update(u_j, lambda1, lambda2, 
                                        secondary_beta_hat[j], 
                                        one_plus_delta);

      if (logging) {
        myfile << "new_beta_hat_j = " + std::to_string(new_beta_j) + "   \n";
        myfile << "\n";
      }

      if (new_beta_j != 0) df++;

      double shift = new_beta_j - curr_beta[j];
      if (shift != 0) {
        if (conv && std::abs(shift) > tol) conv = false;
        curr_beta[j] = new_beta_j;
        dotprods = sfbm->incr_mult_col(j, dotprods, shift);
      }
    }

    if ((check_divergence) && (gap > gap0)) { curr_beta.fill(NA_REAL); break; }
    if (conv || df > dfmax) break;
    if (logging) {
      myfile << "\n";
    }
  }

  if (logging) {
    myfile.close();
  }

  return List::create(_["beta_est"] = curr_beta, _["num_iter"] = k + 1);
}

/******************************************************************************/
