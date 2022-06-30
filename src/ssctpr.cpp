/******************************************************************************/

#include <string>
#include <bigstatsr/arma-strict-R-headers.h>
#include <bigstatsr/utils.h>
#include <bigsparser/SFBM.h>
#include <iostream>
#include <fstream>
#include <sstream>

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


double loss_j(double beta_j, double one_plus_delta, double beta_hat_j, 
                     double dotprods_j, double lambda1, double lambda2,
		     double secondary_beta_hat_j)
{
  float loss = std::pow(beta_j,2) * one_plus_delta;
  loss += (-2) * (beta_j) * beta_hat_j;
  loss += 2 * beta_j * dotprods_j;
  loss += (-2) * std::pow(beta_j, 2);
  loss += 2 * lambda1 * std::abs(beta_j);
  loss += lambda2 * std::pow(beta_j - secondary_beta_hat_j, 2);
  return (loss);
}

// the first element is \beta^T \tilde{X}^T \tilde{X}\beta
// the second element is -2\beta^T r
// the third element is \delta \beta^T \beta
// the fourth element is 2\lambda ||\beta\\_1^1
// the fifth element is \lambda_2\sum_t\sum_i(\beta_i - s_{ti})^2
arma::vec termwise_loss_j(
	double beta_j, double one_plus_delta, double beta_hat_j,
        double dotprods_j, double lambda1, double lambda2,
        double secondary_beta_hat_j) 
{
  arma::vec termwise_loss(5, arma::fill::zeros);

  termwise_loss[0] = std::pow(beta_j, 2) ;
  termwise_loss[0] += (-2) * std::pow(beta_j, 2);
  termwise_loss[0] += 2 * beta_j * dotprods_j;
  termwise_loss[1] = (-2) * (beta_j) * beta_hat_j;
  termwise_loss[2] = std::pow(beta_j, 2) * (one_plus_delta - 1);
  termwise_loss[3] = 2 * lambda1 * std::abs(beta_j);
  termwise_loss[4] = lambda2 * std::pow(beta_j - secondary_beta_hat_j, 2);
  
return (termwise_loss);
}

double likelihood_j(double beta_j, double one_plus_delta, double beta_hat_j, 
                  double dotprods_j)
{
  float loss = std::pow(beta_j,2) * one_plus_delta;
  loss += (-2) * (beta_j) * beta_hat_j;
  loss += 2 * beta_j * dotprods_j;
  loss += (-2) * std::pow(beta_j, 2);
  return (loss);
}

double penalty_j(double beta_j, double lambda1, double lambda2,
	       double secondary_beta_hat_j)
{
  float loss = 2 * lambda1 * std::abs(beta_j);
  loss += lambda2 * std::pow(beta_j - secondary_beta_hat_j, 2);
  return (loss);
}

std::string double_to_string(double dub) {
  std::ostringstream ss;
  ss.precision(12);
  ss << std::fixed << dub;
  std::string my_string = ss.str();
  return(my_string);
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

  if ((beta_hat.max() < lambda1) & (beta_hat.min() > (-1 * lambda1))) {
    return List::create(_["beta_est"] = curr_beta, _["num_iter"] = 0);
  }

  double one_plus_delta = 1 + delta;
  double gap0 = arma::dot(beta_hat, beta_hat);
  double loss = 0, likelihood = 0, penalty = 0;
  arma::vec termwise_loss(5, arma::fill::zeros);
  
  int k = 0;

  bool logging = !logfile.empty();
  //std::ofstream myfile(logfile.c_str(), arma::ios::app);
  std::ofstream myfile(logfile.c_str());
  if (logging) {
    loss = lambda2 * arma::sum(arma::pow(curr_beta - secondary_beta_hat, 2));
    termwise_loss[4] = loss;
    penalty = loss;
    myfile << "k, j, cor, secondary_cor, u_j, beta_hat_old, beta_hat_new, dotprods_j, likelihood, penalty, loss, loss_term_1, loss_term_2, loss_term_3, loss_term_4, loss_term_5\n";
  }

  for (; k < maxiter; k++) {

    bool conv = true;
    double df = 0;
    double gap = 0;


    for (int j = 0; j < m; j++) {

      double resid = beta_hat[j] - dotprods[j]; // I think dotprods is R_{.,j}^T\beta
      gap += resid * resid;
      double u_j = curr_beta[j] + resid;

      
      if (logging) {

        myfile << std::to_string(k) + ", ";
        myfile << std::to_string(j) + ", ";
        myfile << double_to_string(beta_hat[j]) + ", ";
        myfile << double_to_string(secondary_beta_hat[j]) + ", ";
        myfile << double_to_string(u_j) + ", ";
        myfile << double_to_string(curr_beta[j]) + ", ";
        loss += (-1) * loss_j(curr_beta[j], one_plus_delta, beta_hat[j],
                              dotprods[j], lambda1, lambda2, secondary_beta_hat[j]);
        likelihood += (-1) * likelihood_j(curr_beta[j], one_plus_delta, beta_hat[j],
                                          dotprods[j]);
        penalty += (-1) * penalty_j(curr_beta[j], lambda1, lambda2, secondary_beta_hat[j]);
	termwise_loss -= termwise_loss_j(curr_beta[j], one_plus_delta, beta_hat[j],
                                         dotprods[j], lambda1, lambda2, secondary_beta_hat[j]);
      }

      double new_beta_j = ssctpr_update(u_j, lambda1, lambda2, 
                                        secondary_beta_hat[j], 
                                        one_plus_delta);


      if (new_beta_j != 0) df++;

      double shift = new_beta_j - curr_beta[j];
      if (shift != 0) {
        if (conv && std::abs(shift) > tol) conv = false;
        curr_beta[j] = new_beta_j;
        dotprods = sfbm->incr_mult_col(j, dotprods, shift);
      }

      if (logging) {
        myfile << double_to_string(new_beta_j) + ", ";
        myfile << double_to_string(dotprods[j]) + ", ";
        loss += loss_j(curr_beta[j], one_plus_delta, beta_hat[j],
                       dotprods[j], lambda1, lambda2, secondary_beta_hat[j]);
        likelihood += likelihood_j(curr_beta[j], one_plus_delta, beta_hat[j],
                                   dotprods[j]);
        penalty += penalty_j(curr_beta[j], lambda1, lambda2, secondary_beta_hat[j]);
	termwise_loss += termwise_loss_j(curr_beta[j], one_plus_delta, beta_hat[j],
                                         dotprods[j], lambda1, lambda2, secondary_beta_hat[j]);


        myfile << double_to_string(likelihood) + ", ";
        myfile << double_to_string(penalty) + ", ";
        myfile << double_to_string(loss) + ", ";
        myfile << double_to_string(termwise_loss[0]) + ", ";
        myfile << double_to_string(termwise_loss[1]) + ", ";
        myfile << double_to_string(termwise_loss[2]) + ", ";
        myfile << double_to_string(termwise_loss[3]) + ", ";
        myfile << double_to_string(termwise_loss[4]);

        myfile << "\n";
      }
    }


    if ((k != 0) && (check_divergence) && (gap > gap0)) { 
      if (logging) {
        myfile.close();
      }
      curr_beta.fill(NA_REAL); break; 
    }
    if (conv || df > dfmax) break;
  }

  if (logging) {
    myfile.close();
  }

  return List::create(_["beta_est"] = curr_beta, _["num_iter"] = k + 1);
}

/******************************************************************************/
