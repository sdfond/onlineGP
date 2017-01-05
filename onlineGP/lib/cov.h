/** @file pgpr_cov.h
 *  @brief This file provides the covariance class: pgpr_cov, which compute the
 *  covariance matrix.
 */
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
//#include "pgpr_type.h"
using namespace Eigen;

/** @class pgpr_cov
 *  @brief The pgpr_cov class provides the informaiton of covariance.
 */
template<class T>
inline T SQR(const T a) {
  return a * a;
}

class gp_kern
{
public:
        double mean, var;  /**< the mean and variance of training output, for normalization the prediction result */
	/*******
		Note that sig, nos, lsc are all in log form
	********/
	double nos;   /**< the noise variance of the data */	
	VectorXd lsc;  /**< vector of the length scale for each dimension */
	double sig;   /**< the signal variance of the data */
	int dim;
	//int dim, num_train, num_test, num_sup;    /**< the input dimension, number of training, testing and support samples */

  /** @brief this functions initialized the class with a hyperparameter file
   *
   */
    gp_kern(char * hypf) {
    FILE *fp = fopen(hypf, "r");
    if(fp == NULL) {
      throw("Fail to open file of hyperparameters\n");
    }
    //fscanf(fp, "%lf %lf %lf %lf", &sig, &nos, &mean, &var);
    fscanf(fp, "%lf %lf %lf %lf %d", &sig, &nos, &mean, &var, &dim);
    //fscanf(fp, "%d %d %d", &dim, &num_train, &num_test);
    lsc.resize(dim);
    double tmp;
    for (int i = 0; i < dim; i++) {
      fscanf(fp, "%lf ", &tmp);
      lsc[i] = exp(tmp);
    }
    fclose(fp);
    sig = SQR(exp(sig));
    nos = SQR(exp(nos));
  }


  inline double se_ard_n(const VectorXd &x, const VectorXd &y) {
    double val = 0;
    for (int i = 0; i < dim; i++) {
      val += SQR((x[i] - y[i]) / lsc[i]);
    }

    return sig * exp(-0.5 * val) + nos;
  }


  inline double se_ard(const VectorXd &x, const VectorXd &y) {
    double val = 0;
    for (int i = 0; i < dim; i++) {
      val += SQR((x[i] - y[i]) / lsc[i]);
    }

    return sig * exp(-0.5 * val);
  }

  inline void se_ard_n(const MatrixXd &a, MatrixXd &k) {
    int ss = a.rows();
    if (k.rows() != ss) {
      k.resize(ss, ss);
    }
    for (int i = 0; i < ss; i++) {
      for (int j = i; j < ss; j++) {
	VectorXd r_i = a.row(i);
	VectorXd r_j = a.row(j);
	k(i,j) = se_ard(r_i, r_j);
	if(i == j) {
	  k(i,j) += nos;
	} else {
	  k(j,i) = k(i,j);
	}
      }
    }
  }
  inline void se_ard(const MatrixXd &a, MatrixXd &k) {
    int ss = a.rows();
    if (k.rows() != ss) {
      k.resize(ss, ss);
    }
    for (int i = 0; i < ss; i++) {
      for (int j = i; j < ss; j++) {
	VectorXd r_i = a.row(i);
	VectorXd r_j = a.row(j);
	k(i,j) = se_ard(r_i, r_j);
	if(i != j) {
	  k(j,i) = k(i,j);
	}
      }
    }
  }

  //K_AB Eigen version
  inline void se_ard(const MatrixXd &a, const MatrixXd &b, MatrixXd &k) {
    int ssa = a.rows();
    int ssb = b.rows();
    k.resize(ssa, ssb);
    for (int i = 0; i < ssa; i++) {
      for (int j = 0; j < ssb; j++) {
	VectorXd r_i = a.row(i);
	VectorXd r_j = b.row(j);
	k(i,j) = se_ard(r_i,r_j);
      }
    }
  }
};
