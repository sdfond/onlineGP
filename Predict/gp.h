#include "cov.h"

class FullGP
{
public:
    gp_kern *kern;
    void predict(const MatrixXd &K_yy, const MatrixXd &k_star, const VectorXd &K_ss, const VectorXd &trainY, VectorXd &pmean, VectorXd &pvar);
    void kernCreate(const MatrixXd &trainX, const VectorXd &trainY, const MatrixXd &testX, MatrixXd &K_yy, MatrixXd &k_star, VectorXd &K_ss);
    FullGP(char * hypf) {
      kern = new gp_kern(hypf);
  }

    ~FullGP() {
      delete kern;
    }
};

//sequence of hyper: length_scale_1, ..., length_scale_d, signal, noise
//totally d + 2 hyper-parameters where d is the input dimension

//trainX is a nxd matrix where n is size of data, d is size of input dimension
void FullGP::kernCreate(const MatrixXd &trainX, const VectorXd &trainY, const MatrixXd &testX, MatrixXd &K_yy, MatrixXd &k_star, VectorXd &K_ss) {
  MatrixXd cov_star;

    //compute kernel matrix K_yy
    kern->se_ard_n(trainX, K_yy);
    //compute the prediction
    kern->se_ard(testX, trainX, k_star);
    kern->se_ard_n(testX, cov_star);
    K_ss = cov_star.diagonal();
}

//k_star compute kernel matrix between test data and training data
//K_yy is the kernel matrix of training data
//K_ss is the kernel matrix of testing data
void FullGP::predict(const MatrixXd &K_yy, const MatrixXd &k_star, const VectorXd &K_ss, const VectorXd &trainY, VectorXd &pmean, VectorXd &pvar) {
    //k_star_T is the transpose of k_star
    //y is the normalized training output

    VectorXd alpha;
    MatrixXd k_star_T, beta, pcov;
    

    LLT<MatrixXd> chol_kyy;
    //get K_yy^-1
    chol_kyy.compute(K_yy);

    //alpha = K_yy^-1 * y
    alpha = chol_kyy.solve(trainY);

    //predict mean
    pmean = k_star * alpha;
    //scale the mean back
    for (int i = 0; i < pmean.size(); i++)
      pmean[i] = pmean[i] * kern->var + kern->mean;
    
    k_star_T = k_star.transpose();
    //beta = K_yy^-1 * k_star^T

    beta = chol_kyy.solve(k_star_T);
    pcov = k_star * beta;
 
    //predict variance
    pvar = K_ss - pcov.diagonal();
    //scale variance back
    pvar = pvar * kern->var * kern->var;
}
