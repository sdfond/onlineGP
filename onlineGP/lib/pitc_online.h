/*
 *  @brief This file provides the predictor with PITC sparse Gaussian Process.
 *  @the prediction is in online
 */
#include "cov.h"

class pitc_online
{
private:
  LLT<MatrixXd> chol_kuu;
  MatrixXd gs_kuu, aset;
  VectorXd gs_zu;
  gp_kern *kern;
  // the block size can be changed accordingly
  const int block_size;

  // initialize global and local summary
  // they are used for prediction
  void init(const MatrixXd &D) {
    aset = D;
    MatrixXd kuu;
    kern->se_ard(aset, kuu);
    chol_kuu.compute(kuu);

    gs_kuu = kuu;
    gs_zu = VectorXd::Zero(aset.rows());
  }

  void pitc_update(const MatrixXd &D) {
    MatrixXd kff, kfu, kuf, tmp, ls_kuu;
    VectorXd ls_zu;

    kern->se_ard_n(D, kff);
    kern->se_ard(D, aset, kfu);
    kuf = kfu.transpose();
    tmp = chol_kuu.solve(kuf);
    tmp = kfu * tmp;
    kff -= tmp;
    // Now we have got kff - kfu * kuu^-1 * kuf + noise

    //start to obtain local summary
    LLT<MatrixXd> chol_sdd(kff);
    VectorXd v = D.col(kern->dim);
    for (int i = 0; i < v.size(); i++) {
      v[i] = (v[i] - kern->mean) / kern->var;
    }
    chol_sdd.solveInPlace(v);
    ls_zu = kuf * v;

    tmp = chol_sdd.solve(kfu);
    ls_kuu = kuf * tmp;

    //global summary update
    gs_zu += ls_zu;
    gs_kuu += ls_kuu;
  }

  void read_data(char * fname, MatrixXd &D, bool flag) {
    FILE * fp = fopen(fname, "r");
    int num, n_col = flag ? kern->dim + 1 : kern->dim;
    double tmp;
    fscanf(fp, "%d", &num);
    D.resize(num, kern->dim+1);
    for (int i = 0; i < num; i++) {
      for (int j = 0; j < n_col; j++) {
	fscanf(fp, "%lf", &tmp);
	D(i,j) = tmp;
      }
    }
    fclose(fp);
  }

public:
  // aset is the support set
  // tset is the testing data
  void pitc_predict(const MatrixXd &tset,
		    VectorXd &pmu, VectorXd &pvar) {
    LLT<MatrixXd> chol_suu(gs_kuu);
    MatrixXd K_tu, K_ut, tmp;
    int ts = tset.rows();

    pmu.resize(ts);
    pvar.resize(ts);
    kern->se_ard(tset, aset, K_tu);
    pmu = chol_suu.solve(gs_zu);
    pmu = K_tu * pmu;
    for (int i = 0; i < pmu.size(); i++) {
      pmu[i] = pmu[i] * kern->var + kern->mean;
    }

    K_ut = K_tu.transpose();
    tmp = chol_suu.solve(K_ut);
    for (int i = 0; i < K_tu.rows(); i++) {
      pvar[i] = kern->sig + kern->nos;
    }
    tmp = K_tu * tmp;
    pvar += tmp.diagonal();
    tmp = chol_kuu.solve(K_ut);
    tmp = K_tu * tmp;
    pvar -= tmp.diagonal();

    for (int i = 0; i < pvar.size(); i++) {
      pvar[i] = pvar[i] * kern->var * kern->var;
    }
  }


 pitc_online(char * hypf): block_size(8) {
    kern = new gp_kern(hypf);
  }

  ~pitc_online() {
    delete kern;
  }


  // at any time step, can do either of the following operation:
  // 1. do prediction on testing point T_i
  // 2. insert training point D_i
  void regress(char * data_name, char * aset_name) {
    MatrixXd D;
    VectorXd pmu;
    VectorXd pvar;

    // obtain the support set
    // read in the support set
    // support set can be selected in multiple ways, such as kmeans or random selection
    // its selection is not implemented here, it is obtained from a specified data file
    read_data(aset_name, D, false);
    init(D);
    // obtain the data
    read_data(data_name, D, true);


    // repeat the following process:
    // 1. do prediction on data point D_i (D_i is the testing set)
    // 2. insert D_i into the model (D_i will be training data for further prediction)
    // 3. i++
    // the update operation will be done once the new data size equals block_size

    FILE * res = fopen("res.txt", "w");
    for (int i = 0; i < D.rows(); i++) {
      pitc_predict(D.block(i, 0, 1, kern->dim+1), pmu, pvar);
      for (int j = 0; j < pmu.size(); j++) {
	fprintf(res, "%lf %lf %lf\n", pmu[j], D(i,kern->dim), pvar[j]);
      }
      if ((i + 1) % block_size == 0)
	pitc_update(D.block(i-block_size+1, 0, block_size, kern->dim+1));
    }
    fclose(res);
  }

};

