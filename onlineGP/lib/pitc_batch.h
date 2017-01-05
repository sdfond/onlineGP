/*
 *  @brief This file provides the predictor with PITC sparse Gaussian Process.
 *  @the prediction is in batch
 */
#include "cov.h"

class pitc_batch
{
private:
  LLT<MatrixXd> chol_kuu;
  MatrixXd gs_kuu;
  VectorXd gs_zu;
  gp_kern *kern;

public:
  void pitc_predict(const MatrixXd &aset,
		    const MatrixXd &tset,
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
      pmu[i] += kern->mean;
    }

    K_ut = K_tu.transpose();
    tmp = chol_suu.solve(K_ut);
    for (int i = 0; i < K_tu.rows(); i++) {
      pvar[i] = kern->sig + kern->nos;
    }
    printf("%lf %lf\n", kern->sig, kern->nos);
    tmp = K_tu * tmp;
    pvar += tmp.diagonal();
    tmp = chol_kuu.solve(K_ut);
    tmp = K_tu * tmp;
    pvar -= tmp.diagonal();
  }
  
  void pitc_update(const MatrixXd &D,
		   const MatrixXd &aset) {
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
      v[i] = v[i] - kern->mean;
    }
    chol_sdd.solveInPlace(v);
    ls_zu = kuf * v;

    tmp = chol_sdd.solve(kfu);
    ls_kuu = kuf * tmp;

    //global summary update
    gs_zu += ls_zu;
    gs_kuu += ls_kuu;
  }

  pitc_batch(char * hypf) {
    kern = new gp_kern(hypf);
  }

  ~pitc_batch() {
    delete kern;
  }

  void init(const MatrixXd &aset) {
    MatrixXd kuu;
    kern->se_ard(aset, kuu);
    chol_kuu.compute(kuu);

    gs_kuu = kuu;
    gs_zu = VectorXd::Zero(aset.rows());
  }
  void regress(char * train, char * test, char * support, char * output) {
    FILE * fp = fopen(support, "r");
    MatrixXd D, tset, uset;
    int ss, ts, ds, blk_size;
    double tmp;
 
    fscanf(fp, "%d", &ss);
    uset.resize(ss, kern->dim);
    for (int i = 0; i < ss; i++) {
      for (int j = 0; j < kern->dim; j++) {
	fscanf(fp, "%lf", &tmp);
	uset(i,j) = tmp;
      }
    }
    fclose(fp);

    fp = fopen(train, "r");
    fscanf(fp, "%d %d", &ds, &blk_size);
    D.resize(ds, kern->dim+1);
    for (int i = 0; i < ds; i++) {
      for (int j = 0; j < kern->dim + 1; j++) {
	fscanf(fp, "%lf", &tmp);
	D(i,j) = tmp;
      }
    }
    fclose(fp);

    fp = fopen(test, "r");
    fscanf(fp, "%d", &ts);
    tset.resize(ts, kern->dim+1);
    for (int i = 0; i < ts; i++) {
      for (int j = 0; j < kern->dim + 1; j++) {
	fscanf(fp, "%lf", &tmp);
	tset(i,j) = tmp;
      }
    }
    fclose(fp);
    
    VectorXd pmu(ts);
    VectorXd pvar(ts);
    
    init(uset);
    int blk_num = ds / blk_size;
    
    for (int i = 0; i < blk_num; i++) {
      MatrixXd blk_data = D.block(i*blk_size, 0, blk_size, kern->dim+1);
      pitc_update(blk_data, uset);
    }

    pitc_predict(uset, tset, pmu, pvar);


    fp = fopen(output, "w");
    if(fp == NULL) {
      throw("Fail to open output file\n");
    }
    
    for(int i = 0; i < pmu.size(); i++) {
      fprintf(fp, "%.8lf %.8lf\n", pmu[i], pvar[i]);
    }
   
    fclose(fp);
  }

};

