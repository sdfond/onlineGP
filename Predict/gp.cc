#include "gp.h"

// argc has three parameters, training file, test file and hyper file name
int main(int argc, char *argv[])
{
  MatrixXd trainX, testX;
  VectorXd trainY, testY;
  
  FullGP fgp(argv[3]);
  FILE *fp_train = fopen(argv[1], "r");
  FILE *fp_test = fopen(argv[2], "r");

  if (fp_train == NULL || fp_test == NULL)
    throw("cannot open training or testing file.\n");
  double tmp;
  trainX.resize(fgp.kern->num_train, fgp.kern->dim);
  testX.resize(fgp.kern->num_test, fgp.kern->dim);
  trainY.resize(fgp.kern->num_train);
  testY.resize(fgp.kern->num_test);
  for (int i = 0; i < fgp.kern->num_train; i++) {
    for (int j = 0; j < fgp.kern->dim; j++) {
      fscanf(fp_train, "%lf ", &tmp);
      trainX(i,j) = tmp;
    }
    fscanf(fp_train, "%lf ", &tmp);
    trainY[i] = tmp;
  }
  for (int i = 0; i < fgp.kern->num_test; i++) {
    for (int j = 0; j < fgp.kern->dim; j++) {
      fscanf(fp_test, "%lf ", &tmp);
      testX(i,j) = tmp;
    }
    fscanf(fp_test, "%lf ", &tmp);
    testY[i] = tmp;
  }

  fclose(fp_train);
  fclose(fp_test);

  MatrixXd K_yy, k_star;
  VectorXd K_ss, pmean, pvar;

  fgp.kernCreate(trainX, trainY, testX, K_yy, k_star, K_ss);
  fgp.predict(K_yy, k_star, K_ss, trainY, pmean, pvar);

  // change the name of output file if necessary
  char resf[] = "result.txt";
  FILE *res = fopen(resf, "w");
  for (int i = 0; i < pmean.size(); i++)
    fprintf(res, "%lf %lf\n", pmean[i], pvar[i]);
  return 0;
}
