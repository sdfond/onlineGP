#include "../lib/pitc_batch.h"

int main() {
  char data_name[32] = {"../dom/traindata.txt"};
  char test_name[32] = {"../dom/testdata.txt"};
  char kern_name[32] = {"../dom/spGPkern.txt"};
  char sup_name[32] = {"../dom/support.txt"};
  char output_name[32] = {"Rst.txt"};

  pitc_batch gp(kern_name);
  gp.regress(data_name, test_name, sup_name, output_name);

  return 0;
}
