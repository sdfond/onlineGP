#include "../lib/pitc_online.h"

int main() {
  char data_name[32] = {"../dom/traindata.txt"};
  char kern_name[32] = {"../dom/spGPkern.txt"};
  char sup_name[32] = {"../dom/support.txt"};

  pitc_online gp(kern_name);
  gp.regress(data_name, sup_name);

  return 0;
}
