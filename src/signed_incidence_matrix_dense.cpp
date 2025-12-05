#include "signed_incidence_matrix_dense.h"

void signed_incidence_matrix_dense(
  const int n,
  const Eigen::MatrixXi & E,
  Eigen::MatrixXd & A)
{
  const int m = static_cast<int>(E.rows());
  A.resize(m, n);
  A.setZero();

  for(int e = 0; e < m; ++e)
  {
    const int i = E(e,0);
    const int j = E(e,1);
    // Oriented edge: i -> j  ⇒  row = e_i - e_j
    A(e, i) =  1.0;
    A(e, j) = -1.0;
  }
}
