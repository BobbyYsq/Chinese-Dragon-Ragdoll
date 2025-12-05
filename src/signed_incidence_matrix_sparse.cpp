#include "signed_incidence_matrix_sparse.h"
#include <vector>

void signed_incidence_matrix_sparse(
  const int n,
  const Eigen::MatrixXi & E,
  Eigen::SparseMatrix<double>  & A)
{
  const int m = static_cast<int>(E.rows());
  std::vector<Eigen::Triplet<double> > ijv;
  ijv.reserve(2 * m);

  for(int e = 0; e < m; ++e)
  {
    const int i = E(e,0);
    const int j = E(e,1);
    ijv.emplace_back(e, i,  1.0);
    ijv.emplace_back(e, j, -1.0);
  }

  A.resize(m, n);
  A.setFromTriplets(ijv.begin(), ijv.end());
}
