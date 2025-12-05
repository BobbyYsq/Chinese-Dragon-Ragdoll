#include "fast_mass_springs_step_sparse.h"
#include <igl/matlab_format.h>

void fast_mass_springs_step_sparse(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXi & b,
  const double delta_t,
  const Eigen::MatrixXd & fext,
  const Eigen::VectorXd & r,
  const Eigen::SparseMatrix<double>  & M,
  const Eigen::SparseMatrix<double>  & A,
  const Eigen::SparseMatrix<double>  & C,
  const Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > & prefactorization,
  const Eigen::MatrixXd & Uprev,
  const Eigen::MatrixXd & Ucur,
  Eigen::MatrixXd & Unext)
{
  const int n  = static_cast<int>(V.rows());
  const int ne = static_cast<int>(E.rows());
  const double h  = delta_t;
  const double h2 = h * h;
  const double inv_h2 = 1.0 / h2;
  const double w = 1e10;
  const int max_iter = 50;

  Eigen::VectorXd mass_diag = M.diagonal();

  Eigen::MatrixXd Y(n,3);
  Y.setZero();
  for(int i = 0; i < n; ++i)
  {
    const double mi = mass_diag(i);
    Eigen::RowVector3d term = 2.0 * Ucur.row(i) - Uprev.row(i);
    if(mi > 0.0)
    {
      term += (h2 / mi) * fext.row(i);
    }
    Y.row(i) = term;
  }

  Eigen::MatrixXd B_mass(n,3);
  for(int i = 0; i < n; ++i)
  {
    const double coeff = mass_diag(i) * inv_h2;
    B_mass.row(i) = coeff * Y.row(i);
  }

  Eigen::MatrixXd CV = C * V;
  Eigen::MatrixXd B_pin = w * C.transpose() * CV;

  Eigen::MatrixXd U = Ucur;
  Eigen::MatrixXd d(ne,3);

  for(int iter = 0; iter < max_iter; ++iter)
  {
    Eigen::MatrixXd AU = A * U;
    for(int e = 0; e < ne; ++e)
    {
      Eigen::RowVector3d v = AU.row(e);
      const double len = v.norm();
      if(len > 1e-8)
      {
        d.row(e) = (r(e) / len) * v;
      }
      else
      {
        d.row(e).setZero();
      }
    }

    Eigen::MatrixXd B_spring = k * A.transpose() * d;
    Eigen::MatrixXd B = B_mass + B_spring + B_pin;

    Unext.resize(n,3);
    for(int dim = 0; dim < 3; ++dim)
    {
      Unext.col(dim) = prefactorization.solve(B.col(dim));
    }

    U = Unext;
  }
}
