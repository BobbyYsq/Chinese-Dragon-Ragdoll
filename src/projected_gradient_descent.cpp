#include "projected_gradient_descent.h"
#include "line_search.h"
#include <cmath>

void projected_gradient_descent(
  const std::function<double(const Eigen::VectorXd &)> & f,
  const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> & grad_f,
  const std::function<void(Eigen::VectorXd &)> & proj_z,
  const int max_iters,
  Eigen::VectorXd & z)
{
  proj_z(z);
  if(max_iters <= 0) return;

  for(int iter = 0; iter < max_iters; ++iter)
  {
    Eigen::VectorXd g = grad_f(z);
    double gnorm = g.norm();
    if(gnorm == 0.0 || !std::isfinite(gnorm))
    {
      break;
    }

    Eigen::VectorXd dz = -g;
    double step = line_search(f, proj_z, z, dz, 10000.0);
    if(step <= 0.0)
    {
      break;
    }

    z = z + step * dz;
    proj_z(z);
  }
}
