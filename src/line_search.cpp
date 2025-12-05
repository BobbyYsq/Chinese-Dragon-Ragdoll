#include "line_search.h"
#include <iostream>
#include <cmath>

double line_search(
  const std::function<double(const Eigen::VectorXd &)> & f,
  const std::function<void(Eigen::VectorXd &)> & proj_z,
  const Eigen::VectorXd & z,
  const Eigen::VectorXd & dz,
  const double max_step)
{
  if(max_step <= 0.0) return 0.0;
  if(dz.norm() == 0.0) return 0.0;

  const double factor = 0.5;
  double step = max_step;
  const double f0 = f(z);

  Eigen::VectorXd z_new(z.size());
  for(int iter = 0; iter < 64; ++iter)
  {
    z_new = z + step * dz;
    proj_z(z_new);
    double f_new = f(z_new);
    if(f_new < f0)
    {
      return step;
    }
    step *= factor;
    if(step <= 1e-10)
    {
      break;
    }
  }
  return 0.0;
}
