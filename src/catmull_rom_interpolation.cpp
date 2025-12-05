#include "catmull_rom_interpolation.h"
#include <Eigen/Dense>
#include <algorithm>

Eigen::Vector3d catmull_rom_interpolation(
  const std::vector<std::pair<double, Eigen::Vector3d> > & keyframes,
  double t)
{
  if(keyframes.empty())
  {
    return Eigen::Vector3d(0,0,0);
  }
  if(keyframes.size() == 1)
  {
    return keyframes[0].second;
  }

  const int n = static_cast<int>(keyframes.size());
  if(t <= keyframes.front().first)
  {
    return keyframes.front().second;
  }
  if(t >= keyframes.back().first)
  {
    return keyframes.back().second;
  }

  int i = 0;
  for(int k = 0; k < n-1; ++k)
  {
    if(t >= keyframes[k].first && t <= keyframes[k+1].first)
    {
      i = k;
      break;
    }
  }

  double t0 = keyframes[i].first;
  double t1 = keyframes[i+1].first;
  double dt = t1 - t0;
  if(dt <= 0)
  {
    return keyframes[i].second;
  }
  double u = (t - t0) / dt;

  int i0 = std::max(i-1, 0);
  int i1 = i;
  int i2 = i+1;
  int i3 = std::min(i+2, n-1);

  const Eigen::Vector3d &p0 = keyframes[i0].second;
  const Eigen::Vector3d &p1 = keyframes[i1].second;
  const Eigen::Vector3d &p2 = keyframes[i2].second;
  const Eigen::Vector3d &p3 = keyframes[i3].second;

  double u2 = u*u;
  double u3 = u2*u;

  Eigen::Vector3d result =
    0.5 * ((2.0*p1) +
    (-p0 + p2) * u +
    (2.0*p0 - 5.0*p1 + 4.0*p2 - p3) * u2 +
    (-p0 + 3.0*p1 - 3.0*p2 + p3) * u3);

  return result;
}
