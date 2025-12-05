#include "forward_kinematics.h"
#include "euler_angles_to_transform.h"
#include <functional>

void forward_kinematics(
  const Skeleton & skeleton,
  std::vector<Eigen::Affine3d,Eigen::aligned_allocator<Eigen::Affine3d> > & T)
{
  const int n = static_cast<int>(skeleton.size());
  T.assign(n, Eigen::Affine3d::Identity());
  if(n == 0) return;

  std::vector<bool> done(n,false);
  std::function<void(int)> dfs = [&](int i)
  {
    if(done[i]) return;
    Eigen::Affine3d parent_T = Eigen::Affine3d::Identity();
    int p = skeleton[i].parent_index;
    if(p >= 0)
    {
      if(!done[p]) dfs(p);
      parent_T = T[p];
    }
    Eigen::Affine3d R = euler_angles_to_transform(skeleton[i].xzx);
    Eigen::Affine3d rel = skeleton[i].rest_T * R * skeleton[i].rest_T.inverse();
    T[i] = parent_T * rel;
    done[i] = true;
  };

  for(int i = 0; i < n; ++i)
  {
    if(!done[i]) dfs(i);
  }
}
