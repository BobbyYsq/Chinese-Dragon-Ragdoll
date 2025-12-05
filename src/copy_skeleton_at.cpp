#include "copy_skeleton_at.h"
#include <cassert>

Skeleton copy_skeleton_at(
  const Skeleton & skeleton, 
  const Eigen::VectorXd & A)
{
  Skeleton copy = skeleton;
  assert(static_cast<int>(copy.size())*3 == A.size());
  for(int i = 0; i < static_cast<int>(copy.size()); ++i)
  {
    copy[i].xzx = A.segment<3>(3*i);
  }
  return copy;
}
