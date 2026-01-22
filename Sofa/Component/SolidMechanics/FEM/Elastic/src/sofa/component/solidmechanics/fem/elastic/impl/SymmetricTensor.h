#pragma once
#include <sofa/config.h>

namespace sofa::component::solidmechanics::fem::elastic::symmetric_tensor
{

/// The number of independent elements in a symmetric 2nd-order tensor of size (N x N)
template<sofa::Size N>
constexpr sofa::Size NumberOfIndependentElements = N * (N + 1) / 2;

}
