#pragma once
#include <cwchar>

namespace sofa::component::solidmechanics::fem::elastic
{

template<typename Real>
constexpr static Real kroneckerDelta(std::size_t i, std::size_t j)
{
    return static_cast<Real>(i == j);
}

}
