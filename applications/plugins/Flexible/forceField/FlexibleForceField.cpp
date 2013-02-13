#define SOFA_FLEXIBLECOMPATIBLITY_CPP

#include "FlexibleForceField.h"
#include <sofa/core/behavior/ForceField.inl>

namespace sofa
{

namespace core
{

namespace behavior
{

using namespace sofa::defaulttype;

template class SOFA_Flexible_API ForceField<Affine3Types>;

} // namespace behavior

} // namespace core

} // namespace sofa