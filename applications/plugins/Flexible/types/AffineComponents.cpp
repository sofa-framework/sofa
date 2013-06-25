#define FLEXIBLE_AffineComponents_CPP // warning this define must follow the shape FLEXIBLE_TYPEABSTRACTNAMEComponents_CPP

#include "AffineComponents.h"

#define TYPEABSTRACTNAME Affine
#include "ComponentSpecializations.cpp.inl"

namespace sofa
{

namespace core
{

namespace behavior
{

#ifndef SOFA_FLOAT
template class SOFA_Flexible_API ForceField<defaulttype::Affine3dTypes>;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Flexible_API ForceField<defaulttype::Affine3fTypes>;
#endif

} // namespace behavior

} // namespace core

} // namespace sofa
