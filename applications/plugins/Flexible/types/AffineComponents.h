#ifndef FLEXIBLE_AffineComponents_H
#define FLEXIBLE_AffineComponents_H



#include "AffineTypes.h"

#define TYPEABSTRACTNAME Affine
#include "ComponentSpecializations.h.inl"

// instanciation
namespace sofa
{

namespace core
{

namespace behavior
{

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FLEXIBLE_AffineComponents_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_Flexible_API ForceField<defaulttype::Affine3dTypes>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_Flexible_API ForceField<defaulttype::Affine3fTypes>;
#endif
#endif

} // namespace behavior

} // namespace core

} // namespace sofa

#endif // FLEXIBLE_AffineComponents_H
