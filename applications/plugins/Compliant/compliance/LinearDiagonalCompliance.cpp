#include "LinearDiagonalCompliance.inl"
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace forcefield
{

using namespace sofa::defaulttype;

// Register in the Factory
int LinearDiagonalComplianceClass = core::RegisterObject("Linear Diagonal compliance")
#ifndef SOFA_FLOAT
        .add< LinearDiagonalCompliance< Vec1dTypes > >(true)
        .add< LinearDiagonalCompliance< Vec3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< LinearDiagonalCompliance< Vec1fTypes > >()
        .add< LinearDiagonalCompliance< Vec3fTypes > >()
#endif
        ;

SOFA_DECL_CLASS(LinearDiagonalCompliance)

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API LinearDiagonalCompliance<Vec1dTypes>;
template class SOFA_Compliant_API LinearDiagonalCompliance<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API LinearDiagonalCompliance<Vec1fTypes>;
template class SOFA_Compliant_API LinearDiagonalCompliance<Vec3fTypes>;
#endif

}
}
}
