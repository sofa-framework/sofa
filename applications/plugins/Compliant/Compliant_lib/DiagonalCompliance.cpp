#include "DiagonalCompliance.inl"
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
int DiagonalComplianceClass = core::RegisterObject("Diagonal compliance")
#ifndef SOFA_FLOAT
        .add< DiagonalCompliance< Vec1dTypes > >(true)
        .add< DiagonalCompliance< Vec3dTypes > >(true)
        .add< DiagonalCompliance< Vec6dTypes > >(true)
#endif
#ifndef SOFA_DOUBLE
        .add< DiagonalCompliance< Vec1fTypes > >(true)
        .add< DiagonalCompliance< Vec3fTypes > >(true)
        .add< DiagonalCompliance< Vec6fTypes > >(true)
#endif
        ;

SOFA_DECL_CLASS(DiagonalCompliance)

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API DiagonalCompliance<Vec1dTypes>;
template class SOFA_Compliant_API DiagonalCompliance<Vec3dTypes>;
template class SOFA_Compliant_API DiagonalCompliance<Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API DiagonalCompliance<Vec1fTypes>;
template class SOFA_Compliant_API DiagonalCompliance<Vec3fTypes>;
template class SOFA_Compliant_API DiagonalCompliance<Vec6fTypes>;
#endif

}
}
}
