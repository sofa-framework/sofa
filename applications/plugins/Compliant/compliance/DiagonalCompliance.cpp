#define SOFA_COMPONENT_COMPLIANCE_DIAGONALCOMPLIANCE_CPP
#include "DiagonalCompliance.inl"
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
        .add< DiagonalCompliance< Vec3dTypes > >()
        .add< DiagonalCompliance< Vec6dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< DiagonalCompliance< Vec1fTypes > >()
        .add< DiagonalCompliance< Vec3fTypes > >()
        .add< DiagonalCompliance< Vec6fTypes > >()
#endif
        ;


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
