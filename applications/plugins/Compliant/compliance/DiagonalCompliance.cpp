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
        .add< DiagonalCompliance< Vec1Types > >(true)
        .add< DiagonalCompliance< Vec3Types > >()
        .add< DiagonalCompliance< Vec6Types > >()

        ;

template class SOFA_Compliant_API DiagonalCompliance<Vec1Types>;
template class SOFA_Compliant_API DiagonalCompliance<Vec3Types>;
template class SOFA_Compliant_API DiagonalCompliance<Vec6Types>;


}
}
}
