#define SOFA_COMPONENT_COMPLIANCE_LINEARDIAGONALCOMPLIANCE_CPP

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
        .add< LinearDiagonalCompliance< Vec1Types > >(true)
        .add< LinearDiagonalCompliance< Vec3Types > >()

        ;

template class SOFA_Compliant_API LinearDiagonalCompliance<Vec1Types>;
template class SOFA_Compliant_API LinearDiagonalCompliance<Vec3Types>;


}
}
}
