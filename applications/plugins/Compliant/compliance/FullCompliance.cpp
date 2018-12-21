#define SOFA_COMPONENT_COMPLIANCE_FULLCOMPLIANCE_CPP

#include "FullCompliance.inl"
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
int FullComplianceClass = core::RegisterObject("User provided full compliance or stiffness matrix")
        .add< FullCompliance< Vec1Types > >(true)
        .add< FullCompliance< Vec3Types > >()
        .add< FullCompliance< Vec6Types > >()

        ;

template class SOFA_Compliant_API FullCompliance<Vec1Types>;
template class SOFA_Compliant_API FullCompliance<Vec3Types>;
template class SOFA_Compliant_API FullCompliance<Vec6Types>;


}
}
}
