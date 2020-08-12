#define SOFA_COMPONENT_COMPLIANCE_DAMPINGCOMPLIANCE_CPP
#include "DampingCompliance.h"

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace forcefield
{

using namespace defaulttype;

// Register in the Factory
int DampingComplianceClass = core::RegisterObject("Damping Compliance")
    .add< DampingCompliance< Vec6Types > >(true)
    .add< DampingCompliance< Vec2Types > >()
    .add< DampingCompliance< Vec1Types > >()

	;

template class SOFA_Compliant_API DampingCompliance<Vec6Types>;
template class SOFA_Compliant_API DampingCompliance<Vec2Types>;
template class SOFA_Compliant_API DampingCompliance<Vec1Types>;

}
}
}

