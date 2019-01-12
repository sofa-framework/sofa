#define SOFA_COMPONENT_COMPLIANCE_UNIFORMCOMPLIANCE_CPP

#include "UniformCompliance.inl"
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
int UniformComplianceClass = core::RegisterObject("Uniform compliance")
        .add< UniformCompliance< Vec1Types > >()
        .add< UniformCompliance< Vec2Types > >()
        .add< UniformCompliance< Vec3Types > >()
        .add< UniformCompliance< Vec6Types > >()

        ;

template class SOFA_Compliant_API UniformCompliance<Vec1Types>;
template class SOFA_Compliant_API UniformCompliance<Vec2Types>;
template class SOFA_Compliant_API UniformCompliance<Vec3Types>;
template class SOFA_Compliant_API UniformCompliance<Vec6Types>;


}
}
}
