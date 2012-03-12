#include "UniformCompliance.inl"
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace compliance
{

using namespace sofa::defaulttype;

// Register in the Factory
int UniformComplianceClass = core::RegisterObject("Compute edge extensions")
#ifndef SOFA_FLOAT
        .add< UniformCompliance< Vec1dTypes > >(true)
#endif
#ifndef SOFA_DOUBLE
        .add< UniformCompliance< Vec1fTypes > >(true)
#endif
        ;

SOFA_DECL_CLASS(UniformCompliance)


template class SOFA_Compliant_API UniformCompliance<Vec1dTypes>;

template class SOFA_Compliant_API UniformCompliance<Vec1fTypes>;




}
}
}
