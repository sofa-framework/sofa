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
#ifndef SOFA_FLOAT
        .add< UniformCompliance< Vec1dTypes > >(true)
	.add< UniformCompliance< Vec3dTypes > >(true)
        .add< UniformCompliance< Vec6dTypes > >(true)
#endif
#ifndef SOFA_DOUBLE
        .add< UniformCompliance< Vec1fTypes > >(true)
	.add< UniformCompliance< Vec3fTypes > >(true)
        .add< UniformCompliance< Vec6fTypes > >(true)
#endif
        ;

SOFA_DECL_CLASS(UniformCompliance)


template class SOFA_Compliant_API UniformCompliance<Vec1dTypes>;
template class SOFA_Compliant_API UniformCompliance<Vec3dTypes>;
template class SOFA_Compliant_API UniformCompliance<Vec6dTypes>;

template class SOFA_Compliant_API UniformCompliance<Vec1fTypes>;
template class SOFA_Compliant_API UniformCompliance<Vec3fTypes>;
template class SOFA_Compliant_API UniformCompliance<Vec6fTypes>;


}
}
}
