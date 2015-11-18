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
#ifndef SOFA_FLOAT
        .add< FullCompliance< Vec1dTypes > >(true)
        .add< FullCompliance< Vec3dTypes > >()
        .add< FullCompliance< Vec6dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< FullCompliance< Vec1fTypes > >()
        .add< FullCompliance< Vec3fTypes > >()
        .add< FullCompliance< Vec6fTypes > >()
#endif
        ;

SOFA_DECL_CLASS(FullCompliance)

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API FullCompliance<Vec1dTypes>;
template class SOFA_Compliant_API FullCompliance<Vec3dTypes>;
template class SOFA_Compliant_API FullCompliance<Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API FullCompliance<Vec1fTypes>;
template class SOFA_Compliant_API FullCompliance<Vec3fTypes>;
template class SOFA_Compliant_API FullCompliance<Vec6fTypes>;
#endif

}
}
}
