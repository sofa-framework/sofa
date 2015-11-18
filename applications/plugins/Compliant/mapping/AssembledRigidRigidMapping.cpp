#include "AssembledRigidRigidMapping.h"


#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(AssembledRigidRigidMapping)

using namespace defaulttype;

// Register in the Factory
int AssembledRigidRigidMappingClass = core::RegisterObject("Assembled rigid displacement mapping")

#ifndef SOFA_FLOAT
.add< AssembledRigidRigidMapping< Rigid3dTypes, Rigid3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< AssembledRigidRigidMapping< Rigid3fTypes, Rigid3fTypes > >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API AssembledRigidRigidMapping<  Rigid3dTypes, Rigid3dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API AssembledRigidRigidMapping< Rigid3fTypes, Rigid3fTypes >;

#endif



} // namespace mapping

} // namespace component

} // namespace sofa

