#include "GearMapping.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{


SOFA_DECL_CLASS(GearMultiMapping)

using namespace defaulttype;

// Register in the Factory
int GearMultiMappingClass = core::RegisterObject("Compute scaled velocity differences between two different kinematic dofs")
#ifndef SOFA_FLOAT
.add< GearMultiMapping< Rigid3dTypes, Vec1dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< GearMultiMapping< Rigid3fTypes, Vec1fTypes > >()
#endif
;


#ifndef SOFA_FLOAT
template class SOFA_Compliant_API GearMultiMapping< Rigid3dTypes, Vec1dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API GearMultiMapping< Rigid3fTypes, Vec1fTypes >;
#endif



}
}
}


