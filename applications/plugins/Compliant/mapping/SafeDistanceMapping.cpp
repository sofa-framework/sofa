#include "SafeDistanceMapping.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(SafeDistanceMapping)

using namespace defaulttype;

// Register in the Factory
int SafeDistanceMappingClass = core::RegisterObject("Compute position SafeDistances between dofs")
#ifndef SOFA_FLOAT
.add< SafeDistanceMapping< Vec3dTypes, Vec1dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< SafeDistanceMapping< Vec3fTypes, Vec1fTypes > >()
#endif
;


#ifndef SOFA_FLOAT
template class SOFA_Compliant_API SafeDistanceMapping< Vec3dTypes, Vec1dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API SafeDistanceMapping< Vec3fTypes, Vec1fTypes >;
#endif


/////////////////////////


}
}
}


