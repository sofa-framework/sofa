#include "SafeDistanceMapping.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

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



// Register in the Factory
int SafeDistanceFromTargetMappingClass = core::RegisterObject("Compute position SafeDistanceFromTargets between dofs")
#ifndef SOFA_FLOAT
.add< SafeDistanceFromTargetMapping< Vec3dTypes, Vec1dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< SafeDistanceFromTargetMapping< Vec3fTypes, Vec1fTypes > >()
#endif
;


#ifndef SOFA_FLOAT
template class SOFA_Compliant_API SafeDistanceFromTargetMapping< Vec3dTypes, Vec1dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API SafeDistanceFromTargetMapping< Vec3fTypes, Vec1fTypes >;
#endif


}
}
}


