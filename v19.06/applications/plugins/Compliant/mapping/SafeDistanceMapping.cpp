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
.add< SafeDistanceMapping< Vec3Types, Vec1Types > >()

;


template class SOFA_Compliant_API SafeDistanceMapping< Vec3Types, Vec1Types >;



/////////////////////////



// Register in the Factory
int SafeDistanceFromTargetMappingClass = core::RegisterObject("Compute position SafeDistanceFromTargets between dofs")
.add< SafeDistanceFromTargetMapping< Vec3Types, Vec1Types > >()

;


template class SOFA_Compliant_API SafeDistanceFromTargetMapping< Vec3Types, Vec1Types >;



}
}
}


