#include "NormalizationMapping.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace defaulttype;

// Register in the Factory
int NormalizationMappingClass = core::RegisterObject("Compute 3d vector normalization")
#ifndef SOFA_FLOAT
.add< NormalizationMapping< Vec3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< NormalizationMapping< Vec3fTypes > >()
#endif
;


#ifndef SOFA_FLOAT
template class SOFA_Compliant_API NormalizationMapping< Vec3dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API NormalizationMapping< Vec3fTypes >;
#endif



}
}
}


