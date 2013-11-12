#include "DifferenceMapping.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(DifferenceMapping)

using namespace defaulttype;

// Register in the Factory
int DifferenceMappingClass = core::RegisterObject("Compute position differences between dofs lol")
#ifndef SOFA_FLOAT
.add< DifferenceMapping< Vec3dTypes, Vec3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< DifferenceMapping< Vec3fTypes, Vec3fTypes > >()
#endif
;


#ifndef SOFA_FLOAT
template class SOFA_Compliant_API DifferenceMapping<  Vec3dTypes, Vec3dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API DifferenceMapping< Vec3fTypes, Vec3fTypes >;
#endif


///////////////////////


SOFA_DECL_CLASS(DifferenceMultiMapping)

using namespace defaulttype;

// Register in the Factory
int DifferenceMultiMappingClass = core::RegisterObject("Compute position differences between two different mstates")
#ifndef SOFA_FLOAT
.add< DifferenceMultiMapping< Vec3dTypes, Vec3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< DifferenceMultiMapping< Vec3fTypes, Vec3fTypes > >()
#endif
;


#ifndef SOFA_FLOAT
template class SOFA_Compliant_API DifferenceMultiMapping<  Vec3dTypes, Vec3dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API DifferenceMultiMapping< Vec3fTypes, Vec3fTypes >;
#endif



}
}
}

