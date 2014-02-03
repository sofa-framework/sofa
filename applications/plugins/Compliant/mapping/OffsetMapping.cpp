#include "OffsetMapping.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(OffsetMapping)

using namespace defaulttype;

// Register in the Factory
int OffsetMappingClass = core::RegisterObject("Compute position differences between dofs")
#ifndef SOFA_FLOAT
.add< OffsetMapping< Vec3dTypes, Vec3dTypes > >()
.add< OffsetMapping< Vec1dTypes, Vec1dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< OffsetMapping< Vec3fTypes, Vec3fTypes > >()
.add< OffsetMapping< Vec1fTypes, Vec1fTypes > >()
#endif
;


#ifndef SOFA_FLOAT
template class SOFA_Compliant_API OffsetMapping< Vec3dTypes, Vec3dTypes >;
template class SOFA_Compliant_API OffsetMapping< Vec1dTypes, Vec1dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API OffsetMapping< Vec3fTypes, Vec3fTypes >;
template class SOFA_Compliant_API OffsetMapping< Vec1fTypes, Vec1fTypes >;
#endif


}
}
}


