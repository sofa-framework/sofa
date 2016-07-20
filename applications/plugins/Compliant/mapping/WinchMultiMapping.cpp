#include "WinchMultiMapping.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{




SOFA_DECL_CLASS(WinchMultiMapping)

using namespace defaulttype;

// Register in the Factory
int WinchMultiMappingClass = core::RegisterObject("Compute position differences between two different mstates")
#ifndef SOFA_FLOAT
.add< WinchMultiMapping< Vec1dTypes, Vec1dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< WinchMultiMapping< Vec1fTypes, Vec1fTypes > >()
#endif
;


#ifndef SOFA_FLOAT
template class SOFA_Compliant_API WinchMultiMapping< Vec1dTypes, Vec1dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API WinchMultiMapping< Vec1fTypes, Vec1fTypes >;
#endif



}
}
}


