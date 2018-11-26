#include "DotProductMapping.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace defaulttype;

// Register in the Factory
int DotProductMappingClass = core::RegisterObject("Compute Dot Products between dofs")
#ifndef SOFA_FLOAT
.add< DotProductMapping< Vec3dTypes, Vec1dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< DotProductMapping< Vec3fTypes, Vec1fTypes > >()
#endif
;


#ifndef SOFA_FLOAT
template class SOFA_Compliant_API DotProductMapping< Vec3dTypes, Vec1dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API DotProductMapping< Vec3fTypes, Vec1fTypes >;
#endif


///////////////////////


using namespace defaulttype;

// Register in the Factory
int DotProductMultiMappingClass = core::RegisterObject("Compute Dot Products between dofs in different mstates")
#ifndef SOFA_FLOAT
.add< DotProductMultiMapping< Vec3dTypes, Vec1dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< DotProductMultiMapping< Vec3fTypes, Vec1fTypes > >()
#endif
;


#ifndef SOFA_FLOAT
template class SOFA_Compliant_API DotProductMultiMapping< Vec3dTypes, Vec1dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API DotProductMultiMapping< Vec3fTypes, Vec1fTypes >;
#endif


///////////////////////

using namespace defaulttype;

// Register in the Factory
int DotProductFromTargetMappingClass = core::RegisterObject("Compute Dot Products from dofs to targets")
#ifndef SOFA_FLOAT
.add< DotProductFromTargetMapping< Vec3dTypes, Vec1dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< DotProductFromTargetMapping< Vec3fTypes, Vec1fTypes > >()
#endif
;


#ifndef SOFA_FLOAT
template class SOFA_Compliant_API DotProductFromTargetMapping< Vec3dTypes, Vec1dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API DotProductFromTargetMapping< Vec3fTypes, Vec1fTypes >;
#endif

}
}
}


