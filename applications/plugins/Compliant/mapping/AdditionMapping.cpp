#include "AdditionMapping.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{


using namespace defaulttype;

// Register in the Factory
int AdditionMappingClass = core::RegisterObject("Compute position Additions between dofs")
#ifndef SOFA_FLOAT
.add< AdditionMapping< Vec3dTypes, Vec3dTypes > >()
.add< AdditionMapping< Vec1dTypes, Vec1dTypes > >()
.add< AdditionMapping< Rigid3dTypes, Vec3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< AdditionMapping< Vec3fTypes, Vec3fTypes > >()
.add< AdditionMapping< Vec1fTypes, Vec1fTypes > >()
.add< AdditionMapping< Rigid3fTypes, Vec3fTypes > >()
#endif
;


#ifndef SOFA_FLOAT
template class SOFA_Compliant_API AdditionMapping< Vec3dTypes, Vec3dTypes >;
template class SOFA_Compliant_API AdditionMapping< Vec1dTypes, Vec1dTypes >;
template class SOFA_Compliant_API AdditionMapping< Rigid3dTypes, Vec3dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API AdditionMapping< Vec3fTypes, Vec3fTypes >;
template class SOFA_Compliant_API AdditionMapping< Vec1fTypes, Vec1fTypes >;
template class SOFA_Compliant_API AdditionMapping< Rigid3fTypes, Vec3fTypes >;
#endif


///////////////////////



using namespace defaulttype;

// Register in the Factory
int AdditionMultiMappingClass = core::RegisterObject("Compute position Additions between two different mstates")
#ifndef SOFA_FLOAT
.add< AdditionMultiMapping< Vec3dTypes, Vec3dTypes > >()
.add< AdditionMultiMapping< Vec1dTypes, Vec1dTypes > >()
.add< AdditionMultiMapping< Rigid3dTypes, Vec3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< AdditionMultiMapping< Vec3fTypes, Vec3fTypes > >()
.add< AdditionMultiMapping< Vec1fTypes, Vec1fTypes > >()
.add< AdditionMultiMapping< Rigid3fTypes, Vec3fTypes > >()
#endif
;


#ifndef SOFA_FLOAT
template class SOFA_Compliant_API AdditionMultiMapping< Vec3dTypes, Vec3dTypes >;
template class SOFA_Compliant_API AdditionMultiMapping< Vec1dTypes, Vec1dTypes >;
template class SOFA_Compliant_API AdditionMultiMapping< Rigid3dTypes, Vec3dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API AdditionMultiMapping< Vec3fTypes, Vec3fTypes >;
template class SOFA_Compliant_API AdditionMultiMapping< Vec1fTypes, Vec1fTypes >;
template class SOFA_Compliant_API AdditionMultiMapping< Rigid3fTypes, Vec3fTypes >;
#endif



}
}
}


