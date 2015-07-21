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
int DifferenceMappingClass = core::RegisterObject("Compute position differences between dofs")
#ifndef SOFA_FLOAT
.add< DifferenceMapping< Vec3dTypes, Vec3dTypes > >()
.add< DifferenceMapping< Vec1dTypes, Vec1dTypes > >()
.add< DifferenceMapping< Rigid3dTypes, Vec3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< DifferenceMapping< Vec3fTypes, Vec3fTypes > >()
.add< DifferenceMapping< Vec1fTypes, Vec1fTypes > >()
.add< DifferenceMapping< Rigid3fTypes, Vec3fTypes > >()
#endif
;


#ifndef SOFA_FLOAT
template class SOFA_Compliant_API DifferenceMapping< Vec3dTypes, Vec3dTypes >;
template class SOFA_Compliant_API DifferenceMapping< Vec1dTypes, Vec1dTypes >;
template class SOFA_Compliant_API DifferenceMapping< Rigid3dTypes, Vec3dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API DifferenceMapping< Vec3fTypes, Vec3fTypes >;
template class SOFA_Compliant_API DifferenceMapping< Vec1fTypes, Vec1fTypes >;
template class SOFA_Compliant_API DifferenceMapping< Rigid3fTypes, Vec3fTypes >;
#endif


///////////////////////


SOFA_DECL_CLASS(DifferenceMultiMapping)

using namespace defaulttype;

// Register in the Factory
int DifferenceMultiMappingClass = core::RegisterObject("Compute position differences between two different mstates")
#ifndef SOFA_FLOAT
.add< DifferenceMultiMapping< Vec3dTypes, Vec3dTypes > >()
.add< DifferenceMultiMapping< Vec1dTypes, Vec1dTypes > >()
.add< DifferenceMultiMapping< Rigid3dTypes, Vec3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< DifferenceMultiMapping< Vec3fTypes, Vec3fTypes > >()
.add< DifferenceMultiMapping< Vec1fTypes, Vec1fTypes > >()
.add< DifferenceMultiMapping< Rigid3fTypes, Vec3fTypes > >()
#endif
;


#ifndef SOFA_FLOAT
template class SOFA_Compliant_API DifferenceMultiMapping< Vec3dTypes, Vec3dTypes >;
template class SOFA_Compliant_API DifferenceMultiMapping< Vec1dTypes, Vec1dTypes >;
template class SOFA_Compliant_API DifferenceMultiMapping< Rigid3dTypes, Vec3dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API DifferenceMultiMapping< Vec3fTypes, Vec3fTypes >;
template class SOFA_Compliant_API DifferenceMultiMapping< Vec1fTypes, Vec1fTypes >;
template class SOFA_Compliant_API DifferenceMultiMapping< Rigid3fTypes, Vec3fTypes >;
#endif



}
}
}


