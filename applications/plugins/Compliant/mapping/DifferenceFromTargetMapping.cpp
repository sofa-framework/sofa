#include "DifferenceFromTargetMapping.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(DifferenceFromTargetMapping)

using namespace defaulttype;

// Register in the Factory
int DifferenceFromTargetMappingClass = core::RegisterObject("Compute position differences between dofs and targets")
        .addAlias("OffsetMapping") // backward compatibility with a previous identical mapping
#ifndef SOFA_FLOAT
.add< DifferenceFromTargetMapping< Vec3dTypes, Vec3dTypes > >()
.add< DifferenceFromTargetMapping< Vec1dTypes, Vec1dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< DifferenceFromTargetMapping< Vec3fTypes, Vec3fTypes > >()
.add< DifferenceFromTargetMapping< Vec1fTypes, Vec1fTypes > >()
#endif
;


#ifndef SOFA_FLOAT
template class SOFA_Compliant_API DifferenceFromTargetMapping< Vec3dTypes, Vec3dTypes >;
template class SOFA_Compliant_API DifferenceFromTargetMapping< Vec1dTypes, Vec1dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API DifferenceFromTargetMapping< Vec3fTypes, Vec3fTypes >;
template class SOFA_Compliant_API DifferenceFromTargetMapping< Vec1fTypes, Vec1fTypes >;
#endif




}
}
}


