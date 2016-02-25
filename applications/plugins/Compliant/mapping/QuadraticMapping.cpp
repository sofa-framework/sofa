#include "QuadraticMapping.h"
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(QuadraticMapping)

using namespace defaulttype;

// Register in the Factory
int QuadraticMappingClass = core::RegisterObject("scaled squared norm")

#ifndef SOFA_FLOAT
.add< QuadraticMapping< Vec6dTypes, Vec1dTypes > >()        
.add< QuadraticMapping< Vec3dTypes, Vec1dTypes > >()
.add< QuadraticMapping< Vec2dTypes, Vec1dTypes > >()    
.add< QuadraticMapping< Vec1dTypes, Vec1dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< QuadraticMapping< Vec6fTypes, Vec1fTypes > >()        
.add< QuadraticMapping< Vec3fTypes, Vec1fTypes > >()
.add< QuadraticMapping< Vec2fTypes, Vec1fTypes > >()    
.add< QuadraticMapping< Vec1fTypes, Vec1fTypes > >()
#endif
;


#ifndef SOFA_FLOAT
template class SOFA_Compliant_API QuadraticMapping< Vec6dTypes, Vec1dTypes >;
template class SOFA_Compliant_API QuadraticMapping< Vec3dTypes, Vec1dTypes >;
template class SOFA_Compliant_API QuadraticMapping< Vec2dTypes, Vec1dTypes >;
template class SOFA_Compliant_API QuadraticMapping< Vec1dTypes, Vec1dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API QuadraticMapping< Vec6fTypes, Vec1fTypes >;
template class SOFA_Compliant_API QuadraticMapping< Vec3fTypes, Vec1fTypes >;
template class SOFA_Compliant_API QuadraticMapping< Vec2fTypes, Vec1fTypes >;
template class SOFA_Compliant_API QuadraticMapping< Vec1fTypes, Vec1fTypes >;
#endif


}
}
}


