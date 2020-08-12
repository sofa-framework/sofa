#include "QuadraticMapping.h"
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace defaulttype;

// Register in the Factory
int QuadraticMappingClass = core::RegisterObject("scaled squared norm")

.add< QuadraticMapping< Vec6Types, Vec1Types > >()        
.add< QuadraticMapping< Vec3Types, Vec1Types > >()
.add< QuadraticMapping< Vec2Types, Vec1Types > >()    
.add< QuadraticMapping< Vec1Types, Vec1Types > >()

;


template class SOFA_Compliant_API QuadraticMapping< Vec6Types, Vec1Types >;
template class SOFA_Compliant_API QuadraticMapping< Vec3Types, Vec1Types >;
template class SOFA_Compliant_API QuadraticMapping< Vec2Types, Vec1Types >;
template class SOFA_Compliant_API QuadraticMapping< Vec1Types, Vec1Types >;



}
}
}


