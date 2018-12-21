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
.add< AdditionMapping< Vec3Types, Vec3Types > >()
.add< AdditionMapping< Vec1Types, Vec1Types > >()
.add< AdditionMapping< Rigid3Types, Vec3Types > >()

;


template class SOFA_Compliant_API AdditionMapping< Vec3Types, Vec3Types >;
template class SOFA_Compliant_API AdditionMapping< Vec1Types, Vec1Types >;
template class SOFA_Compliant_API AdditionMapping< Rigid3Types, Vec3Types >;



///////////////////////


using namespace defaulttype;

// Register in the Factory
int AdditionMultiMappingClass = core::RegisterObject("Compute position Additions between two different mstates")
.add< AdditionMultiMapping< Vec3Types, Vec3Types > >()
.add< AdditionMultiMapping< Vec1Types, Vec1Types > >()
.add< AdditionMultiMapping< Rigid3Types, Vec3Types > >()

;


template class SOFA_Compliant_API AdditionMultiMapping< Vec3Types, Vec3Types >;
template class SOFA_Compliant_API AdditionMultiMapping< Vec1Types, Vec1Types >;
template class SOFA_Compliant_API AdditionMultiMapping< Rigid3Types, Vec3Types >;




}
}
}


