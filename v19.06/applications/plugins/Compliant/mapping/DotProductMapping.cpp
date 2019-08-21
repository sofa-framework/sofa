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
.add< DotProductMapping< Vec3Types, Vec1Types > >()

;


template class SOFA_Compliant_API DotProductMapping< Vec3Types, Vec1Types >;



///////////////////////


using namespace defaulttype;

// Register in the Factory
int DotProductMultiMappingClass = core::RegisterObject("Compute Dot Products between dofs in different mstates")
.add< DotProductMultiMapping< Vec3Types, Vec1Types > >()

;


template class SOFA_Compliant_API DotProductMultiMapping< Vec3Types, Vec1Types >;



///////////////////////

using namespace defaulttype;

// Register in the Factory
int DotProductFromTargetMappingClass = core::RegisterObject("Compute Dot Products from dofs to targets")
.add< DotProductFromTargetMapping< Vec3Types, Vec1Types > >()

;


template class SOFA_Compliant_API DotProductFromTargetMapping< Vec3Types, Vec1Types >;


}
}
}


