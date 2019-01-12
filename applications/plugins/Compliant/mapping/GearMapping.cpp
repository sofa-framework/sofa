#include "GearMapping.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{


using namespace defaulttype;

// Register in the Factory
int GearMultiMappingClass = core::RegisterObject("Compute scaled velocity differences between two different kinematic dofs")
.add< GearMultiMapping< Rigid3Types, Vec1Types > >()

;


template class SOFA_Compliant_API GearMultiMapping< Rigid3Types, Vec1Types >;




}
}
}


