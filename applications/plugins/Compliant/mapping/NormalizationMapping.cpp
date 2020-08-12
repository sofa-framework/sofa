#include "NormalizationMapping.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace defaulttype;

// Register in the Factory
int NormalizationMappingClass = core::RegisterObject("Compute 3d vector normalization")
.add< NormalizationMapping< Vec3Types > >()

;


template class SOFA_Compliant_API NormalizationMapping< Vec3Types >;




}
}
}


