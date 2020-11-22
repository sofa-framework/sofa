#include "DifferenceMapping.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace defaulttype;

// Register in the Factory
int DifferenceMappingClass = core::RegisterObject("Compute position differences between dofs")
.add< DifferenceMapping< Vec3Types, Vec3Types > >()
.add< DifferenceMapping< Vec1Types, Vec1Types > >()
.add< DifferenceMapping< Rigid3Types, Vec3Types > >()

;


template class SOFA_Compliant_API DifferenceMapping< Vec3Types, Vec3Types >;
template class SOFA_Compliant_API DifferenceMapping< Vec1Types, Vec1Types >;
template class SOFA_Compliant_API DifferenceMapping< Rigid3Types, Vec3Types >;



///////////////////////


using namespace defaulttype;

// Register in the Factory
int DifferenceMultiMappingClass = core::RegisterObject("Compute position differences between two different mstates")
.add< DifferenceMultiMapping< Vec3Types, Vec3Types > >()
.add< DifferenceMultiMapping< Vec1Types, Vec1Types > >()
.add< DifferenceMultiMapping< Rigid3Types, Vec3Types > >()

;


template class SOFA_Compliant_API DifferenceMultiMapping< Vec3Types, Vec3Types >;
template class SOFA_Compliant_API DifferenceMultiMapping< Vec1Types, Vec1Types >;
template class SOFA_Compliant_API DifferenceMultiMapping< Rigid3Types, Vec3Types >;




}
}
}


