#include "DifferenceFromTargetMapping.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace defaulttype;

// Register in the Factory
int DifferenceFromTargetMappingClass = core::RegisterObject("Compute position differences between dofs and targets")
        .addAlias("OffsetMapping") // backward compatibility with a previous identical mapping
.add< DifferenceFromTargetMapping< Vec3Types, Vec3Types > >()
.add< DifferenceFromTargetMapping< Vec1Types, Vec1Types > >()

;


template class SOFA_Compliant_API DifferenceFromTargetMapping< Vec3Types, Vec3Types >;
template class SOFA_Compliant_API DifferenceFromTargetMapping< Vec1Types, Vec1Types >;





}
}
}


