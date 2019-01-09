#include "AffineMultiMapping.h"

#include <sofa/core/ObjectFactory.h>


namespace sofa {
namespace component {
namespace mapping {

using namespace defaulttype;

// Register in the Factory
int AffineMultiMappingClass = core::RegisterObject("Arbitrary affine mapping")

.add< AffineMultiMapping< Vec6Types, Vec1Types > >()
.add< AffineMultiMapping< Vec3Types, Vec1Types > >()
.add< AffineMultiMapping< Vec1Types, Vec1Types > >()

;

template class SOFA_Compliant_API AffineMultiMapping<  Vec6Types, Vec1Types >;
template class SOFA_Compliant_API AffineMultiMapping<  Vec3Types, Vec1Types >;
template class SOFA_Compliant_API AffineMultiMapping<  Vec1Types, Vec1Types >;




}
}
}
