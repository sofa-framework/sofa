#include "AffineMultiMapping.h"

#include <sofa/core/ObjectFactory.h>


namespace sofa {
namespace component {
namespace mapping {

SOFA_DECL_CLASS(AffineMultiMapping)

using namespace defaulttype;

// Register in the Factory
int AffineMultiMappingClass = core::RegisterObject("Arbitrary affine mapping")

#ifndef SOFA_FLOAT
.add< AffineMultiMapping< Vec6dTypes, Vec1dTypes > >()
.add< AffineMultiMapping< Vec3dTypes, Vec1dTypes > >()
.add< AffineMultiMapping< Vec1dTypes, Vec1dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< AffineMultiMapping< Vec6fTypes, Vec1fTypes > >()
.add< AffineMultiMapping< Vec3fTypes, Vec1fTypes > >()
.add< AffineMultiMapping< Vec1fTypes, Vec1fTypes > >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API AffineMultiMapping<  Vec6dTypes, Vec1dTypes >;
template class SOFA_Compliant_API AffineMultiMapping<  Vec3dTypes, Vec1dTypes >;
template class SOFA_Compliant_API AffineMultiMapping<  Vec1dTypes, Vec1dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API AffineMultiMapping< Vec6fTypes, Vec1fTypes >;
template class SOFA_Compliant_API AffineMultiMapping< Vec3fTypes, Vec1fTypes >;
template class SOFA_Compliant_API AffineMultiMapping< Vec1fTypes, Vec1fTypes >;
#endif



}
}
}
