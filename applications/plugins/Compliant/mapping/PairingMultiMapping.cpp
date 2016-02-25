#include "PairingMultiMapping.h"

#include <sofa/defaulttype/VecTypes.h>

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(PairingMultiMapping)

using namespace defaulttype;

// Register in the Factory
const int PairingMultiMappingClass = core::RegisterObject("dot product between vec dofs")

#ifndef SOFA_FLOAT
    .add< PairingMultiMapping< Vec6dTypes, Vec1dTypes > >()    
    .add< PairingMultiMapping< Vec3dTypes, Vec1dTypes > >()
    .add< PairingMultiMapping< Vec2dTypes, Vec1dTypes > >()    
    .add< PairingMultiMapping< Vec1dTypes, Vec1dTypes > >()
#endif
#ifndef SOFA_DOUBLE
    .add< PairingMultiMapping< Vec6fTypes, Vec1fTypes > >()    
    .add< PairingMultiMapping< Vec3fTypes, Vec1fTypes > >()
    .add< PairingMultiMapping< Vec2fTypes, Vec1fTypes > >()
    .add< PairingMultiMapping< Vec1fTypes, Vec1fTypes > >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API PairingMultiMapping<  Vec3dTypes, Vec1dTypes >;
template class SOFA_Compliant_API PairingMultiMapping<  Vec6dTypes, Vec1dTypes >;
template class SOFA_Compliant_API PairingMultiMapping<  Vec2dTypes, Vec1dTypes >;
template class SOFA_Compliant_API PairingMultiMapping<  Vec1dTypes, Vec1dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API PairingMultiMapping<  Vec3fTypes, Vec1fTypes >;
template class SOFA_Compliant_API PairingMultiMapping<  Vec6fTypes, Vec1fTypes >;
template class SOFA_Compliant_API PairingMultiMapping<  Vec2fTypes, Vec1fTypes >;
template class SOFA_Compliant_API PairingMultiMapping<  Vec1fTypes, Vec1fTypes >;

#endif



} // namespace mapping

} // namespace component

} // namespace sofa

