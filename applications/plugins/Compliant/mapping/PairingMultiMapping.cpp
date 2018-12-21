#include "PairingMultiMapping.h"

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
int PairingMultiMappingClass = core::RegisterObject("dot product between vec dofs")

    .add< PairingMultiMapping< Vec6Types, Vec1Types > >()    
    .add< PairingMultiMapping< Vec3Types, Vec1Types > >()
    .add< PairingMultiMapping< Vec2Types, Vec1Types > >()    
    .add< PairingMultiMapping< Vec1Types, Vec1Types > >()

;

template class SOFA_Compliant_API PairingMultiMapping<  Vec3Types, Vec1Types >;
template class SOFA_Compliant_API PairingMultiMapping<  Vec6Types, Vec1Types >;
template class SOFA_Compliant_API PairingMultiMapping<  Vec2Types, Vec1Types >;
template class SOFA_Compliant_API PairingMultiMapping<  Vec1Types, Vec1Types >;




} // namespace mapping

} // namespace component

} // namespace sofa

