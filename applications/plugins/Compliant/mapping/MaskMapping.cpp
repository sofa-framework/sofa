#include "MaskMapping.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace defaulttype;

// Register in the Factory
int MaskMappingClass = core::RegisterObject("Filters out dofs. You need to map to 1d dofs.")

    .add< MaskMapping< Vec6Types, Vec1Types > >()
    .add< MaskMapping< Vec3Types, Vec1Types > >()

;

template class SOFA_Compliant_API MaskMapping< Vec6Types, Vec1Types >;
template class SOFA_Compliant_API MaskMapping< Vec3Types, Vec1Types >;




} // namespace mapping

} // namespace component

} // namespace sofa

