#include "MaskMapping.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/Mapping.inl>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(MaskMapping)

using namespace defaulttype;

// Register in the Factory
int MaskMappingClass = core::RegisterObject("Filters out dofs. You need to map to 1d dofs.")

#ifndef SOFA_FLOAT
	.add< MaskMapping< Vec6dTypes, Vec1dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< MaskMapping< Vec6fTypes, Vec1fTypes > >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API MaskMapping<  Vec6dTypes, Vec1dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API MaskMapping< Vec6fTypes, Vec1fTypes >;

#endif



} // namespace mapping

} // namespace component

} // namespace sofa

