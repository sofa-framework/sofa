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
int MaskMappingClass = core::RegisterObject("Filters out dofs by term-wise multiplication")

#ifndef SOFA_FLOAT
	.add< MaskMapping< Vec6dTypes, Vec6dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< MaskMapping< Vec6fTypes, Vec6fTypes > >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API MaskMapping<  Vec6dTypes, Vec6dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API MaskMapping< Vec6fTypes, Vec6fTypes >;

#endif



} // namespace mapping

} // namespace component

} // namespace sofa

