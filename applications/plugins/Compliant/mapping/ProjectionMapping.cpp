#include "ProjectionMapping.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(ProjectionMapping)

using namespace defaulttype;

// Register in the Factory
int ProjectionMappingClass = core::RegisterObject("Projects dofs on a given vector set.")

#ifndef SOFA_FLOAT
	.add< ProjectionMapping< Vec6dTypes, Vec1dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< ProjectionMapping< Vec6fTypes, Vec1fTypes > >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API ProjectionMapping<  Vec6dTypes, Vec1dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API ProjectionMapping< Vec6fTypes, Vec1fTypes >;

#endif



} // namespace mapping

} // namespace component

} // namespace sofa

