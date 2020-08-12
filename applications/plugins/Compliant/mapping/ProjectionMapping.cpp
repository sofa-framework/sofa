#define SOFA_COMPONENT_COMPLIANCE_PROJECTIONMAPPING_CPP

#include "ProjectionMapping.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace defaulttype;

// Register in the Factory
int ProjectionMappingClass = core::RegisterObject("Projects dofs on a given vector set.")

	.add< ProjectionMapping< Vec6Types, Vec1Types > >()

;

template class SOFA_Compliant_API ProjectionMapping<  Vec6Types, Vec1Types >;




} // namespace mapping

} // namespace component

} // namespace sofa

