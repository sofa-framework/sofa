#define SOFA_COMPONENT_MAPPING_FRAME_MAPPING

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec3Types.h>
#include "../Flexible/deformationMapping/FrameMapping.inl"
#include "../initFlexible.h"


namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(FrameMapping)

using namespace defaulttype;

int FrameMappingClass = core::RegisterObject("Maps positions using a frame").add< FrameMapping< Rigid3fTypes, Vec3fTypes > >();

template class SOFA_Flexible_API FrameMapping< Rigid3fTypes, Vec3fTypes >;

} //namespace mapping
} //namespace component
} //namespace sofa
