#ifndef SOFA_COMPONENT_COLLISION_ADAPTATIVEATTACHPERFORMER_CPP
#define SOFA_COMPONENT_COLLISION_ADAPTATIVEATTACHPERFORMER_CPP
#include <sofa/component/component.h>
#include <sofa/component/collision/AdaptativeAttachPerformer.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/helper/Factory.inl>

namespace sofa
{

namespace component
{

namespace collision
{
SOFA_DECL_CLASS(AdaptativeAttachPerformer)
#ifndef SOFA_DOUBLE
template class SOFA_USER_INTERACTION_API  AdaptativeAttachPerformer<defaulttype::Vec3fTypes>;
#endif
#ifndef SOFA_FLOAT
template class SOFA_USER_INTERACTION_API  AdaptativeAttachPerformer<defaulttype::Vec3dTypes>;
#endif


#ifndef SOFA_DOUBLE
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AdaptativeAttachPerformer<defaulttype::Vec3fTypes> >  AdaptativeAttachPerformerVec3fClass("AdaptativeAttach",true);
#endif
#ifndef SOFA_FLOAT
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AdaptativeAttachPerformer<defaulttype::Vec3dTypes> >  AdaptativeAttachPerformerVec3dClass("AdaptativeAttach",true);
#endif
}
}
}
#endif
