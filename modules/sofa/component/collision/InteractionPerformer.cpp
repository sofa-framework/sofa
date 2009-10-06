#ifndef SOFA_COMPONENT_COLLISION_INTERACTIONPERFOMER_CPP
#define SOFA_COMPONENT_COLLISION_INTERACTIONPERFOMER_CPP

#include <sofa/component/collision/InteractionPerformer.h>
#include <sofa/component/component.h>
#include <sofa/helper/Factory.inl>

namespace sofa
{

namespace component
{

namespace collision
{
//explicit instanciation of our factory class.
template class SOFA_COMPONENT_COLLISION_API helper::Factory<std::string, InteractionPerformer, BaseMouseInteractor*>;

} // collision
} // component
} // sofa
#endif
