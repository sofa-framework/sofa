#ifndef SOFA_COMPONENT_COLLISION_INTERACTIONPERFOMER_CPP
#define SOFA_COMPONENT_COLLISION_INTERACTIONPERFOMER_CPP

#include <sofa/component/collision/InteractionPerformer.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/component.h>
#include <sofa/helper/Factory.inl>

namespace sofa
{

namespace helper
{
//explicit instanciation of our factory class.
template class SOFA_COMPONENT_COLLISION_API helper::Factory<std::string, component::collision::InteractionPerformer, component::collision::BaseMouseInteractor*>;
} // helper
} // sofa
#endif
