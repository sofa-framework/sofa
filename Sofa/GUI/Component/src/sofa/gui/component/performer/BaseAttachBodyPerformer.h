#pragma once

#include <sofa/core/objectmodel/BaseObject.h>


namespace sofa::gui::component::performer
{
struct BodyPicked;


/**
 * This class is a virtualization of attachment performer used to allow the blind use of either "AttachBodyPerformer" based on springs and "ConstraintAttachBodyPerformer" based on lagrangian
 * constraints. An example of use can be found in the external plugin Sofa.IGTLink in the component "iGTLinkMouseInteractor"
 */
class BaseAttachBodyPerformer
{
public:
    virtual ~BaseAttachBodyPerformer() = default;
    virtual sofa::core::objectmodel::BaseObject* getInteractionObject() = 0;
    virtual void clear() = 0;
    virtual bool start_partial(const BodyPicked& picked) = 0;
};
}
