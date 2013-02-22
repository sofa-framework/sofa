#ifndef SOFA_COMPONENT_COLLISION_ADAPTATIVEATTACHPERFORMER_H
#define SOFA_COMPONENT_COLLISION_ADAPTATIVEATTACHPERFORMER_H

#include <sofa/component/collision/InteractionPerformer.h>
#include <sofa/component/collision/BaseContactMapper.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/visual/DisplayFlags.h>

#include "./../../applications-dev/plugins/ARPlugin/ARPSStiffSpringForceField.inl"
#include "./../../applications-dev/plugins/ARPlugin/ARPSSpringForceField.inl"
#include <sofa/component/configurationsetting/AdaptativeAttachButtonSetting.h>
#include <sofa/component/interactionforcefield/StiffSpringForceField.h>

#include <sofa/defaulttype/Vec3Types.h>
#include <../applications/plugins/frame/Blending.h>


namespace sofa
{

namespace component
{

namespace collision
{

struct BodyPicked;

using sofa::defaulttype::StdRigidTypes;

template <class DataTypes>
class AdaptativeAttachPerformer: public TInteractionPerformer<DataTypes>
{
public:
    typedef typename sofa::defaulttype::BaseFrameBlendingMapping<true> FBMapping;
    typedef sofa::component::collision::BaseContactMapper< DataTypes >        MouseContactMapper;
    typedef sofa::core::behavior::MechanicalState< DataTypes >         MouseContainer;
    typedef sofa::core::behavior::BaseForceField              MouseForceField;

public:
    AdaptativeAttachPerformer(BaseMouseInteractor *i);
    virtual ~AdaptativeAttachPerformer();

    void start();
    void execute();
    void draw(const core::visual::VisualParams* vparams);
    void clear();

    void setStiffness(double s) {stiffness=s;}
    void setArrowSize(float s) {size=s;}

    virtual void configure(configurationsetting::MouseButtonSetting* setting)
        {
            configurationsetting::AdaptativeAttachButtonSetting* s = dynamic_cast<configurationsetting::AdaptativeAttachButtonSetting*>(setting);
            if (s)
            {
                setStiffness((double)s->stiffness.getValue());
                setArrowSize((float)s->arrowSize.getValue());
            }
        }

    protected:
        SReal stiffness;
        SReal size;
        int index; //Index of the attached DOF

        virtual bool start_partial(const BodyPicked& picked);
        MouseContactMapper  *mapper;
        MouseForceField::SPtr m_forcefield;

        core::visual::DisplayFlags flags;
};



#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_COLLISION_ADAPTATIVEATTACHPERFORMER_CPP)
#ifndef SOFA_DOUBLE
extern template class SOFA_USER_INTERACTION_API  AdaptativeAttachPerformer<defaulttype::Vec3fTypes>;
#endif
#ifndef SOFA_FLOAT
extern template class SOFA_USER_INTERACTION_API  AdaptativeAttachPerformer<defaulttype::Vec3dTypes>;
#endif
#endif


}
}
}

#endif
