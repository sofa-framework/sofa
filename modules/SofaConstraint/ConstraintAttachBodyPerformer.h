/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_CONSTRAINTATTACHBODYPERFORMER_H
#define SOFA_COMPONENT_COLLISION_CONSTRAINTATTACHBODYPERFORMER_H
#include "config.h"

#include <SofaUserInteraction/InteractionPerformer.h>
#include <SofaBaseCollision/BaseContactMapper.h>
//#include <sofa/core/behavior/BaseForceField.h>
#include <SofaDeformable/SpringForceField.h>
#include <SofaDeformable/StiffSpringForceField.h>
#include <SofaGraphComponent/AttachBodyButtonSetting.h>
#include <sofa/core/visual/DisplayFlags.h>
#include <SofaConstraint/BilateralInteractionConstraint.h>
#include <SofaGraphComponent/AttachBodyButtonSetting.h>

namespace sofa
{
namespace component
{

namespace collision
{

class ConstraintAttachBodyButtonSetting : public sofa::component::configurationsetting::AttachBodyButtonSetting
{
public:
    SOFA_CLASS(ConstraintAttachBodyButtonSetting,sofa::component::configurationsetting::AttachBodyButtonSetting);
protected:
    ConstraintAttachBodyButtonSetting() {}
public:
//        Data<SReal> snapDistance;
    std::string getOperationType() override {return  "ConstraintAttachBody";}
};

struct BodyPicked;

template <class DataTypes>
class SOFA_CONSTRAINT_API ConstraintAttachBodyPerformer: public TInteractionPerformer<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef sofa::component::collision::BaseContactMapper< DataTypes >        MouseContactMapper;
    typedef sofa::core::behavior::MechanicalState< DataTypes >         MouseContainer;
//        typedef sofa::component::constraintset::BilateralInteractionConstraint< DataTypes > MouseConstraint;

//        typedef sofa::core::behavior::BaseForceField              MouseForceField;

    ConstraintAttachBodyPerformer(BaseMouseInteractor *i);
    virtual ~ConstraintAttachBodyPerformer();

    void start();
    void execute();
    void draw(const core::visual::VisualParams* vparams);
    void clear();

    void setStiffness(SReal s) {stiffness=s;}
    void setArrowSize(float s) {size=s;}
    void setShowFactorSize(float s) {showFactorSize = s;}

    virtual void configure(configurationsetting::MouseButtonSetting* setting)
    {
        ConstraintAttachBodyButtonSetting* s = dynamic_cast<ConstraintAttachBodyButtonSetting*>(setting);
        if (s)
        {
            setStiffness((double)s->stiffness.getValue());
            setArrowSize((float)s->arrowSize.getValue());
            setShowFactorSize((float)s->showFactorSize.getValue());
        }
    }

protected:
    SReal stiffness;
    SReal size;
    SReal showFactorSize;

    virtual bool start_partial(const BodyPicked& picked);
    /*
    initialise MouseForceField according to template.
    StiffSpringForceField for Vec3
    JointSpringForceField for Rigid3
    */

    MouseContactMapper  *mapper;
    constraintset::BilateralInteractionConstraint<defaulttype::Vec3Types>::SPtr m_constraint;

    core::visual::DisplayFlags flags;

    sofa::core::behavior::MechanicalState<DataTypes> *mstate1, *mstate2;
};

/*#ifndef SOFA_FLOAT
      template<>
      bool ConstraintAttachBodyPerformer<defaulttype::Rigid3dTypes>::start_partial(const BodyPicked& picked);
#endif

#ifndef SOFA_DOUBLE
      template<>
      bool ConstraintAttachBodyPerformer<defaulttype::Rigid3fTypes>::start_partial(const BodyPicked& picked);
#endif*/


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_COLLISION_CONSTRAINTATTACHBODYPERFORMER_CPP)
// #ifndef SOFA_DOUBLE
// extern template class SOFA_CONSTRAINT_API ConstraintAttachBodyPerformer<defaulttype::Vec3fTypes>;
// extern template class SOFA_CONSTRAINT_API ConstraintAttachBodyPerformer<defaulttype::Rigid3fTypes>;
// #endif
// #ifndef SOFA_FLOAT
// extern template class SOFA_CONSTRAINT_API ConstraintAttachBodyPerformer<defaulttype::Vec3dTypes>;
// extern template class SOFA_CONSTRAINT_API ConstraintAttachBodyPerformer<defaulttype::Rigid3dTypes>;
// #endif

extern template class SOFA_CONSTRAINT_API ConstraintAttachBodyPerformer<defaulttype::Vec3Types>;
#endif



}
}
}

#endif
