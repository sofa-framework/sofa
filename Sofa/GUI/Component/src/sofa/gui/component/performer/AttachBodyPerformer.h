/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/gui/component/config.h>

#include <sofa/gui/component/performer/BaseAttachBodyPerformer.h>
#include <sofa/gui/component/AttachBodyButtonSetting.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/visual/DisplayFlags.h>

namespace sofa::gui::component::performer
{

struct BodyPicked;

template <class DataTypes>
class AttachBodyPerformer: public BaseAttachBodyPerformer<DataTypes>
{
public:

    typedef sofa::component::collision::response::mapper::BaseContactMapper< DataTypes >        MouseContactMapper;
    typedef sofa::core::behavior::MechanicalState< DataTypes >         MouseContainer;
    typedef sofa::core::behavior::BaseForceField              MouseForceField;

    AttachBodyPerformer(BaseMouseInteractor *i);
    virtual ~AttachBodyPerformer() = default;

    virtual bool startPartial(const BodyPicked& picked) override;
    /*
    initialise MouseForceField according to template.
    StiffSpringForceField for Vec3
    JointSpringForceField for Rigid3
    */

    void setStiffness(SReal s) {m_stiffness=s;}
    void setArrowSize(float s) {m_size=s;}

    virtual void configure(sofa::component::setting::MouseButtonSetting* setting)
    {
        const auto* s = dynamic_cast<sofa::gui::component::AttachBodyButtonSetting*>(setting);
        if (s)
        {
            setStiffness(s->d_stiffness.getValue());
            setArrowSize((float)s->d_arrowSize.getValue());
        }
    }

protected:
    SReal m_stiffness;
    SReal m_size;
};

#if !defined(SOFA_COMPONENT_COLLISION_ATTACHBODYPERFORMER_CPP)
extern template class SOFA_GUI_COMPONENT_API  AttachBodyPerformer<defaulttype::Vec2Types>;
extern template class SOFA_GUI_COMPONENT_API  AttachBodyPerformer<defaulttype::Vec3Types>;
extern template class SOFA_GUI_COMPONENT_API  AttachBodyPerformer<defaulttype::Rigid3Types>;
#endif

} // namespace sofa::gui::component::performer
