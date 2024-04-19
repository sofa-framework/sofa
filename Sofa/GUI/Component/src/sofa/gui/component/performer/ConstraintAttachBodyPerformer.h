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

#include <sofa/component/constraint/lagrangian/model/BilateralLagrangianConstraint.h>
#include <sofa/gui/component/performer/BaseAttachBodyPerformer.h>
#include <sofa/gui/component/ConstraintAttachButtonSetting.h>



namespace sofa::gui::component::performer
{

struct BodyPicked;

template <class DataTypes>
class ConstraintAttachBodyPerformer: public BaseAttachBodyPerformer<DataTypes>
{
public:

    typedef typename DataTypes::VecCoord VecCoord;
    typedef sofa::component::collision::response::mapper::BaseContactMapper< DataTypes >        MouseContactMapper;
    typedef sofa::core::behavior::MechanicalState< DataTypes >         MouseContainer;

    ConstraintAttachBodyPerformer(BaseMouseInteractor *i);
    virtual ~ConstraintAttachBodyPerformer() = default;

    virtual bool startPartial(const BodyPicked& picked) override;

protected:

    sofa::core::behavior::MechanicalState<DataTypes> *m_mstate1, *m_mstate2;
};

#if !defined(SOFA_COMPONENT_COLLISION_CONSTRAINTATTACHBODYPERFORMER_CPP)
extern template class SOFA_GUI_COMPONENT_API ConstraintAttachBodyPerformer<defaulttype::Vec3Types>;
#endif

} // namespace sofa::gui::component::performer
