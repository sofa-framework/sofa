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

#include <sofa/component/solidmechanics/spring/StiffSpringForceField.h>
#include <sofa/component/solidmechanics/spring/SpringForceField.h>
#include <sofa/component/constraint/projective/FixedProjectiveConstraint.h>

#include <sofa/gui/component/performer/InteractionPerformer.h>
#include <sofa/gui/component/performer/MouseInteractor.h>

namespace sofa::gui::component::performer
{

class SuturePointPerformerConfiguration
{
public:
    void setStiffness (double f) {stiffness=f;}
    void setDamping (double f) {damping=f;}

protected:
    double stiffness;
    double damping;
};


template <class DataTypes>
class SOFA_GUI_COMPONENT_API SuturePointPerformer: public TInteractionPerformer<DataTypes>, public SuturePointPerformerConfiguration
{
public:
    typedef typename DataTypes::Real Real;
    typedef sofa::component::solidmechanics::spring::LinearSpring<Real> Spring;
    typedef sofa::component::solidmechanics::spring::StiffSpringForceField<DataTypes> SpringObjectType;
    typedef sofa::component::constraint::projective::FixedProjectiveConstraint<DataTypes> FixObjectType;

    SuturePointPerformer(BaseMouseInteractor *i);
    ~SuturePointPerformer();

    void start();
    void execute() {}

protected:
    bool first;
    unsigned int fixedIndex;

    sofa::type::vector<Spring> addedSprings;

    BodyPicked firstPicked;
    SpringObjectType *SpringObject;
    FixObjectType *FixObject;
};

#if !defined(SOFA_COMPONENT_COLLISION_SUTUREPOINTPERFORMER_CPP)
extern template class SOFA_GUI_COMPONENT_API  SuturePointPerformer<defaulttype::Vec3Types>;

#endif

} // namespace sofa::gui::component::performer
