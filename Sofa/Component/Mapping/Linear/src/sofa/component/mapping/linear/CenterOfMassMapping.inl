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

#include <sofa/component/mapping/linear/CenterOfMassMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>

#include <string>
#include <iostream>

namespace sofa::component::mapping::linear
{

template <class TIn, class TOut>
void CenterOfMassMapping<TIn, TOut>::init()
{
    //get the pointer on the input dofs mass
    masses = this->fromModel->getContext()->getMass();
    if(!masses)
        return;

    totalMass = 0.0;

    //compute the total mass of the object
    for (unsigned int i=0, size = this->fromModel->getSize() ; i< size; i++)
        totalMass += masses->getElementMass(i);

    Inherit::init();
}


template <class TIn, class TOut>
void CenterOfMassMapping<TIn, TOut>::apply( const sofa::core::MechanicalParams* mparams, OutDataVecCoord& outData, const InDataVecCoord& inData)
{
    SOFA_UNUSED(mparams);

    OutVecCoord& childPositions = *outData.beginEdit();
    const InVecCoord& parentPositions = inData.getValue();

    if(!masses || totalMass==0.0)
    {
        msg_error() << "CenterOfMassMapping : no mass found corresponding to the DOFs";
        return;
    }

    OutCoord outX;

    //compute the center of mass position with the relation X = sum(Xi*Mi)/Mt
    //with Xi: position of the dof i, Mi: mass of the dof i, and Mt : total mass of the object
    for (unsigned int i=0 ; i<parentPositions.size() ; i++)
    {
        outX += parentPositions[i].getCenter() * masses->getElementMass(i);
    }

    childPositions[0] = outX / totalMass;

    outData.endEdit();
}


template <class TIn, class TOut>
void CenterOfMassMapping<TIn, TOut>::applyJ( const sofa::core::MechanicalParams* mparams, OutDataVecDeriv& outData, const InDataVecDeriv& inData)
{
    SOFA_UNUSED(mparams);

    OutVecDeriv& childForces = *outData.beginEdit();
    const InVecDeriv& parentForces = inData.getValue();

    if(!masses || totalMass==0.0)
    {
        msg_error() << "CenterOfMassMapping : no mass found corresponding to the DOFs";
        return;
    }

    OutDeriv outF;

    //compute the forces applied on the center of mass with the relation F = sum(Fi*Mi)/Mt
    //with Fi: force of the dof i, Mi: mass of the dof i, and Mt : total mass of the object
    for (unsigned int i=0 ; i<parentForces.size() ; i++)
    {
        outF += getVCenter(parentForces[i]) * masses->getElementMass(i);
    }

    childForces[0] = outF / totalMass;

    outData.endEdit();
}


template <class TIn, class TOut>
void CenterOfMassMapping<TIn, TOut>::applyJT( const sofa::core::MechanicalParams* mparams, InDataVecDeriv& outData, const OutDataVecDeriv& inData)
{
    SOFA_UNUSED(mparams);

    InVecDeriv& parentForces = *outData.beginEdit();
    const OutVecDeriv& childForces = inData.getValue();

    if(!masses || totalMass==0.0)
    {
        msg_error() << "CenterOfMassMapping : no mass found corresponding to the DOFs";
        return;
    }

    //compute the force applied on each object Dof from the force applied on the center of mass
    //the force on a dof is proportional to its mass
    //relation is Fi = Fc * (Mi/Mt), with Fc: force of center of mass, Mi: dof mass, Mt: total mass
    for (unsigned int i=0 ; i<parentForces.size() ; i++)
        getVCenter(parentForces[i]) += childForces[0] * (masses->getElementMass(i) / totalMass);

    outData.endEdit();
}


template <class TIn, class TOut>
void CenterOfMassMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowMapping()) return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    const typename Out::VecCoord &X = this->toModel->read(core::vec_id::read_access::position)->getValue();

    std::vector< sofa::type::Vec3 > points;
    for(unsigned int i=0 ; i<OutCoord::spatial_dimensions ; i++)
    {
        OutCoord v;
        v[i] = 0.1;
        points.push_back(toVec3(Out::getCPos((X[0] -v))));
        points.push_back(toVec3(Out::getCPos((X[0] +v))));
    }

    vparams->drawTool()->drawLines(points, 1, sofa::type::RGBAColor::yellow());


}

} // namespace sofa::component::mapping::linear
