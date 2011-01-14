/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_CENTEROFMASSMAPPING_INL
#define SOFA_COMPONENT_MAPPING_CENTEROFMASSMAPPING_INL

#include <sofa/component/mapping/CenterOfMassMapping.h>

#include <sofa/core/Mapping.inl>

#include <sofa/simulation/common/Simulation.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/helper/gl/template.h>

#include <string>
#include <iostream>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;


template <class TIn, class TOut>
void CenterOfMassMapping<TIn, TOut>::init()
{
    //get the pointer on the input dofs mass
    masses = dynamic_cast<BaseMass*> (this->fromModel->getContext()->getMass());
    if(!masses)
        return;

    totalMass = 0.0;

    //compute the total mass of the object
    for (unsigned int i=0 ; i<this->fromModel->getX()->size() ; i++)
        totalMass += masses->getElementMass(i);
}


template <class TIn, class TOut>
void CenterOfMassMapping<TIn, TOut>::apply ( typename Out::VecCoord& childPositions, const typename In::VecCoord& parentPositions )
{
    if(!masses || totalMass==0.0)
    {
        serr<<"Error in CenterOfMassMapping : no mass found corresponding to the DOFs"<<sendl;
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
}


template <class TIn, class TOut>
void CenterOfMassMapping<TIn, TOut>::applyJ ( typename Out::VecDeriv& childForces, const typename In::VecDeriv& parentForces )
{
    if(!masses || totalMass==0.0)
    {
        serr<<"Error in CenterOfMassMapping : no mass found corresponding to the DOFs"<<sendl;
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
}


template <class TIn, class TOut>
void CenterOfMassMapping<TIn, TOut>::applyJT ( typename In::VecDeriv& parentForces, const typename Out::VecDeriv& childForces )
{
    if(!masses || totalMass==0.0)
    {
        serr<<"Error in CenterOfMassMapping : no mass found corresponding to the DOFs"<<sendl;
        return;
    }

    //compute the force applied on each object Dof from the force applied on the center of mass
    //the force on a dof is proportional to its mass
    //relation is Fi = Fc * (Mi/Mt), with Fc: force of center of mass, Mi: dof mass, Mt: total mass
    for (unsigned int i=0 ; i<parentForces.size() ; i++)
        getVCenter(parentForces[i]) += childForces[0] * (masses->getElementMass(i) / totalMass);
}


template <class TIn, class TOut>
void CenterOfMassMapping<TIn, TOut>::draw()
{
    const typename Out::VecCoord &X = *this->toModel->getX();

    std::vector< Vector3 > points;
    Vector3 point1,point2;
    for(unsigned int i=0 ; i<OutCoord::spatial_dimensions ; i++)
    {
        OutCoord v;
        v[i] = (Real)0.1;
        point1 = OutDataTypes::getCPos((X[0] -v));
        point2 = OutDataTypes::getCPos((X[0] +v));
        points.push_back(point1);
        points.push_back(point2);
    }

    simulation::getSimulation()->DrawUtility().drawLines(points, 1, Vec<4,float>(1,1,0,1));
}


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
