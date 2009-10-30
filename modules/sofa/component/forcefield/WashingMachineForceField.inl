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
#ifndef SOFA_COMPONENT_FORCEFIELD_WASHINGMACHINEFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_WASHINGMACHINEFORCEFIELD_INL

#include <sofa/core/componentmodel/behavior/ForceField.inl>
#include <sofa/component/forcefield/WashingMachineForceField.h>
#include <sofa/helper/system/config.h>
#include <sofa/helper/system/gl.h>
#include <assert.h>
#include <iostream>



namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
void WashingMachineForceField<DataTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    for(int i=0; i<6; ++i)
    {
        _planes[i]->addForce(f,x,v);
        _planes[i]->rotate(Deriv(1,0,0),_speed.getValue());
    }
}

template<class DataTypes>
void WashingMachineForceField<DataTypes>::addDForce(VecDeriv& f1, const VecDeriv& dx1, double kFactor, double bFactor)
{
    for(int i=0; i<6; ++i)
        _planes[i]->addDForce(f1, dx1, kFactor, bFactor);
}



template <class DataTypes>
double WashingMachineForceField<DataTypes>::getPotentialEnergy(const VecCoord&x)
{
    double energy = 0.0;
    for(int i=0; i<6; ++i)
        energy += _planes[i]->getPotentialEnergy(x);
    return energy;
}



template<class DataTypes>
void WashingMachineForceField<DataTypes>::draw()
{
    if (!this->getContext()->getShowForceFields() || !_alreadyInit ) return;
    for(int i=0; i<6; ++i)
// 				_planes[i]->drawPlane(_size.getValue()[0]);
        _planes[i]->draw();
}

template<class DataTypes>
bool WashingMachineForceField<DataTypes>::addBBox(double* minBBox, double* maxBBox)
{
    Deriv corner0 = _center.getValue() - _size.getValue() * .5;
    Deriv corner1 = _center.getValue() + _size.getValue() * .5;
    for (int c=0; c<3; c++)
    {
        if (minBBox[c] > corner0[c]) minBBox[c] = corner0[c];
        if (maxBBox[c] < corner1[c]) maxBBox[c] = corner1[c];
    }
    return true;
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
