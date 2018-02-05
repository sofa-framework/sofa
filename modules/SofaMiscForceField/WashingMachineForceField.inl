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
#ifndef SOFA_COMPONENT_FORCEFIELD_WASHINGMACHINEFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_WASHINGMACHINEFORCEFIELD_INL

#include <SofaMiscForceField/WashingMachineForceField.h>
#include <sofa/core/visual/VisualParams.h>
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
void WashingMachineForceField<DataTypes>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v)
{
    for(int i=0; i<6; ++i)
    {
        _planes[i]->rotate(_axis.getValue(),_speed.getValue());
        _planes[i]->addForce(mparams,f,x,v);
    }
}

template<class DataTypes>
void WashingMachineForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    for(int i=0; i<6; ++i)
        _planes[i]->addDForce(mparams, df, dx);
}

template<class DataTypes>
void WashingMachineForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields() || !_alreadyInit ) return;
    for(int i=0; i<6; ++i)
// 				_planes[i]->drawPlane(_size.getValue()[0]);
        _planes[i]->draw(vparams);
}

template<class DataTypes>
bool WashingMachineForceField<DataTypes>::addBBox(SReal* minBBox, SReal* maxBBox)
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
