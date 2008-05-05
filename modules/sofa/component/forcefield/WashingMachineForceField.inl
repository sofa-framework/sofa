/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_FORCEFIELD_WASHINGMACHINEFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_WASHINGMACHINEFORCEFIELD_INL

#include <sofa/core/componentmodel/behavior/ForceField.inl>
#include <sofa/component/forcefield/WashingMachineForceField.h>
#include <sofa/helper/system/config.h>
#include <sofa/helper/system/gl.h>
#include <assert.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
void WashingMachineForceField<DataTypes>::addForce(VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1)
{
    for(int i=0; i<6; ++i)
    {
        _planes[i]->addForce(f1,p1,v1);
        _planes[i]->rotate(Deriv(1,0,0),_speed.getValue());
    }
}

template<class DataTypes>
void WashingMachineForceField<DataTypes>::addDForce(VecDeriv& f1, const VecDeriv& dx1)
{
    for(int i=0; i<6; ++i)
        _planes[i]->addDForce(f1,dx1);
}



template <class DataTypes>
double WashingMachineForceField<DataTypes>::getPotentialEnergy(const VecCoord&)
{
    cerr<<"WashingMachineForceField::getPotentialEnergy-not-implemented !!!"<<endl;
    return 0;
}



template<class DataTypes>
void WashingMachineForceField<DataTypes>::draw()
{
    if (!getContext()->getShowForceFields()) return;
    for(int i=0; i<6; ++i)
// 				_planes[i]->draw2(_size.getValue()[0]);
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
