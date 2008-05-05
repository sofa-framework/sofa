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
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_ConstantForceField_INL
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_ConstantForceField_INL

#include <sofa/core/componentmodel/behavior/ForceField.inl>
#include "ConstantForceField.h"
#include <sofa/helper/system/config.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/gl/template.h>
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
ConstantForceField<DataTypes>::ConstantForceField()
    : points(initData(&points, "points", "points where the forces are applied"))
    , forces(initData(&forces, "forces", "applied forces"))
{}


template<class DataTypes>
void ConstantForceField<DataTypes>::addForce(VecDeriv& f1, const VecCoord& p1, const VecDeriv& )
{
    f1.resize(p1.size());
    const VecIndex& indices = points.getValue();
    const VecDeriv& f = forces.getValue();
    for (unsigned int i=0; i<indices.size(); i++)
    {
        f1[indices[i]]+=f[i];
    }
}



template <class DataTypes>
double ConstantForceField<DataTypes>::getPotentialEnergy(const VecCoord& x)
{
    const VecIndex& indices = points.getValue();
    const VecDeriv& f = forces.getValue();
    double e=0;
    for (unsigned int i=0; i<indices.size(); i++)
    {
        e -= f[i]*x[indices[i]];
    }
    return e;
}




template<class DataTypes>
void ConstantForceField<DataTypes>::draw()
{
    if (!getContext()->getShowForceFields()) return;  /// \todo put this in the parent class
    const VecIndex& indices = points.getValue();
    const VecDeriv& f = forces.getValue();
    const VecCoord& x = *this->mstate->getX();
    glDisable(GL_LIGHTING);
    glBegin(GL_LINES);
    glColor3f(0,1,0);
    for (unsigned int i=0; i<indices.size(); i++)
    {
        Real xx,xy,xz,fx,fy,fz;
        DataTypes::get(xx,xy,xz,x[indices[i]]);
        DataTypes::get(fx,fy,fz,f[i]);
        glVertex3f( (GLfloat)xx, (GLfloat)xy, (GLfloat)xz );
        glVertex3f( (GLfloat)(xx+fx), (GLfloat)(xy+fy), (GLfloat)(xz+fz) );
    }
    glEnd();
}


template <class DataTypes>
bool ConstantForceField<DataTypes>::addBBox(double*, double* )
{
    return false;
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif



