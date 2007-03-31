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
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_PLANEFORCEFIELD_INL
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_PLANEFORCEFIELD_INL

#include <sofa/core/componentmodel/behavior/ForceField.inl>
#include "PlaneForceField.h"
#include <sofa/helper/system/config.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/gl/template.h>
#include <assert.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace forcefield
{


template<class DataTypes>
void PlaneForceField<DataTypes>::addForce(VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1)
{
    //this->dfdd.resize(p1.size());
    this->contacts.clear();
    f1.resize(p1.size());
    for (unsigned int i=0; i<p1.size(); i++)
    {
        Real d = p1[i]*planeNormal.getValue()-planeD.getValue();
        if (d<0)
        {
            Real forceIntensity = -this->stiffness.getValue()*d;
            Real dampingIntensity = -this->damping.getValue()*d;
            Deriv force = planeNormal.getValue()*forceIntensity - v1[i]*dampingIntensity;
            f1[i]+=force;
            //this->dfdd[i] = -this->stiffness;
            this->contacts.push_back(i);
        }
    }
}

template<class DataTypes>
void PlaneForceField<DataTypes>::addDForce(VecDeriv& f1, const VecDeriv& dx1)
{
    f1.resize(dx1.size());
    for (unsigned int i=0; i<this->contacts.size(); i++)
    {
        unsigned int p = this->contacts[i];
        assert(p<dx1.size());
        f1[p] += planeNormal.getValue() * (-this->stiffness.getValue() * (dx1[p]*planeNormal.getValue()));
    }
}

template<class DataTypes>
void PlaneForceField<DataTypes>::updateStiffness( const VecCoord& x )
{
    this->contacts.clear();
    for (unsigned int i=0; i<x.size(); i++)
    {
        Real d = x[i]*planeNormal.getValue()-planeD.getValue();
        if (d<0)
        {
            this->contacts.push_back(i);
        }
    }
}


template <class DataTypes>
double PlaneForceField<DataTypes>::getPotentialEnergy(const VecCoord&)
{
    std::cerr<<"PlaneForceField::getPotentialEnergy-not-implemented !!!"<<std::endl;
    return 0;
}


// Rotate the plane. Note that the rotation is only applied on the 3 first coordinates
template<class DataTypes>
void PlaneForceField<DataTypes>::rotate( Deriv axe, Real angle )
{
    defaulttype::Vec3d axe3d(1,1,1); axe3d = axe;
    defaulttype::Vec3d normal3d; normal3d = planeNormal.getValue();
    defaulttype::Vec3d v = normal3d.cross(axe3d);
    v.normalize();
    v = normal3d * cos ( angle ) + v * sin ( angle );
    *planeNormal.beginEdit() = v;
    planeNormal.endEdit();
}


template<class DataTypes>
void PlaneForceField<DataTypes>::draw()
{
    if (!getContext()->getShowForceFields()) return;
    if (!bDraw.getValue()) return;
    draw2();
}


template<class DataTypes>
void PlaneForceField<DataTypes>::draw2(float size)
{
    if (!getContext()->getShowForceFields()) return;

    const VecCoord& p1 = *this->mstate->getX();

    defaulttype::Vec3d normal; normal = planeNormal.getValue();

    // find a first vector inside the plane
    defaulttype::Vec3d v1;
    if( 0.0 != normal[0] ) v1 = defaulttype::Vec3d((-normal[2]-normal[1])/normal[0], 1.0, 1.0);
    else if ( 0.0 != normal[1] ) v1 = defaulttype::Vec3d(1.0, (-normal[0]-normal[2])/normal[1],1.0);
    else if ( 0.0 != normal[2] ) v1 = defaulttype::Vec3d(1.0, 1.0, (-normal[0]-normal[1])/normal[2]);
    v1.normalize();
    // find a second vector inside the plane and orthogonal to the first
    defaulttype::Vec3d v2;
    v2 = v1.cross(normal);
    v2.normalize();

    defaulttype::Vec3d center = normal*planeD.getValue();
    defaulttype::Vec3d corners[4];
    corners[0] = center-v1*size-v2*size;
    corners[1] = center+v1*size-v2*size;
    corners[2] = center+v1*size+v2*size;
    corners[3] = center-v1*size+v2*size;

    // glEnable(GL_LIGHTING);
    glEnable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glCullFace(GL_FRONT);

    glColor3f(color.getValue()[0],color.getValue()[1],color.getValue()[2]);

    glBegin(GL_QUADS);
    helper::gl::glVertexT(corners[0]);
    helper::gl::glVertexT(corners[1]);
    helper::gl::glVertexT(corners[2]);
    helper::gl::glVertexT(corners[3]);
    glEnd();

    glDisable(GL_CULL_FACE);

    glColor4f(1,0,0,1);
    glDisable(GL_LIGHTING);
    // lignes pour les points pass� dessous
    glBegin(GL_LINES);
    for (unsigned int i=0; i<p1.size(); i++)
    {
        Real d = p1[i]*planeNormal.getValue()-planeD.getValue();
        Coord p2 = p1[i];
        p2 += planeNormal.getValue()*(-d);
        if (d<0)
        {
            helper::gl::glVertexT(p1[i]);
            helper::gl::glVertexT(p2);
        }
    }
    glEnd();
}

template <class DataTypes>
bool PlaneForceField<DataTypes>::addBBox(double* minBBox, double* maxBBox)
{
    if (!bDraw.getValue()) return false;

    defaulttype::Vec3d normal; normal = planeNormal.getValue();
    double size=10.0;

    // find a first vector inside the plane
    defaulttype::Vec3d v1;
    if( 0.0 != normal[0] ) v1 = defaulttype::Vec3d((-normal[2]-normal[1])/normal[0], 1.0, 1.0);
    else if ( 0.0 != normal[1] ) v1 = defaulttype::Vec3d(1.0, (-normal[0]-normal[2])/normal[1],1.0);
    else if ( 0.0 != normal[2] ) v1 = defaulttype::Vec3d(1.0, 1.0, (-normal[0]-normal[1])/normal[2]);
    v1.normalize();
    // find a second vector inside the plane and orthogonal to the first
    defaulttype::Vec3d v2;
    v2 = v1.cross(normal);
    v2.normalize();

    defaulttype::Vec3d center = normal*planeD.getValue();
    defaulttype::Vec3d corners[4];
    corners[0] = center-v1*size-v2*size;
    corners[1] = center+v1*size-v2*size;
    corners[2] = center+v1*size+v2*size;
    corners[3] = center-v1*size+v2*size;

    for (unsigned int i=0; i<4; i++)
    {
        for (int c=0; c<3; c++)
        {
            if (corners[i][c] > maxBBox[c]) maxBBox[c] = corners[i][c];
            if (corners[i][c] < minBBox[c]) minBBox[c] = corners[i][c];
        }
    }
    return true;
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
