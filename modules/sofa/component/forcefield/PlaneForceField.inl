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
#include <assert.h>
#include <GL/gl.h>
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


template<class DataTypes>
void PlaneForceField<DataTypes>::rotate( Deriv axe, Real angle )
{
    Deriv v;
    v = planeNormal.getValue().cross(axe);
    v.normalize();

    planeNormal.setValue( planeNormal.getValue() * cos( angle ) + v * sin( angle ) );
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

    const Coord normal = planeNormal.getValue();

    // un vecteur quelconque du plan
    Deriv v1;
    if( 0.0 != normal[0] ) v1 = Deriv((-normal[2]-normal[1])/normal[0], 1.0, 1.0);
    else if ( 0.0 != normal[1] ) v1 = Deriv(1.0, (-normal[0]-normal[2])/normal[1],1.0);
    else if ( 0.0 != normal[2] ) v1 = Deriv(1.0, 1.0, (-normal[0]-normal[1])/normal[2]);
    v1.normalize();
    // un deuxiement vecteur quelconque du plan orthogonal au premier
    Deriv v2;
    v2 = v1.cross(normal);
    v2.normalize();

    Coord center = normal*planeD.getValue();
    Coord q0 = center-v1*size-v2*size;
    Coord q1 = center+v1*size-v2*size;
    Coord q2 = center+v1*size+v2*size;
    Coord q3 = center-v1*size+v2*size;

// 	glEnable(GL_LIGHTING);
    glEnable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glCullFace(GL_FRONT);

    glColor3d(color.getValue()[0],color.getValue()[1],color.getValue()[2]);



    glBegin(GL_QUADS);
    glVertex3d(q0[0],q0[1],q0[2]);
    glVertex3d(q1[0],q1[1],q1[2]);
    glVertex3d(q2[0],q2[1],q2[2]);
    glVertex3d(q3[0],q3[1],q3[2]);
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
            glVertex3d(p1[i][0],p1[i][1],p1[i][2]);
            glVertex3d(p2[0],p2[1],p2[2]);
        }
    }
    glEnd();



    /*
    glPointSize(1);
    glColor4f(0,1,0,1);
    glBegin(GL_POINTS);
    for (unsigned int i=0; i<p1.size(); i++)
    {
    Real d = p1[i]*planeNormal-planeD;
    Coord p2 = p1[i];
    p2 += planeNormal*(-d);
    if (d>=0)
    glVertex3d(p2[0],p2[1],p2[2]);
    }
    glEnd();
    */
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
