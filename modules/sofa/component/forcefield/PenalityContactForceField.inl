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
#ifndef SOFA_COMPONENT_FORCEFIELD_PENALITYCONTACTFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_PENALITYCONTACTFORCEFIELD_INL

#include <sofa/core/componentmodel/behavior/ForceField.inl>
#include <sofa/component/forcefield/PenalityContactForceField.h>
#include <sofa/helper/system/config.h>
#include <assert.h>
#include <GL/gl.h>
#include <sofa/helper/gl/template.h>
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
void PenalityContactForceField<DataTypes>::clear(int reserve)
{
    prevContacts.swap(contacts); // save old contacts in prevContacts
    contacts.clear();
    if (reserve)
        contacts.reserve(reserve);
}

template<class DataTypes>
void PenalityContactForceField<DataTypes>::addContact(int m1, int m2, const Deriv& norm, Real dist, Real ks, Real mu_s, Real mu_v, int oldIndex)
{
    int i = contacts.size();
    contacts.resize(i+1);
    Contact& c = contacts[i];
    c.m1 = m1;
    c.m2 = m2;
    c.norm = norm;
    c.dist = dist;
    c.ks = ks;
    c.mu_s = mu_s;
    c.mu_v = mu_v;
    c.pen = 0;
    if (oldIndex > 0 && oldIndex <= (int)prevContacts.size())
    {
        c.age = prevContacts[oldIndex-1].age+1;
    }
    else
    {
        c.age = 0;
    }
}

template<class DataTypes>
void PenalityContactForceField<DataTypes>::addForce()
{
    assert(this->object1);
    assert(this->object2);
    VecDeriv& f1 = *this->object1->getF();
    const VecCoord& p1 = *this->object1->getX();
    //const VecDeriv& v1 = *this->object1->getV();
    VecDeriv& f2 = *this->object2->getF();
    const VecCoord& p2 = *this->object2->getX();
    //const VecDeriv& v2 = *this->object2->getV();
    f1.resize(p1.size());
    f2.resize(p2.size());
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        Contact& c = contacts[i];
        Coord u = p2[c.m2]-p1[c.m1];
        c.pen = c.dist - u*c.norm;
        if (c.pen > 0)
        {
            Real fN = c.ks * c.pen;
            Deriv force = -c.norm*fN;
            f1[c.m1]+=force;
            f2[c.m2]-=force;
        }
    }
}

template<class DataTypes>
void PenalityContactForceField<DataTypes>::addDForce()
{
    VecDeriv& f1  = *this->object1->getF();
    //const VecCoord& p1 = *this->object1->getX();
    const VecDeriv& dx1 = *this->object1->getDx();
    VecDeriv& f2  = *this->object2->getF();
    //const VecCoord& p2 = *this->object2->getX();
    const VecDeriv& dx2 = *this->object2->getDx();
    f1.resize(dx1.size());
    f2.resize(dx2.size());
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        const Contact& c = contacts[i];
        if (c.pen > 0) // + dpen > 0)
        {
            Coord du = dx2[c.m2]-dx1[c.m1];
            Real dpen = - du*c.norm;
            //if (c.pen < 0) dpen += c.pen; // start penality at distance 0
            Real dfN = c.ks * dpen;
            Deriv dforce = -c.norm*dfN;
            f1[c.m1]+=dforce;
            f2[c.m2]-=dforce;
        }
    }
}

template <class DataTypes>
double PenalityContactForceField<DataTypes>::getPotentialEnergy()
{
    cerr<<"PenalityContactForceField::getPotentialEnergy-not-implemented !!!"<<endl;
    return 0;
}

template<class DataTypes>
void PenalityContactForceField<DataTypes>::draw()
{
    if (!((this->object1 == this->object2)?getContext()->getShowForceFields():getContext()->getShowInteractionForceFields())) return;
    const VecCoord& p1 = *this->object1->getX();
    const VecCoord& p2 = *this->object2->getX();
    glDisable(GL_LIGHTING);

    glBegin(GL_LINES);
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        const Contact& c = contacts[i];
        Real d = c.dist - (p2[c.m2]-p1[c.m1])*c.norm;
        if (c.age > 10) //c.spen > c.mu_s * c.ks * 0.99)
            if (d > 0)
                glColor4f(1,0,1,1);
            else
                glColor4f(0,1,1,1);
        else if (d > 0)
            glColor4f(1,0,0,1);
        else
            glColor4f(0,1,0,1);
        helper::gl::glVertexT(p1[c.m1]);
        helper::gl::glVertexT(p2[c.m2]);
    }
    glEnd();

    if (getContext()->getShowNormals())
    {
        glColor4f(1,1,0,1);
        glBegin(GL_LINES);
        for (unsigned int i=0; i<contacts.size(); i++)
        {
            const Contact& c = contacts[i];
            Coord p = p1[c.m1] - c.norm;
            helper::gl::glVertexT(p1[c.m1]);
            helper::gl::glVertexT(p);
            p = p2[c.m2] + c.norm;
            helper::gl::glVertexT(p2[c.m2]);
            helper::gl::glVertexT(p);
        }
        glEnd();
    }
}


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
