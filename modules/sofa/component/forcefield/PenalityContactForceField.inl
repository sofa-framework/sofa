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
    prevContacts.swap(*contacts.beginEdit()); // save old contacts in prevContacts
    contacts.beginEdit()->clear();
    if (reserve)
        contacts.beginEdit()->reserve(reserve);
    contacts.endEdit();
}

template<class DataTypes>
void PenalityContactForceField<DataTypes>::addContact(int m1, int m2, int index1, int index2, const Deriv& norm, Real dist, Real ks, Real mu_s, Real mu_v, int oldIndex)
{
    int i = contacts.getValue().size();
    contacts.beginEdit()->resize(i+1);
    Contact& c = (*contacts.beginEdit())[i];
    c.m1 = m1;
    c.m2 = m2;
    c.index1 = index1;
    c.index2 = index2;
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
    contacts.endEdit();
}

template<class DataTypes>
void PenalityContactForceField<DataTypes>::addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& /*v1*/, const VecDeriv& /*v2*/)
{
    f1.resize(x1.size());
    f2.resize(x2.size());
    for (unsigned int i=0; i<contacts.getValue().size(); i++)
    {
        Contact& c = (*contacts.beginEdit())[i];
        Coord u = x2[c.m2]-x1[c.m1];
        c.pen = c.dist - u*c.norm;
        if (c.pen > 0)
        {
            Real fN = c.ks * c.pen;
            Deriv force = -c.norm*fN;
            f1[c.m1]+=force;
            f2[c.m2]-=force;
        }
    }
    contacts.endEdit();
}

template<class DataTypes>
void PenalityContactForceField<DataTypes>::addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2, double kFactor, double /*bFactor*/)
{
    df1.resize(dx1.size());
    df2.resize(dx2.size());
    for (unsigned int i=0; i<contacts.getValue().size(); i++)
    {
        const Contact& c = contacts.getValue()[i];
        if (c.pen > 0) // + dpen > 0)
        {
            Coord du = dx2[c.m2]-dx1[c.m1];
            Real dpen = - du*c.norm;
            //if (c.pen < 0) dpen += c.pen; // start penality at distance 0
            Real dfN = c.ks * dpen * kFactor;
            Deriv dforce = -c.norm*dfN;
            df1[c.m1]+=dforce;
            df2[c.m2]-=dforce;
        }
    }
}

template <class DataTypes>
double PenalityContactForceField<DataTypes>::getPotentialEnergy(const VecCoord&, const VecCoord&)
{
    cerr<<"PenalityContactForceField::getPotentialEnergy-not-implemented !!!"<<endl;
    return 0;
}

template<class DataTypes>
void PenalityContactForceField<DataTypes>::draw()
{
    if (!((this->mstate1 == this->mstate2)?getContext()->getShowForceFields():getContext()->getShowInteractionForceFields())) return;
    const VecCoord& p1 = *this->mstate1->getX();
    const VecCoord& p2 = *this->mstate2->getX();
    glDisable(GL_LIGHTING);

    glBegin(GL_LINES);
    for (unsigned int i=0; i<contacts.getValue().size(); i++)
    {
        const Contact& c = contacts.getValue()[i];
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
        for (unsigned int i=0; i<contacts.getValue().size(); i++)
        {
            const Contact& c = contacts.getValue()[i];
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


template<class DataTypes>
void PenalityContactForceField<DataTypes>::grabPoint(
    const core::componentmodel::behavior::MechanicalState<defaulttype::Vec3Types> *tool,
    const helper::vector< unsigned int > &index,
    helper::vector< std::pair< core::objectmodel::BaseObject*, defaulttype::Vec3f> > &result,
    helper::vector< unsigned int > &triangle,
    helper::vector< unsigned int > &index_point)
{
    if (static_cast< core::objectmodel::BaseObject *>(this->mstate1) == static_cast< const core::objectmodel::BaseObject *>(tool))
    {
        for (unsigned int i=0; i<contacts.getValue().size(); i++)
        {
            for (unsigned int j=0; j<index.size(); j++)
            {
                if (contacts.getValue()[i].m1  == (int)index[j])
                {
                    result.push_back(std::make_pair(static_cast< core::objectmodel::BaseObject *>(this),
                            (*this->mstate2->getX())[contacts.getValue()[i].m2])
                                    );
                    triangle.push_back(contacts.getValue()[i].index2);
                    index_point.push_back(index[j]);
                }
            }
        }
    }
    else if (static_cast< core::objectmodel::BaseObject *>(this->mstate2) == static_cast< const core::objectmodel::BaseObject *>(tool))
    {

        for (unsigned int i=0; i<contacts.getValue().size(); i++)
        {
            for (unsigned int j=0; j<index.size(); j++)
            {
                if (contacts.getValue()[i].m2  == (int)index[j])
                {
                    result.push_back(std::make_pair(static_cast< core::objectmodel::BaseObject *>(this),
                            (*this->mstate1->getX())[contacts.getValue()[i].m1])
                                    );

                    triangle.push_back(contacts.getValue()[i].index1);
                    index_point.push_back(index[j]);
                }
            }
        }
    }


}


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
