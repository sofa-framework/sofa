/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_LAGRANGIANMULTIPLIERCONTACTCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINTSET_LAGRANGIANMULTIPLIERCONTACTCONSTRAINT_INL

#include <sofa/component/constraintset/LagrangianMultiplierContactConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/behavior/Constraint.inl>
#include <SofaBaseMechanics/MechanicalObject.inl>
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

namespace constraintset
{

template<class DataTypes>
void LagrangianMultiplierContactConstraint<DataTypes>::addContact(int m1, int m2, const Deriv& norm, Real dist, Real ks, Real mu_s, Real mu_v)
{
    int i = contacts.size();
    contacts.resize(i+1);
    this->lambda->resize(i+1);
    (*this->lambda->getX())[i] = 0;
    (*this->lambda->getV())[i] = 0;
    Contact& c = contacts[i];
    c.m1 = m1;
    c.m2 = m2;
    c.norm = norm;
    c.dist = dist;
    c.ks = ks;
    c.mu_s = mu_s;
    c.mu_v = mu_v;
    c.pen = 0;
}

template<class DataTypes>
void LagrangianMultiplierContactConstraint<DataTypes>::addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& p1, const VecCoord& p2, const VecDeriv& /*v1*/, const VecDeriv& /*v2*/)
{
    f1.resize(p1.size());
    f2.resize(p2.size());

    LMVecCoord& lambda = *this->lambda->getX();
    LMVecDeriv& vlambda = *this->lambda->getV();
    LMVecDeriv& flambda = *this->lambda->getF();
    flambda.resize(lambda.size());

    // Create list of active contact
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        Contact& c = contacts[i];
        c.pen = c.pen0 = (p2[c.m2]-p1[c.m1])*c.norm - c.dist;//c.dist - (p1[c.m1]-p2[c.m2])*c.norm;
        lambda[i] = 0;
        vlambda[i] = 0;
    }

    // flamdba += d - C . DOF
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        Contact& c = contacts[i];
        if (c.pen0 > 0.001) continue;
        flambda[i] += c.dist - (p2[c.m2] - p1[c.m1])*c.norm;
    }

    // f -= Ct . lambda
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        Contact& c = contacts[i];
        if (c.pen0 > 0.001) continue;
        Real v = lambda[i];
        f1[c.m1] += c.norm * v;
        f2[c.m2] -= c.norm * v;
    }
}

template<class DataTypes>
void LagrangianMultiplierContactConstraint<DataTypes>::addDForce(VecDeriv& f1, VecDeriv& f2, const VecDeriv& dx1, const VecDeriv& dx2)
{
    f1.resize(dx1.size());
    f2.resize(dx2.size());

    //LMVecCoord& lambda = *this->lambda->getX();
    LMVecCoord& dlambda = *this->lambda->getDx();
    LMVecDeriv& flambda = *this->lambda->getF();
    flambda.resize(dlambda.size());

    // dflamdba -= C . dX
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        Contact& c = contacts[i];
        if (c.pen0 > 0.001) continue;
        flambda[i] -= (dx2[c.m2] - dx1[c.m1])*c.norm;
    }

    // df -= Ct . dlambda
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        Contact& c = contacts[i];
        if (c.pen0 > 0.001) continue;
        Real v = dlambda[i];
        f1[c.m1] += c.norm * v;
        f2[c.m2] -= c.norm * v;
    }
}


template<class DataTypes>
void LagrangianMultiplierContactConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!((this->mstate1 == this->mstate2)?vparams->displayFlags().getShowForceFields():vparams->displayFlags().getShowInteractionForceFields())) return;
    const VecCoord& p1 = *this->mstate1->getX();
    const VecCoord& p2 = *this->mstate2->getX();
    const LMVecCoord& lambda = *this->lambda->getX();
    glDisable(GL_LIGHTING);

    glBegin(GL_LINES);
    for (unsigned int i=0; i<contacts.size(); i++)
    {
        const Contact& c = contacts[i];
        Real d = c.dist - (p2[c.m2]-p1[c.m1])*c.norm;
        if (d > 0)
            glColor4f(1,0,0,1);
        else
            glColor4f(0,1,0,1);
        helper::gl::glVertexT(p1[c.m1]);
        helper::gl::glVertexT(p2[c.m2]);
    }
    glEnd();
    glLineWidth(5);
    //if (vparams->displayFlags().getShowNormals())
    {
        glColor4f(1,1,0,1);
        glBegin(GL_LINES);
        for (unsigned int i=0; i<contacts.size(); i++)
        {
            const Contact& c = contacts[i];
            //if (c.pen > 0) continue;
            //std::cout << " lambda["<<i<<"]="<<lambda[i]<<std::endl;
            Coord p = p1[c.m1] - c.norm * lambda[i];
            helper::gl::glVertexT(p1[c.m1]);
            helper::gl::glVertexT(p);
            p = p2[c.m2] + c.norm * lambda[i];
            helper::gl::glVertexT(p2[c.m2]);
            helper::gl::glVertexT(p);
        }
        glEnd();
    }
    glLineWidth(1);
}

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
