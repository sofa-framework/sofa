/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_LAGRANGIANMULTIPLIERATTACHCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINTSET_LAGRANGIANMULTIPLIERATTACHCONSTRAINT_INL

#include <sofa/component/constraintset/LagrangianMultiplierAttachConstraint.h>
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
void LagrangianMultiplierAttachConstraint<DataTypes>::addConstraint(int m1, int m2)
{
    int i = constraints.size();
    constraints.resize(i+1);
    this->lambda->resize(3*(i+1));
    (*this->lambda->getX())[i] = 0;
    (*this->lambda->getV())[i] = 0;
    ConstraintData& c = constraints[i];
    c.m1 = m1;
    c.m2 = m2;
}

template<class DataTypes>
void LagrangianMultiplierAttachConstraint<DataTypes>::addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& p1, const VecCoord& p2, const VecDeriv& /*v1*/, const VecDeriv& /*v2*/)
{
    f1.resize(p1.size());
    f2.resize(p2.size());

    LMVecCoord& lambda = *this->lambda->getX();
    LMVecDeriv& vlambda = *this->lambda->getV();
    LMVecDeriv& flambda = *this->lambda->getF();
    flambda.resize(lambda.size());

    // Initialize constraints
    for (unsigned int i=0; i<constraints.size(); i++)
    {
        //ConstraintData& c = constraints[i];
        //Coord val = p2[c.m2]-p1[c.m1];
        lambda[3*i+0] = 0; //val[0];
        lambda[3*i+1] = 0; //val[1];
        lambda[3*i+2] = 0; //val[2];
        vlambda[3*i+0] = 0;
        vlambda[3*i+1] = 0;
        vlambda[3*i+2] = 0;
    }

    // flamdba -= C . DOF
    for (unsigned int i=0; i<constraints.size(); i++)
    {
        ConstraintData& c = constraints[i];
        Coord val = (p2[c.m2]-p1[c.m1]);
        val *= 16;
        flambda[3*i+0] -= val[0];
        flambda[3*i+1] -= val[1];
        flambda[3*i+2] -= val[2];
    }

    // f -= Ct . lambda
    for (unsigned int i=0; i<constraints.size(); i++)
    {
        ConstraintData& c = constraints[i];
        Deriv val(lambda[3*i+0],lambda[3*i+1],lambda[3*i+2]);
        val *= 16;
        f1[c.m1] += val;
        f2[c.m2] -= val;
    }
}

template<class DataTypes>
void LagrangianMultiplierAttachConstraint<DataTypes>::addDForce(VecDeriv& f1, VecDeriv& f2, const VecDeriv& dx1, const VecDeriv& dx2)
{
    f1.resize(dx1.size());
    f2.resize(dx2.size());

    //LMVecCoord& lambda = *this->lambda->getX();
    LMVecCoord& dlambda = *this->lambda->getDx();
    LMVecDeriv& flambda = *this->lambda->getF();
    flambda.resize(dlambda.size());

    // dflamdba -= C . dX
    for (unsigned int i=0; i<constraints.size(); i++)
    {
        ConstraintData& c = constraints[i];
        Deriv val = (dx2[c.m2] - dx1[c.m1]);
        val *= 16;
        flambda[3*i+0] -= val[0];
        flambda[3*i+1] -= val[1];
        flambda[3*i+2] -= val[2];
    }

    // df -= Ct . dlambda
    for (unsigned int i=0; i<constraints.size(); i++)
    {
        ConstraintData& c = constraints[i];
        Deriv val(dlambda[3*i+0],dlambda[3*i+1],dlambda[3*i+2]);
        val *= 16;
        f1[c.m1] += val;
        f2[c.m2] -= val;
    }
}


template<class DataTypes>
void LagrangianMultiplierAttachConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!((this->mstate1 == this->mstate2)?vparams->displayFlags().getShowForceFields():vparams->displayFlags().getShowInteractionForceFields())) return;
    const VecCoord& p1 = *this->mstate1->getX();
    const VecCoord& p2 = *this->mstate2->getX();
    const LMVecCoord& lambda = *this->lambda->getX();
    glDisable(GL_LIGHTING);

    glColor4f(1,0,0,1);
    glBegin(GL_LINES);
    for (unsigned int i=0; i<constraints.size(); i++)
    {
        const ConstraintData& c = constraints[i];
        helper::gl::glVertexT(p1[c.m1]);
        helper::gl::glVertexT(p2[c.m2]);
    }
    glEnd();

    //if (vparams->displayFlags().getShowNormals())
    {
        glColor4f(1,1,0,1);
        glBegin(GL_LINES);
        for (unsigned int i=0; i<constraints.size(); i++)
        {
            const ConstraintData& c = constraints[i];
            Coord dp ( lambda[3*i+0], lambda[3*i+1], lambda[3*i+2] );
            dp*=1.0/16;
            dp*=0.001;
            Coord p = p1[c.m1] - dp;
            helper::gl::glVertexT(p1[c.m1]);
            helper::gl::glVertexT(p);
            p = p2[c.m2] + dp;
            helper::gl::glVertexT(p2[c.m2]);
            helper::gl::glVertexT(p);
        }
        glEnd();
    }
}


} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
