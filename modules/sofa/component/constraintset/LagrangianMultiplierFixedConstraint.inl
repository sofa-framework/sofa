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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_LAGRANGIANMULTIPLIERFIXEDCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINTSET_LAGRANGIANMULTIPLIERFIXEDCONSTRAINT_INL

#include <sofa/component/constraintset/LagrangianMultiplierFixedConstraint.h>
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
void LagrangianMultiplierFixedConstraint<DataTypes>::addConstraint(int indice, const Coord& pos)
{
    int i = constraints.size();
    constraints.resize(3*(i+1)); // 3 lamdba elements are requires to fix the X, Y and Z
    this->lambda->resize(i+1);
    (*this->lambda->getX())[i] = 0;
    PointConstraint& c = constraints[i];
    c.indice = indice;
    c.pos = pos;
}

template<class DataTypes>
void LagrangianMultiplierFixedConstraint<DataTypes>::init()
{
    this->core::behavior::ForceField<DataTypes>::init();
    //this->core::behavior::Constraint<DataTypes>::init();
}

template<class DataTypes>
void LagrangianMultiplierFixedConstraint<DataTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/)
{
    f.resize(x.size());

    LMVecCoord& lambda = *this->lambda->getX();
    LMVecDeriv& flambda = *this->lambda->getF();
    flambda.resize(lambda.size());

    // Initialize constraints
    for (unsigned int i=0; i<constraints.size(); i++)
    {
        PointConstraint& c = constraints[i];
        Coord val = x[c.indice] - c.pos;
        lambda[3*i+0] = val[0];
        lambda[3*i+1] = val[1];
        lambda[3*i+2] = val[2];
    }

    // flamdba += C . DOF
    for (unsigned int i=0; i<constraints.size(); i++)
    {
        PointConstraint& c = constraints[i];
        Coord val = x[c.indice] - c.pos;
        flambda[3*i+0] += val[0];
        flambda[3*i+1] += val[1];
        flambda[3*i+2] += val[2];
    }

    // f += Ct . lambda
    for (unsigned int i=0; i<constraints.size(); i++)
    {
        PointConstraint& c = constraints[i];
        Deriv val(lambda[3*i+0],lambda[3*i+1],lambda[3*i+2]);
        f[c.indice] += val;
    }
}

template<class DataTypes>
void LagrangianMultiplierFixedConstraint<DataTypes>::addDForce(VecDeriv& df, const VecDeriv& dx)
{
    df.resize(dx.size());

    //LMVecCoord& lambda = *this->lambda->getX();
    LMVecCoord& dlambda = *this->lambda->getDx();
    LMVecDeriv& flambda = *this->lambda->getF();
    flambda.resize(dlambda.size());

    // dflamdba += C . dX
    for (unsigned int i=0; i<constraints.size(); i++)
    {
        PointConstraint& c = constraints[i];
        Deriv val = dx[c.indice];
        flambda[3*i+0] += val[0];
        flambda[3*i+1] += val[1];
        flambda[3*i+2] += val[2];
    }

    // df += Ct . dlambda
    for (unsigned int i=0; i<constraints.size(); i++)
    {
        PointConstraint& c = constraints[i];
        Deriv val(dlambda[3*i+0],dlambda[3*i+1],dlambda[3*i+2]);
        df[c.indice] += val;
    }
}

template <class DataTypes>
double LagrangianMultiplierFixedConstraint<DataTypes>::getPotentialEnergy(const VecCoord& )
{
    cerr<<"LagrangianMultiplierFixedConstraint::getPotentialEnergy-not-implemented !!!"<<endl;
    return 0;
}


template<class DataTypes>
void LagrangianMultiplierFixedConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;
    const VecCoord& p = *this->mstate->getX();
    const LMVecCoord& lambda = *this->lambda->getX();
    glDisable(GL_LIGHTING);
    glColor4f(1,1,0,1);
    glBegin(GL_LINES);
    for (unsigned int i=0; i<constraints.size(); i++)
    {
        const PointConstraint& c = constraints[i];
        Coord p2 = p[c.indice] + Coord(lambda[3*i+0],lambda[3*i+1],lambda[3*i+2]);
        helper::gl::glVertexT(p[c.indice]);
        helper::gl::glVertexT(p2);
    }
    glEnd();
}

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
