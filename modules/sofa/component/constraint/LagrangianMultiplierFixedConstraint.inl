#ifndef SOFA_COMPONENT_CONSTRAINT_LAGRANGIANMULTIPLIERFIXEDCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINT_LAGRANGIANMULTIPLIERFIXEDCONSTRAINT_INL

#include <sofa/component/constraint/LagrangianMultiplierFixedConstraint.h>
#include <sofa/core/componentmodel/behavior/Constraint.inl>
#include <sofa/component/MechanicalObject.inl>
#include <sofa/helper/system/config.h>
#include <assert.h>
#if defined (__APPLE__)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif
#include <sofa/helper/gl/template.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace constraint
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
    this->core::componentmodel::behavior::ForceField<DataTypes>::init();
    //this->core::componentmodel::behavior::Constraint<DataTypes>::init();
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
void LagrangianMultiplierFixedConstraint<DataTypes>::draw()
{
    if (!getContext()->getShowForceFields()) return;
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

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
