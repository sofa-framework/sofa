#ifndef SOFA_COMPONENTS_LagrangianMultiplierAttachConstraint_INL
#define SOFA_COMPONENTS_LagrangianMultiplierAttachConstraint_INL

#include "LagrangianMultiplierAttachConstraint.h"
#include "Sofa/Core/Constraint.inl"
#include "Sofa/Core/MechanicalObject.inl"
#include "Common/config.h"
#include <assert.h>
#include <GL/gl.h>
#include "GL/template.h"

namespace Sofa
{

namespace Components
{

template<class DataTypes>
void LagrangianMultiplierAttachConstraint<DataTypes>::addConstraint(int m1, int m2)
{
    int i = constraints.size();
    constraints.resize(i+1);
    this->lambda->resize(3*(i+1));
    (*this->lambda->getX())[i] = 0;
    ConstraintData& c = constraints[i];
    c.m1 = m1;
    c.m2 = m2;
}

template<class DataTypes>
void LagrangianMultiplierAttachConstraint<DataTypes>::addForce()
{
    assert(this->object1);
    assert(this->object2);
    VecDeriv& f1 = *this->object1->getF();
    VecCoord& p1 = *this->object1->getX();
    //VecDeriv& v1 = *this->object1->getV();
    VecDeriv& f2 = *this->object2->getF();
    VecCoord& p2 = *this->object2->getX();
    //VecDeriv& v2 = *this->object2->getV();
    f1.resize(p1.size());
    f2.resize(p2.size());

    LMVecCoord& lambda = *this->lambda->getX();
    LMVecDeriv& flambda = *this->lambda->getF();
    flambda.resize(lambda.size());

    // Initialize constraints
    for (unsigned int i=0; i<constraints.size(); i++)
    {
        ConstraintData& c = constraints[i];
        Coord val = p2[c.m2]-p1[c.m1];
        lambda[3*i+0] = val[0];
        lambda[3*i+1] = val[1];
        lambda[3*i+2] = val[2];
    }

    // flamdba += C . DOF
    for (unsigned int i=0; i<constraints.size(); i++)
    {
        ConstraintData& c = constraints[i];
        Coord val = p2[c.m2]-p1[c.m1];
        flambda[3*i+0] += val[0];
        flambda[3*i+1] += val[1];
        flambda[3*i+2] += val[2];
    }

    // f += Ct . lambda
    for (unsigned int i=0; i<constraints.size(); i++)
    {
        ConstraintData& c = constraints[i];
        Deriv val(lambda[3*i+0],lambda[3*i+1],lambda[3*i+2]);
        f1[c.m1] += val;
        f2[c.m2] -= val;
    }
}

template<class DataTypes>
void LagrangianMultiplierAttachConstraint<DataTypes>::addDForce()
{
    VecDeriv& f1  = *this->object1->getF();
    //VecCoord& p1 = *this->object1->getX();
    VecDeriv& dx1 = *this->object1->getDx();
    VecDeriv& f2  = *this->object2->getF();
    //VecCoord& p2 = *this->object2->getX();
    VecDeriv& dx2 = *this->object2->getDx();
    f1.resize(dx1.size());
    f2.resize(dx2.size());

    //LMVecCoord& lambda = *this->lambda->getX();
    LMVecCoord& dlambda = *this->lambda->getDx();
    LMVecDeriv& flambda = *this->lambda->getF();
    flambda.resize(dlambda.size());

    // dflamdba += C . dX
    for (unsigned int i=0; i<constraints.size(); i++)
    {
        ConstraintData& c = constraints[i];
        Deriv val = dx2[c.m2] - dx1[c.m1];
        flambda[3*i+0] += val[0];
        flambda[3*i+1] += val[1];
        flambda[3*i+2] += val[2];
    }

    // df += Ct . dlambda
    for (unsigned int i=0; i<constraints.size(); i++)
    {
        ConstraintData& c = constraints[i];
        Deriv val(dlambda[3*i+0],dlambda[3*i+1],dlambda[3*i+2]);
        f1[c.m1] += val;
        f2[c.m2] -= val;
    }
}

template<class DataTypes>
void LagrangianMultiplierAttachConstraint<DataTypes>::draw()
{
    if (!((this->object1 == this->object2)?getContext()->getShowForceFields():getContext()->getShowInteractionForceFields())) return;
    VecCoord& p1 = *this->object1->getX();
    VecCoord& p2 = *this->object2->getX();
    LMVecCoord& lambda = *this->lambda->getX();
    glDisable(GL_LIGHTING);

    glColor4f(1,0,0,1);
    glBegin(GL_LINES);
    for (unsigned int i=0; i<constraints.size(); i++)
    {
        const ConstraintData& c = constraints[i];
        GL::glVertexT(p1[c.m1]);
        GL::glVertexT(p2[c.m2]);
    }
    glEnd();

    //if (getContext()->getShowNormals())
    {
        glColor4f(1,1,0,1);
        glBegin(GL_LINES);
        for (unsigned int i=0; i<constraints.size(); i++)
        {
            const ConstraintData& c = constraints[i];
            Coord dp ( lambda[3*i+0], lambda[3*i+1], lambda[3*i+2] );
            dp*=0.001;
            Coord p = p1[c.m1] - dp;
            GL::glVertexT(p1[c.m1]);
            GL::glVertexT(p);
            p = p2[c.m2] + dp;
            GL::glVertexT(p2[c.m2]);
            GL::glVertexT(p);
        }
        glEnd();
    }
}


} // namespace Components

} // namespace Sofa

#endif
