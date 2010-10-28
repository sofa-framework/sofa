/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_BILATERALINTERACTIONCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINTSET_BILATERALINTERACTIONCONSTRAINT_INL

#include <sofa/component/constraintset/BilateralInteractionConstraint.h>
#include <sofa/core/behavior/Constraint.inl>

#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/gl/template.h>

namespace sofa
{

namespace component
{

namespace constraintset
{

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::init()
{
    assert(this->mstate1);
    assert(this->mstate2);
    prevForces.clear();
}


template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::buildConstraintMatrix(DataMatrixDeriv &c1_d, DataMatrixDeriv &c2_d, unsigned int &constraintId
        , const DataVecCoord &/*x1*/, const DataVecCoord &/*x2*/, const core::ConstraintParams* /*cParams*/)
{
    int tm1 = m1.getValue();
    int tm2 = m2.getValue();

    MatrixDeriv &c1 = *c1_d.beginEdit();
    MatrixDeriv &c2 = *c2_d.beginEdit();

    const defaulttype::Vec<3, Real> cx(1,0,0), cy(0,1,0), cz(0,0,1);

    cid = constraintId;
    constraintId += 3;

    MatrixDerivRowIterator c1_it = c1.writeLine(cid);
    c1_it.addCol(tm1, -cx);

    MatrixDerivRowIterator c2_it = c2.writeLine(cid);
    c2_it.addCol(tm2, cx);

    c1_it = c1.writeLine(cid + 1);
    c1_it.setCol(tm1, -cy);

    c2_it = c2.writeLine(cid + 1);
    c2_it.setCol(tm2, cy);

    c1_it = c1.writeLine(cid + 2);
    c1_it.setCol(tm1, -cz);

    c2_it = c2.writeLine(cid + 2);
    c2_it.setCol(tm2, cz);

    c1_d.endEdit();
    c2_d.endEdit();
}


template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::getConstraintViolation(defaulttype::BaseVector *v, const DataVecCoord &x1, const DataVecCoord &x2
        , const DataVecDeriv &/*v1*/, const DataVecDeriv &/*v2*/, const core::ConstraintParams*)
{
    dfree = x2.getValue()[m2.getValue()] - x1.getValue()[m1.getValue()];

    v->set(cid, dfree[0]);
    v->set(cid+1, dfree[1]);
    v->set(cid+2, dfree[2]);
}


template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::getConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset)
{
    resTab[offset] = new BilateralConstraintResolution3Dof(&prevForces);
    offset += 3;
}


template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::draw()
{
    if (!this->getContext()->getShowInteractionForceFields()) return;

    glDisable(GL_LIGHTING);
    glPointSize(10);
    glBegin(GL_POINTS);
    glColor4f(1,0,1,1);
    helper::gl::glVertexT((*this->mstate1->getX())[m1.getValue()]);
    helper::gl::glVertexT((*this->mstate2->getX())[m2.getValue()]);
    glEnd();
    glPointSize(1);
}

#ifndef SOFA_FLOAT
template<>
void BilateralInteractionConstraint<defaulttype::Rigid3dTypes>::buildConstraintMatrix(DataMatrixDeriv &c1_d, DataMatrixDeriv &c2_d, unsigned int &cIndex
        , const DataVecCoord &x1, const DataVecCoord &x2, const core::ConstraintParams *cParams);

template<>
void BilateralInteractionConstraint<defaulttype::Rigid3dTypes>::getConstraintViolation(defaulttype::BaseVector *v, const DataVecCoord &x1_d, const DataVecCoord &x2_d
        , const DataVecDeriv &v1_d, const DataVecDeriv &v2_d, const core::ConstraintParams *cParams);
#endif

#ifndef SOFA_DOUBLE
template<>
void BilateralInteractionConstraint<defaulttype::Rigid3fTypes>::buildConstraintMatrix(DataMatrixDeriv &c1_d, DataMatrixDeriv &c2_d, unsigned int &cIndex
        , const DataVecCoord &x1_d, const DataVecCoord &x2_d, const core::ConstraintParams *cParams);

template<>
void BilateralInteractionConstraint<defaulttype::Rigid3fTypes>::getConstraintViolation(defaulttype::BaseVector *v, const DataVecCoord &x1_d, const DataVecCoord &x2_d
        , const DataVecDeriv &v1_d, const DataVecDeriv &v2_d, const core::ConstraintParams *cParams);
#endif

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONSTRAINTSET_BILATERALINTERACTIONCONSTRAINT_INL
