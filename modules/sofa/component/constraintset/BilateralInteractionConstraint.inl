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
    assert(this->object1);
    assert(this->object2);
    prevForces.clear();
}

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::buildConstraintMatrix(unsigned int &constraintId, core::VecId)
{
    int tm1, tm2;
    tm1 = m1.getValue();
    tm2 = m2.getValue();

    assert(this->object1);
    assert(this->object2);

    VecConst& c1 = *this->object1->getC();
    VecConst& c2 = *this->object2->getC();

    defaulttype::Vec<3, Real> cx(1,0,0), cy(0,1,0), cz(0,0,1);

    cid = constraintId;
    constraintId+=3;

    SparseVecDeriv svd1;
    SparseVecDeriv svd2;

    this->object1->setConstraintId(cid);
    svd1.add(tm1, -cx);
    c1.push_back(svd1);

    this->object2->setConstraintId(cid);
    svd2.add(tm2, cx);
    c2.push_back(svd2);

    this->object1->setConstraintId(cid+1);
    svd1.set(tm1, -cy);
    c1.push_back(svd1);

    this->object2->setConstraintId(cid+1);
    svd2.set(tm2, cy);
    c2.push_back(svd2);

    this->object1->setConstraintId(cid+2);
    svd1.set(tm1, -cz);
    c1.push_back(svd1);

    this->object2->setConstraintId(cid+2);
    svd2.set(tm2, cz);
    c2.push_back(svd2);
}

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::getConstraintValue(defaulttype::BaseVector* v, bool freeMotion)
{
    if (!freeMotion)
        sout<<"WARNING has to be implemented for method based on non freeMotion"<<sendl;

    if (freeMotion)
        dfree = (*this->object2->getXfree())[m2.getValue()] - (*this->object1->getXfree())[m1.getValue()];
    else
        dfree = (*this->object2->getX())[m2.getValue()] - (*this->object1->getX())[m1.getValue()];

    v->set(cid, dfree[0]);
    v->set(cid+1, dfree[1]);
    v->set(cid+2, dfree[2]);
}

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::getConstraintId(long* id, unsigned int &offset)
{
    if (!yetIntegrated)
    {
        id[offset++] = -(int)cid;

        yetIntegrated = true;
    }
    else
    {
        id[offset++] = cid;
    }
}

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::getConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset)
{
    resTab[offset] = new BilateralConstraintResolution3Dof(&prevForces);
    offset += 3;

//	for(int i=0; i<3; i++)
//		resTab[offset++] = new BilateralConstraintResolution(); //&prevForces
}

template<class DataTypes>
void BilateralInteractionConstraint<DataTypes>::draw()
{
    if (!this->getContext()->getShowInteractionForceFields()) return;

    glDisable(GL_LIGHTING);
    glPointSize(10);
    glBegin(GL_POINTS);
    glColor4f(1,0,1,1);
    helper::gl::glVertexT((*this->object1->getX())[m1.getValue()]);
    helper::gl::glVertexT((*this->object2->getX())[m2.getValue()]);
    glEnd();
    glPointSize(1);
}

#ifndef SOFA_FLOAT
template<>
void BilateralInteractionConstraint<defaulttype::Rigid3dTypes>::buildConstraintMatrix(unsigned int &constraintId, core::VecId);
template<>
void BilateralInteractionConstraint<defaulttype::Rigid3dTypes>::getConstraintValue(defaulttype::BaseVector* v, bool freeMotion);
#endif

#ifndef SOFA_DOUBLE
template<>
void BilateralInteractionConstraint<defaulttype::Rigid3fTypes>::buildConstraintMatrix(unsigned int &constraintId, core::VecId);
template<>
void BilateralInteractionConstraint<defaulttype::Rigid3fTypes>::getConstraintValue(defaulttype::BaseVector* v, bool freeMotion);
#endif

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
