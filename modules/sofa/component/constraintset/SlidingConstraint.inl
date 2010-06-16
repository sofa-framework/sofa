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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_SLIDINGCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINTSET_SLIDINGCONSTRAINT_INL

#include <sofa/component/constraintset/SlidingConstraint.h>
#include <sofa/component/constraintset/BilateralInteractionConstraint.h>
#include <sofa/component/constraintset/UnilateralInteractionConstraint.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/gl/template.h>
namespace sofa
{

namespace component
{

namespace constraintset
{

template<class DataTypes>
void SlidingConstraint<DataTypes>::init()
{
    assert(this->object1);
    assert(this->object2);

    thirdConstraint = 0;
}

template<class DataTypes>
void SlidingConstraint<DataTypes>::buildConstraintMatrix(unsigned int &constraintId, core::VecId)
{
    int tm1, tm2a, tm2b;
    tm1 = m1.getValue();
    tm2a = m2a.getValue();
    tm2b = m2b.getValue();

    assert(this->object1);
    assert(this->object2);

    VecConst& c1 = *this->object1->getC();
    VecConst& c2 = *this->object2->getC();

    Coord A, B, P, uniAB, dir1, dir2, proj;
    P = (*this->object1->getXfree())[tm1];
    A = (*this->object2->getXfree())[tm2a];
    B = (*this->object2->getXfree())[tm2b];

    // the axis
    uniAB = (B - A);
    Real ab = uniAB.norm();
    uniAB.normalize();

    // projection of the point on the axis
    Real r = (P-A) * uniAB;
    Real r2 = r / ab;
    proj = A + uniAB * r;

    // We move the constraint point onto the projection
    dir1 = P-proj;
    dist = dir1.norm(); // constraint violation
    dir1.normalize(); // direction of the constraint

    dir2 = cross(dir1, uniAB);
    dir2.normalize();

    cid = constraintId;
    constraintId+=2;

    SparseVecDeriv svd1;
    SparseVecDeriv svd2;

    this->object1->setConstraintId(cid);
    svd1.add(tm1, dir1);
    c1.push_back(svd1);

    this->object2->setConstraintId(cid);
    svd2.add(tm2a, -dir1 * (1-r2));
    svd2.add(tm2b, -dir1 * r2);
    c2.push_back(svd2);
    svd2.clear();

    this->object1->setConstraintId(cid+1);
    svd1.set(tm1, dir2);
    c1.push_back(svd1);

    this->object2->setConstraintId(cid+1);
    svd2.add(tm2a, -dir2 * (1-r2));
    svd2.add(tm2b, -dir2 * r2);
    c2.push_back(svd2);
    svd2.clear();

    thirdConstraint = 0;
    if(r<0)
    {
        thirdConstraint = r;
        constraintId++;

        this->object1->setConstraintId(cid+2);
        svd1.set(tm1, uniAB);
        c1.push_back(svd1);

        this->object2->setConstraintId(cid+2);
        svd2.add(tm2a, -uniAB);
        c2.push_back(svd2);
    }
    else if(r>ab)
    {
        thirdConstraint = r-ab;
        constraintId++;

        this->object1->setConstraintId(cid+2);
        svd1.set(tm1, -uniAB);
        c1.push_back(svd1);

        this->object2->setConstraintId(cid+2);
        svd2.add(tm2b, uniAB);
        c2.push_back(svd2);
    }
}

template<class DataTypes>
void SlidingConstraint<DataTypes>::getConstraintValue(defaulttype::BaseVector* v, bool freeMotion)
{
    if (!freeMotion)
        sout<<"WARNING has to be implemented for method based on non freeMotion"<<sendl;

    v->set(cid, dist);
    v->set(cid+1, 0.0);

    if(thirdConstraint)
    {
        if(thirdConstraint>0)
            v->set(cid+2, -thirdConstraint);
        else
            v->set(cid+2, thirdConstraint);
    }
}

template<class DataTypes>
void SlidingConstraint<DataTypes>::getConstraintId(long* id, unsigned int &offset)
{
    if (!yetIntegrated)
    {
        id[offset++] = -(int)cid;

        yetIntegrated =  true;
    }
    else
    {
        id[offset++] = cid;
    }
}
#ifdef SOFA_DEV
template<class DataTypes>
void SlidingConstraint<DataTypes>::getConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset)
{
    for(int i=0; i<2; i++)
        resTab[offset++] = new BilateralConstraintResolution();
    if(thirdConstraint)
        resTab[offset++] = new UnilateralConstraintResolution();
}
#endif

template<class DataTypes>
void SlidingConstraint<DataTypes>::draw()
{
    if (!this->getContext()->getShowInteractionForceFields()) return;

    glDisable(GL_LIGHTING);
    glPointSize(10);
    glBegin(GL_POINTS);
    if(thirdConstraint<0)
        glColor4f(1,1,0,1);
    else if(thirdConstraint>0)
        glColor4f(0,1,0,1);
    else
        glColor4f(1,0,1,1);
    helper::gl::glVertexT((*this->object1->getX())[m1.getValue()]);
//      helper::gl::glVertexT((*this->object2->getX())[m3]);
//      helper::gl::glVertexT(proj);
    glEnd();

    glBegin(GL_LINES);
    glColor4f(0,0,1,1);
    helper::gl::glVertexT((*this->object2->getX())[m2a.getValue()]);
    helper::gl::glVertexT((*this->object2->getX())[m2b.getValue()]);
    glEnd();
    glPointSize(1);
}

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
