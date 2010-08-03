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

    MatrixDeriv& c1 = *this->object1->getC();
    MatrixDeriv& c2 = *this->object2->getC();

    const Coord P = (*this->object1->getXfree())[tm1];
    const Coord A = (*this->object2->getXfree())[tm2a];
    const Coord B = (*this->object2->getXfree())[tm2b];

    // the axis
    Coord uniAB = B - A;
    const Real ab = uniAB.norm();
    uniAB.normalize();

    // projection of the point on the axis
    Real r = (P-A) * uniAB;
    Real r2 = r / ab;
    const Coord proj = A + uniAB * r;

    // We move the constraint point onto the projection
    Coord dir1 = P - proj;
    m_dist = dir1.norm(); // constraint violation
    dir1.normalize(); // direction of the constraint

    Coord dir2 = cross(dir1, uniAB);
    dir2.normalize();

    cid = constraintId;
    constraintId += 2;

    MatrixDerivRowIterator c1_it = c1.writeLine(cid);
    c1_it.addCol(tm1, dir1);

    MatrixDerivRowIterator c2_it = c2.writeLine(cid);
    c2_it.addCol(tm2a, -dir1 * (1-r2));
    c2_it.addCol(tm2b, -dir1 * r2);

    c1_it = c1.writeLine(cid + 1);
    c1_it.addCol(tm1, dir2);

    c2_it = c2.writeLine(cid + 1);
    c2_it.addCol(tm2a, -dir2 * (1-r2));
    c2_it.addCol(tm2b, -dir2 * r2);

    thirdConstraint = 0;

    if (r < 0)
    {
        thirdConstraint = r;
        constraintId++;

        c1_it = c1.writeLine(cid + 2);
        c1_it.addCol(tm1, uniAB);

        c2_it = c2.writeLine(cid + 2);
        c2_it.addCol(tm2a, -uniAB);
    }
    else if (r > ab)
    {
        thirdConstraint = r - ab;
        constraintId++;

        c1_it = c1.writeLine(cid + 2);
        c1_it.addCol(tm1, -uniAB);

        c2_it = c2.writeLine(cid + 2);
        c2_it.addCol(tm2b, uniAB);
    }
}

template<class DataTypes>
void SlidingConstraint<DataTypes>::getConstraintValue(defaulttype::BaseVector* v, bool freeMotion)
{
    if (!freeMotion)
        sout<<"WARNING has to be implemented for method based on non freeMotion"<<sendl;

    v->set(cid, m_dist);
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
