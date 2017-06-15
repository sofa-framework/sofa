/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_CONSTRAINTSET_SLIDINGCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINTSET_SLIDINGCONSTRAINT_INL

#include <SofaConstraint/SlidingConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaConstraint/BilateralInteractionConstraint.h>
#include <SofaConstraint/UnilateralInteractionConstraint.h>

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
    assert(this->mstate1);
    assert(this->mstate2);

    thirdConstraint = 0;
}


template<class DataTypes>
void SlidingConstraint<DataTypes>::buildConstraintMatrix(const core::ConstraintParams*, DataMatrixDeriv &c1_d, DataMatrixDeriv &c2_d, unsigned int &cIndex
        , const DataVecCoord &x1, const DataVecCoord &x2)
{
    int tm1 = m1.getValue();
    int tm2a = m2a.getValue();
    int tm2b = m2b.getValue();

    MatrixDeriv &c1 = *c1_d.beginEdit();
    MatrixDeriv &c2 = *c2_d.beginEdit();

    const Coord P = x1.getValue()[tm1];
    const Coord A = x2.getValue()[tm2a];
    const Coord B = x2.getValue()[tm2b];

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

    cid = cIndex;
    cIndex += 2;

    MatrixDerivRowIterator c1_it = c1.writeLine(cid);
    c1_it.addCol(tm1, dir1);

    MatrixDerivRowIterator c2_it = c2.writeLine(cid);
    c2_it.addCol(tm2a, -dir1 * (1-r2));
    c2_it.addCol(tm2b, -dir1 * r2);

    c1_it = c1.writeLine(cid + 1);
    c1_it.setCol(tm1, dir2);

    c2_it = c2.writeLine(cid + 1);
    c2_it.addCol(tm2a, -dir2 * (1-r2));
    c2_it.addCol(tm2b, -dir2 * r2);

    thirdConstraint = 0;

    if (r < 0)
    {
        thirdConstraint = r;
        cIndex++;

        c1_it = c1.writeLine(cid + 2);
        c1_it.setCol(tm1, uniAB);

        c2_it = c2.writeLine(cid + 2);
        c2_it.addCol(tm2a, -uniAB);
    }
    else if (r > ab)
    {
        thirdConstraint = r - ab;
        cIndex++;

        c1_it = c1.writeLine(cid + 2);
        c1_it.setCol(tm1, -uniAB);

        c2_it = c2.writeLine(cid + 2);
        c2_it.addCol(tm2b, uniAB);
    }

    c1_d.endEdit();
    c2_d.endEdit();
}


template<class DataTypes>
void SlidingConstraint<DataTypes>::getConstraintViolation(const core::ConstraintParams *, defaulttype::BaseVector *v, const DataVecCoord &, const DataVecCoord &
        , const DataVecDeriv &, const DataVecDeriv &)
{
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
void SlidingConstraint<DataTypes>::getConstraintResolution(const ConstraintParams*,
                                                           std::vector<core::behavior::ConstraintResolution*>& resTab,
                                                           unsigned int& offset)
{
    resTab[offset++] = new BilateralConstraintResolution();
    resTab[offset++] = new BilateralConstraintResolution();

    if(thirdConstraint)
        resTab[offset++] = new UnilateralConstraintResolution();
}


template<class DataTypes>
void SlidingConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowInteractionForceFields()) return;

    glDisable(GL_LIGHTING);
    glPointSize(10);
    glBegin(GL_POINTS);
    if(thirdConstraint<0)
        glColor4f(1,1,0,1);
    else if(thirdConstraint>0)
        glColor4f(0,1,0,1);
    else
        glColor4f(1,0,1,1);
    helper::gl::glVertexT((this->mstate1->read(core::ConstVecCoordId::position())->getValue())[m1.getValue()]);
    //      helper::gl::glVertexT((*this->object2->read(sofa::core::ConstVecCoordId::position())->getValue())[m3]);
    //      helper::gl::glVertexT(proj);
    glEnd();

    glBegin(GL_LINES);
    glColor4f(0,0,1,1);
    helper::gl::glVertexT((this->mstate2->read(core::ConstVecCoordId::position())->getValue())[m2a.getValue()]);
    helper::gl::glVertexT((this->mstate2->read(core::ConstVecCoordId::position())->getValue())[m2b.getValue()]);
    glEnd();
    glPointSize(1);
#endif /* SOFA_NO_OPENGL */
}

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
