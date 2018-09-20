/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/defaulttype/RGBAColor.h>
#include <sofa/defaulttype/Vec.h>
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

    Deriv m_dirAxe = B - A;
    const Real ab = m_dirAxe.norm();
    m_dirAxe.normalize();

    // projection of the point on the axis
    Real r = (P-A) * m_dirAxe;
    Real r2 = r / ab;
    const Deriv proj = A + m_dirAxe * r;

    // We move the constraint point onto the projection
    m_dirProj = P - proj;
    m_dist = m_dirProj.norm(); // constraint violation
    m_dirProj.normalize(); // direction of the constraint

    Deriv m_dirOrtho = cross(m_dirProj, m_dirAxe);
    m_dirOrtho.normalize();

    cid = cIndex;
    cIndex += 2;

    MatrixDerivRowIterator c1_it = c1.writeLine(cid);
    c1_it.addCol(tm1, m_dirProj);

    MatrixDerivRowIterator c2_it = c2.writeLine(cid);
    c2_it.addCol(tm2a, -m_dirProj * (1-r2));
    c2_it.addCol(tm2b, -m_dirProj * r2);

    c1_it = c1.writeLine(cid + 1);
    c1_it.setCol(tm1, m_dirOrtho);

    c2_it = c2.writeLine(cid + 1);
    c2_it.addCol(tm2a, -m_dirOrtho * (1-r2));
    c2_it.addCol(tm2b, -m_dirOrtho * r2);

    thirdConstraint = 0;

    if (r < 0)
    {
        thirdConstraint = r;
        cIndex++;

        c1_it = c1.writeLine(cid + 2);
        c1_it.setCol(tm1, m_dirAxe);

        c2_it = c2.writeLine(cid + 2);
        c2_it.addCol(tm2a, -m_dirAxe);
    }
    else if (r > ab)
    {
        thirdConstraint = r - ab;
        cIndex++;

        c1_it = c1.writeLine(cid + 2);
        c1_it.setCol(tm1, -m_dirAxe);

        c2_it = c2.writeLine(cid + 2);
        c2_it.addCol(tm2b, m_dirAxe);
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
void SlidingConstraint<DataTypes>::storeLambda(const ConstraintParams* /*cParams*/, sofa::core::MultiVecDerivId /*res*/, const sofa::defaulttype::BaseVector* lambda)
{
    Real lamb1,lamb2, lamb3;

    lamb1 = lambda->element(cid);
    lamb2 = lambda->element(cid+1);

    if(thirdConstraint)
    {

        lamb3 = lambda->element(cid+2);
        mforce.setValue( m_dirProj* lamb1 + m_dirOrtho * lamb2 + m_dirAxe * lamb3);
    }
    else
    {
        mforce.setValue( m_dirProj* lamb1 + m_dirOrtho * lamb2 );
    }



}

template<class DataTypes>
void SlidingConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowInteractionForceFields()) return;

    vparams->drawTool()->saveLastState();

    vparams->drawTool()->disableLighting();

    sofa::defaulttype::RGBAColor color;

    if(thirdConstraint<0)
        color = sofa::defaulttype::RGBAColor::yellow();
    else if(thirdConstraint>0)
        color = sofa::defaulttype::RGBAColor::green();
    else
        color = sofa::defaulttype::RGBAColor::magenta();

    std::vector<sofa::defaulttype::Vector3> vertices;
    vertices.push_back(DataTypes::getCPos((this->mstate1->read(core::ConstVecCoordId::position())->getValue())[m1.getValue()]));

    vparams->drawTool()->drawPoints(vertices, 10, color);
    vertices.clear();

    color = sofa::defaulttype::RGBAColor::blue();
    vertices.push_back(DataTypes::getCPos((this->mstate2->read(core::ConstVecCoordId::position())->getValue())[m2a.getValue()]));
    vertices.push_back(DataTypes::getCPos((this->mstate2->read(core::ConstVecCoordId::position())->getValue())[m2b.getValue()]));
    vparams->drawTool()->drawLines(vertices, 1, color);

    vparams->drawTool()->restoreLastState();
}

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
