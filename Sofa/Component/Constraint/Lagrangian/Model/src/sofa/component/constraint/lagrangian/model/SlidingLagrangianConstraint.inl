/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/component/constraint/lagrangian/model/SlidingLagrangianConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/constraint/lagrangian/model/BilateralLagrangianConstraint.h>
#include <sofa/component/constraint/lagrangian/model/UnilateralLagrangianConstraint.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/type/Vec.h>
namespace sofa::component::constraint::lagrangian::model
{

template<class DataTypes>
SlidingLagrangianConstraint<DataTypes>::SlidingLagrangianConstraint()
    : SlidingLagrangianConstraint(nullptr, nullptr)
{
}

template<class DataTypes>
SlidingLagrangianConstraint<DataTypes>::SlidingLagrangianConstraint(MechanicalState* object)
    : SlidingLagrangianConstraint(object, object)
{
}

template<class DataTypes>
SlidingLagrangianConstraint<DataTypes>::SlidingLagrangianConstraint(MechanicalState* object1, MechanicalState* object2)
    : Inherit(object1, object2)
    , d_m1(initData(&d_m1, 0, "sliding_point","index of the spliding point on the first model"))
    , d_m2a(initData(&d_m2a, 0, "axis_1","index of one end of the sliding axis"))
    , d_m2b(initData(&d_m2b, 0, "axis_2","index of the other end of the sliding axis"))
    , d_force(initData(&d_force,"force","force (impulse) used to solve the constraint"))
{
}

template<class DataTypes>
void SlidingLagrangianConstraint<DataTypes>::init()
{
    assert(this->mstate1);
    assert(this->mstate2);

    m_thirdConstraint = 0;
}


template<class DataTypes>
void SlidingLagrangianConstraint<DataTypes>::buildConstraintMatrix(const core::ConstraintParams*, DataMatrixDeriv &c1_d, DataMatrixDeriv &c2_d, unsigned int &cIndex
        , const DataVecCoord &x1, const DataVecCoord &x2)
{
    int tm1 = d_m1.getValue();
    int tm2a = d_m2a.getValue();
    int tm2b = d_m2b.getValue();

    MatrixDeriv &c1 = *c1_d.beginEdit();
    MatrixDeriv &c2 = *c2_d.beginEdit();

    const Coord P = x1.getValue()[tm1];
    const Coord A = x2.getValue()[tm2a];
    const Coord B = x2.getValue()[tm2b];

    // the axis
    m_dirAxe = B - A;
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

    m_dirOrtho = cross(m_dirProj, m_dirAxe);
    m_dirOrtho.normalize();

    m_cid = cIndex;
    cIndex += 2;

    MatrixDerivRowIterator c1_it = c1.writeLine(m_cid);
    c1_it.addCol(tm1, m_dirProj);

    MatrixDerivRowIterator c2_it = c2.writeLine(m_cid);
    c2_it.addCol(tm2a, -m_dirProj * (1-r2));
    c2_it.addCol(tm2b, -m_dirProj * r2);

    c1_it = c1.writeLine(m_cid + 1);
    c1_it.setCol(tm1, m_dirOrtho);

    c2_it = c2.writeLine(m_cid + 1);
    c2_it.addCol(tm2a, -m_dirOrtho * (1-r2));
    c2_it.addCol(tm2b, -m_dirOrtho * r2);

    m_thirdConstraint = 0;

    if (r < 0)
    {
        m_thirdConstraint = r;
        cIndex++;

        c1_it = c1.writeLine(m_cid + 2);
        c1_it.setCol(tm1, m_dirAxe);

        c2_it = c2.writeLine(m_cid + 2);
        c2_it.addCol(tm2a, -m_dirAxe);
    }
    else if (r > ab)
    {
        m_thirdConstraint = r - ab;
        cIndex++;

        c1_it = c1.writeLine(m_cid + 2);
        c1_it.setCol(tm1, -m_dirAxe);

        c2_it = c2.writeLine(m_cid + 2);
        c2_it.addCol(tm2b, m_dirAxe);
    }

    c1_d.endEdit();
    c2_d.endEdit();
}


template<class DataTypes>
void SlidingLagrangianConstraint<DataTypes>::getConstraintViolation(const core::ConstraintParams *, linearalgebra::BaseVector *v, const DataVecCoord &, const DataVecCoord &
        , const DataVecDeriv &, const DataVecDeriv &)
{
    v->set(m_cid, m_dist);
    v->set(m_cid+1, 0.0);

    if(m_thirdConstraint)
    {
        if(m_thirdConstraint>0)
            v->set(m_cid+2, -m_thirdConstraint);
        else
            v->set(m_cid+2, m_thirdConstraint);
    }
}


template<class DataTypes>
void SlidingLagrangianConstraint<DataTypes>::getConstraintResolution(const ConstraintParams*,
                                                           std::vector<core::behavior::ConstraintResolution*>& resTab,
                                                           unsigned int& offset)
{
    resTab[offset++] = new BilateralConstraintResolution();
    resTab[offset++] = new BilateralConstraintResolution();

    if(m_thirdConstraint)
        resTab[offset++] = new UnilateralConstraintResolution();
}


template<class DataTypes>
void SlidingLagrangianConstraint<DataTypes>::storeLambda(const ConstraintParams* /*cParams*/, sofa::core::MultiVecDerivId /*res*/, const sofa::linearalgebra::BaseVector* lambda)
{
    Real lamb1,lamb2, lamb3;

    lamb1 = lambda->element(m_cid);
    lamb2 = lambda->element(m_cid+1);

    if(m_thirdConstraint)
    {
        lamb3 = lambda->element(m_cid+2);
        d_force.setValue( m_dirProj* lamb1 + m_dirOrtho * lamb2 + m_dirAxe * lamb3);
    }
    else
    {
        d_force.setValue( m_dirProj* lamb1 + m_dirOrtho * lamb2 );
    }
}

template<class DataTypes>
void SlidingLagrangianConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowInteractionForceFields())
        return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    vparams->drawTool()->disableLighting();

    sofa::type::RGBAColor color;

    if(m_thirdConstraint<0)
        color = sofa::type::RGBAColor::yellow();
    else if(m_thirdConstraint>0)
        color = sofa::type::RGBAColor::green();
    else
        color = sofa::type::RGBAColor::magenta();

    std::vector<sofa::type::Vec3> vertices;
    vertices.push_back(DataTypes::getCPos((this->mstate1->read(core::ConstVecCoordId::position())->getValue())[d_m1.getValue()]));

    vparams->drawTool()->drawPoints(vertices, 10, color);
    vertices.clear();

    color = sofa::type::RGBAColor::blue();
    vertices.push_back(DataTypes::getCPos((this->mstate2->read(core::ConstVecCoordId::position())->getValue())[d_m2a.getValue()]));
    vertices.push_back(DataTypes::getCPos((this->mstate2->read(core::ConstVecCoordId::position())->getValue())[d_m2b.getValue()]));
    vparams->drawTool()->drawLines(vertices, 1, color);


}

} //namespace sofa::component::constraint::lagrangian::model
