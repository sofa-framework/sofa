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
#include <sofa/component/constraint/lagrangian/model/FixedLagrangianConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/accessor/WriteAccessor.h>

#include <sofa/type/Vec.h>

namespace sofa::component::constraint::lagrangian::model
{

template<class DataTypes>
FixedLagrangianConstraint<DataTypes>::FixedLagrangianConstraint(MechanicalState* object)
    : Inherit(object)
    , d_indices(initData(&d_indices,  "indices", "Indices of points to fix"))
    , d_fixAll(initData(&d_fixAll, false,  "fixAll", "If true, fix all points"))
{}

template<class DataTypes>
void FixedLagrangianConstraint<DataTypes>::init()
{
    Inherit1::init();
}


// Vec3D specialization
template<class DataTypes>
void FixedLagrangianConstraint<DataTypes>::buildConstraintMatrix(const core::ConstraintParams* /*cParams*/, DataMatrixDeriv &c_d, unsigned int &cIndex, const DataVecCoord &/*x*/)
{
    if(d_fixAll.getValue())
        m_cid.resize(this->getMState()->getSize());
    else
        m_cid.resize(d_indices.getValue().size());

    helper::WriteAccessor<DataMatrixDeriv> c = c_d;

    for(unsigned i=0; i<m_cid.size(); ++i)
    {
        m_cid[i] = cIndex;
        cIndex += Deriv::total_size;

        doBuildConstraintLine(c,i);
    }
    c_d.endEdit();
}

template<class DataTypes>
void FixedLagrangianConstraint<DataTypes>::getConstraintViolation(const core::ConstraintParams* /*cParams*/, linearalgebra::BaseVector *resV, const DataVecCoord &x, const DataVecDeriv &/*v*/)
{
    const DataVecCoord * freePos = this->getMState()->read(sofa::core::VecId::freePosition());
    const DataVecCoord * restPos = this->getMState()->read(sofa::core::VecId::restPosition());

    for(unsigned i=0; i<m_cid.size(); ++i)
    {
        doGetSingleConstraintViolation(resV,freePos, restPos, i);
    }
}

template<class DataTypes>
void FixedLagrangianConstraint<DataTypes>::getConstraintResolution(const core::ConstraintParams *, std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset)
{
    m_prevForces.resize(m_cid.size());
    for(unsigned i=0; i<m_cid.size(); ++i)
    {
        doGetSingleConstraintResolution(resTab,offset,i);
    }

}


} //namespace sofa::component::constraint::lagrangian::model
