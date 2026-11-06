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
#include <sofa/component/constraint/lagrangian/model/StopperLagrangianConstraint.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/type/Vec.h>

namespace sofa::component::constraint::lagrangian::model
{

template<class DataTypes>
StopperLagrangianConstraint<DataTypes>::StopperLagrangianConstraint(MechanicalState* object)
    : Inherit(object)
    , d_indices(initData(&d_indices, "indices", "indices of the stop constraint"))
    , d_min(initData(&d_min, -100.0_sreal, "min", "minimum value accepted"))
    , d_max(initData(&d_max, 100.0_sreal, "max", "maximum value accepted"))
{
}

template<class DataTypes>
void StopperLagrangianConstraint<DataTypes>::init()
{
    Inherit::init();
    if(!getMState())
    {
        msg_error() << "There is no mstate to constraint";
        d_componentState = sofa::core::objectmodel::ComponentState::Invalid;
        return;
    }

    auto x = getMState()->readPositions();
    if(!d_indices.isSet())
    {
        auto indices = sofa::helper::getWriteOnlyAccessor(d_indices);
        indices.resize(x.size());
        for(sofa::Index i=0;i<x.size();i++)
        {
            indices[i] = i;
        }
    }

    d_componentState = sofa::core::objectmodel::ComponentState::Valid;
}

template<class DataTypes>
void StopperLagrangianConstraint<DataTypes>::buildConstraintMatrix(const core::ConstraintParams* /*cParams*/, DataMatrixDeriv &c_d, unsigned int &cIndex, const DataVecCoord &/*x*/)
{
    if(isComponentStateInvalid())
        return;

    auto c = sofa::helper::getWriteOnlyAccessor(c_d);
    auto indices = sofa::helper::getReadAccessor(d_indices);

    m_cid.resize(indices.size());
    for(unsigned i=0; i<m_cid.size(); ++i)
    {
        m_cid[i] = cIndex;
        MatrixDerivRowIterator c_it = c->writeLine(m_cid[i]);
        c_it.setCol(indices[i], Coord(1));
        cIndex++;
    }
}

template<class DataTypes>
void StopperLagrangianConstraint<DataTypes>::getConstraintViolation(const core::ConstraintParams* /*cParams*/, linearalgebra::BaseVector *resV, const DataVecCoord& x_, const DataVecDeriv &/*v*/)
{
    if(isComponentStateInvalid())
        return;

    auto x = x_.getValue();
    auto indices = sofa::helper::getReadAccessor(d_indices);

    for(unsigned int i =0 ; i<indices.size();i++)
    {
        resV->set(m_cid[i], x[indices[i]][0]);
    }
}

template<class DataTypes>
void StopperLagrangianConstraint<DataTypes>::getConstraintResolution(const core::ConstraintParams *, std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset)
{
    if(isComponentStateInvalid())
        return;

    auto min = d_min.getValue();
    auto max = d_max.getValue();
    auto indices = sofa::helper::getReadAccessor(d_indices);
    for(auto _ : indices)
        resTab[offset++] = new StopperLagrangianConstraintResolution1Dof(min, max);
}

} //namespace sofa::component::constraint::lagrangian::model
