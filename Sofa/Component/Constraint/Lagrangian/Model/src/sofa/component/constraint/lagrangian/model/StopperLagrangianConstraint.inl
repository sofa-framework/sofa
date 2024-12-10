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
    , d_index(initData(&d_index, 0, "index", "index of the stop constraint"))
    , d_min(initData(&d_min, -100.0_sreal, "min", "minimum value accepted"))
    , d_max(initData(&d_max, 100.0_sreal, "max", "maximum value accepted"))
{
        index.setOriginalData(&d_index);
        min.setOriginalData(&d_min);
        max.setOriginalData(&d_max);
}

template<class DataTypes>
void StopperLagrangianConstraint<DataTypes>::init()
{
    this->mstate = dynamic_cast<MechanicalState*>(this->getContext()->getMechanicalState());
    assert(this->mstate);

    helper::WriteAccessor<Data<VecCoord> > xData = *this->mstate->write(core::vec_id::write_access::position);
    VecCoord& x = xData.wref();
    if (x[d_index.getValue()].x() < d_min.getValue())
        x[d_index.getValue()].x() = (Real) d_min.getValue();
    if (x[d_index.getValue()].x() > d_max.getValue())
        x[d_index.getValue()].x() = (Real) d_max.getValue();
}

template<class DataTypes>
void StopperLagrangianConstraint<DataTypes>::buildConstraintMatrix(const core::ConstraintParams* /*cParams*/, DataMatrixDeriv &c_d, unsigned int &cIndex, const DataVecCoord &/*x*/)
{
    auto c = sofa::helper::getWriteAccessor(c_d);

    MatrixDerivRowIterator c_it = c->writeLine(cIndex++);
    c_it.setCol(d_index.getValue(), Coord(1));
}

template<class DataTypes>
void StopperLagrangianConstraint<DataTypes>::getConstraintViolation(const core::ConstraintParams* /*cParams*/, linearalgebra::BaseVector *resV, const DataVecCoord &x, const DataVecDeriv &/*v*/)
{
    const auto constraintIndex = this->d_constraintIndex.getValue();
    resV->set(constraintIndex, x.getValue()[d_index.getValue()][0]);
}

template<class DataTypes>
void StopperLagrangianConstraint<DataTypes>::getConstraintResolution(const core::ConstraintParams *, std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset)
{
    for(int i=0; i<1; i++)
        resTab[offset++] = new StopperLagrangianConstraintResolution1Dof(d_min.getValue(), d_max.getValue());
}

} //namespace sofa::component::constraint::lagrangian::model
