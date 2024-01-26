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
    , index(initData(&index, 0, "index", "index of the stop constraint"))
    , min(initData(&min, -100.0_sreal, "min", "minimum value accepted"))
    , max(initData(&max, 100.0_sreal, "max", "maximum value accepted"))
{
}

template<class DataTypes>
void StopperLagrangianConstraint<DataTypes>::init()
{
    this->mstate = dynamic_cast<MechanicalState*>(this->getContext()->getMechanicalState());
    assert(this->mstate);

    helper::WriteAccessor<Data<VecCoord> > xData = *this->mstate->write(core::VecCoordId::position());
    VecCoord& x = xData.wref();
    if (x[index.getValue()].x() < min.getValue())
        x[index.getValue()].x() = (Real) min.getValue();
    if (x[index.getValue()].x() > max.getValue())
        x[index.getValue()].x() = (Real) max.getValue();
}

template<class DataTypes>
void StopperLagrangianConstraint<DataTypes>::buildConstraintMatrix(const core::ConstraintParams* /*cParams*/, DataMatrixDeriv &c_d, unsigned int &cIndex, const DataVecCoord &/*x*/)
{
    cid = cIndex;

    MatrixDeriv& c = *c_d.beginEdit();

    MatrixDerivRowIterator c_it = c.writeLine(cid);
    c_it.setCol(index.getValue(), Coord(1));

    cIndex++;
    c_d.endEdit();
}

template<class DataTypes>
void StopperLagrangianConstraint<DataTypes>::getConstraintViolation(const core::ConstraintParams* /*cParams*/, linearalgebra::BaseVector *resV, const DataVecCoord &x, const DataVecDeriv &/*v*/)
{
    resV->set(cid, x.getValue()[index.getValue()][0]);
}

template<class DataTypes>
void StopperLagrangianConstraint<DataTypes>::getConstraintResolution(const core::ConstraintParams *, std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset)
{
    for(int i=0; i<1; i++)
        resTab[offset++] = new StopperLagrangianConstraintResolution1Dof(min.getValue(), max.getValue());
}

} //namespace sofa::component::constraint::lagrangian::model
