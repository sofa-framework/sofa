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
#include <sofa/component/constraint/lagrangian/model/UniformLagrangianConstraint.h>

#include <sofa/core/behavior/Constraint.inl>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/component/constraint/lagrangian/model/BilateralConstraintResolution.h>

namespace sofa::component::constraint::lagrangian::model
{

template< class DataTypes >
UniformLagrangianConstraint<DataTypes>::UniformLagrangianConstraint()
    :d_iterative(initData(&d_iterative, true, "iterative", "Iterate over the bilateral constraints, otherwise a block factorisation is computed."))
    ,d_constraintRestPos(initData(&d_constraintRestPos, false, "constrainToRestPos", "if false, constrains the pos to be zero / if true constraint the current position to stay at rest position"))
    ,m_constraintIndex(0)
{

}

template< class DataTypes >
void UniformLagrangianConstraint<DataTypes>::buildConstraintMatrix(const sofa::core::ConstraintParams* cParams, DataMatrixDeriv & c, unsigned int &cIndex, const DataVecCoord &x)
{
    SOFA_UNUSED(cParams);

    const auto N = Deriv::size(); // MatrixDeriv is a container of Deriv types.

    auto& jacobian = sofa::helper::getWriteAccessor(c).wref();
    auto  xVec     = sofa::helper::getReadAccessor(x);

    m_constraintIndex = cIndex; // we should not have to remember this, it should be available through the API directly.

    for (std::size_t i = 0; i < xVec.size(); ++i)
    {
        for (std::size_t j = 0; j < N; ++j)
        {
            auto row = jacobian.writeLine(N*i + j + m_constraintIndex);
            Deriv d;
            d[j] = Real(1);
            row.setCol(i, d);
            ++cIndex;
        }
    }
}

template<class DstV, class Free>
void computeViolation(DstV& resV, unsigned int constraintIndex, const
                      Free& free, size_t N, std::function<double(int i, int j)> f)
{
    for (std::size_t i = 0; i < free.size(); ++i)
    {
        for (std::size_t j = 0; j < N; ++j)
        {
            resV->set(constraintIndex + i*N + j, f(i, j) );
        }
    }
}

template< class DataTypes >
void UniformLagrangianConstraint<DataTypes>::getConstraintViolation(const sofa::core::ConstraintParams* cParams, sofa::linearalgebra::BaseVector *resV, const DataVecCoord &x, const DataVecDeriv &v)
{
    auto xfree = sofa::helper::getReadAccessor(x);
    auto vfree = sofa::helper::getReadAccessor(v);
    const SReal dt = this->getContext()->getDt();
    const SReal invDt = 1.0 / dt;

    auto pos     = this->getMState()->readPositions();
    auto restPos = this->getMState()->readRestPositions();

    if (cParams->constOrder() == sofa::core::ConstraintOrder::VEL)
    {
        if (d_constraintRestPos.getValue()){
            computeViolation(resV, m_constraintIndex, vfree, Deriv::size(),[&invDt,&pos,&vfree,&restPos](int i, int j)
            { return vfree[i][j] + invDt *(pos[i][j]-restPos[i][j]); });
        }
        else {
            computeViolation(resV, m_constraintIndex, vfree, Deriv::size(),[&invDt,&pos,&vfree](int i, int j)
            { return vfree[i][j] + invDt *pos[i][j]; });
        }
    }
    else
    {
        if( d_constraintRestPos.getValue() )
            computeViolation(resV, m_constraintIndex, xfree, Coord::size(),
                             [&xfree,&restPos](int i, int j){ return xfree[i][j] - restPos[i][j]; });
        else
            computeViolation(resV, m_constraintIndex, xfree, Coord::size(),[&xfree](int i, int j){ return xfree[i][j]; });
    }
}

template< class DataTypes >
void UniformLagrangianConstraint<DataTypes>::getConstraintResolution(const sofa::core::ConstraintParams* cParams, std::vector<sofa::core::behavior::ConstraintResolution*>& crVector, unsigned int& offset)
{
    SOFA_UNUSED(cParams);

    if (d_iterative.getValue())
    {
        for (std::size_t i = 0; i < this->getMState()->getSize(); ++i)
        {
            for (std::size_t j = 0; j < Deriv::size(); ++j)
            {
                auto* cr = new sofa::component::constraint::lagrangian::model::BilateralConstraintResolution();
                crVector[offset++] = cr;
            }
        }
    }
    else
    {
        const std::size_t nbLines = this->getMState()->getSize() * Deriv::size();
        auto* cr = new sofa::component::constraint::lagrangian::model::BilateralConstraintResolutionNDof(nbLines);
        crVector[offset] = cr;
        offset += nbLines;
    }
}

} // namespace sofa::component::constraint::lagrangian::model
