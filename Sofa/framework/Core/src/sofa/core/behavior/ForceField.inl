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

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/core/MechanicalParams.h>
#include <iostream>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>

namespace sofa::core::behavior
{


template<class DataTypes>
ForceField<DataTypes>::ForceField(MechanicalState<DataTypes> *mm)
    : BaseForceField(), SingleStateAccessor<DataTypes>(mm)
{
}

template<class DataTypes>
ForceField<DataTypes>::~ForceField() = default;

template<class DataTypes>
void ForceField<DataTypes>::addForce(const MechanicalParams* mparams, MultiVecDerivId fId )
{
    auto mstate = this->mstate.get();
    if (mparams && mstate)
    {
        addForce(mparams, *fId[mstate].write() , *mparams->readX(mstate), *mparams->readV(mstate));
    }
}


/// Instead of accumulating the matrix contributions in a matrix data structure, this class
/// directly multiplies the contribution by the dx vector
template<class DataTypes>
class MatrixFreeAccumulator final : public StiffnessMatrixAccumulator
{
public:
    using Deriv = typename DataTypes::Deriv;
    using VecDeriv = typename DataTypes::VecDeriv;
    using Real = typename DataTypes::Real;

private:
    const VecDeriv& m_dx;
    VecDeriv& m_df;
    const Real m_kFactor;

    template<sofa::Size L, sofa::Size C, class real>
    void blockAdd(sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<L, C, real>& value)
    {
        const auto DerivSize = Deriv::total_size;
        assert(row % DerivSize == 0);
        assert(col % DerivSize == 0);
        const auto dfId = row / DerivSize;
        const auto dxId = col / DerivSize;
        auto& df = m_df[dfId];
        const auto& dx = m_dx[dxId];

        for (sofa::Size i = 0; i < L; ++i)
        {
            for (sofa::Size j = 0; j < C; ++j)
            {
                df[i] += m_kFactor * static_cast<Real>(value(i, j)) * dx[j];
            }
        }
    }

    template<typename real>
    void scalarAdd(sofa::SignedIndex row, sofa::SignedIndex col, real value)
    {
        const auto DerivSize = Deriv::total_size;
        const auto dfId = row / DerivSize;
        const auto dfDim = row - dfId * DerivSize;
        const auto dxId = col / DerivSize;
        const auto dxDim = col - dxId * DerivSize;
        m_df[dfId][dfDim] += m_kFactor * static_cast<Real>(value) * m_dx[dxId][dxDim];
    }

public:
    MatrixFreeAccumulator(const VecDeriv& dx, VecDeriv& df, const Real kFactor) : m_dx(dx), m_df(df), m_kFactor(kFactor) {}

    void add(sofa::SignedIndex row, sofa::SignedIndex col, float value) override
    {
        scalarAdd(row, col, value);
    }
    void add(sofa::SignedIndex row, sofa::SignedIndex col, double value) override
    {
        scalarAdd(row, col, value);
    }

    void add(sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<1, 1, double>& value) override
    {
        blockAdd(row, col, value);
    }

    void add(sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<1, 1, float>& value) override
    {
        blockAdd(row, col, value);
    }

    void add(sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<2, 2, double>& value) override
    {
        blockAdd(row, col, value);
    }

    void add(sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<2, 2, float>& value) override
    {
        blockAdd(row, col, value);
    }

    void add(sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<3, 3, double>& value) override
    {
        blockAdd(row, col, value);
    }

    void add(sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<3, 3, float>& value) override
    {
        blockAdd(row, col, value);
    }
};

template<class DataTypes>
void ForceField<DataTypes>::addDForce(const MechanicalParams* mparams, MultiVecDerivId dfId )
{
    if (mparams && this->mstate)
    {
        const auto dx = sofa::helper::getReadAccessor(*mparams->readDx(this->mstate.get()));
        auto df = sofa::helper::getWriteOnlyAccessor(*dfId[this->mstate.get()].write());
        const Real kFactor = static_cast<Real>(
            sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue()));

        MatrixFreeAccumulator<DataTypes> acc(dx.ref(), df.wref(), kFactor);

        StiffnessMatrix matrix;
        matrix.setMechanicalParams(mparams);
        matrix.setMatrixAccumulator(&acc, this->mstate, this->mstate);
        buildStiffnessMatrix(&matrix);
    }
}

template<class DataTypes>
void ForceField<DataTypes>::addClambda(const MechanicalParams* mparams, MultiVecDerivId resId, MultiVecDerivId lambdaId, SReal cFactor )
{
    if (mparams && this->mstate)
    {
        addClambda(mparams, *resId[this->mstate.get()].write(), *lambdaId[this->mstate.get()].read(), cFactor);
    }
}

template<class DataTypes>
void ForceField<DataTypes>::addClambda(const MechanicalParams* /*mparams*/, DataVecDeriv& /*df*/, const DataVecDeriv& /*lambda*/, SReal /*cFactor*/ )
{
    msg_error()<<"function 'addClambda' is not implemented";
}



template<class DataTypes>
SReal ForceField<DataTypes>::getPotentialEnergy(const MechanicalParams* mparams) const
{
    if (this->mstate)
        return getPotentialEnergy(mparams, *mparams->readX(this->mstate.get()));
    return 0;
}

template<class DataTypes>
void ForceField<DataTypes>::addKToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix )
{
    if (this->mstate)
    {
        sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
        if (r)
            addKToMatrix(r.matrix, sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams,rayleighStiffness.getValue()), r.offset);
        else msg_error()<<"addKToMatrix found no valid matrix accessor.";
    }
}

template<class DataTypes>
void ForceField<DataTypes>::addKToMatrix(sofa::linearalgebra::BaseMatrix * /*mat*/, SReal /*kFact*/, unsigned int &/*offset*/)
{
    static int i=0;
    if (i < 10)
    {
        // This function is called for implicit time integration where stiffness matrix assembly is expected
        msg_warning() << "This force field does not support stiffness matrix assembly. "
                         "Therefore, the forces are integrated explicitly. "
                         "To support stiffness matrix assembly, addKToMatrix must be implemented.";
        i++;
    }
}

template<class DataTypes>
void ForceField<DataTypes>::addBToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    if (this->mstate)
    {
        sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
        if (r)
            addBToMatrix(r.matrix, sofa::core::mechanicalparams::bFactor(mparams) , r.offset);
    }
}
template<class DataTypes>
void ForceField<DataTypes>::addBToMatrix(sofa::linearalgebra::BaseMatrix * /*mat*/, SReal /*bFact*/, unsigned int &/*offset*/)
{

}

} // namespace sofa::core::behavior
