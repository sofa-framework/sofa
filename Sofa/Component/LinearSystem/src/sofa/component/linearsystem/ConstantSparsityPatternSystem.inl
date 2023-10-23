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
#include <sofa/component/linearsystem/ConstantSparsityPatternSystem.h>
#include <sofa/component/linearsystem/MatrixLinearSystem.inl>
#include <sofa/component/linearsystem/matrixaccumulators/SparsityPatternLocalMappedMatrix.h>
#include <sofa/component/linearsystem/matrixaccumulators/ConstantLocalMappedMatrix.h>

namespace sofa::component::linearsystem
{

template<class TMatrix, class TVector>
ConstantSparsityPatternSystem<TMatrix, TVector>::ConstantSparsityPatternSystem()
    : Inherit1()
{
}

template<class TMatrix, class TVector>
template <core::matrixaccumulator::Contribution c>
void ConstantSparsityPatternSystem<TMatrix, TVector>::replaceLocalMatrixMapped(const core::MechanicalParams* mparams, LocalMatrixMaps<c, Real>& matrixMaps)
{
    for (auto& [component, localMatrixMap] : matrixMaps.mappedLocalMatrix)
    {
        for (auto& [states, localMatrix] : localMatrixMap)
        {
            if (auto* sparsityPatternMatrix = dynamic_cast<SparsityPatternLocalMappedMatrix<c, SReal>*>(localMatrix))
            {
                const auto& insertionOrderList = sparsityPatternMatrix->getInsertionOrderList();

                const auto factor = Inherit1::template getContributionFactor<c>(mparams, component);

                auto mat = sofa::core::objectmodel::New<ConstantLocalMappedMatrix<c, Real>>();
                configureCreatedMatrixComponent<c>(mat, component, factor, !this->notMuted());

                msg_info() << "Replacing " << sparsityPatternMatrix->getPathName() << " (class "
                << sparsityPatternMatrix->getClass()->className << ") by " << mat->getPathName() << " (class "
                << mat->getClass()->className << ")";

                mat->setMatrixSize(localMatrix->getMatrixSize());
                const auto& sharedMatrix = localMatrix->getMatrix();
                mat->shareMatrix(sharedMatrix);

                const auto it = std::find_if(this->m_localMappedMatrices.begin(), this->m_localMappedMatrices.end(),
                                             [&sharedMatrix](const auto& el){ return el.second == sharedMatrix; });
                if (it != this->m_localMappedMatrices.end())
                {
                    const auto id = std::distance(this->m_localMappedMatrices.begin(), it);
                    const auto& mapping = m_constantCRSMappingMappedMatrices[id];
                    mat->insertionOrderList.reserve(insertionOrderList.size());
                    for (const auto& [row, col] : insertionOrderList)
                    {
                        mat->insertionOrderList.push_back(mapping.at(row + col * sharedMatrix->rows()));
                    }
                }
                else
                {
                    dmsg_error() << "Cannot find a matrix for this component";
                }

                component->removeSlave(localMatrix);
                localMatrix = mat.get();
                localMatrixMap[states] = mat.get();
                matrixMaps.componentLocalMatrix[component][states] = mat.get();

                const auto& [mstate0, mstate1] = states;
                if constexpr (c == Contribution::STIFFNESS)
                {
                    this->m_stiffness[component].setMatrixAccumulator(mat.get(), mstate0, mstate1);
                }
                else if constexpr (c == Contribution::DAMPING)
                {
                    this->m_damping[component].setMatrixAccumulator(mat.get(), mstate0, mstate1);
                }
                else if constexpr (c == Contribution::GEOMETRIC_STIFFNESS)
                {
                    this->m_geometricStiffness[component].setMatrixAccumulator(mat.get(), mstate0, mstate1);
                }
                else if constexpr (c == Contribution::MASS)
                {
                    this->m_mass[component] = mat.get();
                }
            }
            else
            {
                dmsg_error() << "not a sparsity pattern matrix";
            }
        }
    }
}

template<class TMatrix, class TVector>
template <core::matrixaccumulator::Contribution c>
void ConstantSparsityPatternSystem<TMatrix, TVector>::replaceLocalMatricesNonMapped(const core::MechanicalParams* mparams, LocalMatrixMaps<c, Real>& matrixMaps)
{
    for (auto& [component, localMatrixMap] : matrixMaps.componentLocalMatrix)
    {
        for (auto& [states, localMatrix] : localMatrixMap)
        {
            if (auto* sparsityPatternMatrix = dynamic_cast<SparsityPatternLocalMatrix<c>*>(localMatrix))
            {
                const auto& insertionOrderList = sparsityPatternMatrix->getInsertionOrderList();

                SReal factor = Inherit1::template getContributionFactor<c>(mparams, component);

                auto mat = sofa::core::objectmodel::New<ConstantLocalMatrix<TMatrix, c>>();
                configureCreatedMatrixComponent<c>(mat, component, factor, !this->notMuted());

                msg_info() << "Replacing " << sparsityPatternMatrix->getPathName() << " (class "
                << sparsityPatternMatrix->getClass()->className << ") by " << mat->getPathName() << " (class "
                << mat->getClass()->className << ")";

                mat->setMatrixSize(localMatrix->getMatrixSize());
                mat->setGlobalMatrix(this->getSystemMatrix());
                mat->setPositionInGlobalMatrix(localMatrix->getPositionInGlobalMatrix());

                mat->insertionOrderList.reserve(insertionOrderList.size());
                for (const auto& [row, col] : insertionOrderList)
                {
                    const auto flatIndex = row + col * this->getSystemMatrix()->rows();
                    const auto it = m_constantCRSMapping->find(flatIndex);
                    if (it != m_constantCRSMapping->end())
                    {
                        mat->insertionOrderList.push_back(it->second);
                    }
                    else
                    {
                        msg_error() << "Could not find index " << flatIndex << " (row " << row <<
                                ", col " << col << ") in the hash table";
                    }
                }

                component->removeSlave(localMatrix);

                const auto& [mstate0, mstate1] = states;
                if constexpr (c == Contribution::STIFFNESS)
                {
                    this->m_stiffness[component].setMatrixAccumulator(mat.get(), mstate0, mstate1);
                }
                else if constexpr (c == Contribution::DAMPING)
                {
                    this->m_damping[component].setMatrixAccumulator(mat.get(), mstate0, mstate1);
                }
                else if constexpr (c == Contribution::GEOMETRIC_STIFFNESS)
                {
                    this->m_geometricStiffness[component].setMatrixAccumulator(mat.get(), mstate0, mstate1);
                }
                else if constexpr (c == Contribution::MASS)
                {
                    this->m_mass[component] = mat.get();
                }

                localMatrix = mat.get();
            }
        }
    }
}

template<class TMatrix, class TVector>
template<core::matrixaccumulator::Contribution c>
void ConstantSparsityPatternSystem<TMatrix, TVector>::replaceLocalMatrices(const core::MechanicalParams* mparams,
    LocalMatrixMaps<c, Real>& matrixMaps)
{
    replaceLocalMatrixMapped<c>(mparams, matrixMaps);
    replaceLocalMatricesNonMapped<c>(mparams, matrixMaps);
}

template<class TMatrix, class TVector>
template <core::matrixaccumulator::Contribution c>
void ConstantSparsityPatternSystem<TMatrix, TVector>::reinitLocalMatrices(LocalMatrixMaps<c, Real>& matrixMaps)
{
    for (auto& [component, localMatrixMap] : matrixMaps.componentLocalMatrix)
    {
        for (auto& [states, localMatrix] : localMatrixMap)
        {
            if (auto* local = dynamic_cast<ConstantLocalMatrix<TMatrix, c>* >(localMatrix))
            {
                local->currentId = 0;
            }
        }
    }
}

template<class TMatrix, class TVector>
void ConstantSparsityPatternSystem<TMatrix, TVector>::buildHashTable(linearalgebra::CompressedRowSparseMatrix<SReal>& M, ConstantCRSMapping& mapping)
{
    for (unsigned int it_rows_k = 0; it_rows_k < M.rowIndex.size() ; it_rows_k ++)
    {
        const auto row = M.rowIndex[it_rows_k];
        typename Matrix::Range rowRange( M.rowBegin[it_rows_k], M.rowBegin[it_rows_k+1] );
        for(auto xj = rowRange.begin() ; xj < rowRange.end() ; ++xj )  // for each non-null block
        {
            const auto col = M.colsIndex[xj];
            mapping.emplace(row + col * M.rows(), xj);
        }
    }
}

template<class TMatrix, class TVector>
void ConstantSparsityPatternSystem<TMatrix, TVector>::applyProjectiveConstraints(const core::MechanicalParams* mparams)
{
    if (!isConstantSparsityPatternUsedYet())
    {
        auto& M = *this->getSystemMatrix();
        M.compress();

        m_constantCRSMapping = std::make_unique<ConstantCRSMapping>();

        // build the hash table from the compressed matrix
        buildHashTable(M, *m_constantCRSMapping);

        m_constantCRSMappingMappedMatrices.resize(this->m_localMappedMatrices.size());
        std::size_t i {};
        for (const auto& mat : this->m_localMappedMatrices)
        {
            mat.second->fullRows();
            buildHashTable(*mat.second, m_constantCRSMappingMappedMatrices[i++]);
        }

        //replace the local matrix components by new ones that use the hash table
        replaceLocalMatrices(mparams, this->template getLocalMatrixMap<Contribution::STIFFNESS>());
        replaceLocalMatrices(mparams, this->template getLocalMatrixMap<Contribution::MASS>());
        replaceLocalMatrices(mparams, this->template getLocalMatrixMap<Contribution::DAMPING>());
        replaceLocalMatrices(mparams, this->template getLocalMatrixMap<Contribution::GEOMETRIC_STIFFNESS>());

        m_isConstantSparsityPatternUsedYet = true;
    }
    else
    {
        dmsg_error_when(!this->getSystemMatrix()->btemp.empty()) << "Matrix is not compressed";
        for (const auto& mat : this->m_localMappedMatrices)
        {
            dmsg_error_when(!mat.second->btemp.empty()) << "Matrix is not compressed";
        }
    }

    //the hash table and the ordered lists must be created BEFORE the application of the projective constraints
    Inherit1::applyProjectiveConstraints(mparams);
}

template<class TMatrix, class TVector>
void ConstantSparsityPatternSystem<TMatrix, TVector>::resizeSystem(sofa::Size n)
{
    this->allocateSystem();

    if (this->getSystemMatrix())
    {
        Index nIndex = sofa::helper::narrow_cast<Index>(n);
        if (nIndex != this->getSystemMatrix()->rowSize() || nIndex != this->getSystemMatrix()->colSize())
        {
            this->getSystemMatrix()->resize(n, n);
        }
        else
        {
            // In the CRS format, the pattern is unchanged from a time step to the next
            // Only the values are reset to 0

            auto& values = this->getSystemMatrix()->colsValue;
            std::fill(values.begin(), values.end(), 0_sreal);

            for (auto& m : this->m_localMappedMatrices)
            {
                std::fill(m.second->colsValue.begin(), m.second->colsValue.end(), 0_sreal);
            }
        }
    }

    this->resizeVectors(n);
}

template<class TMatrix, class TVector>
void ConstantSparsityPatternSystem<TMatrix, TVector>::clearSystem()
{
    this->allocateSystem();

    if (this->getSystemMatrix())
    {
        if (!isConstantSparsityPatternUsedYet())
        {
            this->getSystemMatrix()->clear();
        }
        else
        {
            auto& values = this->getSystemMatrix()->colsValue;
            std::fill(values.begin(), values.end(), 0_sreal);

            unsigned int i = 0;
            for (auto& m : this->m_localMappedMatrices)
            {
                std::fill(m.second->colsValue.begin(), m.second->colsValue.end(), 0_sreal);
                ++i;
            }


        }
    }

    if (this->getRHSVector())
    {
        this->getRHSVector()->clear();
    }

    if (this->getSolutionVector())
    {
        this->getSolutionVector()->clear();
    }
}

template<class TMatrix, class TVector>
bool ConstantSparsityPatternSystem<TMatrix, TVector>::isConstantSparsityPatternUsedYet() const
{
    return m_isConstantSparsityPatternUsedYet;
}

template<class TMatrix, class TVector>
void ConstantSparsityPatternSystem<TMatrix, TVector>::preAssembleSystem(const core::MechanicalParams* mechanical_params)
{
    Inherit1::preAssembleSystem(mechanical_params);

    if (isConstantSparsityPatternUsedYet())
    {
        reinitLocalMatrices(this->template getLocalMatrixMap<Contribution::STIFFNESS>());
        reinitLocalMatrices(this->template getLocalMatrixMap<Contribution::MASS>());
        reinitLocalMatrices(this->template getLocalMatrixMap<Contribution::DAMPING>());
        reinitLocalMatrices(this->template getLocalMatrixMap<Contribution::GEOMETRIC_STIFFNESS>());
    }
}

template<class TMatrix, class TVector>
void ConstantSparsityPatternSystem<TMatrix, TVector>::makeCreateDispatcher()
{
    std::get<std::unique_ptr<CreateMatrixDispatcher<Contribution::STIFFNESS          >>>(this->m_createDispatcher) = makeCreateDispatcher<Contribution::STIFFNESS          >();
    std::get<std::unique_ptr<CreateMatrixDispatcher<Contribution::MASS               >>>(this->m_createDispatcher) = makeCreateDispatcher<Contribution::MASS               >();
    std::get<std::unique_ptr<CreateMatrixDispatcher<Contribution::DAMPING            >>>(this->m_createDispatcher) = makeCreateDispatcher<Contribution::DAMPING            >();
    std::get<std::unique_ptr<CreateMatrixDispatcher<Contribution::GEOMETRIC_STIFFNESS>>>(this->m_createDispatcher) = makeCreateDispatcher<Contribution::GEOMETRIC_STIFFNESS>();
}

template<class TMatrix, class TVector>
template <Contribution c>
std::unique_ptr<CreateMatrixDispatcher<c>> ConstantSparsityPatternSystem<TMatrix, TVector>::makeCreateDispatcher()
{
    struct SparsityPatternMatrixDispatcher : CreateMatrixDispatcher<c>
    {
        typename BaseAssemblingMatrixAccumulator<c>::SPtr
        createLocalMappedMatrix() override
        {
            return sofa::core::objectmodel::New<SparsityPatternLocalMappedMatrix<c, Real>>();
        }

    protected:

        typename BaseAssemblingMatrixAccumulator<c>::SPtr
        createLocalMatrix() const override
        {
            return sofa::core::objectmodel::New<SparsityPatternLocalMatrix<c>>();
        }

        typename BaseAssemblingMatrixAccumulator<c>::SPtr
        createLocalMatrixWithIndexChecking() const override
        {
            return sofa::core::objectmodel::New<SparsityPatternLocalMatrix<c, core::matrixaccumulator::RangeVerification>>();
        }
    };

    return std::make_unique<SparsityPatternMatrixDispatcher>();
}

}
