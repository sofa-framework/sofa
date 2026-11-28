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

#include <sofa/component/linearsolver/direct/SparseLDLSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <cmath>
#include <fstream>
#include <iomanip>      // std::setprecision
#include <string>
#include <sofa/simulation/task/MainTaskSchedulerFactory.h>
#include <sofa/simulation/task/ParallelForEach.h>


namespace sofa::component::linearsolver::direct 
{

template<class TMatrix, class TVector, class TThreadManager>
SparseLDLSolver<TMatrix,TVector,TThreadManager>::SparseLDLSolver()
    : numStep(0)
{}

template <class TMatrix, class TVector, class TThreadManager>
void SparseLDLSolver<TMatrix, TVector, TThreadManager>::init()
{
    Inherit::init();

    if (this->d_componentState.getValue() != core::objectmodel::ComponentState::Invalid)
    {
        this->d_componentState.setValue(core::objectmodel::ComponentState::Valid);
    }
}

template <class TMatrix, class TVector, class TThreadManager>
void SparseLDLSolver<TMatrix, TVector, TThreadManager>::parse(sofa::core::objectmodel::BaseObjectDescription* arg)
{
    Inherit1::parse(arg);

    if (!arg->getAttribute("template"))
    {
        std::string header = this->getClassName();
        if (const std::string& name = this->getName(); !name.empty())
        {
            header.append("(" + name + ")");
        }

        static const char* blocksType =
        sofa::linearalgebra::CompressedRowSparseMatrix<sofa::type::Mat<3, 3, SReal> >::Name();

        msg_advice(header) << "Template is empty\n"
                           << "By default " << this->getClassName() << " uses blocks with a single scalar (to handle all cases of simulations).\n"
                           << "If you are using only 3D DOFs, you may consider using blocks of Matrix3 to speedup the calculations.\n"
                           << "If it is the case, add template=\"" << blocksType << "\" to this object in your scene\n"
                           << "Otherwise, if you want to disable this message, add " << "template=\"" << this->getTemplateName() << "\" " << ".";
    }

    if (arg->getAttribute("savingMatrixToFile"))
    {
        msg_warning() << "It is no longer possible to export the linear system matrix from within " << this->getClassName() <<  ". Instead, use the component GlobalSystemMatrixExporter (from the SofaMatrix plugin).";
    }
}

template<class TMatrix, class TVector, class TThreadManager>
void SparseLDLSolver<TMatrix,TVector,TThreadManager>::solve (Matrix& M, Vector& z, Vector& r)
{
    SCOPED_TIMER_VARNAME(solveTimer, "solve");
    Inherit::solve_cpu(z.ptr(), r.ptr(), (InvertData *) this->getMatrixInvertData(&M));
}

template <class TMatrix, class TVector, class TThreadManager>
bool SparseLDLSolver<TMatrix, TVector, TThreadManager>::filterMatrix(Matrix& M, int nBlocks, int systemSize)
{
    using traits = typename BlockOperations::traits;
    constexpr int N = Inherit::BlockSize;

    M.compress();

    const auto& rowIndex = M.getRowIndex();
    const auto& rowBegin = M.getRowBegin();
    const auto& colsIndex = M.getColsIndex();
    const auto& colsValue = M.getColsValue();

    m_filteredRowBegin.resize(nBlocks + 1);
    m_filteredColsIndex.clear();
    m_filteredValues.clear();
    m_filteredColsIndex.reserve(colsIndex.size());
    m_filteredValues.reserve(colsIndex.size());

    // copy the non-zero blocks, keeping the empty rows
    int row = 0;
    m_filteredRowBegin[0] = 0;
    for (std::size_t xi = 0; xi < rowIndex.size(); ++xi)
    {
        const int r = static_cast<int>(rowIndex[xi]);
        while (row < r)
        {
            m_filteredRowBegin[++row] = static_cast<int>(m_filteredColsIndex.size());
        }
        for (auto j = rowBegin[xi]; j < rowBegin[xi + 1]; ++j)
        {
            if (!traits::empty(colsValue[j]))
            {
                m_filteredColsIndex.push_back(static_cast<int>(colsIndex[j]));
                m_filteredValues.push_back(colsValue[j]);
            }
        }
        m_filteredRowBegin[++row] = static_cast<int>(m_filteredColsIndex.size());
    }
    while (row < nBlocks)
    {
        m_filteredRowBegin[++row] = static_cast<int>(m_filteredColsIndex.size());
    }

    // If the system size is not a multiple of the block size, the last block is incomplete: the
    // unused rows and columns are zero. Identity is set on their diagonal so that the padded
    // system remains invertible. The unused entries of the right-hand side are zero, so the
    // padded entries of the solution are zero and do not interfere with the actual system.
    if (const int nbUsed = systemSize - (nBlocks - 1) * N; nbUsed < N)
    {
        const int lastRow = nBlocks - 1;
        // the last row is stored at the end of the arrays, and its columns are sorted
        const bool hasDiagonal =
            m_filteredRowBegin[lastRow + 1] > m_filteredRowBegin[lastRow] &&
            m_filteredColsIndex.back() == lastRow;
        if (!hasDiagonal)
        {
            m_filteredColsIndex.push_back(lastRow);
            m_filteredValues.emplace_back();
            traits::clear(m_filteredValues.back());
            m_filteredRowBegin[nBlocks] = static_cast<int>(m_filteredColsIndex.size());
        }
        for (int c = nbUsed; c < N; ++c)
        {
            traits::vset(m_filteredValues.back(), c, c, 1);
        }
    }

    for (int r = 0; r < nBlocks; ++r)
    {
        if (m_filteredRowBegin[r + 1] == m_filteredRowBegin[r])
        {
            return false;
        }
    }

    return true;
}

template <class TMatrix, class TVector, class TThreadManager>
bool SparseLDLSolver<TMatrix, TVector, TThreadManager>::factorize(
    Matrix& M, InvertData * invertData)
{
    constexpr int N = Inherit::BlockSize;

    const int n = M.colSize();

    if (n == 0)
    {
        showInvalidSystemMessage("null size");
        return true;
    }

    const int nBlocks = (n + N - 1) / N;

    if (!filterMatrix(M, nBlocks, n))
    {
        showInvalidSystemMessage("empty row");
        return true;
    }

    Inherit::factorize(nBlocks, m_filteredRowBegin.data(), m_filteredColsIndex.data(), m_filteredValues.data(), invertData, n);

    numStep++;

    return false;
}

template <class TMatrix, class TVector, class TThreadManager>
void SparseLDLSolver<TMatrix, TVector, TThreadManager>::showInvalidSystemMessage(const std::string& reason) const
{
    msg_warning() << "Invalid Linear System to solve (" << reason << "). Please insure that there is enough constraints (not rank deficient).";
}

template<class TMatrix, class TVector, class TThreadManager>
void SparseLDLSolver<TMatrix,TVector,TThreadManager>::invert(Matrix& M)
{
    factorize(M, (InvertData *) this->getMatrixInvertData(&M));
}

template <class TMatrix, class TVector, class TThreadManager>
bool SparseLDLSolver<TMatrix, TVector, TThreadManager>::doAddJMInvJtLocal(ResMatrixType* result, const JMatrixType* J, SReal fact, InvertData* data)
{
    if (!this->isComponentStateValid())
    {
        return true;
    }

    /*
    J * M^-1 * J^T = J * (L*D*L^T)^-1 * J^t
                   = (J * (L^T)^-1) * D^-1 * (L^-1 * J^T)
                   = (L^-1 * J^T)^T * D^-1 * (L^-1 * J^T)
    */

    if (J->rowSize() == 0)
    {
        return true;
    }

    Jlocal2global.clear();
    Jlocal2global.reserve(J->rowSize());
    for (auto jit = J->begin(), jitend = J->end(); jit != jitend; ++jit)
    {
        sofa::SignedIndex l = jit->first;
        Jlocal2global.push_back(l);
    }

    if (Jlocal2global.empty())
    {
        return true;
    }

    const unsigned int JlocalRowSize = (unsigned int)Jlocal2global.size();

    const simulation::ForEachExecutionPolicy execution = this->d_parallelInverseProduct.getValue() ?
        simulation::ForEachExecutionPolicy::PARALLEL :
        simulation::ForEachExecutionPolicy::SEQUENTIAL;

    simulation::TaskScheduler* taskScheduler = simulation::MainTaskSchedulerFactory::createInRegistry();
    assert(taskScheduler);

    using traits = typename BlockOperations::traits;
    constexpr int N = Inherit::BlockSize;

    // size of the factorized system (it includes the padding of the last block, if any)
    const int n = data->n * N;

    JLinv.clear();
    JLinv.resize(J->rowSize(), n);
    JLinvDinv.resize(J->rowSize(), n);

    // copy J in to JLinv taking into account the (block) permutation
    unsigned int localRow = 0;
    for (auto jit = J->begin(), jitend = J->end(); jit != jitend; ++jit, ++localRow)
    {
        Real* line = JLinv[localRow];
        for (auto it = jit->second.begin(), i2end = jit->second.end(); it != i2end; ++it)
        {
            sofa::SignedIndex block = it->first;
            sofa::SignedIndex component = 0;
            traits::split_col_index(block, component);

            const int col = data->invperm[block] * N + component;
            Real val = it->second;

            line[col] = val;
        }
    }

    {
        SCOPED_TIMER("LowerSystem");
        simulation::forEachRange(execution, *taskScheduler, 0u, JlocalRowSize,
            [&data, this](const auto& range)
            {
                SCOPED_TIMER("Lower");
                for (auto i = range.start; i != range.end; ++i)
                {
                    Real* line = JLinv[i];
                    sofa::linearalgebra::solveLowerUnitriangularSystemCSR(data->n, line, line, data->LT_colptr.data(), data->LT_rowind.data(), data->LT_values.data());
                }
            });
    }

    {
        SCOPED_TIMER("Diagonal");
        simulation::forEachRange(execution, *taskScheduler, 0u, JlocalRowSize,
           [&data, this](const auto& range)
           {
               for (auto i = range.start; i != range.end; ++i)
               {
                   Real* lineD = JLinv[i];
                   Real* lineM = JLinvDinv[i];
                   sofa::linearalgebra::solveDiagonalSystemUsingInvertedValues(data->n, lineD, lineM, data->invD.data());
               }
           });
    }

    const auto nbTriplets = JlocalRowSize * (JlocalRowSize+1) / 2;
    std::vector<Triplet> tripletsBuffer(nbTriplets);

    SCOPED_TIMER("UpperSystem");
    std::mutex mutex;

    // Distribution of the tasks according to the number of triplets, i.e. the
    // number of elements in a triangular matrix
    simulation::forEachRange(execution, *taskScheduler, 0u, nbTriplets,
        [n, this, fact, &mutex, result, &tripletsBuffer](const auto& range)
        {
            {
                SCOPED_TIMER("UpperRange");
                for (auto r = range.start; r != range.end; ++r)
                {
                    //convert a triangular matrix (flat) index to row and column coordinates
                    sofa::Index i, j;
                    linearalgebra::computeRowColumnCoordinateFromIndexInLowerTriangularMatrix(r, i, j);

                    auto& [row, col, value] = tripletsBuffer[r];
                    row = Jlocal2global[j];
                    col = Jlocal2global[i];

                    Real* lineI = JLinv[i];
                    Real* lineJ = JLinvDinv[j];

                    value = 0;
                    for (int k = 0; k < n; ++k)
                    {
                        value += lineJ[k] * lineI[k];
                    }
                    value *= fact;
                }
            }

            std::lock_guard guard(mutex);

            SCOPED_TIMER("Assembling");
            for (auto r = range.start; r != range.end; ++r)
            {
                const auto& [row, col, value] = tripletsBuffer[r];
                result->add(row, col, value);
                if (row != col)
                {
                    result->add(col, row, value);
                }
            }
        });

    return true;
}

// Default implementation of Multiply the inverse of the system matrix by the transpose of the given matrix, and multiply the result with the given matrix J
template<class TMatrix, class TVector, class TThreadManager>
bool SparseLDLSolver<TMatrix,TVector,TThreadManager>::addJMInvJtLocal(TMatrix * M, ResMatrixType * result,const JMatrixType * J, SReal fact) 
{

    InvertData* data = (InvertData*)this->getMatrixInvertData(M);

    return doAddJMInvJtLocal(result, J, fact, data);
}

} // namespace sofa::component::linearsolver::direct
