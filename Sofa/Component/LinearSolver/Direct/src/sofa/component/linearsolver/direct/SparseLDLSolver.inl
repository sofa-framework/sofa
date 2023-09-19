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
#include <sofa/simulation/MainTaskSchedulerFactory.h>
#include <sofa/simulation/ParallelForEach.h>


namespace sofa::component::linearsolver::direct 
{

template<class TMatrix, class TVector, class TThreadManager>
SparseLDLSolver<TMatrix,TVector,TThreadManager>::SparseLDLSolver()
    : numStep(0)
    , d_parallelInverseProduct(initData(&d_parallelInverseProduct, false,
        "parallelInverseProduct", "Parallelize the computation of the product J*M^{-1}*J^T "
                                  "where M is the matrix of the linear system and J is any "
                                  "matrix with compatible dimensions"))
{
    this->addUpdateCallback("parallelODESolving", {&d_parallelInverseProduct},
    [this](const core::DataTracker& tracker) -> sofa::core::objectmodel::ComponentState
    {
        SOFA_UNUSED(tracker);
        if (d_parallelInverseProduct.getValue())
        {
            simulation::TaskScheduler* taskScheduler = simulation::MainTaskSchedulerFactory::createInRegistry();
            assert(taskScheduler);

            if (taskScheduler->getThreadCount() < 1)
            {
                taskScheduler->init(0);
                msg_info() << "Task scheduler initialized on " << taskScheduler->getThreadCount() << " threads";
            }
            else
            {
                msg_info() << "Task scheduler already initialized on " << taskScheduler->getThreadCount() << " threads";
            }
        }
        return this->d_componentState.getValue();
    },
    {});
}

template <class TMatrix, class TVector, class TThreadManager>
void SparseLDLSolver<TMatrix, TVector, TThreadManager>::init()
{
    Inherit::init();

    this->d_componentState.setValue(core::objectmodel::ComponentState::Valid);
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
    sofa::helper::ScopedAdvancedTimer solveTimer("solve");
    Inherit::solve_cpu(z.ptr(), r.ptr(), (InvertData *) this->getMatrixInvertData(&M));
}

template <class TMatrix, class TVector, class TThreadManager>
bool SparseLDLSolver<TMatrix, TVector, TThreadManager>::factorize(
    Matrix& M, InvertData * invertData)
{
    Mfiltered.copyNonZeros(M);
    Mfiltered.compress();

    int n = M.colSize();

    if (n == 0)
    {
        showInvalidSystemMessage("null size");
        return true;
    }

    int * M_colptr = (int *)Mfiltered.getRowBegin().data();
    int * M_rowind = (int *)Mfiltered.getColsIndex().data();
    Real * M_values = (Real *)Mfiltered.getColsValue().data();

    if (M_colptr == nullptr || M_rowind == nullptr || M_values == nullptr)
    {
        showInvalidSystemMessage("invalid matrix data structure");
        return true;
    }

    if (Mfiltered.getRowBegin().size() < (size_t)n)
    {
        showInvalidSystemMessage("size mismatch");
        return true;
    }

    Inherit::factorize(n,M_colptr,M_rowind,M_values, invertData);

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

    const simulation::ForEachExecutionPolicy execution = d_parallelInverseProduct.getValue() ?
        simulation::ForEachExecutionPolicy::PARALLEL :
        simulation::ForEachExecutionPolicy::SEQUENTIAL;

    simulation::TaskScheduler* taskScheduler = simulation::MainTaskSchedulerFactory::createInRegistry();
    assert(taskScheduler);

    JLinv.clear();
    JLinv.resize(J->rowSize(), data->n);
    JLinvDinv.resize(J->rowSize(), data->n);

    // copy J in to JLinv taking into account the permutation
    unsigned int localRow = 0;
    for (auto jit = J->begin(), jitend = J->end(); jit != jitend; ++jit, ++localRow)
    {
        Real* line = JLinv[localRow];
        for (auto it = jit->second.begin(), i2end = jit->second.end(); it != i2end; ++it)
        {
            int col = data->invperm[it->first];
            Real val = it->second;

            line[col] = val;
        }
    }

    simulation::forEachRange(execution, *taskScheduler, 0u, JlocalRowSize,
        [&data, this](const auto& range)
        {
            for (auto i = range.start; i != range.end; ++i)
            {
                Real* line = JLinv[i];
                sofa::linearalgebra::solveLowerUnitriangularSystemCSR(data->n, line, line, data->LT_colptr.data(), data->LT_rowind.data(), data->LT_values.data());
            }
        });

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

    std::mutex mutex;
    simulation::forEachRange(execution, *taskScheduler, 0u, JlocalRowSize,
        [&data, this, fact, &mutex, result, JlocalRowSize](const auto& range)
        {
            std::vector<std::tuple<sofa::SignedIndex, sofa::SignedIndex, Real> > triplets;
            triplets.reserve(JlocalRowSize * (range.end - range.start));

            for (auto j = range.start; j != range.end; ++j)
            {
                Real* lineJ = JLinvDinv[j];
                sofa::SignedIndex globalRowJ = Jlocal2global[j];
                for (unsigned i = j; i < JlocalRowSize; ++i)
                {
                    Real* lineI = JLinv[i];
                    int globalRowI = Jlocal2global[i];

                    Real acc = 0;
                    for (unsigned k = 0; k < (unsigned)data->n; k++)
                    {
                        acc += lineJ[k] * lineI[k];
                    }
                    acc *= fact;

                    triplets.emplace_back(globalRowJ, globalRowI, acc);
                }
            }

            std::lock_guard guard(mutex);

            for (const auto& [row, col, value] : triplets)
            {
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
