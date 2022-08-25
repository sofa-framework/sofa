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
#include <sofa/simulation/TaskScheduler.h>
#include <sofa/component/linearsolver/direct/ComplianceTask.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

namespace sofa::component::linearsolver::direct 
{

template<class TMatrix, class TVector, class TThreadManager>
SparseLDLSolver<TMatrix,TVector,TThreadManager>::SparseLDLSolver()
    : numStep(0){}

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
    Inherit::solve_cpu(&z[0],&r[0],(InvertData *) this->getMatrixInvertData(&M));
}

template <class TMatrix, class TVector, class TThreadManager>
bool SparseLDLSolver<TMatrix, TVector, TThreadManager>::factorize(
    Matrix& M, InvertData * invertData)
{
    Mfiltered.copyNonZeros(M);
    Mfiltered.compress();

    int n = M.colSize();

    int * M_colptr = (int *) &Mfiltered.getRowBegin()[0];
    int * M_rowind = (int *) &Mfiltered.getColsIndex()[0];
    Real * M_values = (Real *) &Mfiltered.getColsValue()[0];

    if(M_colptr==nullptr || M_rowind==nullptr || M_values==nullptr || Mfiltered.getRowBegin().size() < (size_t)n )
    {
        msg_warning() << "Invalid Linear System to solve. Please insure that there is enough constraints (not rank deficient)." ;
        return true;
    }

    Inherit::factorize(n,M_colptr,M_rowind,M_values, invertData);

    numStep++;

    return false;
}

template<class TMatrix, class TVector, class TThreadManager>
void SparseLDLSolver<TMatrix,TVector,TThreadManager>::invert(Matrix& M)
{
    factorize(M, (InvertData *) this->getMatrixInvertData(&M));
}

template <class TMatrix, class TVector, class TThreadManager>
bool SparseLDLSolver<TMatrix, TVector, TThreadManager>::doAddJMInvJtLocal(ResMatrixType* result, const JMatrixType* J, SReal fact, InvertData* data)
{
    /*
    J*Minv*J^t = J*(L*D*L^t)^-1 * J^t
               = J*(L^t)^-1*D^-1*L^-1*J^t
               = J*(L^t)^-1*D^-1*(J*L^-1)^t
               = (L^-1 *J^t)^t * D^-1 * (L^-1*J^t )
    */


    const bool mutlithread  = this->d_multithreading.getValue() ;

    if (J->rowSize()==0) return true;

    Jlocal2global.clear();
    Jlocal2global.reserve(J->rowSize());
    for (auto jit = J->begin(), jitend = J->end(); jit != jitend; ++jit) {
        int l = jit->first;
        Jlocal2global.push_back(l);
    }

    if (Jlocal2global.empty()) return true;

    const unsigned int JlocalRowSize = (unsigned int)Jlocal2global.size();



    JLTinv.clear();
    JLTinv.resize(J->rowSize(), data->n);
    JLTinvDinv.resize(J->rowSize(), data->n);

    // copy J into JLTinv
    unsigned int localRow = 0;
    for (auto jit = J->begin(), jitend = J->end(); jit != jitend; ++jit, ++localRow) {
        Real* line = JLTinv[localRow];
        for (auto it = jit->second.begin(), i2end = jit->second.end(); it != i2end; ++it) {
            int col = data->invperm[it->first];
            double val = it->second;

            line[col] = val;
        }
    }

    auto* taskScheduler = sofa::simulation::TaskScheduler::getInstance();

    if( mutlithread )
    {
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
    else
    {
       if (taskScheduler->getThreadCount() != 1) taskScheduler->init(1);
    }

    std::vector< SolverTask<TMatrix,TVector> > solveTaskList;
    if( mutlithread ) solveTaskList.reserve( JlocalRowSize );

    sofa::simulation::CpuTask::Status status;

    if (this->linearSystem.needInvert)
    {
        this->invert(*(this->linearSystem.systemMatrix));
        this->linearSystem.needInvert = false;
    }

// Compute JLtinv column by column and then compute JLtinvDinv 
    {
        sofa::helper::ScopedAdvancedTimer solveTimer("solve");
        // row(JLtinv) = col( LinvJt ) = Linv*col(Jt) = Linv*row(J)
        for (unsigned c = 0; c < JlocalRowSize; c++) 
        {
            SReal* row = JLTinv[c];
            SReal* rowD = JLTinvDinv[c] ;

            if( mutlithread )
            {
                // CSC Lt <-> CSR L
                solveTaskList.emplace_back( data->n, c,row , rowD ,data->LT_colptr.data(), data->LT_rowind.data() ,
                                            data->LT_values.data(), data->invD.data(), &status );
                taskScheduler->addTask( &(solveTaskList.back()) );
            }
            else
            {
                SolverTask<Matrix,Vector> solTask( data->n, c,row , rowD ,data->LT_colptr.data(), data->LT_rowind.data() ,
                                            data->LT_values.data(), data->invD.data(), &status );
                solTask.run();
            }
        }

        taskScheduler->workUntilDone(&status);
    }


    sofa::linearalgebra::FullMatrix<Real> JMinvJt( JlocalRowSize, JlocalRowSize );
    
    std::vector< MultiplyTask<TMatrix,TVector> > multiplyTaskList;
    if( mutlithread ) multiplyTaskList.reserve( JlocalRowSize );

    {

        sofa::helper::ScopedAdvancedTimer multiplyTimer("product");
        //compute the marix product JLtinvDinv*(JLtinv)^t
        for (unsigned j = 0; j < JlocalRowSize; j++)
        {
            Real* lineJ = JLTinvDinv[j];
            int globalRowJ = Jlocal2global[j];

            if( mutlithread )
            {
                multiplyTaskList.emplace_back(data->n, JlocalRowSize, j, &JLTinvDinv, &JLTinv, &JMinvJt, Jlocal2global.data(), &status );
                taskScheduler->addTask( &(multiplyTaskList.back()) );
            }
            else
            {
                MultiplyTask<Matrix,Vector> mulTask(data->n, JlocalRowSize, j, &JLTinvDinv, &JLTinv, &JMinvJt, Jlocal2global.data(), &status );
                mulTask.run();
            } 
        }
        taskScheduler->workUntilDone(&status);
    }

    {
        sofa::helper::ScopedAdvancedTimer projectTimer("project");
        //project the data
        for (unsigned j = 0; j < JlocalRowSize; j++) {
            int globalRowJ = Jlocal2global[j];
            for (unsigned i = j; i < JlocalRowSize; i++) {
                int globalRowI = Jlocal2global[i];

                result->add(globalRowJ, globalRowI, JMinvJt.element(globalRowI,globalRowJ)*fact);
                if (globalRowI != globalRowJ) result->add(globalRowI, globalRowJ, JMinvJt.element(globalRowI,globalRowJ)*fact);
            }
        }

    }

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
