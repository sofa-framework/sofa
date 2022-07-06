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

#include <sofa/simulation/CpuTask.h>

#include <sofa/simulation/fwd.h>
#include <sofa/core/fwd.h>
#include <sofa/core/MultiVecId.h>

#include<sofa/component/linearsolver/iterative/MatrixLinearSolver.h>

namespace sofa::component::linearsolver
{

template<class Matrix, class Vector>
class solverTask: public sofa::simulation::CpuTask
{

public:

    typedef typename MatrixLinearSolverInternalData<Vector>::JMatrixType JMatrixType;

    int m_row;
    Vector  *m_taskRH;
    Vector  *m_taskLH;
    const JMatrixType *m_J;
    MatrixLinearSolver<Matrix,Vector> *m_solver;
    

    solverTask(sofa::simulation::CpuTask::Status *status);
    solverTask(int row
                    ,Vector  *taskRH
                    ,Vector  *taskLH
                    ,sofa::simulation::CpuTask::Status *status
                    ,const JMatrixType *J
                    ,MatrixLinearSolver<Matrix,Vector> *solver
                     );
    ~solverTask() override = default;
    sofa::simulation::Task::MemoryAlloc run() final;  
};

template<class Matrix, class Vector>
solverTask<Matrix,Vector>::solverTask(sofa::simulation::CpuTask::Status *status)
:sofa::simulation::CpuTask(status)
{}

template<class Matrix, class Vector>
solverTask<Matrix,Vector>::solverTask(
    int row,
    Vector *taskRH, 
    Vector *taskLH, 
    sofa::simulation::CpuTask::Status *status,
    const JMatrixType *J,
    MatrixLinearSolver<Matrix,Vector> *solver)
:sofa::simulation::CpuTask(status)
,m_row(row)
,m_taskRH(taskRH)
,m_taskLH(taskLH)
,m_J(J)
,m_solver(solver)
{}

template<class Matrix, class Vector>
sofa::simulation::Task::MemoryAlloc solverTask<Matrix,Vector>::run()
{

for(int col=0;col<m_J->colSize();col++)
    {
        m_taskRH->set( col , m_J->element(m_row,col) );
    }

m_solver->solve( *(m_solver->linearSystem.systemMatrix), *m_taskLH, *m_taskRH );

return simulation::Task::Stack;
};



template<class Matrix, class Vector>
class productTask: public sofa::simulation::CpuTask
{

public:

    typedef typename MatrixLinearSolverInternalData<Vector>::JMatrixType JMatrixType;

    int m_row;
    Vector* m_colMinvJt;
    sofa::linearalgebra::FullMatrix<double> *m_product;
    const JMatrixType *m_J;

    productTask(sofa::simulation::CpuTask::Status *status);
    productTask(int row
                ,Vector* colMinvJt
                ,const JMatrixType *J
                ,sofa::linearalgebra::FullMatrix<double> *product
                ,sofa::simulation::CpuTask::Status *status
                );
    ~productTask() override = default;
    sofa::simulation::Task::MemoryAlloc run() final;  
};

template<class Matrix, class Vector>
productTask<Matrix,Vector>::productTask(
    int row
    ,Vector* colMinvJt
    ,const JMatrixType *J
    ,sofa::linearalgebra::FullMatrix<double> *product
    ,sofa::simulation::CpuTask::Status *status)
:sofa::simulation::CpuTask(status)
,m_row(row)
,m_colMinvJt(colMinvJt)
,m_product(product)
,m_J(J)
{}

template<class Matrix, class Vector>
sofa::simulation::Task::MemoryAlloc productTask<Matrix,Vector>::run()
{
    
    const typename linearalgebra::SparseMatrix<SReal>::LineConstIterator jitend = m_J->end();
    for (typename linearalgebra::SparseMatrix<SReal>::LineConstIterator jit = m_J->begin(); jit != jitend; ++jit)
    {
        auto row2 = jit->first;
        m_product->set(row2,m_row , 0 );

        for (typename linearalgebra::SparseMatrix<SReal>::LElementConstIterator i2 = jit->second.begin(), i2end = jit->second.end(); i2 != i2end; ++i2)
        {
            auto col2 = i2->first;
            double val2 = i2->second;
            m_product->add(row2,m_row, val2 * m_colMinvJt->element(col2) );
        }
    }



    return simulation::Task::Stack;
}

} // namespace sofa::component::linearsolver