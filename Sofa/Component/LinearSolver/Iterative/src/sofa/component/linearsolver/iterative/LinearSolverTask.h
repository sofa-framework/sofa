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
class LinearSolverTask: public sofa::simulation::CpuTask
{

public:

    typedef typename MatrixLinearSolverInternalData<Vector>::JMatrixType JMatrixType;

    int m_row;
    Vector  *m_taskRH;
    Vector  *m_taskLH;
    const JMatrixType *m_J;
    MatrixLinearSolver<Matrix,Vector> *m_solver;
    

    LinearSolverTask(sofa::simulation::CpuTask::Status *status);
    LinearSolverTask(int row
                    ,Vector  *taskRH
                    ,Vector  *taskLH
                    ,sofa::simulation::CpuTask::Status *status
                    ,const JMatrixType *J
                    ,MatrixLinearSolver<Matrix,Vector> *solver
                     );
    ~LinearSolverTask() override = default;
    sofa::simulation::Task::MemoryAlloc run() final;  
};

template<class Matrix, class Vector>
LinearSolverTask<Matrix,Vector>::LinearSolverTask(sofa::simulation::CpuTask::Status *status)
:sofa::simulation::CpuTask(status)
{}

template<class Matrix, class Vector>
LinearSolverTask<Matrix,Vector>::LinearSolverTask(
    int row
    ,Vector *taskRH 
    ,Vector *taskLH 
    ,sofa::simulation::CpuTask::Status *status
    ,const JMatrixType *J
    ,MatrixLinearSolver<Matrix,Vector> *solver)
:sofa::simulation::CpuTask(status)
,m_row(row)
,m_taskRH(taskRH)
,m_taskLH(taskLH)
,m_J(J)
,m_solver(solver)
{}

template<class Matrix, class Vector>
sofa::simulation::Task::MemoryAlloc LinearSolverTask<Matrix,Vector>::run()
{

for(int col=0;col<m_J->colSize();col++)
    {
            // col,row                row,col
        //listRH[row][col] = J->element(row,col) ; //copy Jt
        m_taskRH->set( col , m_J->element(m_row,col) );
    }

m_solver->solve( *(m_solver->linearSystem.systemMatrix), *m_taskLH, *m_taskRH );

return simulation::Task::Stack;
};
} // namespace sofa::component::linearsolver