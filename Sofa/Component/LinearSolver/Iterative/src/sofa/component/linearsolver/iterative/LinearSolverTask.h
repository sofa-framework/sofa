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
    typedef typename MatrixLinearSolverInternalData<Vector>::JMatrixType JMatrixType;

public:

    Vector *specificThreadRHVector;
    std::vector<SReal> m_taskRH;
    MatrixLinearSolver<Matrix,Vector> *m_solver;
    const JMatrixType * m_J;
    int m_row;

    LinearSolverTask(sofa::simulation::CpuTask::Status *status);
    LinearSolverTask(sofa::simulation::CpuTask::Status *status,const JMatrixType * J
                    ,MatrixLinearSolver<Matrix,Vector> *solver
                    ,int row,std::vector<SReal> taskRH );
    ~LinearSolverTask() override = default;
    sofa::simulation::Task::MemoryAlloc run() final;  
};


template<>
class LinearSolverTask<GraphScatteredVector,GraphScatteredVector>:public sofa::simulation::CpuTask
{
    typedef typename MatrixLinearSolverInternalData<GraphScatteredVector>::JMatrixType JMatrixType;

    GraphScatteredVector *specificThreadRHVector;
    MatrixLinearSolver<GraphScatteredVector,GraphScatteredVector> *m_solver;
    const JMatrixType * m_J;
    int m_row;


    LinearSolverTask(sofa::simulation::CpuTask::Status *status):sofa::simulation::CpuTask(status){};

    LinearSolverTask(sofa::simulation::CpuTask::Status *status,const JMatrixType * J
                    ,MatrixLinearSolver<GraphScatteredVector,GraphScatteredVector> *solver
                    ,int row,std::vector<SReal> taskRH )
                    :sofa::simulation::CpuTask(status){};
    ~LinearSolverTask() ;
    sofa::simulation::Task::MemoryAlloc run(){return {};} ;
    
}; // dummy class for the compiler, must not be used


template<class Matrix, class Vector>
LinearSolverTask<Matrix,Vector>::LinearSolverTask(sofa::simulation::CpuTask::Status *status)
:sofa::simulation::CpuTask(status)
{}

template<class Matrix, class Vector>
LinearSolverTask<Matrix,Vector>::LinearSolverTask(sofa::simulation::CpuTask::Status *status,const JMatrixType * J
                    ,MatrixLinearSolver<Matrix,Vector> *solver,int row,std::vector<SReal> taskRH )
:sofa::simulation::CpuTask(status)
,m_J(J)
,m_solver(solver)
,m_row(row)
,m_taskRH(taskRH)
{}



template<class Matrix, class Vector>
sofa::simulation::Task::MemoryAlloc LinearSolverTask<Matrix,Vector>::run(){
    //std::cout<< m_taskRH[0] << std::endl;

    //for (typename JMatrixType::Index i=0; i<m_J->colSize(); i++) m_taskRH[i]= m_J->element(m_row, i);
   // m_taskRH[0] = 1;

    for(int i=0; i<m_J->colSize(); i++){
        std::cout<<m_taskRH[i] << ' ';
    }
    std::cout << std::endl;

//std::cout<< "called" << std::endl;
//std::cout << m_J->element(0,0) << std::endl;
//std::cout << m_J->colSize() <<  /* ' ' << specificThreadRHVector->size() <<*/ std::endl;
//for(int i=0; i<m_J->colSize();i++) std::cout<< m_J->element(0,i)<<' ';
//std::cout<<std::endl;
//std::cout << m_row << std::endl;
return simulation::Task::Stack;
};
} // namespace sofa::component::linearsolver