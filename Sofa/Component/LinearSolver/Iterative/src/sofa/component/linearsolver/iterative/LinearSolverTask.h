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
class ComputeColumnTask: public sofa::simulation::CpuTask
{
public:

    typedef typename MatrixLinearSolverInternalData<Vector>::JMatrixType JMatrixType;

    int m_row;
    Vector* m_taskRH;
    Vector* m_taskLH;
    const JMatrixType* m_J;
    MatrixLinearSolver<Matrix,Vector> *m_solver;
    
    ComputeColumnTask(int row,
                      Vector  *taskRH,
                      Vector  *taskLH,
                      sofa::simulation::CpuTask::Status *status,
                      const JMatrixType *J,
                      MatrixLinearSolver<Matrix,Vector> *solver
                     );
    ~ComputeColumnTask() override = default;
    sofa::simulation::Task::MemoryAlloc run() final;
};

template<class Matrix, class Vector>
class ProductTask: public sofa::simulation::CpuTask
{

public:

    typedef typename MatrixLinearSolverInternalData<Vector>::JMatrixType JMatrixType;

    int m_row;
    Vector* m_colMinvJt;
    sofa::linearalgebra::FullMatrix<double>* m_product;
    const JMatrixType* m_J;

    ProductTask(int row
                ,Vector* colMinvJt
                ,const JMatrixType *J
                ,sofa::linearalgebra::FullMatrix<double> *product
                ,sofa::simulation::CpuTask::Status *status
                );
    ~ProductTask() override = default;
    sofa::simulation::Task::MemoryAlloc run() final;  
};


} // namespace sofa::component::linearsolver
