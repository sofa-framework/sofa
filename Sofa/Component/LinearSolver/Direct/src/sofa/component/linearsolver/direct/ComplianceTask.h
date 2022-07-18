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
#include <sofa/linearalgebra/FullMatrix.h>


namespace sofa::component::linearsolver
{

template<class Matrix,class Vector>
class SolverTask : public sofa::simulation::CpuTask
{
    public:

    typedef typename Matrix::Real Real;

    int m_n;
    int m_numRow;
    Real *m_row; //row(JLTinv) = col(LinvJt) = Linv*col(Jt) = Linv*row(J)
    Real *m_rowD;
    const int* m_L_colptr;
    const int* m_L_rowind;
    const Real* m_L_values;
    const SReal* m_invD;

    SolverTask(
                int n,
                int numRow,
                Real *row,
                Real *rowD,
                const int* L_colptr,
                const int* L_rowind,
                const Real* L_values,
                Real* invD,
                sofa::simulation::CpuTask::Status *status );
    ~SolverTask() override = default;
    sofa::simulation::Task::MemoryAlloc run() final;

};

template <class Matrix, class Vector>
SolverTask<Matrix,Vector>::SolverTask(
    int n,
    int numRow,
    Real *row,
    Real *rowD,
    const int* L_colptr,
    const int* L_rowind,
    const Real* L_values,
    Real* invD,
    sofa::simulation::CpuTask::Status *status)
:sofa::simulation::CpuTask(status),
m_n(n),
m_numRow(numRow),
m_row(row),
m_rowD(rowD),
m_L_colptr(L_colptr),
m_L_rowind(L_rowind),
m_L_values(L_values),
m_invD(invD)
{}


template <class Matrix, class Vector>
sofa::simulation::Task::MemoryAlloc SolverTask<Matrix,Vector>::run()
{
    
    // Solve the triangular system with forward substitution
    for (int j=0; j<m_n; j++) 
    {
        for (int p = m_L_colptr[j] ; p<m_L_colptr[j+1] ; p++) //  CSC L <-> CSR L^t
        {
            int col = m_L_rowind[p];
            double val = m_L_values[p];
            m_row[j] -= val * m_row[col];
        }

    }

    //apply the diagonal
    for (unsigned i = 0; i < (unsigned)m_n; i++) 
    {
        m_rowD[i] = m_invD[i] * m_row[i];
    }

    return simulation::Task::Stack;
}



template<class Matrix,class Vector>
class MultiplyTask : public sofa::simulation::CpuTask
{
    public:

    typedef typename Matrix::Real Real;   

    int m_n;
    int m_RowSize;
    int m_lineIndex;
    sofa::linearalgebra::FullMatrix<Real> *m_LHfactor;
    sofa::linearalgebra::FullMatrix<Real> *m_RHfactor;
    sofa::linearalgebra::FullMatrix<Real>  *m_result;
    int *m_local2global;

    MultiplyTask(
                int n,
                int RowSize,
                int lineIndex,
                sofa::linearalgebra::FullMatrix<Real> *LHfactor,
                sofa::linearalgebra::FullMatrix<Real> *RHfactor,
                sofa::linearalgebra::FullMatrix<Real>  *result,
                int *local2global,
                sofa::simulation::CpuTask::Status *status 
                );
    ~MultiplyTask() override = default;
    sofa::simulation::Task::MemoryAlloc run() final;

};

template <class Matrix, class Vector>
MultiplyTask<Matrix,Vector>::MultiplyTask(
                int n,
                int RowSize,
                int lineIndex,
                sofa::linearalgebra::FullMatrix<Real> *LHfactor,
                sofa::linearalgebra::FullMatrix<Real> *RHfactor,
                sofa::linearalgebra::FullMatrix<Real>  *result,
                int *local2global,
                sofa::simulation::CpuTask::Status *status 
):sofa::simulation::CpuTask(status),
    m_n(n),
    m_RowSize(RowSize),
    m_lineIndex(lineIndex),
    m_LHfactor(LHfactor),
    m_RHfactor(RHfactor),
    m_result(result),
    m_local2global(local2global)
    {}

template <class Matrix, class Vector>
sofa::simulation::Task::MemoryAlloc MultiplyTask<Matrix,Vector>::run()
{
    
    unsigned j = (unsigned)m_lineIndex;
    int globalRowJ = m_local2global[j];

    for (unsigned i = j; i < m_RowSize; i++) {

                int globalRowI = m_local2global[i];

                //m_result->set(globalRowI,globalRowJ, 0.0);
                double coeff =0;
                for (unsigned k = 0; k < (unsigned)m_n; k++) 
                {
                    //m_result->add(globalRowI,globalRowJ, m_LHfactor->element( i, k ) * m_RHfactor->element(i,k)  ); // we use the transpose here
                    coeff += m_LHfactor->element( i, k ) * m_RHfactor->element(i,k);
                }
                m_result->add(globalRowI,globalRowJ, coeff);
            }
    

   /*
    unsigned j = (unsigned)m_lineIndex;
    //Real* lineJ = m_LHfactor[j];
    int globalRowJ = m_local2global[j];
    for (unsigned i = j; i < (unsigned)m_RowSize; i++) 
    {
        //Real* lineI = m_RHfactor[i];
        int globalRowI = m_local2global[i];

            double acc = 0.0;
            for (unsigned k = 0; k < (unsigned)m_n; k++) 
            {
                //acc += lineJ[k] * lineI[k];
                acc += m_LHfactor->element(i,k) * m_RHfactor->element(i,k);
            }
            m_result->add(globalRowI,globalRowJ, acc  );
    }
    */

/*
for (unsigned i = j; i < JlocalRowSize; i++) {
    Real* lineI = JLTinv[i];
    int globalRowI = Jlocal2global[i];

    JMinvJt.set(globalRowI,globalRowJ, 0.0);
    for (unsigned k = 0; k < (unsigned)data->n; k++) {
        JMinvJt.add(globalRowI,globalRowJ, lineJ[k] * lineI[k]);
    }
            }
*/

    return simulation::Task::Stack;
}

} // namespace sofa::component::linearsolver