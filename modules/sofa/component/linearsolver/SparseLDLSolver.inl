/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
// Author: Hadrien Courtecuisse

#ifndef SOFA_COMPONENT_LINEARSOLVER_SparseLDLSolver_INL
#define SOFA_COMPONENT_LINEARSOLVER_SparseLDLSolver_INL

#include <sofa/component/linearsolver/SparseLDLSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include "sofa/helper/system/thread/CTime.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <math.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.inl>

namespace sofa {

namespace component {

namespace linearsolver {

template<class TMatrix, class TVector, class TThreadManager>
SparseLDLSolver<TMatrix,TVector,TThreadManager>::SparseLDLSolver() {}

template<class TMatrix, class TVector, class TThreadManager>
void SparseLDLSolver<TMatrix,TVector,TThreadManager>::solve (Matrix& M, Vector& z, Vector& r) {
    Inherit::solve_cpu(&z[0],&r[0],(InvertData *) this->getMatrixInvertData(&M));
}

template<class TMatrix, class TVector, class TThreadManager>
void SparseLDLSolver<TMatrix,TVector,TThreadManager>::invert(Matrix& M) {
    Inherit::factorize(M,(InvertData *) this->getMatrixInvertData(&M));
}

/// Default implementation of Multiply the inverse of the system matrix by the transpose of the given matrix, and multiply the result with the given matrix J
template<class TMatrix, class TVector, class TThreadManager>
bool SparseLDLSolver<TMatrix,TVector,TThreadManager>::addJMInvJtLocal(TMatrix * M, ResMatrixType * result, JMatrixType * J, double fact) {
    if (J->rowSize()==0) return true;

    InvertData * data = (InvertData *) getMatrixInvertData(M);

    Jdense.clear();
    Jdense.resize(J->rowSize(),data->n);
    Jminv.resize(J->rowSize(),data->n);

    for (typename SparseMatrix<Real>::LineConstIterator jit = J->begin() , jitend = J->end(); jit != jitend; ++jit) {
        int l = jit->first;
        Real * line = Jdense[l];
        for (typename SparseMatrix<Real>::LElementConstIterator it = jit->second.begin(), i2end = jit->second.end(); it != i2end; ++it) {
            int col = data->invperm[it->first];
            double val = it->second;

            line[col] = val;
        }
    }

    //Solve the lower triangular system
    for (unsigned c=0;c<J->rowSize();c++) {
        Real * line = Jdense[c];

        for (int j=0; j<data->n; j++) {
            for (int p = data->LT_colptr[j] ; p<data->LT_colptr[j+1] ; p++) {
                int col = data->LT_rowind[p];
                double val = data->LT_values[p];
                line[j] -= val * line[col];
            }
        }
    }

    //apply diagonal
    for (unsigned j=0; j<J->rowSize(); j++) {
        Real * lineD = Jdense[j];
        Real * lineM = Jminv[j];
        for (unsigned i=0;i<J->colSize();i++) {
            lineM[i] = lineD[i] * data->invD[i];
        }
    }

    for (unsigned j=0; j<J->rowSize(); j++) {
        Real * lineJ = Jminv[j];
        for (unsigned i=j;i<J->rowSize();i++) {
            Real * lineI = Jdense[i];

            double acc = 0.0;
            for (unsigned k=0;k<J->colSize();k++) {
                acc += lineJ[k] * lineI[k];
            }
            result->add(j,i,acc*fact);
            if(i!=j) result->add(i,j,acc*fact);
        }
    }


//    //Solve the lower triangular system
//    res.resize(data->n);
//    for (typename SparseMatrix<Real>::LineConstIterator jit = J->begin() , jitend = J->end(); jit != jitend; ++jit) {
//        int row = jit->first;

//        line.clear();
//        line.resize(data->n);

//        for (typename SparseMatrix<Real>::LElementConstIterator it = jit->second.begin(), i2end = jit->second.end(); it != i2end; ++it) {
//            int col = data->invperm[it->first];
//            double val = it->second;
//            line[col] = val;
//        }

//        for (int j=0; j<data->n; j++) {
//            for (int p = data->LT_colptr[j] ; p<data->LT_colptr[j+1] ; p++) {
//                int col = data->LT_rowind[p];
//                double val = data->LT_values[p];
//                line[j] -= val * line[col];
//            }
//        }

//        for (int j = data->n-1 ; j >= 0 ; j--) {
//            line[j] *= data->invD[j];

//            for (int p = data->L_colptr[j] ; p < data->L_colptr[j+1] ; p++) {
//                int col = data->L_rowind[p];
//                double val = data->L_values[p];
//                line[j] -= val * line[col];
//            }

//            res[data->perm[j]] = line[j];
//        }

//        for (typename SparseMatrix<Real>::LineConstIterator jit = J->begin() , jitend = J->end(); jit != jitend; ++jit) {
//            int row2 = jit->first;
//            double acc = 0.0;
//            for (typename SparseMatrix<Real>::LElementConstIterator i2 = jit->second.begin(), i2end = jit->second.end(); i2 != i2end; ++i2) {
//                int col2 = i2->first;
//                double val2 = i2->second;
//                acc += val2 * res[col2];
//            }
//            acc *= fact;
//            result->add(row2,row,acc);
//        }
//    }


    return true;
}

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
