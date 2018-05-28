/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
// Author: Hadrien Courtecuisse

#ifndef SOFA_COMPONENT_LINEARSOLVER_SparseLDLSolver_INL
#define SOFA_COMPONENT_LINEARSOLVER_SparseLDLSolver_INL

#include <SofaSparseSolver/SparseLDLSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include "sofa/helper/system/thread/CTime.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <math.h>
#include <sofa/helper/system/thread/CTime.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.inl>
#include <fstream>

namespace sofa {

namespace component {

namespace linearsolver {

template<class TMatrix, class TVector, class TThreadManager>
SparseLDLSolver<TMatrix,TVector,TThreadManager>::SparseLDLSolver()
    : numStep(0)
    , f_saveMatrixToFile( initData(&f_saveMatrixToFile, false, "saveMatrixToFile", "save matrix to a text file (can be very slow, as full matrix is stored"))
{}

template<class TMatrix, class TVector, class TThreadManager>
void SparseLDLSolver<TMatrix,TVector,TThreadManager>::solve (Matrix& M, Vector& z, Vector& r) {
    Inherit::solve_cpu(&z[0],&r[0],(InvertData *) this->getMatrixInvertData(&M));
}

template<class TMatrix, class TVector, class TThreadManager>
void SparseLDLSolver<TMatrix,TVector,TThreadManager>::invert(Matrix& M) {
    if (f_saveMatrixToFile.getValue()) {
        std::ofstream f;
        char name[100];
        sprintf(name, "matrixInLDLInvert_%04d.txt", numStep);
        f.open(name);
        f << M;
        f.close();
    }

    Mfiltered.copyNonZeros(M);
    Mfiltered.compress();

    int n = M.colSize();

    int * M_colptr = (int *) &Mfiltered.getRowBegin()[0];
    int * M_rowind = (int *) &Mfiltered.getColsIndex()[0];
    Real * M_values = (Real *) &Mfiltered.getColsValue()[0];

    if(M_colptr==nullptr || M_rowind==nullptr || M_values==nullptr || Mfiltered.getRowBegin().size() < n )
    {
        msg_warning() << "Invalid Linear System to solve. Please insure that there is enough constraints (not rank deficient)." ;
        return ;
    }

    Inherit::factorize(n,M_colptr,M_rowind,M_values,(InvertData *) this->getMatrixInvertData(&M));

    numStep++;
}

/// Default implementation of Multiply the inverse of the system matrix by the transpose of the given matrix, and multiply the result with the given matrix J
template<class TMatrix, class TVector, class TThreadManager>
bool SparseLDLSolver<TMatrix,TVector,TThreadManager>::addJMInvJtLocal(TMatrix * M, ResMatrixType * result,const JMatrixType * J, double fact) {
    if (J->rowSize()==0) return true;

    InvertData * data = (InvertData *) this->getMatrixInvertData(M);

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
    for (unsigned c=0;c<(unsigned)J->rowSize();c++) {
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
    for (unsigned j=0; j<(unsigned)J->rowSize(); j++) {
        Real * lineD = Jdense[j];
        Real * lineM = Jminv[j];
        for (unsigned i=0;i<(unsigned)J->colSize();i++) {
            lineM[i] = lineD[i] * data->invD[i];
        }
    }

    for (unsigned j=0; j<(unsigned)J->rowSize(); j++) {
        Real * lineJ = Jminv[j];
        for (unsigned i=j;i<(unsigned)J->rowSize();i++) {
            Real * lineI = Jdense[i];

            double acc = 0.0;
            for (unsigned k=0;k<(unsigned)J->colSize();k++) {
                acc += lineJ[k] * lineI[k];
            }
            result->add(j,i,acc*fact);
            if(i!=j) result->add(i,j,acc*fact);
        }
    }

    return true;
}

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
