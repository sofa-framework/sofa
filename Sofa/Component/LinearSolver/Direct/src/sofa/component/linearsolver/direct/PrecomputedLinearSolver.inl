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

#include <sofa/component/linearsolver/direct/PrecomputedLinearSolver.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <cmath>
#include <sofa/component/linearsolver/direct/EigenSimplicialLLT.h>
#include <sofa/component/linearsolver/direct/EigenDirectSparseSolver.inl>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/core/behavior/LinearSolver.h>

#include <sofa/core/behavior/OdeSolver.h>

#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>

namespace sofa::component::linearsolver::direct
{

template<class TMatrix,class TVector>
PrecomputedLinearSolver<TMatrix,TVector>::PrecomputedLinearSolver()
    : jmjt_twostep( initData(&jmjt_twostep,true,"jmjt_twostep","Use two step algorithm to compute JMinvJt") )
    , use_file( initData(&use_file,true,"use_file","Dump system matrix in a file") )
{
    first = true;
}

template<class TMatrix,class TVector>
void PrecomputedLinearSolver<TMatrix,TVector>::setSystemMBKMatrix(const core::MechanicalParams* mparams)
{
    // Update the matrix only the first time
    if (first)
    {
        first = false;
        Inherit::setSystemMBKMatrix(mparams);
        loadMatrix(*this->getSystemMatrix());
    }
}

//Solve x = R * M^-1 * R^t * b
template<class TMatrix,class TVector>
void PrecomputedLinearSolver<TMatrix,TVector>::solve (TMatrix& , TVector& z, TVector& r)
{
    z = internalData.Minv * r;
}

template<class TMatrix,class TVector>
void PrecomputedLinearSolver<TMatrix,TVector >::loadMatrix(TMatrix& M)
{
    systemSize = this->getSystemMatrix()->rowSize();
    internalData.Minv.resize(systemSize,systemSize);
    dt = this->getContext()->getDt();

    sofa::core::behavior::OdeSolver::SPtr odeSolver;
    this->getContext()->get(odeSolver);
    factInt = 1.0; // christian : it is not a compliance... but an admittance that is computed !
    if (odeSolver) factInt = odeSolver->getPositionIntegrationFactor(); // here, we compute a compliance

    std::stringstream ss;
    ss << this->getContext()->getName() << "-" << systemSize << "-" << dt << ".comp";
    if(! use_file.getValue() || ! internalData.readFile(ss.str().c_str(),systemSize) )
    {
        loadMatrixWithCholeskyDecomposition(M);
        if (use_file.getValue()) internalData.writeFile(ss.str().c_str(),systemSize);
    }

    for (unsigned int j=0; j<systemSize; j++)
    {
        for (unsigned i=0; i<systemSize; i++)
        {
            internalData.Minv.set(j,i,internalData.Minv.element(j,i)/factInt);
        }
    }
}

template<class TMatrix,class TVector>
void PrecomputedLinearSolver<TMatrix,TVector>::loadMatrixWithCholeskyDecomposition(TMatrix& M)
{
    using namespace sofa::linearalgebra;
    msg_info() << "Compute the initial invert matrix with CS_PARSE" ;

    CompressedRowSparseMatrix<SReal> matSolv;
    FullVector<SReal> r;
    FullVector<SReal> b;

// 	unsigned systemSize = internalData.Minv.colSize();

    matSolv.resize(systemSize,systemSize);
    r.resize(systemSize);
    b.resize(systemSize);
    EigenSimplicialLLT<SReal> solver;

    for (unsigned int j=0; j<systemSize; j++)
    {
        for (unsigned int i=0; i<systemSize; i++)
        {
            if (M.element(j,i)!=0) matSolv.set(j,i,M.element(j,i));
        }
        b.set(j,0.0);
    }

    msg_info() << "Precomputing constraint correction LU decomposition " ;
    solver.invert(matSolv);

    for (unsigned int j=0; j<systemSize; j++)
    {
        std::stringstream tmp;
        tmp.precision(2);
        tmp << "Precomputing constraint correction : " << std::fixed << (float)j/(float)systemSize*100.0f << " %   " << '\xd';
        msg_info() << tmp.str() ;

        if (j>0) b.set(j-1,0.0);
        b.set(j,1.0);

        solver.solve(matSolv,r,b);
        for (unsigned int i=0; i<systemSize; i++)
        {
            internalData.Minv.set(j,i,r.element(i) * factInt);
        }
    }
    msg_info() << "Precomputing constraint correction : " << std::fixed << 100.0f << " %   " << '\xd';

}

template<class TMatrix,class TVector>
void PrecomputedLinearSolver<TMatrix,TVector>::invert(TMatrix& /*M*/) {}

template<class TMatrix,class TVector> template<class JMatrix>
void PrecomputedLinearSolver<TMatrix,TVector>::computeActiveDofs(JMatrix& J)
{
    isActiveDofs.clear();
    isActiveDofs.resize(systemSize);

    //compute JR = J * R
    for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != J.end(); jit1++)
    {
        for (typename JMatrix::LElementConstIterator i1 = jit1->second.begin(); i1 != jit1->second.end(); i1++)
        {
            isActiveDofs[i1->first] = true;
        }
    }

    internalData.invActiveDofs.clear();
    internalData.invActiveDofs.resize(systemSize);
    internalData.idActiveDofs.clear();

    for (unsigned c=0; c<systemSize; c++)
    {
        if (isActiveDofs[c])
        {
            internalData.invActiveDofs[c] = internalData.idActiveDofs.size();
            internalData.idActiveDofs.push_back(c);
        }
    }
}

template<class TMatrix,class TVector>
bool PrecomputedLinearSolver<TMatrix,TVector>::addJMInvJt(linearalgebra::BaseMatrix* result, linearalgebra::BaseMatrix* J, SReal fact)
{
    using namespace sofa::linearalgebra;

    if (first)
    {
        const core::MechanicalParams mparams = *core::mechanicalparams::defaultInstance();
        //TODO get the m b k factor from euler

        msg_error() << "The construction of the matrix when the solver is used only as cvonstraint "
                       "correction is not implemented. You first need to save the matrix into a file. " ;
        setSystemMBKMatrix(&mparams);
    }

    if (SparseMatrix<double>* j = dynamic_cast<SparseMatrix<double>*>(J))
    {
        computeActiveDofs(*j);
        ComputeResult(result, *j, (float) fact);
    }
    else if (SparseMatrix<float>* j = dynamic_cast<SparseMatrix<float>*>(J))
    {
        computeActiveDofs(*j);
        ComputeResult(result, *j, (float) fact);
    } return false;

    return true;
}

template <class TMatrix, class TVector>
void PrecomputedLinearSolver<TMatrix, TVector>::parse(core::objectmodel::BaseObjectDescription* arg)
{
    if (arg->getAttribute("verbose"))
    {
        msg_warning() << "Attribute 'verbose' has no use in this component. "
                         "To disable this warning, remove the attribute from the scene.";
    }

    Inherit::parse(arg);
}

template<class TMatrix,class TVector> template<class JMatrix>
void PrecomputedLinearSolver<TMatrix,TVector>::ComputeResult(linearalgebra::BaseMatrix * result,JMatrix& J, SReal fact)
{
    unsigned nl = 0;
    internalData.JMinv.clear();
    internalData.JMinv.resize(J.rowSize(),internalData.idActiveDofs.size());

    nl=0;
    for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != J.end(); jit1++)
    {
        for (unsigned c = 0; c<internalData.idActiveDofs.size(); c++)
        {
            int col = internalData.idActiveDofs[c];
            Real v = 0.0;
            for (typename JMatrix::LElementConstIterator i1 = jit1->second.begin(); i1 != jit1->second.end(); i1++)
            {
                v += internalData.Minv.element(i1->first,col) * i1->second;
            }
            internalData.JMinv.set(nl,c,v);
        }
        nl++;
    }
    //compute Result = JRMinv * (JR)t
    nl = 0;
    for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != J.end(); jit1++)
    {
        int row = jit1->first;
        for (typename JMatrix::LineConstIterator jit2 = J.begin(); jit2 != J.end(); jit2++)
        {
            int col = jit2->first;
            Real res = 0.0;
            for (typename JMatrix::LElementConstIterator i1 = jit2->second.begin(); i1 != jit2->second.end(); i1++)
            {
                res += internalData.JMinv.element(nl,internalData.invActiveDofs[i1->first]) * i1->second;
            }
            result->add(row,col,res*fact);
        }
        nl++;
    }
}

} // namespace sofa::component::linearsolver::direct
