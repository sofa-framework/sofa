/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_COLLISION_PRECOMPUTEDLINEARSOLVER_INL
#define SOFA_COMPONENT_COLLISION_PRECOMPUTEDLINEARSOLVER_INL

#include "PrecomputedLinearSolver.h"
#include <sofa/component/linearsolver/NewMatMatrix.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <math.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/component/forcefield/TetrahedronFEMForceField.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/linearsolver/MatrixLinearSolver.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/component/container/RotationFinder.h>
#include <sofa/core/behavior/LinearSolver.h>

#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/component/linearsolver/PCGLinearSolver.h>

#include <sofa/component/linearsolver/SparseCholeskySolver.h>
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.h>
#include <sofa/component/linearsolver/CholeskySolver.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

using namespace sofa::component::odesolver;
using namespace sofa::component::linearsolver;

template<class TMatrix,class TVector>
PrecomputedLinearSolver<TMatrix,TVector>::PrecomputedLinearSolver()
    : jmjt_twostep( initData(&jmjt_twostep,true,"jmjt_twostep","Use two step algorithm to compute JMinvJt") )
    , f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , use_file( initData(&use_file,true,"use_file","Dump system matrix in a file") )
{
    first = true;
    usePrecond = true;
}

template<class TMatrix,class TVector>
void PrecomputedLinearSolver<TMatrix,TVector>::setSystemMBKMatrix(const core::MechanicalParams* mparams)
{
    // Update the matrix only the first time

    if (first)
    {
        Inherit::setSystemMBKMatrix(mparams);
        loadMatrix();
        first = false;
    }

    this->currentGroup->needInvert = usePrecond;
}

//Solve x = R * M^-1 * R^t * b
template<class TMatrix,class TVector>
void PrecomputedLinearSolver<TMatrix,TVector>::solve (TMatrix& , TVector& z, TVector& r)
{
    if (usePrecond) z = internalData.Minv * r;
    else z = r;
}

template<class TMatrix,class TVector>
void PrecomputedLinearSolver<TMatrix,TVector >::loadMatrix()
{
    unsigned systemSize = this->currentGroup->systemMatrix->rowSize();
    internalData.Minv.resize(systemSize,systemSize);
    dt = this->getContext()->getDt();

    EulerImplicitSolver* EulerSolver;
    this->getContext()->get(EulerSolver);
    factInt = 1.0; // christian : it is not a compliance... but an admittance that is computed !
    if (EulerSolver) factInt = EulerSolver->getPositionIntegrationFactor(); // here, we compute a compliance

    std::stringstream ss;
    ss << this->getContext()->getName() << "-" << systemSize << "-" << dt << ".comp";
    std::ifstream compFileIn(ss.str().c_str(), std::ifstream::binary);

    if(compFileIn.good() && use_file.getValue())
    {
        cout << "file open : " << ss.str() << " compliance being loaded" << endl;
        compFileIn.read((char*) internalData.Minv[0], systemSize * systemSize * sizeof(Real));
        compFileIn.close();
    }
    else
    {
        loadMatrixWithCSparse();

        if (use_file.getValue())
        {
            std::ofstream compFileOut(ss.str().c_str(), std::fstream::out | std::fstream::binary);
            compFileOut.write((char*)internalData.Minv[0], systemSize * systemSize*sizeof(Real));
            compFileOut.close();
        }
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
void PrecomputedLinearSolver<TMatrix,TVector>::loadMatrixWithCSparse()
{
    cout << "Compute the initial invert matrix with CS_PARSE" << endl;

    CompressedRowSparseMatrix<double> matSolv;
    FullVector<double> r;
    FullVector<double> b;

    unsigned systemSize = internalData.Minv.colSize();

    matSolv.resize(systemSize,systemSize);
    r.resize(systemSize);
    b.resize(systemSize);
    SparseCholeskySolver<CompressedRowSparseMatrix<double>, FullVector<double> > solver;

    for (unsigned int j=0; j<systemSize; j++)
    {
        for (unsigned int i=0; i<systemSize; i++)
        {
            if (this->currentGroup->systemMatrix->element(j,i)!=0) matSolv.set(j,i,this->currentGroup->systemMatrix->element(j,i));
        }
        b.set(j,0.0);
    }

    std::cout << "Precomputing constraint correction LU decomposition " << std::endl;
    solver.invert(matSolv);

    for (unsigned int j=0; j<systemSize; j++)
    {
        std::cout.precision(2);
        std::cout << "Precomputing constraint correction : " << std::fixed << (float)j/(float)systemSize*100.0f << " %   " << '\xd';
        std::cout.flush();

        if (j>0) b.set(j-1,0.0);
        b.set(j,1.0);

        solver.solve(matSolv,r,b);
        for (unsigned int i=0; i<systemSize; i++)
        {
            internalData.Minv.set(j,i,r.element(i) * factInt);
        }
    }
    std::cout << "Precomputing constraint correction : " << std::fixed << 100.0f << " %   " << '\xd';
    std::cout.flush();
}

template<class TMatrix,class TVector>
void PrecomputedLinearSolver<TMatrix,TVector>::invert(TMatrix& /*M*/) {}

template<class TMatrix,class TVector>
bool PrecomputedLinearSolver<TMatrix,TVector>::addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact)
{
    if (SparseMatrix<double>* j = dynamic_cast<SparseMatrix<double>*>(J))
    {
        ComputeResult(result, *j, (float) fact);
    }
    else if (SparseMatrix<float>* j = dynamic_cast<SparseMatrix<float>*>(J))
    {
        ComputeResult(result, *j, (float) fact);
    } return false;

    return true;
}

template<class TMatrix,class TVector> template<class JMatrix>
void PrecomputedLinearSolver<TMatrix,TVector>::ComputeResult(defaulttype::BaseMatrix * result,JMatrix& J, float fact)
{
    unsigned nl = 0;
    for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != J.end(); jit1++) nl++;

    internalData.JMinv.clear();
    internalData.JMinv.resize(nl,internalData.Minv.rowSize());

    nl = 0;
    for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != J.end(); jit1++)
    {
        for (unsigned c = 0; c<internalData.Minv.rowSize(); c++)
        {
            Real v = 0.0;
            for (typename JMatrix::LElementConstIterator i1 = jit1->second.begin(); i1 != jit1->second.end(); i1++)
            {
                v += internalData.Minv.element(i1->first,c) * i1->second;
            }
            internalData.JMinv.add(nl,c,v);
        }
        nl++;
    }

    //compute Result = JRMinv * Jt

    nl = 0;
    for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != J.end(); jit1++)
    {
        int l = jit1->first;
        for (typename JMatrix::LineConstIterator jit2 = J.begin(); jit2 != J.end(); jit2++)
        {
            int c = jit2->first;
            Real res = 0.0;
            for (typename JMatrix::LElementConstIterator i1 = jit2->second.begin(); i1 != jit2->second.end(); i1++)
            {
                res += internalData.JMinv.element(nl,i1->first) * i1->second;
            }
            result->add(l,c,res*fact);
        }
        nl++;
    }
}

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
