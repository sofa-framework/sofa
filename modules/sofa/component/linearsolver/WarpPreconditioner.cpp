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
#include <sofa/component/linearsolver/PrecomputedWarpPreconditioner.h>
#include <sofa/component/linearsolver/NewMatMatrix.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/componentmodel/behavior/LinearSolver.h>
#include <math.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/component/forcefield/TetrahedronFEMForceField.h>
#include <sofa/defaulttype/Vec3Types.h>

#ifdef SOFA_HAVE_CSPARSE
#include <sofa/component/linearsolver/SparseCholeskySolver.h>
#include <sofa/component/linearsolver/SparseLDLSolver.h>
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.h>

#else
#include <sofa/component/linearsolver/CholeskySolver.h>
#endif

namespace sofa
{

namespace component
{

namespace linearsolver
{

template<class TMatrix, class TVector,class TDataTypes>
PrecomputedWarpPreconditioner<TMatrix,TVector,TDataTypes>::PrecomputedWarpPreconditioner()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , f_graph( initData(&f_graph,"graph","Graph of residuals at each iteration") )
{
    f_graph.setWidget("graph");
    f_graph.setReadOnly(true);
    first = true;
}

template<class TMatrix, class TVector,class TDataTypes>
void PrecomputedWarpPreconditioner<TMatrix,TVector,TDataTypes>::setSystemMBKMatrix(double mFact, double bFact, double kFact)
{
    // Update the matrix only the first time
    if (first) Inherit::setSystemMBKMatrix(mFact,bFact,kFact);
    else Inherit::needInvert = true;
}

//Solve x = R * M^-1 * R^t * b
template<class TMatrix, class TVector,class TDataTypes>
void PrecomputedWarpPreconditioner<TMatrix,TVector,TDataTypes>::solve (Matrix& M, Vector& z, Vector& r)
{
    //Solve z = R^t * b
    for (unsigned i=0; i<R.size(); i++)
    {
        z[i * 3 + 0] = R[i][0][0] * r[i * 3 + 0] + R[i][1][0] * r[i * 3 + 1] + R[i][2][0] * r[i * 3 + 2];
        z[i * 3 + 1] = R[i][0][1] * r[i * 3 + 0] + R[i][1][1] * r[i * 3 + 1] + R[i][2][1] * r[i * 3 + 2];
        z[i * 3 + 2] = R[i][0][2] * r[i * 3 + 0] + R[i][1][2] * r[i * 3 + 1] + R[i][2][2] * r[i * 3 + 2];
    }

    //Solve tmp = M^-1 * z
    for (unsigned j=0; j<M.rowSize(); j++)
    {
        tmp[j] = 0.0;
        for (unsigned i=0; i<M.colSize(); i++)
        {
            //initial matrix + current rotations
            tmp[j] += M.element(i,j) * z[i];
        }
    }

    //Solve z = R * tmp
    for (unsigned i=0; i<R.size(); i++)
    {
        z[i * 3 + 0] = R[i][0][0] * tmp[i * 3] + R[i][0][1] * tmp[i * 3 + 1] + R[i][0][2] * tmp[i * 3 + 2];
        z[i * 3 + 1] = R[i][1][0] * tmp[i * 3] + R[i][1][1] * tmp[i * 3 + 1] + R[i][1][2] * tmp[i * 3 + 2];
        z[i * 3 + 2] = R[i][2][0] * tmp[i * 3] + R[i][2][1] * tmp[i * 3 + 1] + R[i][2][2] * tmp[i * 3 + 2];
    }
}

#ifdef SOFA_HAVE_CSPARSE

template<class TMatrix, class TVector,class TDataTypes>
void PrecomputedWarpPreconditioner<TMatrix,TVector,TDataTypes>::precompute(Matrix& M)
{
    std::stringstream ss;
    ss << this->getContext()->getName() << ".comp";
    std::ifstream compFileIn(ss.str().c_str(), std::ifstream::binary);
    unsigned int n=0, nbCol=0;
    this->getMatrixDimension(&n, &nbCol);

    if(compFileIn.good())
    {
        cout << "file open : " << ss.str() << " compliance being loaded" << endl;
        compFileIn.read((char*)M[0], n * n * sizeof(double));
        compFileIn.close();
    }
    else
    {
        cout << "Compute the initial invert matrix" << endl;

        sofa::component::linearsolver::SparseCholeskySolver< CompressedRowSparseMatrix<double>, FullVector<double> > sparsecholesky;

        FullVector<double> b;
        FullVector<double> x;
        CompressedRowSparseMatrix<double> Mc;

        b.resize(n);
        x.resize(n);
        Mc.resize(n,n);

        for (unsigned j=0; j<n; j++)
        {
            for (unsigned i=0; i<n; i++)
            {
                if (M.element(i,j)) Mc.set(i,j,M.element(i,j));
            }
        }

        /*
        std::ifstream comp("toto", std::ifstream::binary);
        if(comp.good()) {
        	if (!sparsecholesky.readFile(comp)) {
        		std::ofstream comp2("toto", std::fstream::binary);
        		sparsecholesky.invert(Mc);
        		sparsecholesky.writeFile(comp2);
        	} else {
        		std::cout << "file loaded" << std::endl;
        	}
        } else {
        	std::ofstream comp2("toto", std::fstream::binary);
        	sparsecholesky.invert(Mc); //creat LU decomposition in cholesky
        	sparsecholesky.writeFile(comp2);
        }
        */

        sparsecholesky.invert(Mc); //creat LU decomposition in cholesky

        //Compute M^-1 in the systemMatrix of PrecomputedWarpPreconditioner (M)
        for (unsigned i=0; i<n; i++)
        {
            b.clear();
            b.set(i,1.0);

            sparsecholesky.solve(Mc,x,b); //M n'est pas utilisé ni modifié, dans l'algo de cholesky

            for (unsigned j=0; j<n; j++) M.set(i,j,x.element(j));
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////
        std::ofstream compFileOut(ss.str().c_str(), std::fstream::out | std::fstream::binary);
        compFileOut.write((char*)M[0], n * n*sizeof(double));
        compFileOut.close();

        Mc.clear();
    }

    first = false;

}

#else

template<class TMatrix, class TVector,class TDataTypes>
void PrecomputedWarpPreconditioner<TMatrix,TVector,TDataTypes>::precompute(Matrix& M)
{
    std::stringstream ss;
    ss << this->getContext()->getName() << ".comp";
    std::ifstream compFileIn(ss.str().c_str(), std::ifstream::binary);
    unsigned int n=0, nbCol=0;
    this->getMatrixDimension(&n, &nbCol);

    if(compFileIn.good())
    {
        cout << "file open : " << ss.str() << " compliance being loaded" << endl;
        compFileIn.read((char*)M[0], n * n * sizeof(double));
        compFileIn.close();
    }
    else
    {
        cout << "Compute the initial invert matrix" << endl;

        sofa::component::linearsolver::CholeskySolver< TMatrix, TVector > cholesky;

        FullVector<double> b;
        FullVector<double> x;

        b.resize(n);
        x.resize(n);

        cholesky.invert(M); //creat LU decomposition in cholesky

        //Compute M^-1 in the systemMatrix of PrecomputedWarpPreconditioner (M)
        for (unsigned i=0; i<n; i++)
        {
            b.clear();
            b.set(i,1.0);

            cholesky.solve(M,x,b); //M n'est pas utilisé ni modifié, dans l'algo de cholesky

            for (unsigned j=0; j<n; j++) M.set(i,j,x.element(j));
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////
        std::ofstream compFileOut(ss.str().c_str(), std::fstream::out | std::fstream::binary);
        compFileOut.write((char*)M[0], n * n*sizeof(double));
        compFileOut.close();
    }

    first = false;

}

#endif

template<class TMatrix, class TVector,class TDataTypes>
void PrecomputedWarpPreconditioner<TMatrix,TVector,TDataTypes>::invert(Matrix& M)
{
    if (first)
    {
        mstate = dynamic_cast< behavior::MechanicalState<TDataTypes>* >(this->getContext()->getMechanicalState());
        assert(mstate);

        this->precompute(M);
        unsigned int n=0, nbCol=0;
        this->getMatrixDimension(&n, &nbCol);

        R.resize(n/3);
        tmp.resize(n);
        //R.clear();

        for (unsigned i=0; i<R.size(); i++)
        {
            R[i][0][0] = R[i][1][1] = R[i][2][2] = 1.0;
            R[i][0][1] = R[i][0][2] = R[i][1][0] = R[i][1][2] = R[i][2][0] = R[i][2][1] = 0.0;
        }
    }

    this->rotateConstraints();
}

template<class TMatrix, class TVector,class TDataTypes>
void PrecomputedWarpPreconditioner<TMatrix,TVector,TDataTypes>::rotateConstraints()
{
    simulation::Node *node = dynamic_cast<simulation::Node *>(this->getContext());

    sofa::component::forcefield::TetrahedronFEMForceField<TDataTypes>* forceField = NULL;
    if (node != NULL)
    {
        forceField = node->get<component::forcefield::TetrahedronFEMForceField<TDataTypes> > ();
    }
    else
    {
        sout << "No rotation defined  !";
        return;
    }

    VecDeriv& dx = *mstate->getDx();
    for(unsigned int j = 0; j < dx.size(); j++)	forceField->getRotation(R[j], j);
}

SOFA_DECL_CLASS(PrecomputedWarpPreconditioner)

int PrecomputedWarpPreconditionerClass = core::RegisterObject("Linear system solver using the conjugate gradient iterative algorithm")
//.add< PrecomputedWarpPreconditioner<GraphScatteredMatrix,GraphScatteredVector> >(true)
//.add< PrecomputedWarpPreconditioner< CompressedRowSparseMatrix<double>, FullVector<double> > >()
//.add< PrecomputedWarpPreconditioner< SparseMatrix<double>, FullVector<double> > >()
//.add< PrecomputedWarpPreconditioner<NewMatBandMatrix,NewMatVector> >(true)
//.add< PrecomputedWarpPreconditioner<NewMatMatrix,NewMatVector> >()
//.add< PrecomputedWarpPreconditioner< NewMatSymmetricMatrix,NewMatVector> >()
//.add< PrecomputedWarpPreconditioner<NewMatSymmetricBandMatrix,NewMatVector> >()
        .add< PrecomputedWarpPreconditioner< FullMatrix<double>, FullVector<double> , defaulttype::Vec3dTypes > >(true)
        .addAlias("PrecomputedWarpPrecond")
        ;

} // namespace linearsolver

} // namespace component

} // namespace sofa

