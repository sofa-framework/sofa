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
#include "PrecomputedLinearSolver.h"
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
#include <sofa/gpu/cuda/CudaTetrahedronFEMForceField.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/linearsolver/MatrixLinearSolver.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/component/container/RotationFinder.h>
#include <sofa/core/componentmodel/behavior/LinearSolver.h>

#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/component/linearsolver/PCGLinearSolver.h>

#ifdef SOFA_HAVE_CSPARSE
#include <sofa/component/linearsolver/SparseCholeskySolver.h>
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

using namespace sofa::component::odesolver;
using namespace sofa::component::linearsolver;

template<class TMatrix,class TVector>
PrecomputedLinearSolver<TMatrix,TVector>::PrecomputedLinearSolver()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , use_file( initData(&use_file,true,"use_file","Dump system matrix in a file") )
    , solverName(initData(&solverName, std::string(""), "solverName", "Name of the solver to use to precompute the first matrix"))
    , init_MaxIter( initData(&init_MaxIter,5000,"init_MaxIter","Max Iter use to precompute the first matrix") )
    , init_Tolerance( initData(&init_Tolerance,1e-20,"init_Tolerance","Tolerance use to precompute the first matrix") )
    , init_Threshold( initData(&init_Threshold,1e-35,"init_Threshold","Threshold use to precompute the first matrix") )
    , f_graph( initData(&f_graph,"graph","Graph of residuals at each iteration") )

{
    f_graph.setWidget("graph");
    f_graph.setReadOnly(true);
    first = true;
}

template<class TMatrix,class TVector>
void PrecomputedLinearSolver<TMatrix,TVector>::setSystemMBKMatrix(double mFact, double bFact, double kFact)
{
    // Update the matrix only the first time

    if (first)
    {
        init_mFact = mFact;
        init_bFact = bFact;
        init_kFact = kFact;
        Inherit::setSystemMBKMatrix(mFact,bFact,kFact);
        loadMatrix();
        first = false;
    }
}

//Solve x = R * M^-1 * R^t * b
template<class TMatrix,class TVector>
void PrecomputedLinearSolver<TMatrix,TVector>::solve (TMatrix& /*M*/, TVector& z, TVector& r)
{
    z = *this->systemMatrix * r;
}

template<class TMatrix,class TVector>
void PrecomputedLinearSolver<TMatrix,TVector>::loadMatrix()
{
    systemSyze = this->systemMatrix->rowSize();
    dt = this->getContext()->getDt();

    EulerImplicitSolver* EulerSolver;
    this->getContext()->get(EulerSolver);
    factInt = 1.0; // christian : it is not a compliance... but an admittance that is computed !
    if (EulerSolver) factInt = EulerSolver->getPositionIntegrationFactor(); // here, we compute a compliance

    std::stringstream ss;
    ss << this->getContext()->getName() << "-" << systemSyze << "-" << dt << ".comp";
    std::ifstream compFileIn(ss.str().c_str(), std::ifstream::binary);

    if(compFileIn.good() && use_file.getValue())
    {
        cout << "file open : " << ss.str() << " compliance being loaded" << endl;
        compFileIn.read((char*) (*this->systemMatrix)[0], systemSyze * systemSyze * sizeof(Real));
        compFileIn.close();
    }
    else
    {
        if (solverName.getValue().empty())
        {
            loadMatrixCSparse();
        }
        else
        {

        }

        if (use_file.getValue())
        {
            std::ofstream compFileOut(ss.str().c_str(), std::fstream::out | std::fstream::binary);
            compFileOut.write((char*)(*this->systemMatrix)[0], systemSyze * systemSyze*sizeof(Real));
            compFileOut.close();
        }
    }

    for (unsigned int j=0; j<systemSyze; j++)
    {
        for (unsigned i=0; i<systemSyze; i++)
        {
            this->systemMatrix->set(j,i,this->systemMatrix->element(j,i)/factInt);
        }
    }
}


template<class TMatrix,class TVector>
void PrecomputedLinearSolver<TMatrix,TVector>::loadMatrixCSparse()
{
#ifdef SOFA_HAVE_CSPARSE
    cout << "Compute the initial invert matrix with CS_PARSE" << endl;

    CompressedRowSparseMatrix<double> matSolv;
    FullVector<double> r;
    FullVector<double> b;

    matSolv.resize(systemSyze,systemSyze);
    r.resize(systemSyze);
    b.resize(systemSyze);
    SparseCholeskySolver<CompressedRowSparseMatrix<double>, FullVector<double> > solver;

    for (unsigned int j=0; j<systemSyze; j++)
    {
        for (unsigned int i=0; i<systemSyze; i++)
        {
            if (this->systemMatrix->element(j,i)!=0) matSolv.set(j,i,(double)this->systemMatrix->element(j,i));
        }
        b.set(j,0.0);
    }

    solver.invert(matSolv);

    for (unsigned int j=0; j<systemSyze; j++)
    {
        if (j>0) b.set(j-1,0.0);
        b.set(j,1.0);

        solver.solve(matSolv,r,b);
        for (unsigned int i=0; i<systemSyze; i++)
        {
            this->systemMatrix->set(j,i,r.element(i) * factInt);
        }
    }

#else
    loadMatrixCG();
#endif
}

template<class TMatrix,class TVector>
void PrecomputedLinearSolver<TMatrix,TVector>::loadMatrixDirectSolver()
{
    cout << "Compute the initial invert matrix with " << solverName.getValue() << endl;

// 	CompressedRowSparseMatrix<double> matSolv;
// 	FullVector<double> r;
// 	FullVector<double> b;
//
// 	matSolv.resize(systemSyze,systemSyze);
// 	r.resize(systemSyze);
// 	b.resize(systemSyze);
// 	SparseCholeskySolver<CompressedRowSparseMatrix<double>, FullVector<double> > solver;
//
// 	for (unsigned int j=0; j<systemSyze; j++) {
// 		for (unsigned int i=0; i<systemSyze; i++) {
// 			if (this->systemMatrix->element(j,i)!=0) matSolv.set(j,i,(double)this->systemMatrix->element(j,i));
// 		}
// 		b.set(j,0.0);
// 	}
//
// 	solver.invert(matSolv);
//
// 	for (unsigned int j=0; j<systemSyze; j++) {
// 		if (j>0) b.set(j-1,0.0);
// 		b.set(j,1.0);
//
// 		solver.solve(matSolv,r,b);
// 		for (unsigned int i=0; i<systemSyze; i++) {
// 			  this->systemMatrix->set(j,i,r.element(i) * factInt);
// 		}
// 	}
}

template<class TMatrix,class TVector>
void PrecomputedLinearSolver<TMatrix,TVector>::loadMatrixCG()
{

// 	behavior::MechanicalState<DataTypes>* mstate = dynamic_cast< behavior::MechanicalState<DataTypes>* >(this->getContext()->getMechanicalState());
// 	if (mstate==NULL) {
// 		serr << "CudaPrecomputedWarpPreconditioner can't find Mstate" << sendl;
// 		return;
// 	}
// 	const VecDeriv& v0 = *mstate->getV();
// 	unsigned dof_on_node = v0[0].size();
// 	unsigned nbNodes = v0.size();

    EulerImplicitSolver* EulerSolver;
    this->getContext()->get(EulerSolver);
    double factInt = 1.0; // christian : it is not a compliance... but an admittance that is computed !
    if (EulerSolver) factInt = EulerSolver->getPositionIntegrationFactor(); // here, we compute a compliance

    cout << "Compute the initial invert matrix" << endl;

// 		// for the initial computation, the gravity has to be put at 0
// 		const Vec3d gravity = this->getContext()->getGravityInWorld();
// 		const Vec3d gravity_zero(0.0,0.0,0.0);
// 		this->getContext()->setGravityInWorld(gravity_zero);
//
// 		PCGLinearSolver<GraphScatteredMatrix,GraphScatteredVector>* PCGlinearSolver;
// 		CGLinearSolver<GraphScatteredMatrix,GraphScatteredVector>* CGlinearSolver;
// 		core::componentmodel::behavior::LinearSolver* linearSolver;

// 		if (solverName.getValue().empty()) {
// 	            this->getContext()->get(CGlinearSolver);
// 	            this->getContext()->get(PCGlinearSolver);
// 	            this->getContext()->get(linearSolver);
// 		} else {
// 		    core::objectmodel::BaseObject* ptr = NULL;
// 		    this->getContext()->get(ptr, solverName.getValue());
// 		    PCGlinearSolver = dynamic_cast<PCGLinearSolver<GraphScatteredMatrix,GraphScatteredVector>*>(ptr);
// 		    CGlinearSolver = dynamic_cast<CGLinearSolver<GraphScatteredMatrix,GraphScatteredVector>*>(ptr);
// 		    linearSolver = dynamic_cast<core::componentmodel::behavior::LinearSolver*>(ptr);
// 		}

// 		if(EulerSolver && CGlinearSolver)
// 			sout << "use EulerImplicitSolver &  CGLinearSolver" << sendl;
// 		else if(EulerSolver && PCGlinearSolver)
// 			sout << "use EulerImplicitSolver &  PCGLinearSolver" << sendl;
// 		else if(EulerSolver && linearSolver)
// 			sout << "use EulerImplicitSolver &  LinearSolver" << sendl;
// 		else if(EulerSolver) {
// 			sout << "use EulerImplicitSolver" << sendl;
// 		} else {
// 			serr<<"PrecomputedContactCorrection must be associated with EulerImplicitSolver+LinearSolver for the precomputation\nNo Precomputation" << sendl;
// 			return;
// 		}
// 		VecId lhId = core::componentmodel::behavior::BaseMechanicalState::VecId::velocity();
// 		VecId rhId = core::componentmodel::behavior::BaseMechanicalState::VecId::force();
//         if (!init_bw.getValue())
//         {
//             mstate->vAvail(lhId);
//             mstate->vAlloc(lhId);
//             mstate->vAvail(rhId);
//             mstate->vAlloc(rhId);
//             std::cout << "System: (" << init_mFact << " * M + " << init_bFact << " * B + " << init_kFact << " * K) " << lhId << " = " << rhId << std::endl;
//             if (linearSolver)
//             {
//                 std::cout << "System Init Solver: " << linearSolver->getName() << " (" << linearSolver->getClassName() << ")" << std::endl;
//                 linearSolver->setSystemMBKMatrix(init_mFact, init_bFact, init_kFact);
//             }
//         }
// 		VecDeriv& force = init_bw.getValue() ? *mstate->getExternalForces() : *mstate->getVecDeriv(rhId.index);
// 		force.clear();
// 		force.resize(nbNodes);
//
// 		///////////////////////// CHANGE THE PARAMETERS OF THE SOLVER /////////////////////////////////
// 		double buf_tolerance=0, buf_threshold=0;
// 		int buf_maxIter=0;
// 		if(CGlinearSolver) {
// 			buf_tolerance = (double) CGlinearSolver->f_tolerance.getValue();
// 			buf_maxIter   = (int) CGlinearSolver->f_maxIter.getValue();
// 			buf_threshold = (double) CGlinearSolver->f_smallDenominatorThreshold.getValue();
// 			CGlinearSolver->f_tolerance.setValue(init_Tolerance.getValue());
// 			CGlinearSolver->f_maxIter.setValue(init_MaxIter.getValue());
// 			CGlinearSolver->f_smallDenominatorThreshold.setValue(init_Threshold.getValue());
// 		} else if(PCGlinearSolver) {
// 			buf_tolerance = (double) PCGlinearSolver->f_tolerance.getValue();
// 			buf_maxIter   = (int) PCGlinearSolver->f_maxIter.getValue();
// 			buf_threshold = (double) PCGlinearSolver->f_smallDenominatorThreshold.getValue();
// 			PCGlinearSolver->f_tolerance.setValue(init_Tolerance.getValue());
// 			PCGlinearSolver->f_maxIter.setValue(init_MaxIter.getValue());
// 			PCGlinearSolver->f_smallDenominatorThreshold.setValue(init_Threshold.getValue());
// 		}
// 		///////////////////////////////////////////////////////////////////////////////////////////////
//
// 		VecDeriv& velocity = init_bw.getValue() ? *mstate->getV() : *mstate->getVecDeriv(lhId.index);
// 		VecDeriv velocity0 = velocity;
// 		VecCoord& pos = *mstate->getX();
// 		VecCoord pos0 = pos;
//
// 		for(unsigned int f = 0 ; f < nbNodes ; f++) {
// 			std::cout.precision(2);
// 			std::cout << "Precomputing constraint correction : " << std::fixed << (float)f/(float)nbNodes*100.0f << " %   " << '\xd';
// 			std::cout.flush();
// 			Deriv unitary_force;
//
// 			for (unsigned int i=0; i<dof_on_node; i++) {
// 				unitary_force.clear();
// 				unitary_force[i]=1.0;
// 				force[f] = unitary_force;
//
// 				velocity.clear();
// 				velocity.resize(nbNodes);
//                 if (init_bw.getValue())
//                     for (unsigned int n=0; n<nbNodes; n++) pos[n] = pos0[n];
//
// 				if(f*dof_on_node+i <2 ) {
// 					EulerSolver->f_verbose.setValue(true);
// 					EulerSolver->f_printLog.setValue(true);
// 					serr<<"getF : "<<force<<sendl;
// 				}
//
// 				if(EulerSolver && init_bw.getValue()){
// 					EulerSolver->solve(dt, core::componentmodel::behavior::BaseMechanicalState::VecId::position(), core::componentmodel::behavior::BaseMechanicalState::VecId::velocity());
// 				}
//                 else if (linearSolver && !init_bw.getValue())
//                 {
//                     linearSolver->setSystemRHVector(rhId);
//                     linearSolver->setSystemLHVector(lhId);
//                     linearSolver->solveSystem();
//                 }
//                 if (linearSolver && f*dof_on_node+i == 0) linearSolver->freezeSystemMatrix(); // do not recompute the matrix for the rest of the precomputation
//
// 				//velocity = *mstate->getV();
// 				double fact = factInt / unitary_force[i];
//
// 				if(f*dof_on_node+i < 2) {
// 					EulerSolver->f_verbose.setValue(false);
// 					EulerSolver->f_printLog.setValue(false);
// 					serr<<"getV : "<<velocity<<sendl;
// 				}
// 				for (unsigned int v=0; v<nbNodes; v++) {
// 					for (unsigned int j=0; j<dof_on_node; j++) {
// 						this->systemMatrix->set(v*dof_on_node+j,f*dof_on_node+i,(Real)(fact * velocity[v][j]));
// 					}
// 				}
//             }
//             unitary_force.clear();
//             force[f] = unitary_force;
// 		}

// 		///////////////////////////////////////////////////////////////////////////////////////////////
// 		if (linearSolver) linearSolver->updateSystemMatrix(); // do not recompute the matrix for the rest of the precomputation
//
// 		///////////////////////// RESET PARAMETERS AT THEIR PREVIOUS VALUE /////////////////////////////////
// 		// gravity is reset at its previous value
// 		this->getContext()->setGravityInWorld(gravity);
//
// 		if(CGlinearSolver) {
// 			CGlinearSolver->f_tolerance.setValue(buf_tolerance);
// 			CGlinearSolver->f_maxIter.setValue(buf_maxIter);
// 			CGlinearSolver->f_smallDenominatorThreshold.setValue(buf_threshold);
// 		} else if(PCGlinearSolver) {
// 			PCGlinearSolver->f_tolerance.setValue(buf_tolerance);
// 			PCGlinearSolver->f_maxIter.setValue(buf_maxIter);
// 			PCGlinearSolver->f_smallDenominatorThreshold.setValue(buf_threshold);
// 		}
//
// 		//Reset the velocity
// 		for (unsigned int i=0; i<velocity0.size(); i++) velocity[i]=velocity0[i];
// 		//Reset the position
// 		for (unsigned int i=0; i<pos0.size(); i++) pos[i]=pos0[i];
//
//         if (!init_bw.getValue())
//         {
//             mstate->vFree(lhId);
//             mstate->vFree(rhId);
//         }
}

template<class TMatrix,class TVector>
void PrecomputedLinearSolver<TMatrix,TVector>::invert(TMatrix& /*M*/) {}



SOFA_DECL_CLASS(PrecomputedLinearSolver)

int PrecomputedLinearSolverClass = core::RegisterObject("linearSolveur M0inv to solve")
#ifndef SOFA_FLOAT
        .add< PrecomputedLinearSolver< FullMatrix<double> , FullVector<double> > >(true)
//#else
//.add< PrecomputedLinearSolver< FullMatrix<float> , FullVector<float> > >(true)
#endif
        ;

} // namespace linearsolver

} // namespace component

} // namespace sofa

