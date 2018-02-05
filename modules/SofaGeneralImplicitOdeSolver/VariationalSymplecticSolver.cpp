/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
// Author: Herve Delingette, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#include "VariationalSymplecticSolver.h"
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/core/ObjectFactory.h>
#include <math.h>
#include <iostream>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/helper/AdvancedTimer.h>


namespace sofa
{

namespace component
{

namespace odesolver
{
using core::VecId;
using namespace sofa::defaulttype;
using namespace core::behavior;

VariationalSymplecticSolver::VariationalSymplecticSolver()
    : f_newtonError( initData(&f_newtonError,0.01,"newtonError","Error tolerance for Newton iterations") )
    , f_newtonSteps( initData(&f_newtonSteps,(unsigned int)5,"steps","Maximum number of Newton steps") )
    , f_rayleighStiffness( initData(&f_rayleighStiffness,(SReal)0.0,"rayleighStiffness","Rayleigh damping coefficient related to stiffness, > 0") )
    , f_rayleighMass( initData(&f_rayleighMass,(SReal)0.0,"rayleighMass","Rayleigh damping coefficient related to mass, > 0"))
    , f_verbose( initData(&f_verbose,false,"verbose","Dump information on the residual errors and number of Newton iterations") )
    , f_saveEnergyInFile( initData(&f_saveEnergyInFile,false,"saveEnergyInFile","If kinetic and potential energies should be dumped in a CSV file at each iteration") )
    , f_explicit( initData(&f_explicit,false,"explicitIntegration","Use explicit integration scheme") )
    , f_fileName(initData(&f_fileName,"file","File name where kinetic and potential energies are saved in a CSV file"))
    , f_computeHamiltonian( initData(&f_computeHamiltonian,true,"computeHamiltonian","Compute hamiltonian") )
    , f_hamiltonianEnergy( initData(&f_hamiltonianEnergy,0.0,"hamiltonianEnergy","hamiltonian energy") )
    , f_useIncrementalPotentialEnergy( initData(&f_useIncrementalPotentialEnergy,true,"use incremental potential Energy","use real potential energy, if false use approximate potential energy"))
{
    cpt=0;
}

void VariationalSymplecticSolver::init()
{
    if (!this->getTags().empty())
    {
        sout << "VariationalSymplecticSolver: responsible for the following objects with tags " << this->getTags() << " :" << sendl;
        helper::vector<core::objectmodel::BaseObject*> objs;
        this->getContext()->get<core::objectmodel::BaseObject>(&objs,this->getTags(),sofa::core::objectmodel::BaseContext::SearchDown);
        for (unsigned int i=0;i<objs.size();++i)
            sout << "  " << objs[i]->getClassName() << ' ' << objs[i]->getName() << sendl;
    }
    sofa::core::behavior::OdeSolver::init();
    energies.open((f_fileName.getValue()).c_str(),std::ios::out);
}

void VariationalSymplecticSolver::solve(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{

    sofa::simulation::common::VectorOperations vop( params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( params, this->getContext() );
    MultiVecCoord pos(&vop, core::VecCoordId::position() );
    MultiVecDeriv f(&vop, core::VecDerivId::force() );
    MultiVecCoord oldpos(&vop);

    MultiVecCoord x_1(&vop, xResult); // vector of final  position

    MultiVecDeriv newp(&vop);
    MultiVecDeriv vel_1(&vop, vResult); // vector of final  velocity
    MultiVecDeriv p(&vop); // vector of momemtum
    // dx is no longer allocated by default (but it will be deleted automatically by the mechanical objects)
    MultiVecDeriv dx(&vop, core::VecDerivId::dx() ); dx.realloc( &vop, true, true );

    const SReal& h = dt;
    const SReal rM = f_rayleighMass.getValue();
    const SReal rK = f_rayleighStiffness.getValue();
    const bool verbose  = f_verbose.getValue();

    if (cpt == 0 || this->getContext()->getTime()==0.0)
    {
		vop.v_alloc(pID); // allocate a new vector in Mechanical Object to store the momemtum
		MultiVecDeriv pInit(&vop, pID); // initialize the first value of the momemtum to M*v
		pInit.clear();
		mop.addMdx(pInit,vel_1,1.0); // momemtum is initialized to M*vinit (assume 0 acceleration)

        // Compute potential energy at time t=0
        SReal KineticEnergy;
        SReal potentialEnergy;
        mop.computeEnergy(KineticEnergy,potentialEnergy);

        // Compute incremental potential energy
        if (f_computeHamiltonian.getValue() && f_useIncrementalPotentialEnergy.getValue())
            m_incrementalPotentialEnergy = potentialEnergy;

		if (f_saveEnergyInFile.getValue()) {
			// header of csv file
            energies << "time,kinetic energy,potential energy, hamiltonian energy"<<std::endl;
		}
    }

	cpt++;
    MultiVecDeriv pPrevious(&vop, pID); // get previous momemtum value
    p.eq(pPrevious); // set p to previous momemtum


    typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
 
	if (f_explicit.getValue()) {
		mop->setImplicit(false); // this solver is explicit only

		MultiVecDeriv acc(&vop, core::VecDerivId::dx() ); acc.realloc( &vop, true, true ); // dx is no longer allocated by default (but it will be deleted automatically by the mechanical objects)

		sofa::helper::AdvancedTimer::stepBegin("ComputeForce");
		mop.computeForce(f);
		sofa::helper::AdvancedTimer::stepEnd("ComputeForce");

		sofa::helper::AdvancedTimer::stepBegin("AccFromF");
		f.peq(p,1.0/h); 

		mop.accFromF(acc, f); // acc= 1/m (f(q(k)+p(k)/h))
		if (rM>0) {
			MultiVecDeriv oldVel(&vop, core::VecDerivId::velocity() );
			// add rayleigh Mass damping if necessary
			acc.peq(oldVel,-rM); // equivalent to adding damping force -rM* M*v(k) 
		}
		sofa::helper::AdvancedTimer::stepEnd("AccFromF");
		mop.projectResponse(acc);

		mop.solveConstraint(acc, core::ConstraintParams::ACC);
		
        VMultiOp ops;
        ops.resize(2);
        // change order of operations depending on the symplectic flag
        ops[0].first = vel_1;
        ops[0].second.push_back(std::make_pair(acc.id(),h)); // v(k+1) = 1/m (h*f(q(k)+p(k))
        ops[1].first = x_1; // q(k+1)=q(k)+h*v(k+1)
        ops[1].second.push_back(std::make_pair(pos.id(),1.0));
        ops[1].second.push_back(std::make_pair(vel_1.id(),h));
		vop.v_multiop(ops);

		// p(k+1)=M v(k+1)
		newp.clear();
		mop.addMdx(newp,vel_1,1.0);

		mop.solveConstraint(vel_1,core::ConstraintParams::VEL);

	} else {

		// implicit time integration with alpha = 0.5
	    mop->setImplicit(true); // this solver is explicit only

		MultiVecCoord resi(&vop);
		MultiVecDeriv b(&vop);
		MultiVecDeriv res(&vop);
		MultiVecCoord x_0(&vop);
		unsigned int nbMaxIterNewton = f_newtonSteps.getValue();
		unsigned int i_newton=0;
		double err_newton =0; // initialisation
		//    MultiVecDeriv F(&vop);

		oldpos.eq(pos); // save initial position
		double positionNorm=oldpos.norm(); // the norm of the position to compute the stopping criterion
		x_0.eq(pos); // this is the previous estimate of the mid-point state
		res.clear(); 
		resi.clear();
		while(((i_newton < nbMaxIterNewton)&&(err_newton>f_newtonError.getValue()))||(i_newton==0)){

			b.clear();
			f.clear();

			/// Solving minimization of Lilyan energy (grad=0): ///
			// q(k,i) is the estimate of the mid point position at iteration k and newton step i
			// q(k,0)= qk i.e. the position at iteration k
			// the position increment res is searched such that q(k,i+1)=q(k,i)+res
			// res must minimize the Lilyan, and is solution of a linearized system around previous estimate 
			// Equation is : matrix * res =  b 
			// where b = f(q(k,i-1)) -K(q(k,i-1)) res(i-1) +(2/h)p^(k)
			// and matrix=-K+4/h^(2)M

			sofa::helper::AdvancedTimer::stepBegin("ComputeForce");
            mop.computeForce(f);

			sofa::helper::AdvancedTimer::stepNext ("ComputeForce", "ComputeRHTerm");

			// we have b=f(q(k,i-1)+(2/h)p(k)
			b.peq(f,1.0);
			b.peq(p,2.0/h);

			// corresponds to do b+=-K*res, where res=res(i-1)=q(k,i-1)-q(k,0)
			mop.propagateDx(res);
			mop.addMBKdx(b,0,0,-1.0);


			mop.projectResponse(b);
			// add left term : matrix=-K+4/h^(2)M, but with dampings rK and rM
            core::behavior::MultiMatrix<simulation::common::MechanicalOperations> matrix(&mop);
//			matrix = MechanicalMatrix::K * (-1.0-rK/h) +  MechanicalMatrix::M * (4.0/(h*h)+rM);
			matrix = MechanicalMatrix::K * (-1.0-4*rK/h) +  MechanicalMatrix::M * (4.0/(h*h)+4*rM/h);

			sofa::helper::AdvancedTimer::stepNext ("MBKBuild", "MBKSolve");

			// resolution of matrix*res=b
			matrix.solve(res,b); //Call to ODE resolution.

			sofa::helper::AdvancedTimer::stepEnd  ("MBKSolve");

			/// Updates of q(k,i) ///
			VMultiOp ops;
			ops.resize(3);

			//x_1=q(k,i)=res(i)+q(k,0)=res(i)+oldpos
			ops[0].first = x_1;
			ops[0].second.push_back(std::make_pair(oldpos.id(),1.0));
			ops[0].second.push_back(std::make_pair(res.id(),1.0));
			// resi=q(k,i)-q(k,i-1)  save the residual as a stopping criterion
			ops[1].first = resi;
			ops[1].second.push_back(std::make_pair(x_1.id(),1.0));
			ops[1].second.push_back(std::make_pair(x_0.id(),-1.0));
			// q(k,i)=q(k,i-1) 
			ops[2].first = x_0; 
			ops[2].second.push_back(std::make_pair(x_1.id(),1.0));
			vop.v_multiop(ops);

			err_newton = resi.norm()/positionNorm; /// this should decrease
			i_newton++;

		}//end of i iterations

        // Compute incremental potential energy
        if (f_computeHamiltonian.getValue() && f_useIncrementalPotentialEnergy.getValue())
        {
            // Compute delta potential Energy
            double deltaPotentialEnergy = -(2.0)*(f.dot(res));

            // potentialEnergy is computed by adding deltaPotentialEnergy to the initial potential energy
            m_incrementalPotentialEnergy = m_incrementalPotentialEnergy + deltaPotentialEnergy;
        }

		if (verbose) 
			std::cout<<" i_newton "<<i_newton<<"    err_newton "<<err_newton<<std::endl;
		/// Updates of v, p and final position ///
		//v(k+1,0)=(2/h)(q(k,i_end)-q(k,0))
		VMultiOp opsfin;
		opsfin.resize(1);
		opsfin[0].first = vel_1;
		opsfin[0].second.push_back(std::make_pair(res.id(),2.0/h));
		vop.v_multiop(opsfin);

		sofa::helper::AdvancedTimer::stepBegin("UpdateVAndX");

        sofa::helper::AdvancedTimer::stepNext ("UpdateVAndX", "CorrectV");
		mop.solveConstraint(vel_1,core::ConstraintParams::VEL);

		// update position
		VMultiOp opsx;

		// here x_1=q(k,0)+(h/2)v(k+1,0)
		opsx.resize(1);
		opsx[0].first = x_1;
		opsx[0].second.push_back(std::make_pair(oldpos.id(),1.0));
		opsx[0].second.push_back(std::make_pair(vel_1.id(),h/2));
		vop.v_multiop(opsx);

		// get p(k+1)= f(q(k,0)+(h/2)v(k+1,0))*h/2 + M*v(k+1)
		mop.computeForce(f);
        mop.projectResponse(f);
		newp.clear();
		newp.peq(f,h/2.0);
		mop.addMdx(newp,vel_1,1.0);

		// adding (h/2)v(k+1,0) so that x_1=q(k+1,0)=q(k,0)+h*v(k+1,0)
		opsx[0].second.push_back(std::make_pair(vel_1.id(),h/2));
		vop.v_multiop(opsx);

        // Compute hamiltonian energy
        if (f_computeHamiltonian.getValue())
        {
            // Compute hamiltonian kinetic energy = 0.5*(newp.dot(Minv*newp))
            MultiVecDeriv b(&vop);
            b.clear();
            core::behavior::MultiMatrix<simulation::common::MechanicalOperations> matrix(&mop);
            // Mass matrix
            matrix = MechanicalMatrix::M;

            // resolution of matrix*b=newp
            matrix.solve(b,newp); // b = inv(matrix)*newp = Minv*newp

            double hamiltonianKineticEnergy = 0.5*(newp.dot(b));

            // Hamiltonian energy with incremental potential energy
            if (f_useIncrementalPotentialEnergy.getValue())
            {
                // Hamiltonian energy
                f_hamiltonianEnergy.setValue(hamiltonianKineticEnergy + m_incrementalPotentialEnergy);

                // Write energy in file
                if (f_saveEnergyInFile.getValue())
                    energies << this->getContext()->getTime()<<","<<hamiltonianKineticEnergy<<","<<m_incrementalPotentialEnergy<<","<<hamiltonianKineticEnergy + m_incrementalPotentialEnergy <<std::endl;

            }

            // Hamiltonian energy with approximate potential energy
            else if (!f_useIncrementalPotentialEnergy.getValue())
            {
                // Compute approximate potential energy
                SReal potentialEnergy;
                SReal KineticEnergy;
                mop.computeEnergy(KineticEnergy,potentialEnergy);

                // Hamiltonian energy
                f_hamiltonianEnergy.setValue(hamiltonianKineticEnergy + potentialEnergy);

                // Write energy in file
                if (f_saveEnergyInFile.getValue())
                    energies << this->getContext()->getTime()<<","<<hamiltonianKineticEnergy<<","<<potentialEnergy<<","<<hamiltonianKineticEnergy+potentialEnergy<<std::endl;
            }



        }

	}

    sofa::helper::AdvancedTimer::stepNext ("CorrectV", "CorrectX");
    mop.solveConstraint(x_1,core::ConstraintParams::POS);
    sofa::helper::AdvancedTimer::stepEnd  ("CorrectX");

	// update the previous momemtum as the current one for next step
    pPrevious.eq(newp);
}

SOFA_DECL_CLASS(VariationalSymplecticSolver)

int VariationalSymplecticSolverClass = core::RegisterObject("Implicit time integrator which conserves linear momentum and mechanical energy")
        .add< VariationalSymplecticSolver >()
        .addAlias("VariationalSolver")
        ;

} // namespace odesolver

} // namespace component

} // namespace sofa


