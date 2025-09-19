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
#include <sofa/component/odesolver/backward/VariationalSymplecticSolver.h>

#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>


namespace sofa::component::odesolver::backward
{

using core::VecId;
using namespace sofa::defaulttype;
using namespace core::behavior;

VariationalSymplecticSolver::VariationalSymplecticSolver()
    : d_newtonError(initData(&d_newtonError, 0.01_sreal, "newtonError", "Error tolerance for Newton iterations") )
    , d_newtonSteps(initData(&d_newtonSteps, (unsigned int)5, "steps", "Maximum number of Newton steps") )
    , d_rayleighStiffness(initData(&d_rayleighStiffness, (SReal)0.0, "rayleighStiffness", "Rayleigh damping coefficient related to stiffness, > 0") )
    , d_rayleighMass(initData(&d_rayleighMass, (SReal)0.0, "rayleighMass", "Rayleigh damping coefficient related to mass, > 0"))
    , d_saveEnergyInFile(initData(&d_saveEnergyInFile, false, "saveEnergyInFile", "If kinetic and potential energies should be dumped in a CSV file at each iteration") )
    , d_explicit(initData(&d_explicit, false, "explicitIntegration", "Use explicit integration scheme") )
    , d_fileName(initData(&d_fileName, "file", "File name where kinetic and potential energies are saved in a CSV file"))
    , d_computeHamiltonian(initData(&d_computeHamiltonian, true, "computeHamiltonian", "Compute hamiltonian") )
    , d_hamiltonianEnergy(initData(&d_hamiltonianEnergy, 0.0_sreal, "hamiltonianEnergy", "hamiltonian energy") )
    , d_useIncrementalPotentialEnergy(initData(&d_useIncrementalPotentialEnergy, true, "useIncrementalPotentialEnergy", "use real potential energy, if false use approximate potential energy"))
    , d_threadSafeVisitor(initData(&d_threadSafeVisitor, false, "threadSafeVisitor", "If true, do not use realloc and free visitors in fwdInteractionForceField."))
{
    cpt=0;
}

void VariationalSymplecticSolver::init()
{
    if (!this->getTags().empty())
    {
        type::vector<core::objectmodel::BaseObject*> objs;
        this->getContext()->get<core::objectmodel::BaseObject>(&objs,this->getTags(),sofa::core::objectmodel::BaseContext::SearchDown);
        std::stringstream tmp;
        for (const auto* obj : objs)
            tmp << "  " << obj->getClassName() << ' ' << obj->getName() << msgendl;

        msg_info() << "Responsible for the following objects with tags " << this->getTags() << " :" << tmp.str();
    }
    sofa::core::behavior::OdeSolver::init();
    sofa::core::behavior::LinearSolverAccessor::init();
    energies.open((d_fileName.getValue()).c_str(), std::ios::out);
}

void VariationalSymplecticSolver::solve(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{

    sofa::simulation::common::VectorOperations vop( params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( params, this->getContext() );
    MultiVecCoord pos(&vop, core::vec_id::write_access::position );
    MultiVecDeriv f(&vop, core::vec_id::write_access::force );
    MultiVecCoord oldpos(&vop);

    MultiVecCoord x_1(&vop, xResult); // vector of final  position

    MultiVecDeriv newp(&vop);
    MultiVecDeriv vel_1(&vop, vResult); // vector of final  velocity
    MultiVecDeriv p(&vop); // vector of momentum
    // dx is no longer allocated by default (but it will be deleted automatically by the mechanical objects)
    MultiVecDeriv dx(&vop, core::vec_id::write_access::dx); dx.realloc(&vop, !d_threadSafeVisitor.getValue(), true);

    const SReal& h = dt;
    const SReal rM = d_rayleighMass.getValue();
    const SReal rK = d_rayleighStiffness.getValue();

    if (cpt == 0 || this->getContext()->getTime()==0.0)
    {
		vop.v_alloc(pID); // allocate a new vector in Mechanical Object to store the momentum
		MultiVecDeriv pInit(&vop, pID); // initialize the first value of the momentum to M*v
		pInit.clear();
		mop.addMdx(pInit,vel_1,1.0); // momentum is initialized to M*vinit (assume 0 acceleration)

        // Compute potential energy at time t=0
        SReal KineticEnergy;
        SReal potentialEnergy;
        mop.computeEnergy(KineticEnergy,potentialEnergy);

        // Compute incremental potential energy
        if (d_computeHamiltonian.getValue() && d_useIncrementalPotentialEnergy.getValue())
            m_incrementalPotentialEnergy = potentialEnergy;

		if (d_saveEnergyInFile.getValue()) {
			// header of csv file
            energies << "time,kinetic energy,potential energy, hamiltonian energy"<<std::endl;
		}
    }

	cpt++;
    MultiVecDeriv pPrevious(&vop, pID); // get previous momentum value
    p.eq(pPrevious); // set p to previous momentum

    typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
 
	if (d_explicit.getValue()) {
		mop->setImplicit(false); // this solver is explicit only

        MultiVecDeriv acc(&vop, core::vec_id::write_access::dx); acc.realloc(&vop, !d_threadSafeVisitor.getValue(), true); // dx is no longer allocated by default (but it will be deleted automatically by the mechanical objects)

		{
		    SCOPED_TIMER("ComputeForce");
		    mop.computeForce(f);
		}

	    {
		    SCOPED_TIMER("AccFromF");

	        f.peq(p,1.0/h);

		    mop.accFromF(acc, f); // acc= 1/m (f(q(k)+p(k)/h))
		    if (rM>0) {
		        MultiVecDeriv oldVel(&vop, core::vec_id::write_access::velocity );
		        // add rayleigh Mass damping if necessary
		        acc.peq(oldVel,-rM); // equivalent to adding damping force -rM* M*v(k)
		    }
	    }

		mop.projectResponse(acc);

		mop.solveConstraint(acc, core::ConstraintOrder::ACC);
		
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

		mop.solveConstraint(vel_1,core::ConstraintOrder::VEL);

	} else {

		// implicit time integration with alpha = 0.5
	    mop->setImplicit(true); // this solver is explicit only

		MultiVecCoord resi(&vop);
		MultiVecDeriv b(&vop);
		MultiVecDeriv res(&vop);
		MultiVecCoord x_0(&vop);
		unsigned int nbMaxIterNewton = d_newtonSteps.getValue();
		unsigned int i_newton=0;
		double err_newton =0; // initialisation

		oldpos.eq(pos); // save initial position
		double positionNorm=oldpos.norm(); // the norm of the position to compute the stopping criterion
		x_0.eq(pos); // this is the previous estimate of the mid-point state
		res.clear(); 
		resi.clear();
		while(((i_newton < nbMaxIterNewton)&&(err_newton > d_newtonError.getValue())) || (i_newton == 0)){

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

			{
			    SCOPED_TIMER("ComputeForce");
			    mop.computeForce(f);
			}

			sofa::helper::AdvancedTimer::stepBegin("ComputeRHTerm");

			// we have b=f(q(k,i-1)+(2/h)p(k)
			b.peq(f,1.0);
			b.peq(p,2.0/h);

			// corresponds to do b+=-K*res, where res=res(i-1)=q(k,i-1)-q(k,0)
			mop.propagateDx(res);
			mop.addMBKdx(b,core::MatricesFactors::M(0),
			    core::MatricesFactors::B(0),
			    core::MatricesFactors::K(-1.0));


			mop.projectResponse(b);
		    sofa::helper::AdvancedTimer::stepEnd("ComputeRHTerm");

			// add left term : matrix=-K+4/h^(2)M, but with dampings rK and rM
		    {
			    SCOPED_TIMER("MBKBuild");
                const core::MatricesFactors::M mFact ( 4.0 / (h * h) + 4 * rM / h );
			    const core::MatricesFactors::K kFact ( -1.0 - 4 * rK / h );
                mop.setSystemMBKMatrix(mFact, core::MatricesFactors::B(0), kFact, l_linearSolver.get());
		    }

            {
			    SCOPED_TIMER("MBKSolve");
                // resolution of matrix*res=b
			    l_linearSolver->getLinearSystem()->setSystemSolution(res);
			    l_linearSolver->getLinearSystem()->setRHS(b);
			    l_linearSolver->solveSystem();
			    l_linearSolver->getLinearSystem()->dispatchSystemSolution(res);
            }

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

			mop.propagateX(x_1);

			err_newton = resi.norm()/positionNorm; /// this should decrease
			i_newton++;

		}//end of i iterations

        // Compute incremental potential energy
        if (d_computeHamiltonian.getValue() && d_useIncrementalPotentialEnergy.getValue())
        {
            // Compute delta potential Energy
            double deltaPotentialEnergy = -(2.0)*(f.dot(res));

            // potentialEnergy is computed by adding deltaPotentialEnergy to the initial potential energy
            m_incrementalPotentialEnergy = m_incrementalPotentialEnergy + deltaPotentialEnergy;
        }

        msg_info() <<" i_newton "<<i_newton<<"    err_newton "<<err_newton ;

        /// Updates of v, p and final position ///
		//v(k+1,0)=(2/h)(q(k,i_end)-q(k,0))
		VMultiOp opsfin;
		opsfin.resize(1);
		opsfin[0].first = vel_1;
		opsfin[0].second.push_back(std::make_pair(res.id(),2.0/h));
		vop.v_multiop(opsfin);

		sofa::helper::AdvancedTimer::stepBegin("CorrectV");
		mop.solveConstraint(vel_1,core::ConstraintOrder::VEL);

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
        if (d_computeHamiltonian.getValue())
        {
            // Compute hamiltonian kinetic energy = 0.5*(newp.dot(Minv*newp))
            MultiVecDeriv b(&vop);
            b.clear();
            // Mass matrix
            mop.setSystemMBKMatrix(core::MatricesFactors::M(1), core::MatricesFactors::B(0), core::MatricesFactors::K(0), l_linearSolver.get());

            // resolution of matrix*b=newp
            l_linearSolver->getLinearSystem()->setSystemSolution(b);
            l_linearSolver->getLinearSystem()->setRHS(newp);
            l_linearSolver->solveSystem(); // b = inv(matrix)*newp = Minv*newp
            l_linearSolver->getLinearSystem()->dispatchSystemSolution(b);

            const auto hamiltonianKineticEnergy = 0.5*(newp.dot(b));

            // Hamiltonian energy with incremental potential energy
            if (d_useIncrementalPotentialEnergy.getValue())
            {
                // Hamiltonian energy
                d_hamiltonianEnergy.setValue(hamiltonianKineticEnergy + m_incrementalPotentialEnergy);

                // Write energy in file
                if (d_saveEnergyInFile.getValue())
                    energies << this->getContext()->getTime()<<","<<hamiltonianKineticEnergy<<","<<m_incrementalPotentialEnergy<<","<<hamiltonianKineticEnergy + m_incrementalPotentialEnergy <<std::endl;

            }

            // Hamiltonian energy with approximate potential energy
            else if (!d_useIncrementalPotentialEnergy.getValue())
            {
                // Compute approximate potential energy
                SReal potentialEnergy;
                SReal KineticEnergy;
                mop.computeEnergy(KineticEnergy,potentialEnergy);

                // Hamiltonian energy
                d_hamiltonianEnergy.setValue(hamiltonianKineticEnergy + potentialEnergy);

                // Write energy in file
                if (d_saveEnergyInFile.getValue())
                    energies << this->getContext()->getTime()<<","<<hamiltonianKineticEnergy<<","<<potentialEnergy<<","<<hamiltonianKineticEnergy+potentialEnergy<<std::endl;
            }
        }
	}

    sofa::helper::AdvancedTimer::stepEnd("CorrectV");
    {
        SCOPED_TIMER("CorrectX");
        mop.solveConstraint(x_1,core::ConstraintOrder::POS);
    }

	// update the previous momentum as the current one for next step
    pPrevious.eq(newp);
}

void VariationalSymplecticSolver::parse(core::objectmodel::BaseObjectDescription* arg)
{
    if (arg->getAttribute("verbose"))
    {
        msg_warning() << "Attribute 'verbose' has no use in this component. "
                         "To disable this warning, remove the attribute from the scene.";
    }

    OdeSolver::parse(arg);
}

void registerVariationalSymplecticSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Implicit time integrator which conserves linear momentum and mechanical energy.")
        .add< VariationalSymplecticSolver >());
}

} // namespace sofa::component::odesolver::backward
