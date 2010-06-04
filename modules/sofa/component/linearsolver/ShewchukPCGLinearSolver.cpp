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
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/ShewchukPCGLinearSolver.h>
#include <sofa/component/linearsolver/NewMatMatrix.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/helper/AdvancedTimer.h>

#include <sofa/core/ObjectFactory.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace linearsolver
{

using namespace sofa::defaulttype;
using namespace sofa::core::behavior;
using namespace sofa::simulation;
using namespace sofa::core::objectmodel;
using sofa::helper::system::thread::CTime;
using sofa::helper::system::thread::ctime_t;
using std::cerr;
using std::endl;

template<class TMatrix, class TVector>
ShewchukPCGLinearSolver<TMatrix,TVector>::ShewchukPCGLinearSolver()
    : f_maxIter( initData(&f_maxIter,(unsigned)25,"iterations","maximum number of iterations of the Conjugate Gradient solution") )
    , f_tolerance( initData(&f_tolerance,1e-5,"tolerance","desired precision of the Conjugate Gradient Solution (ratio of current residual norm over initial residual norm)") )
    , f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , f_update_iteration( initData(&f_update_iteration,(unsigned)0,"update_iteration","Number of CG iterations before next refresh of precondtioner") )
    , f_min_update_iteration( initData(&f_min_update_iteration,(unsigned)1,"min_update_iteration","minimum Number of iterations under the update_iteration is not updated") )
    , f_update_step( initData(&f_update_step,(unsigned)1,"update_step","Number of steps before the next refresh of precondtioners") )
    , f_use_precond( initData(&f_use_precond,true,"use_precond","Use preconditioners") )
    , f_preconditioners( initData(&f_preconditioners, "preconditioners", "If not empty: path to the solvers to use as preconditioners") )
    , f_graph( initData(&f_graph,"graph","Graph of residuals at each iteration") )
{
    f_graph.setWidget("graph");
//    f_graph.setReadOnly(true);
    usePrecond = true;
    first = true;
}

template<class TMatrix, class TVector>
void ShewchukPCGLinearSolver<TMatrix,TVector>::init()
{
    std::vector<sofa::core::behavior::LinearSolver*> solvers;
    BaseContext * c = this->getContext();

    const helper::vector<std::string>& precondNames = f_preconditioners.getValue();
    if (precondNames.empty() && f_use_precond.getValue())
    {
        c->get<sofa::core::behavior::LinearSolver>(&solvers,BaseContext::SearchDown);
    }
    else
    {
        for (unsigned int i=0; i<precondNames.size(); ++i)
        {
            sofa::core::behavior::LinearSolver* s = NULL;
            c->get(s, precondNames[i]);
            if (s) solvers.push_back(s);
            else serr << "Solver \"" << precondNames[i] << "\" not found." << sendl;
        }
    }

    for (unsigned int i=0; i<solvers.size(); ++i)
    {
        if (solvers[i] && solvers[i] != this)
        {
            this->preconditioners.push_back(solvers[i]);
        }
    }

    sout<<"Found " << this->preconditioners.size() << " preconditioners"<<sendl;

    first = true;
}

template<class TMatrix, class TVector>
void ShewchukPCGLinearSolver<TMatrix,TVector>::setSystemMBKMatrix(double mFact, double bFact, double kFact)
{
    sofa::helper::AdvancedTimer::valSet("PCG::buildMBK", 1);
    sofa::helper::AdvancedTimer::stepBegin("PCG::setSystemMBKMatrix");

    Inherit::setSystemMBKMatrix(mFact,bFact,kFact);

    sofa::helper::AdvancedTimer::stepEnd("PCG::setSystemMBKMatrix(Precond)");

    if (preconditioners.size()==0) return;

    if (first)   //We initialize all the preconditioners for the first step
    {
        for (unsigned int i=0; i<this->preconditioners.size(); ++i)
        {
            preconditioners[i]->setSystemMBKMatrix(mFact,bFact,kFact);
        }
        first = false;
        next_refresh_iteration = 1;
        next_refresh_step = 1;
    }
    else if (f_use_precond.getValue())     // We use only the first precond in the list
    {
        sofa::helper::AdvancedTimer::valSet("PCG::PrecondBuildMBK", 1);
        sofa::helper::AdvancedTimer::stepBegin("PCG::PrecondSetSystemMBKMatrix");

        if (f_update_iteration.getValue()>0)
        {
            if (next_refresh_iteration>=f_update_iteration.getValue())
            {
                printf("Update with iterations\n");
                preconditioners[0]->setSystemMBKMatrix(mFact,bFact,kFact);
                next_refresh_iteration=1;
            }
        }
        else if (f_update_step.getValue()>0)
        {
            if (next_refresh_step>=f_update_step.getValue())
            {
                printf("Update with steps\n");
                preconditioners[0]->setSystemMBKMatrix(mFact,bFact,kFact);
                next_refresh_step=1;
            }
            else
            {
                next_refresh_step++;
            }
        }
        sofa::helper::AdvancedTimer::stepEnd("PCG::PrecondSetSystemMBKMatrix");
    }


}

template<>
inline void ShewchukPCGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_beta(Vector& p, Vector& r, double beta)
{
    this->v_op(p,r,p,beta); // p = p*beta + r
}

template<>
inline void ShewchukPCGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_alpha(Vector& x, Vector& p, double alpha)
{
    x.peq(p,alpha);                 // x = x + alpha p
}

/*
template<class TMatrix, class TVector>
void ShewchukPCGLinearSolver<TMatrix,TVector>::solve (Matrix& M, Vector& x, Vector& b) {
	using std::cerr;
	using std::endl;

	Vector& p = *this->createVector();
	Vector& q = *this->createVector();
	Vector& r = *this->createVector();
	Vector& z = *this->createVector();

	const bool printLog =  this->f_printLog.getValue();
	const bool verbose  = f_verbose.getValue();

	// -- solve the system using a conjugate gradient solution
	double rho, rho_1=0, alpha, beta;

	if( verbose ) cerr<<"PCGLinearSolver, b = "<< b <<endl;

	x.clear();
	r = b; // initial residual

	double normb2 = b.dot(b);
	double normb = sqrt(normb2);
	std::map < std::string, sofa::helper::vector<double> >& graph = *f_graph.beginEdit();
	sofa::helper::vector<double>& graph_error = graph["Error"];
	graph_error.clear();
	sofa::helper::vector<double>& graph_den = graph["Denominator"];
	graph_den.clear();
	graph_error.push_back(1);
	unsigned nb_iter;
	const char* endcond = "iterations";

	for( nb_iter=1; nb_iter<=f_maxIter.getValue(); nb_iter++ ) {
		if (this->preconditioners.size()>0 && usePrecond) {
			sofa::helper::AdvancedTimer::stepBegin("PCGLinearSolver::apply Precond");

// 			for (unsigned int i=0;i<this->preconditioners.size();i++) {
// 				preconditioners[i]->setSystemLHVector(z);
// 				preconditioners[i]->setSystemRHVector(r);
// 				preconditioners[i]->solveSystem();
// 			}

			preconditioners[0]->setSystemLHVector(z);
			preconditioners[0]->setSystemRHVector(r);
			preconditioners[0]->solveSystem();

			sofa::helper::AdvancedTimer::stepEnd("PCGLinearSolver::apply Precond");
		} else {
			z = r;
		}

		sofa::helper::AdvancedTimer::stepBegin("PCGLinearSolver::solve");

		rho = r.dot(z);

		if (nb_iter>1) {
			double normr = sqrt(r.dot(r));
			double err = normr/normb;
			graph_error.push_back(err);
			if (err <= f_tolerance.getValue()) {
				endcond = "tolerance";
				break;
			}
		}

		if( nb_iter==1 ) p = z;
		else {
			beta = rho / rho_1;
			//p = p*beta + z;
			cgstep_beta(p,z,beta);
		}

		if( verbose ) cerr<<"p : "<<p<<endl;

		// matrix-vector product
		q = M*p;

		if( verbose ) cerr<<"q = M p : "<<q<<endl;

		double den = p.dot(q);

		graph_den.push_back(den);

		alpha = rho/den;
		//x.peq(p,alpha);                 // x = x + alpha p
		//r.peq(q,-alpha);                // r = r - alpha q
		cgstep_alpha(x,p,alpha);
		cgstep_alpha(r,q,-alpha);

		if( verbose ) {
			cerr<<"den = "<<den<<", alpha = "<<alpha<<endl;
			cerr<<"x : "<<x<<endl;
			cerr<<"r : "<<r<<endl;
		}

		rho_1 = rho;

		sofa::helper::AdvancedTimer::stepEnd("PCGLinearSolver::solve");
	}

	sofa::helper::AdvancedTimer::valSet("PCG iterations", nb_iter);

	f_graph.endEdit();
	// x is the solution of the system

	if( printLog ) cerr<<"PCGLinearSolver::solve, nbiter = "<<nb_iter<<" stop because of "<<endcond<<endl;

	if( verbose ) cerr<<"PCGLinearSolver::solve, solution = "<<x<<endl;

	this->deleteVector(&p);
	this->deleteVector(&q);
	this->deleteVector(&r);
	this->deleteVector(&z);
}
*/

template<class TMatrix, class TVector>
void ShewchukPCGLinearSolver<TMatrix,TVector>::solve (Matrix& M, Vector& x, Vector& b)
{
    sofa::helper::AdvancedTimer::stepBegin("PCGLinearSolver::solve");

    Vector& r = *this->createVector();
    Vector& d = *this->createVector();
    Vector& q = *this->createVector();
    Vector& s = *this->createVector();
    const bool verbose  = f_verbose.getValue();

    unsigned iter=1;
    r = M*x;

    cgstep_beta(r,b,-1);//for (int i=0; i<n; i++) r[i] = b[i] - r[i];
    if (this->preconditioners.size()>0 && usePrecond)
    {
        sofa::helper::AdvancedTimer::stepEnd("PCGLinearSolver::solve");
        sofa::helper::AdvancedTimer::stepBegin("PCGLinearSolver::apply Precond");
        preconditioners[0]->setSystemLHVector(d);
        preconditioners[0]->setSystemRHVector(r);
        preconditioners[0]->freezeSystemMatrix();
        preconditioners[0]->solveSystem();
        //Use freeze boolean to specify the preconditioner that's the fist solve of the step (for example if stepMBK is not call)
        preconditioners[0]->updateSystemMatrix();
        sofa::helper::AdvancedTimer::stepEnd("PCGLinearSolver::apply Precond");
        sofa::helper::AdvancedTimer::stepBegin("PCGLinearSolver::solve");
    }
    else
    {
        d = r;
    }


    double deltaNew = r.dot(d);
    double delta0 = deltaNew;
    double eps = f_tolerance.getValue() * f_tolerance.getValue()  * delta0;
    std::map < std::string, sofa::helper::vector<double> >& graph = * f_graph.beginEdit();
    sofa::helper::vector<double>& graph_error = graph["Error"];
    graph_error.clear();

    while ((iter <= f_maxIter.getValue()) && (deltaNew > eps))
    {
        if (verbose) printf("CG iteration %d: current L2 error vs initial error=%G\n", iter, sqrt(deltaNew/delta0));

        graph_error.push_back(deltaNew);

        q = M * d;
        double dtq = d.dot(q);
        double alpha = deltaNew / dtq;

        cgstep_alpha(x,d,alpha);//for(int i=0; i<n; i++) x[i] += alpha * d[i];

        if (iter % 50 == 0)  // periodically compute the exact residual
        {
            r = M * x;
            cgstep_beta(r,b,-1);//for (int i=0; i<n; i++) r[i] = b[i] - r[i];
        }
        else
        {
            cgstep_alpha(r,q,-alpha);//for (int i=0; i<n; i++) r[i] = r[i] - alpha * q[i];
        }

        if (this->preconditioners.size()>0 && usePrecond)
        {
            sofa::helper::AdvancedTimer::stepEnd("PCGLinearSolver::solve");
            sofa::helper::AdvancedTimer::stepBegin("PCGLinearSolver::apply Precond");
            preconditioners[0]->setSystemLHVector(s);
            preconditioners[0]->setSystemRHVector(r);
            preconditioners[0]->solveSystem();
            sofa::helper::AdvancedTimer::stepEnd("PCGLinearSolver::apply Precond");
            sofa::helper::AdvancedTimer::stepBegin("PCGLinearSolver::solve");

        }
        else
        {
            s = r;
        }


        double deltaOld = deltaNew;
        deltaNew = r.dot(s);
        double beta = deltaNew / deltaOld;

        cgstep_beta(d,s,beta);//for (int i=0; i<n; i++) d[i] = r[i] + beta * d[i];

        iter++;
    }

    if (iter>f_min_update_iteration.getValue()) next_refresh_iteration+=iter-f_min_update_iteration.getValue();
    sofa::helper::AdvancedTimer::valSet("PCG iterations", iter);

    f_graph.endEdit();
    this->deleteVector(&r);
    this->deleteVector(&q);
    this->deleteVector(&d);
    this->deleteVector(&s);
    sofa::helper::AdvancedTimer::stepEnd("PCGLinearSolver::solve");
}

SOFA_DECL_CLASS(ShewchukPCGLinearSolver)

int ShewchukPCGLinearSolverClass = core::RegisterObject("Linear system solver using the conjugate gradient iterative algorithm")
        .add< ShewchukPCGLinearSolver<GraphScatteredMatrix,GraphScatteredVector> >(true)
        ;

} // namespace linearsolver

} // namespace component

} // namespace sofa

