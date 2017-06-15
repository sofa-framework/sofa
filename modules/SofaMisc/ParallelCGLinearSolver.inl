/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_PARALLELCGLINEARSOLVER_INL
#define SOFA_COMPONENT_LINEARSOLVER_PARALLELCGLINEARSOLVER_INL
#include <SofaMisc/ParallelCGLinearSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <math.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"
using
sofa::helper::system::thread::CTime;
using
sofa::helper::system::thread::ctime_t;


using std::cerr;
using std::endl;
#define OPERATION_BEGIN(a)
#define OPERATION_END()
namespace sofa
{

namespace component
{

namespace linearsolver
{

using namespace sofa::defaulttype;
using namespace sofa::core::behavior;
using namespace sofa::core::objectmodel;

template<class TMatrix, class TVector>
ParallelCGLinearSolver<TMatrix, TVector>::ParallelCGLinearSolver()
    : f_maxIter( initData(&f_maxIter,(unsigned)25,"iterations","maximum number of iterations of the Conjugate Gradient solution") )
    , f_tolerance( initData(&f_tolerance,1e-5,"tolerance","desired precision of the Conjugate Gradient Solution (ratio of current residual norm over initial residual norm)") )
    , f_smallDenominatorThreshold( initData(&f_smallDenominatorThreshold,1e-5,"threshold","minimum value of the denominator in the conjugate Gradient solution") )
    , f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    ,rhoSh(new Shared<double>),rho_1Sh(new Shared<double>),alphaSh(new Shared<double>),betaSh(new Shared<double>),denSh(new Shared<double>),breakCondition(new Shared<bool>)
{
    //Iterative::IterativePartition *p= new Iterative::IterativePartition();
    //	setPartition(p);

    //     maxCGIter = 25;
    //     smallDenominatorThreshold = 1e-5;
    //     tolerance = 1e-5;
    //     rayleighStiffness = 0.1;
    //     rayleighMass = 0.1;
    //     velocityDamping = 0;

}

template<class TMatrix, class TVector>
ParallelCGLinearSolver<TMatrix, TVector>::~ParallelCGLinearSolver()
{
    /*	delete rhoSh;
    	delete rho_1Sh;
    	delete alphaSh;
    	delete betaSh;
    	delete denSh;
    */
//delete getPartition();
}

//Auxiliar tasks used by the cg solver
struct ResetRho
{
    void operator()(Shared_w<double> a,Shared_w<double> b,Shared_w<double> c)
    {
        a.write(0.0);
        b.write(0.0);
        c.write(0.0);

    }

};
struct ResetDouble
{
    void operator()(Shared_w<double> a)
    {
        a.write(0.0);

    }

};
struct Print
{
    void operator()(Shared_r<double> a,const std::string str)
    {
        std::cerr<<str<<a.read()<<std::endl;
    }
    void operator()(Shared_r<double> a)
    {
        std::cerr<<"print double:"<<a.read()<<" "<<&a.read()<<std::endl;
    }
};
template<class TMatrix, class TVector>
struct OperationsIteration1
{
    void operator()(Shared_w<double> alpha,Shared_r<double> rho,Shared_rw<double> _den,Shared_w<double> rho_1,double threshold,Shared_w<bool> breakCondition,Shared_w<double> normb)
    {
        double &den=_den.access();
//	if(fabs(den)<threshold)
//	std::cerr<<"BREAK fabs1:"<<fabs(den)<<std::endl;
//	std::cerr<<"CHANGE CONDITION:"<<(fabs(den)>threshold)<<std::endl;
        breakCondition.write(fabs(den)>threshold);

        alpha.write( fabs(den)>threshold ? rho.read()/den : 0.0);
        normb.write(sqrt(rho.read()));
        rho_1.write(rho.read());
        den=0.0;
    }

};

struct DivAlpha
{
    void operator()(Shared_w<double> alpha,Shared_r<double> rho,Shared_rw<double> _den,Shared_w<double> rho_1,double threshold,Shared_w<bool> breakCondition)
    {
        double &den=_den.access();
//	 if(fabs(den)<threshold)
//	std::cerr<<"BREAK fabs"<<fabs(den)<<std::endl;
//	std::cerr<<"CHANGE CONDITION:"<<(fabs(den)>threshold)<<std::endl;
        breakCondition.write(fabs(den)>threshold);
        alpha.write(rho.read()/den);
        rho_1.write(rho.read());
        den=0.0;
    }
};
struct DivBeta
{
    void operator()(Shared_w<double> a,Shared_r<double> b,Shared_r<double> c,Shared_w<bool> condition,Shared_r<double> normb,double tolerance)
    {
        a.write(b.read()/c.read());
//	std::cerr<<"b:"<<b.read()<<" normn:"<<normb.read()<<std::endl;
//	if((sqrt(b.read())/normb.read())<=tolerance)
//	std::cerr<<"BREAK tolerance:"<<sqrt(b.read())/normb.read()<<std::endl;
//	std::cerr<<"CHANGE CONDITION:"<<sqrt(b.read())/normb.read()<<std::endl;
//		std::cerr<<"tolerance:"<<sqrt(b.read())/normb.read()<<" normb: "<<normb.read()<<" rho: "<<b.read()<<std::endl;

        condition.write((sqrt(b.read())/normb.read())>=tolerance);

    }
};


template<class TMatrix, class TVector>
void ParallelCGLinearSolver<TMatrix,TVector>::solve(Matrix& M, Vector& x, Vector& b)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printComment("ConjugateGradient");
#endif

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("VectorAllocation");
#endif

    // -- solve the system using a conjugate gradient solution
//    MultiVector pos(this, VecId::position());
//    MultiVector vel(this, VecId::velocity());
//    MultiVector f(this, VecId::force());
//    MultiVector b2(this, sofa::core::V_DERIV);
    core::ExecParams* params = new core::ExecParams();
    params->setExecMode(core::ExecParams::EXEC_KAAPI);
    typename Inherit::TempVectorContainer vtmp(this, params, M, x, b);
    Vector& p = *vtmp.createTempVector();
    Vector& q = *vtmp.createTempVector();
    Vector& r = *vtmp.createTempVector();
    // MultiVector p(this, sofa::core::V_DERIV);
    // MultiVector q(this, sofa::core::V_DERIV);
    // MultiVector r(this, sofa::core::V_DERIV);
//    MultiVector x2(this, sofa::core::V_DERIV);


    OPERATION_BEGIN("Reset rho");
    BaseObject::Task<ResetRho>(**rho_1Sh,**rhoSh,**denSh);
    OPERATION_END();

    const bool printLog = this->f_printLog.getValue();
    const bool verbose = f_verbose.getValue();


    if ( verbose )
        cerr<<"CGLinearSolver, projected f0 = "<< b <<endl;

    // v_clear( x );
    x.clear();

    r = b;

    if ( verbose )
    {
        // cerr<<"CGLinearSolver, dt = "<< dt <<endl;
        cerr<<"CGLinearSolver, r0 = f0 = "<< b <<endl;
        //cerr<<"CGLinearSolver, r0 = "<< r <<endl;
    }


    OPERATION_BEGIN("CG Loop");

// BEGIN OF FIRST ITERATION
    r.dot(*rhoSh,r);

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("VectorAllocation");
#endif
#ifdef SOFA_DUMP_VISITOR_INFO
    std::ostringstream comment;
    comment << "Iteration_" << "1";
    simulation::Visitor::printNode(comment.str());
#endif

    p = r; //z;

    if ( verbose )
    {
        p.print();
    }

    q=M*p;

    p.dot(*denSh,q);
    BaseObject::Task<OperationsIteration1<TMatrix,TVector> >
    (**alphaSh,**rhoSh,**denSh,**rho_1Sh,f_smallDenominatorThreshold.getValue(),
     **breakCondition,*normbSh);

    x.peq(p,*alphaSh);                 // x = x + alpha p
    r.meq(q,*alphaSh);                // r = r - alpha q

    if ( verbose )
    {

        BaseObject::Task<Print>(**denSh, "den ");
        BaseObject::Task<Print>(**alphaSh, "alpha ");

        x.print();
        r.print();

    }

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode(comment.str());
#endif
//END OF THE FIRST ITERATION

//BEGIN LOOP
    Iterative::Loop::BeginLoop(f_maxIter.getValue()-1,breakCondition);

    BaseObject::Task<ResetDouble>(**rhoSh);
    r.dot(*rhoSh,r);

    if ( verbose )
    {
        BaseObject::Task<Print>(**rhoSh,"rho=r.dot(r):");

    }



    BaseObject::Task<DivBeta>
    (**betaSh,**rhoSh,**rho_1Sh,**breakCondition,*normbSh,f_tolerance.getValue());

    Iterative::Loop::ConditionalBreak();
    //p += r; //z;

    if ( verbose )
    {
        BaseObject::Task<Print>(**betaSh,"beta:");
        p.print();
    }

// TODO : TODO reput this : this->v_op(params /* PARAMS FIRST */, p,r,p,*betaSh); // p = p*beta + r

    // matrix-vector product
    //  	  propagateDx(p);          // dx = p
    //  	  computeDf(q);            // q = df/dx p
    q=M*p;

    if ( verbose )
    {
        q.print();

    }

//  	  projectResponse(q);     // q is projected to the constrained space
    if ( verbose )
    {

        q.print();
    }

    p.dot(*denSh,q); // den = p.dot(q)

    BaseObject::Task<DivAlpha>
    (**alphaSh,**rhoSh,**denSh,**rho_1Sh,
     f_smallDenominatorThreshold.getValue(),**breakCondition); // alpha = rho/den

    Iterative::Loop::ConditionalBreak();

    x.peq(p,*alphaSh);                 // x = x + alpha p
    r.meq(q,*alphaSh);                // r = r - alpha q


    if ( verbose )
    {
        BaseObject::Task<Print>(**denSh,"den:");
        BaseObject::Task<Print>(**alphaSh,"alpha:");

        x.print();
        r.print();
    }

    Iterative::Loop::EndLoop();


    if( printLog )
    {
//        serr<<"CGLinearSolver::solve, nbiter = "<<nb_iter<<" stop because of "<<endcond<<sendl;
    }
    if ( verbose )
    {
        cerr<<"CGLinearSolver::solve, solution = "<<x<<endl;
    }
    vtmp.deleteTempVector(&p);
    vtmp.deleteTempVector(&q);
    vtmp.deleteTempVector(&r);
}// ParallelCGLinearSolver::solve

template<class TMatrix, class TVector>
void ParallelCGLinearSolver<TMatrix,TVector>::resetSystem()
{
    Inherit::resetSystem();
}// ParallelCGLinearSolver::resetSystem

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
