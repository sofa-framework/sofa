/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/component/odesolver/ParallelCGImplicitSolver.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/core/ObjectFactory.h>
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

namespace odesolver
{

using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;

ParallelCGImplicitSolver::ParallelCGImplicitSolver()
    : f_maxIter( initData(&f_maxIter,(unsigned)25,"iterations","maximum number of iterations of the Conjugate Gradient solution") )
    , f_tolerance( initData(&f_tolerance,1e-5,"tolerance","desired precision of the Conjugate Gradient Solution (ratio of current residual norm over initial residual norm)") )
    , f_smallDenominatorThreshold( initData(&f_smallDenominatorThreshold,1e-5,"threshold","minimum value of the denominator in the conjugate Gradient solution") )
    , f_rayleighStiffness( initData(&f_rayleighStiffness,0.1,"rayleighStiffness","Rayleigh damping coefficient related to stiffness") )
    , f_rayleighMass( initData(&f_rayleighMass,0.1,"rayleighMass","Rayleigh damping coefficient related to mass"))
    , f_velocityDamping( initData(&f_velocityDamping,0.,"vdamping","Velocity decay coefficient (no decay if null)") )
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
ParallelCGImplicitSolver::~ParallelCGImplicitSolver()
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



void inline ParallelCGImplicitSolver::cgLoop( MultiVector &x, MultiVector &r,MultiVector& p,MultiVector &q,double &h,const bool verbose)
{



    r.dot(*rhoSh,r);

    p = r; //z;


    if ( verbose )
    {
        p.print();
    }

    // matrix-vector product
    propagateDx(p);          // dx = p

    computeDf(q);            // q = df/dx p
    //q.print();
    if ( verbose )
    {
        q.print( );
    }
    q *= -h*(h+f_rayleighStiffness.getValue());  // q = -h(h+rs) df/dx p
    if ( verbose )
    {
        //       q.print();

    }

    // apply global Rayleigh damping
    if (f_rayleighMass.getValue()==0.0)
    {
        addMdx(q); // no need to propagate p as dx again
    }
    else
    {
        addMdx(q,VecId(),(1+h*f_rayleighMass.getValue())); // no need to propagate p as dx again
    }
    if ( verbose )
    {

        q.print();

    }

    // filter the product to take the constraints into account
    //
    projectResponse(q);     // q is projected to the constrained space
    if ( verbose )
    {
        q.print();


    }


    p.dot(*denSh,q);
    Task<OperationsIteration1>(**alphaSh,**rhoSh,**denSh,**rho_1Sh,f_smallDenominatorThreshold.getValue(),**breakCondition,*normbSh);

    x.peq(p,*alphaSh);                 // x = x + alpha p


    r.meq(q,*alphaSh);                // r = r - alpha q

    if ( verbose )
    {

        Task<Print>(**denSh);
        Task<Print>(**alphaSh);

        x.print();
        r.print();

    }

//END OF THE FIRST ITERATION
    Iterative::Loop::BeginLoop(f_maxIter.getValue()-1,breakCondition);


    Task<ResetDouble>(**rhoSh);
    r.dot(*rhoSh,r);

    if ( verbose )
    {
        Task<Print>(**rhoSh/*,"rho=r.dot(r):"*/);

    }



    Task<DivBeta>(**betaSh,**rhoSh,**rho_1Sh,**breakCondition,*normbSh,f_tolerance.getValue());
    Iterative::Loop::ConditionalBreak();
    //p += r; //z;

    if ( verbose )
    {
        Task<Print>(**betaSh,"beta:");
        p.print();

    }

    v_op(p,r,p,*betaSh); // p = p*beta + r



    if ( verbose )
    {
        p.print();

    }

    // matrix-vector product
    propagateDx(p);          // dx = p
    computeDf(q);            // q = df/dx p

    if ( verbose )
    {
        q.print();

    }

    q *= -h*(h+f_rayleighStiffness.getValue());  // q = -h(h+rs) df/dx p

    if ( verbose )
    {
        q.print();
    }

    // apply global Rayleigh damping
    if (f_rayleighMass.getValue()==0.0)
    {
        addMdx(q); // no need to propagate p as dx again
    }
    else
    {
        addMdx(q,VecId(),(1+h*f_rayleighMass.getValue())); // no need to propagate p as dx again
    }

    if ( verbose )
    {
        q.print();

    }

    // filter the product to take the constraints into account
    //
    projectResponse(q);     // q is projected to the constrained space
    if ( verbose )
    {

        q.print();
    }

    p.dot(*denSh,q);

    Task<DivAlpha>(**alphaSh,**rhoSh,**denSh,**rho_1Sh,f_smallDenominatorThreshold.getValue(),**breakCondition);
    Iterative::Loop::ConditionalBreak();

    x.peq(p,*alphaSh);                 // x = x + alpha p
    r.meq(q,*alphaSh);                // r = r - alpha q


    if ( verbose )
    {
        Task<Print>(**denSh/*,"den:"*/);
        Task<Print>(**alphaSh/*,"alpha:"*/);

        x.print();
        r.print();
    }
    Iterative::Loop::EndLoop();

}

void ParallelCGImplicitSolver::solve(double dt)
{
    MultiVector pos(this, VecId::position());
    MultiVector vel(this, VecId::velocity());
    MultiVector f(this, VecId::force());
    MultiVector b(this, VecId::V_DERIV);
    MultiVector p(this, VecId::V_DERIV);
    MultiVector q(this, VecId::V_DERIV);
    MultiVector x(this, VecId::V_DERIV);

    sofa::simulation::tree::GNode *context=dynamic_cast<sofa::simulation::tree::GNode *>(getContext());
//   if (!getPartition()&&context&&!context->is_partition())
    {
        Iterative::IterativePartition *p;
        Iterative::IterativePartition *firstPartition=context->getFirstPartition();
        if (firstPartition)
        {
            //p->setCPU(firstPartition->getCPU());
            p=firstPartition;
        }
        else if (context->getPartition())
        {
            p=context->getPartition();
        }
        else
        {
            p= new Iterative::IterativePartition();
        }
        setPartition(p);
    }





    double h = dt;
    const bool printLog = f_printLog.getValue();
    const bool verbose  = f_verbose.getValue();

    addSeparateGravity(dt);	// v += dt*g . Used if mass wants to added G separately from the other forces to v.


    // compute the right-hand term of the equation system
    OPERATION_BEGIN("Compute Force");
    computeForce(b);             // b = f0

    OPERATION_END();

    //propagateDx(vel);            // dx = v
    //computeDf(f);                // f = df/dx v
    OPERATION_BEGIN("Compute DfV");
    computeDfV(f);                // f = df/dx v
    OPERATION_END();
    OPERATION_BEGIN("b = f0 + (h+rs)df/dx v");
    b.peq(f,h+f_rayleighStiffness.getValue());      // b = f0 + (h+rs)df/dx v
    OPERATION_END();

    if (f_rayleighMass.getValue() != 0.0)
    {
        //f.clear();
        //addMdx(f,vel);
        //b.peq(f,-f_rayleighMass.getValue());     // b = f0 + (h+rs)df/dx v - rd M v
        //addMdx(b,VecId(),-f_rayleighMass.getValue()); // no need to propagate vel as dx again
        OPERATION_BEGIN("add Mdx");
        addMdx(b,vel,-f_rayleighMass.getValue()); // no need to propagate vel as dx again
        OPERATION_END();
    }

    OPERATION_BEGIN("b = h(f0 + (h+rs)df/dx v - rd M v)");
    b.teq(h);                           // b = h(f0 + (h+rs)df/dx v - rd M v)
    OPERATION_END();

    if ( verbose )
        cerr<<"CGImplicitSolver, f0 = "<< b <<endl;

    OPERATION_BEGIN("Project Response");
    projectResponse(b);          // b is projected to the constrained space
    OPERATION_END();




    // -- solve the system using a conjugate gradient solution
    OPERATION_BEGIN("Reset rho");
    Task<ResetRho>(**rho_1Sh,**rhoSh,**denSh);
    OPERATION_END();



    if ( verbose )
        cerr<<"CGImplicitSolver, projected f0 = "<< b <<endl;

    v_clear( x );
    //v_eq(r,b); // initial residual
    MultiVector& r = b; // b is never used after this point

    if ( verbose )
    {
        cerr<<"CGImplicitSolver, dt = "<< dt <<endl;
        cerr<<"CGImplicitSolver, initial x = "<< pos <<endl;
        cerr<<"CGImplicitSolver, initial v = "<< vel <<endl;
        cerr<<"CGImplicitSolver, r0 = f0 = "<< b <<endl;
        //cerr<<"CGImplicitSolver, r0 = "<< r <<endl;
    }


    OPERATION_BEGIN("CG Loop");

    cgLoop( x, r,p,q,h,verbose);


    OPERATION_END();
    // apply the solution
    OPERATION_BEGIN("Compute Velocity");
    vel.peq( x );                       // vel = vel + x
    OPERATION_END();
    OPERATION_BEGIN("Compute Position");
    pos.peq( vel, h );                  // pos = pos + h vel

    OPERATION_END();
    if (f_velocityDamping.getValue()!=0.0)
        vel *= exp(-h*f_velocityDamping.getValue());

    if ( printLog )
    {
        //cerr<<"CGImplicitSolver::solve, nbiter = "<<nb_iter<<" stop because of "<<endcond<<endl;
    }
    if ( verbose )
    {
        cerr<<"CGImplicitSolver::solve, solution = "<<x<<endl;
        cerr<<"CGImplicitSolver, final x = "<< pos <<endl;
        cerr<<"CGImplicitSolver, final v = "<< vel <<endl;
    }
}

void ParallelCGImplicitSolver::propagatePositionAndVelocity(double t, VecId x, VecId v)
{
    simulation::MechanicalPropagatePositionAndVelocityVisitor(t,x,v).setTags(getTags()).execute( getContext() );
}

SOFA_DECL_CLASS(ParallelCGImplicit)

int ParallelCGImplicitSolverClass = sofa::core::RegisterObject("Implicit time integration using the filtered conjugate gradient")
        .add< ParallelCGImplicitSolver >()
        .addAlias("ParallelCGImplicit");
;

} // namespace odesolver

} // namespace component

} // namespace sofa

