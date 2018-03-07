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
// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>
#include <sofa/core/visual/VisualParams.h>
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

EulerImplicitSolver::EulerImplicitSolver()
    : f_rayleighStiffness( initData(&f_rayleighStiffness,(SReal)0.0,"rayleighStiffness","Rayleigh damping coefficient related to stiffness, > 0") )
    , f_rayleighMass( initData(&f_rayleighMass,(SReal)0.0,"rayleighMass","Rayleigh damping coefficient related to mass, > 0"))
    , f_velocityDamping( initData(&f_velocityDamping,(SReal)0.0,"vdamping","Velocity decay coefficient (no decay if null)") )
    , f_firstOrder (initData(&f_firstOrder, false, "firstOrder", "Use backward Euler scheme for first order ode system."))
    , f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , d_trapezoidalScheme( initData(&d_trapezoidalScheme,false,"trapezoidalScheme","Optional: use the trapezoidal scheme instead of the implicit Euler scheme and get second order accuracy in time") )
    , f_solveConstraint( initData(&f_solveConstraint,false,"solveConstraint","Apply ConstraintSolver (requires a ConstraintSolver in the same node as this solver, disabled by by default for now)") )
{
}

void EulerImplicitSolver::init()
{
    if (!this->getTags().empty())
    {
        sout << "EulerImplicitSolver: responsible for the following objects with tags " << this->getTags() << " :" << sendl;
        helper::vector<core::objectmodel::BaseObject*> objs;
        this->getContext()->get<core::objectmodel::BaseObject>(&objs,this->getTags(),sofa::core::objectmodel::BaseContext::SearchDown);
        for (unsigned int i=0; i<objs.size(); ++i)
            sout << "  " << objs[i]->getClassName() << ' ' << objs[i]->getName() << sendl;
    }
    sofa::core::behavior::OdeSolver::init();
}

void EulerImplicitSolver::cleanup()
{
    // free the locally created vector x (including eventual external mechanical states linked by an InteractionForceField)
    sofa::simulation::common::VectorOperations vop( core::ExecParams::defaultInstance(), this->getContext() );
    vop.v_free( x.id(), true, true );
}

void EulerImplicitSolver::solve(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::printNode("SolverVectorAllocation");
#endif
    sofa::simulation::common::VectorOperations vop( params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( params, this->getContext() );
    MultiVecCoord pos(&vop, core::VecCoordId::position() );
    MultiVecDeriv vel(&vop, core::VecDerivId::velocity() );
    MultiVecDeriv f(&vop, core::VecDerivId::force() );
    MultiVecDeriv b(&vop);
    MultiVecCoord newPos(&vop, xResult );
    MultiVecDeriv newVel(&vop, vResult );

    /// inform the constraint parameters about the position and velocity id
    mop.cparams.setX(xResult);
    mop.cparams.setV(vResult);

    // dx is no longer allocated by default (but it will be deleted automatically by the mechanical objects)
    MultiVecDeriv dx(&vop, core::VecDerivId::dx() ); dx.realloc( &vop, true, true );

    x.realloc( &vop, true, true );


#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::printCloseNode("SolverVectorAllocation");
#endif


    const SReal& h = dt;
    const bool verbose  = f_verbose.getValue();
    const bool firstOrder = f_firstOrder.getValue();

    // the only difference for the trapezoidal rule is the factor tr = 0.5 for some usages of h
    const bool optTrapezoidal = d_trapezoidalScheme.getValue();
    SReal tr;
    if (optTrapezoidal)
        tr = 0.5;
    else
        tr = 1.0;

    if (verbose)
        std::cout<<"trapezoidal factor = "<< tr <<std::endl;

    sofa::helper::AdvancedTimer::stepBegin("ComputeForce");
    mop->setImplicit(true); // this solver is implicit
    // compute the net forces at the beginning of the time step
    mop.computeForce(f);
    if( verbose )
        serr<<"EulerImplicitSolver, initial f = "<< f <<sendl;

    sofa::helper::AdvancedTimer::stepNext ("ComputeForce", "ComputeRHTerm");
    if( firstOrder )
    {
        b.eq(f);
    }
    else
    {
        // new more powerful visitors

        // force in the current configuration
        b.eq(f,1.0/tr);                                                                         // b = f0
        if( verbose )
            serr<<"EulerImplicitSolver, f = "<< f <<sendl;

        // add the change of force due to stiffness + Rayleigh damping
        mop.addMBKv(b, -f_rayleighMass.getValue(), 1, h+f_rayleighStiffness.getValue()); // b =  f0 + ( rm M + B + (h+rs) K ) v

        // integration over a time step
        b.teq(h*tr);                                                                        // b = h(f0 + ( rm M + B + (h+rs) K ) v )
    }

    if( verbose )
        serr<<"EulerImplicitSolver, b = "<< b <<sendl;

    mop.projectResponse(b);          // b is projected to the constrained space

    if( verbose )
        serr<<"EulerImplicitSolver, projected b = "<< b <<sendl;

    sofa::helper::AdvancedTimer::stepNext ("ComputeRHTerm", "MBKBuild");

    core::behavior::MultiMatrix<simulation::common::MechanicalOperations> matrix(&mop);

    if (firstOrder)
        matrix = MechanicalMatrix(1,0,-h*tr); //MechanicalMatrix::K * (-h*tr) + MechanicalMatrix::M;
    else
        matrix = MechanicalMatrix(1+tr*h*f_rayleighMass.getValue(),-tr*h,-tr*h*(h+f_rayleighStiffness.getValue())); // MechanicalMatrix::K * (-tr*h*(h+f_rayleighStiffness.getValue())) + MechanicalMatrix::B * (-tr*h) + MechanicalMatrix::M * (1+tr*h*f_rayleighMass.getValue());

    if( verbose )
    {
        serr<<"EulerImplicitSolver, matrix = "<< (MechanicalMatrix::K * (-h*(h+f_rayleighStiffness.getValue())) + MechanicalMatrix::M * (1+h*f_rayleighMass.getValue())) << " = " << matrix <<sendl;
        serr<<"EulerImplicitSolver, Matrix K = " << MechanicalMatrix::K << sendl;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("SystemSolution");
#endif
    sofa::helper::AdvancedTimer::stepNext ("MBKBuild", "MBKSolve");
    matrix.solve(x, b); //Call to ODE resolution.
    sofa::helper::AdvancedTimer::stepEnd  ("MBKSolve");
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("SystemSolution");
#endif

    // mop.projectResponse(x);
    // x is the solution of the system

    // apply the solution

    const bool solveConstraint = f_solveConstraint.getValue();

#ifndef SOFA_NO_VMULTIOP // unoptimized version
    if (solveConstraint)
#endif
    {
    if (firstOrder)
    {
        sofa::helper::AdvancedTimer::stepBegin("UpdateV");
        newVel.eq(x);                         // vel = x
        sofa::helper::AdvancedTimer::stepNext ("UpdateV", "CorrectV");
        mop.solveConstraint(newVel,core::ConstraintParams::VEL);
        sofa::helper::AdvancedTimer::stepNext ("CorrectV", "UpdateX");
        newPos.eq(pos, newVel, h);            // pos = pos + h vel
        sofa::helper::AdvancedTimer::stepNext ("UpdateX", "CorrectX");
        mop.solveConstraint(newPos,core::ConstraintParams::POS);
        sofa::helper::AdvancedTimer::stepEnd  ("CorrectX");
    }
    else
    {
        sofa::helper::AdvancedTimer::stepBegin("UpdateV");
        //vel.peq( x );                       // vel = vel + x
        newVel.eq(vel, x);
        sofa::helper::AdvancedTimer::stepNext ("UpdateV", "CorrectV");
        mop.solveConstraint(newVel,core::ConstraintParams::VEL);
        sofa::helper::AdvancedTimer::stepNext ("CorrectV", "UpdateX");
        //pos.peq( vel, h );                  // pos = pos + h vel
        newPos.eq(pos, newVel, h);
        sofa::helper::AdvancedTimer::stepNext ("UpdateX", "CorrectX");
        mop.solveConstraint(newPos,core::ConstraintParams::POS);
        sofa::helper::AdvancedTimer::stepEnd  ("CorrectX");
    }
    }
#ifndef SOFA_NO_VMULTIOP
    else
    {
        typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
        VMultiOp ops;
        if (firstOrder)
        {
            ops.resize(2);
            ops[0].first = newVel;
            ops[0].second.push_back(std::make_pair(x.id(),1.0));
            ops[1].first = newPos;
            ops[1].second.push_back(std::make_pair(pos.id(),1.0));
            ops[1].second.push_back(std::make_pair(newVel.id(),h));
        }
        else
        {
            ops.resize(2);
            ops[0].first = newVel;
            ops[0].second.push_back(std::make_pair(vel.id(),1.0));
            ops[0].second.push_back(std::make_pair(x.id(),1.0));
            ops[1].first = newPos;
            ops[1].second.push_back(std::make_pair(pos.id(),1.0));
            ops[1].second.push_back(std::make_pair(newVel.id(),h));
        }

        sofa::helper::AdvancedTimer::stepBegin("UpdateVAndX");
        vop.v_multiop(ops);
        sofa::helper::AdvancedTimer::stepEnd("UpdateVAndX");
    }
#endif

    mop.addSeparateGravity(dt, newVel);	// v += dt*g . Used if mass wants to added G separately from the other forces to v.
    if (f_velocityDamping.getValue()!=0.0)
        newVel *= exp(-h*f_velocityDamping.getValue());

    if( verbose )
    {
        mop.projectPosition(newPos);
        mop.projectVelocity(newVel);
        mop.propagateX(newPos);
        mop.propagateV(newVel);
        serr<<"EulerImplicitSolver, final x = "<< newPos <<sendl;
        serr<<"EulerImplicitSolver, final v = "<< newVel <<sendl;
        mop.computeForce(f);
        serr<<"EulerImplicitSolver, final f = "<< f <<sendl;

    }

}

SOFA_DECL_CLASS(EulerImplicitSolver)

int EulerImplicitSolverClass = core::RegisterObject("Time integrator using implicit backward Euler scheme")
        .add< EulerImplicitSolver >()
        .addAlias("EulerImplicit")
        .addAlias("ImplicitEulerSolver")
        .addAlias("ImplicitEuler")
        ;

} // namespace odesolver

} // namespace component

} // namespace sofa

