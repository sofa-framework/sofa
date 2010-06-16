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
#include <sofa/component/odesolver/NewmarkImplicitSolver.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/core/ObjectFactory.h>
#include <math.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"




namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace sofa::defaulttype;
using namespace core::behavior;

NewmarkImplicitSolver::NewmarkImplicitSolver()
    : f_rayleighStiffness( initData(&f_rayleighStiffness,0.1,"rayleighStiffness","Rayleigh damping coefficient related to stiffness") )
    , f_rayleighMass( initData(&f_rayleighMass,0.1,"rayleighMass","Rayleigh damping coefficient related to mass"))
    , f_velocityDamping( initData(&f_velocityDamping,0.,"vdamping","Velocity decay coefficient (no decay if null)") )
    , f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , f_gamma( initData(&f_gamma, 0.5, "gamma", "Newmark scheme gamma coefficient") )
    , f_beta( initData(&f_beta, 0.25, "beta", "Newmark scheme beta coefficient") )
{
}

void NewmarkImplicitSolver::solve(double dt, sofa::core::behavior::BaseMechanicalState::VecId xResult, sofa::core::behavior::BaseMechanicalState::VecId vResult)
{
    MultiVector pos(this, VecId::position());
    MultiVector vel(this, VecId::velocity());
    MultiVector f(this, VecId::force());
    MultiVector b(this, VecId::V_DERIV);
    MultiVector a(this, VecId::V_DERIV);
    MultiVector aResult(this, VecId::V_DERIV);

    const double h = dt;
    const double gamma = f_gamma.getValue();
    const double beta = f_beta.getValue();
    const double rM = f_rayleighMass.getValue();
    const double rK = f_rayleighStiffness.getValue();
    const bool verbose  = f_verbose.getValue();

    /* This integration scheme is based on the following equations:
    *
    *   $x_{t+h} = x_t + h v_t + h^2/2 ( (1-2\beta) a_t + 2\beta a_{t+h} )$
    *   $v_{t+h} = v_t + h ( (1-\gamma) a_t + \gamma a_{t+h} )$
    *
    * Applied to a mechanical system where $ M a_t + (r_M M + r_K K) v_t + K x_t = f_ext$, we need to solve the following system:
    *
    *   $ M a_{t+h} + (r_M M + r_K K) v_{t+h} + K x_{t+h} = f_ext $
    *   $ M a_{t+h} + (r_M M + r_K K) ( v_t + h ( (1-\gamma) a_t + \gamma a_{t+h} ) ) + K ( x_t + h v_t + h^2/2 ( (1-2\beta) a_t + 2\beta a_{t+h} ) ) = f_ext $
    *   $ ( M + h \gamma (r_M M + r_K K) + h^2 \beta K ) a_{t+h} = f_ext - (r_M M + r_K K) ( v_t + h (1-\gamma) a_t ) - K ( x_t + h v_t + h^2/2 (1-2\beta) a_t ) $
    *   $ ( (1 + h \gamma r_M) M + (h^2 \beta + h \gamma r_K) K ) a_{t+h} = f_ext - (r_M M + r_K K) v_t - K x_t - (r_M M + r_K K) ( h (1-\gamma) a_t ) - K ( h v_t + h^2/2 (1-2\beta) a_t ) $
    *   $ ( (1 + h \gamma r_M) M + (h^2 \beta + h \gamma r_K) K ) a_{t+h} = a_t - (r_M M + r_K K) ( h (1-\gamma) a_t ) - K ( h v_t + h^2/2 (1-2\beta) a_t ) $
    *
    * The current implementation first computes $a_t$ directly (as in the explicit solvers), then solves the previous system to compute $a_{t+dt}$, and finally computes the new position and velocity.
    */

    // 1. Compute a_t (stored in a)

    if (rM == 0.0 && rK == 0.0)
    {
        computeForce(f);                                                        // f = f_ext - K x
    }
    else
    {
        // accumulation through mappings is disabled as it will be done by
        // addMBKv after all factors are computed
        computeForce(f, true, false);                                           //  f = f_ext - K x

        // values are not cleared so that contributions from computeForce
        // are kept and accumulated through mappings once at the end
        addMBKv(f, -rM, 0, rK, false, true);                                   // f -= (r_M M + r_K K) v
    }

    accFromF(a, f);

    projectResponse(a);          // b is projected to the constrained space

    if( verbose )
        serr<<"NewmarkImplicitSolver, a0 = "<< a <<sendl;

    // 2. Compute right hand term of equation on a_{t+h}

    b = a;                                                                      // b = a
    if (rM != 0.0 || rK != 0.0 || beta != 0.5)
    {
        propagateDx(a);
        addMBKdx(b, -h*(1-gamma)*rM, 0, h*(1-gamma)*rK + h*h*(1-2*beta)/2);    // b += ( -h (1-\gamma)(r_M M + r_K K) - h^2/2 (1-2\beta) K ) a
    }
    addMBKv(b, 0, 0, h);                                                       // b += -h K v

    if( verbose )
        serr<<"NewmarkImplicitSolver, b = "<< b <<sendl;

    projectResponse(b);          // b is projected to the constrained space

    if( verbose )
        serr<<"NewmarkImplicitSolver, projected b = "<< b <<sendl;

    // 3. Solve system of equations on a_{t+h}

    MultiMatrix matrix(this);
    matrix = MechanicalMatrix::K * (-h*h*beta - h*rK) + MechanicalMatrix::M * (1 + h*gamma*rM);

    //if( verbose )
    //	serr<<"NewmarkImplicitSolver, matrix = "<< MechanicalMatrix::K * (h*h*beta + h*rK) + MechanicalMatrix::M * (1 + h*gamma*rM) << " = " << matrix <<sendl;

    matrix.solve(aResult, b);
    projectResponse(aResult);

    if( verbose )
        serr<<"NewmarkImplicitSolver, a1 = "<< aResult <<sendl;


    // 4. Compute the new position and velocity

    MultiVector newPos(this, xResult);
    MultiVector newVel(this, vResult);
#ifdef SOFA_NO_VMULTIOP // unoptimized version
    // x_{t+h} = x_t + h v_t + h^2/2 ( (1-2\beta) a_t + 2\beta a_{t+h} )
    b.eq(vel, a, h*(0.5-beta));
    b.peq(aResult, h*beta);
    newPos.eq(pos, b, h);
    solveConstraint(dt,xResult,core::behavior::BaseConstraintSet::POS);
    // v_{t+h} = v_t + h ( (1-\gamma) a_t + \gamma a_{t+h} )
    newVel.eq(vel, a, h*(1-gamma));
    newVel.peq(aResult, h*gamma);
    solveConstraint(dt,vResult,core::behavior::BaseConstraintSet::VEL);

#else // single-operation optimization
    typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
    VMultiOp ops;
    ops.resize(3);
    ops[0].first = (VecId)b;
    ops[0].second.push_back(std::make_pair((VecId)vel,1.0));
    ops[0].second.push_back(std::make_pair((VecId)a, h*(0.5-beta)));
    ops[0].second.push_back(std::make_pair((VecId)aResult, h*beta));
    ops[1].first = (VecId)newPos;
    ops[1].second.push_back(std::make_pair((VecId)pos,1.0));
    ops[1].second.push_back(std::make_pair((VecId)b,h));
    ops[2].first = (VecId)newVel;
    ops[2].second.push_back(std::make_pair((VecId)vel,1.0));
    ops[2].second.push_back(std::make_pair((VecId)a, h*(1-gamma)));
    ops[2].second.push_back(std::make_pair((VecId)aResult, h*gamma));
    simulation::MechanicalVMultiOpVisitor vmop(ops);
    vmop.execute(this->getContext());

    solveConstraint(dt,vResult,core::behavior::BaseConstraintSet::VEL);
    solveConstraint(dt,xResult,core::behavior::BaseConstraintSet::POS);

#endif

    addSeparateGravity(dt, newVel);	// v += dt*g . Used if mass wants to added G separately from the other forces to v.
    if (f_velocityDamping.getValue()!=0.0)
        newVel *= exp(-h*f_velocityDamping.getValue());

    if( verbose )
    {
        serr<<"NewmarkImplicitSolver, final x = "<< newPos <<sendl;
        serr<<"NewmarkImplicitSolver, final v = "<< newVel <<sendl;
    }
}

SOFA_DECL_CLASS(NewmarkImplicitSolver)

int NewmarkImplicitSolverClass = core::RegisterObject("Implicit time integrator using Newmark scheme")
        .add< NewmarkImplicitSolver >()
        .addAlias("Newmark");
;

} // namespace odesolver

} // namespace component

} // namespace sofa

