/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/component/odesolver/EulerKaapiSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <math.h>
#include <iostream>
#include <sofa/simulation/common/MechanicalKaapiAction.h>
#include <sofa/simulation/common/MechanicalVPrintAction.h>

using std::cerr;
using std::endl;
using namespace sofa::simulation::tree;
using namespace sofa::simulation;

namespace sofa
{

namespace component
{

namespace odesolver
{
using core::VecId;
using namespace sofa::defaulttype;
using namespace core::behavior;

int EulerKaapiSolverClass =
    core::RegisterObject("A simple explicit time integrator with kaapi"). add<
    EulerKaapiSolver> ().addAlias("EulerKaapi");

SOFA_DECL_CLASS (EulerKaapi)
;

EulerKaapiSolver::EulerKaapiSolver() :
    symplectic(dataField(&symplectic,
            true,
            "symplectic",
            "If true, the velocities are updated before the velocities and the method is symplectic (more robust). If false, the positions are updated before the velocities (standard Euler, less robust)."))
{
}

EulerKaapiSolver::VecId EulerKaapiSolver::v_alloc(VecId::Type t)
{
    VecId v(t, vectors[t].alloc());
    MechanicalVAllocAction(v).execute(getContext());
    return v;
}
void EulerKaapiSolver::v_clear(VecId v) ///< v=0
{
    MechanicalVOpAction(v).execute(getContext());
}

void EulerKaapiSolver::v_free(VecId v)
{
    if (vectors[v.type].free(v.index))
        MechanicalVFreeAction(v).execute(getContext());
}

void EulerKaapiSolver::v_peq(VecId v, VecId a, double f) ///< v+=f*a
{
    std::cout << "ve peque" << std::endl;
    MechanicalVOpAction(v, v, a, f).execute(getContext());
}

void EulerKaapiSolver::projectResponse(VecId dx, double **W)
{
    MechanicalApplyConstraintsAction(dx, W).execute(getContext());
}

void EulerKaapiSolver::accFromF(VecId a, VecId f)
{
    MechanicalAccFromFAction(a, f).execute(getContext());
}

void EulerKaapiSolver::propagatePositionAndVelocity(double t, VecId x, VecId v)
{
    MechanicalPropagatePositionAndVelocityAction(t, x, v).execute(getContext());
}

void EulerKaapiSolver::computeForce(VecId result)
{
    MechanicalResetForceAction(result).execute(getContext());
    finish();
    MechanicalComputeForceAction(result).execute(getContext());
}

void EulerKaapiSolver::computeAcc(double t, VecId a, VecId x, VecId v)
{
    MultiVector f(this, VecId::force());
    std::cout << "kaapi solver" << std::endl;
    propagatePositionAndVelocity(t, x, v);
    computeForce(f);
    if (this->f_printLog.getValue() == true)
    {
        cerr << "OdeSolver::computeAcc, f = " << f << endl;
    }

    accFromF(a, f);
    projectResponse(a);
}

void EulerKaapiSolver::solve(double dt)
{
    //objectmodel::BaseContext* group = getContext();
    OdeSolver * group = this;
    MultiVector pos(group, VecId::position());
    MultiVector vel(group, VecId::velocity());
    MultiVector acc(group, VecId::dx());
    bool printLog = 1;
    // f_printLog.getValue();

    if (printLog)
    {
        cerr << "EulerKaapiSolver, dt = " << dt << endl;
        cerr << "EulerKaapiSolver, initial x = " << pos << endl;
        cerr << "EulerKaapiSolver, initial v = " << vel << endl;
    }

    computeAcc(getTime(), acc, pos, vel);

    // update state
    if (symplectic.getValue())
    {
        vel.peq(acc, dt);
        pos.peq(vel, dt);
    }
    else
    {
        pos.peq(vel, dt);
        vel.peq(acc, dt);
    }

    if (printLog)
    {
        cerr << "EulerKaapiSolver, acceleration = " << acc << endl;
        cerr << "EulerKaapiSolver, final x = " << pos << endl;
        cerr << "EulerKaapiSolver, final v = " << vel << endl;
    }
}

} // namespace odesolver

} // namespace component

} // namespace sofa
