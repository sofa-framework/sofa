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
#ifndef SOFA_COMPONENT_ODESOLVER_EULERKAAPISOLVER_H
#define SOFA_COMPONENT_ODESOLVER_EULERKAAPISOLVER_H

#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/component/odesolver/OdeSolverImpl.h>

namespace sofa
{

namespace component
{

namespace odesolver
{

/** The simplest time integration.
Two variants are available, depending on the value of field "symplectic".
If true (the default), the symplectic variant of Euler's method is applied:
If false, the basic Euler's method is applied (less robust)
*/
class SOFA_COMPONENT_ODESOLVER_API EulerKaapiSolver : public sofa::component::odesolver::OdeSolverImpl
{
public:
    SOFA_CLASS(EulerKaapiSolver, sofa::component::odesolver::OdeSolverImpl);
    EulerKaapiSolver();
    void solve (double dt);
    void computeAcc (double t, core::VecId a, core::VecId x, core::VecId v);
    DataField<bool> symplectic;
    void v_free(core::VecId v);
    void v_clear(core::VecId v); ///< v=0
    core::VecId v_alloc(core::VecId::Type t);

    void propagatePositionAndVelocity(double t, core::VecId x, core::VecId v);
    void computeForce(core::VecId result);
    void accFromF(core::VecId a, core::VecId f);
    void projectResponse(core::VecId dx, double **W=NULL);
    void v_peq(core::VecId v, core::VecId a, double f); ///< v+=f*a


};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
