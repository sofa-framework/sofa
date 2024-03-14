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
#pragma once

#include <sofa/core/behavior/MultiVec.h>

namespace sofa::core::behavior
{

/**
 *  \brief Component responsible for timestep integration, i.e. advancing the state from time t to t+dt.
 *
 *  This class currently control both the integration scheme (explicit,
 *  implicit, static, etc), and the linear system resolution algorithm
 *  (conjugate gradient, matrix direct inversion, etc). Those two aspect will
 *  propably be separated in a future version.
 *
 *  While all computations required to do the integration step are handled by
 *  this object, they should not be implemented directly in it, but instead
 *  the solver propagates orders (or Visitor) to the other components in the
 *  scenegraph that will locally execute them. This allow for greater
 *  flexibility (the solver can just ask for the forces to be computed without
 *  knowing what type of forces are present), as well as performances
 *  (some computations can be executed in parallel).
 *
 */
class SOFA_CORE_API OdeSolver : public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(OdeSolver, objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(OdeSolver)
protected:
    OdeSolver();

    ~OdeSolver() override;

private:
    OdeSolver(const OdeSolver& n) = delete;
    OdeSolver& operator=(const OdeSolver& n) = delete;

public:
    /// Main computation method.
    ///
    /// Specify and execute all computation for timestep integration, i.e.
    /// advancing the state from time t to t+dt, putting the resulting position and velocity in the provided vectors.
    virtual void solve(const core::ExecParams* /*params*/, SReal /*dt*/, MultiVecCoordId /*xResult*/, MultiVecDerivId /*vResult*/) = 0;

    /// Main computation method.
    ///
    /// Specify and execute all computation for timestep integration, i.e.
    /// advancing the state from time t to t+dt.
    virtual void solve (const core::ExecParams* params, SReal dt) { solve(params, dt, VecCoordId::position(), VecDerivId::velocity()); }


    /// Compute the residual of the newton iteration
    ///
    /// pos_t and vel_t are the position and velocities at the begining of the time step
    /// the result is accumulated in Vecid::force()
    virtual void computeResidual(const core::ExecParams* /*params*/, SReal /*dt*/, sofa::core::MultiVecCoordId /*pos_t*/, sofa::core::MultiVecDerivId /*vel_t*/) { msg_error() << "ComputeResidual is not implemented in " << this->getName(); }


    /// Given an input derivative order (0 for position, 1 for velocity, 2 for acceleration),
    /// how much will it affect the output derivative of the given order.
    ///
    /// This method is used to compute the constraint corrections and adapt the resolution if using baumgart type scheme
    /// For example, a backward-Euler dynamic implicit integrator would use:
    /// Input:      x_t  v_t  a_{t+dt}
    /// x_{t+dt}     1    dt  dt^2
    /// v_{t+dt}     0    1   dt
    ///
    /// If the linear system is expressed on s = a_{t+dt} dt, then the final factors are:
    /// Input:      x_t   v_t    a_t  s
    /// x_{t+dt}     1    dt     0    dt
    /// v_{t+dt}     0    1      0    1
    /// a_{t+dt}     0    0      0    1/dt
    /// The last column is returned by the getSolutionIntegrationFactor method.
    ///
    /// FF: What is the meaning of the parameters ?
    virtual SReal getIntegrationFactor(int /*inputDerivative*/, int /*outputDerivative*/) const { msg_error() << "getIntegrationFactor not implemented !"; return 0; }

    /// Given a solution of the linear system,
    /// how much will it affect the output derivative of the given order.
    ///
    /// FF: What is the meaning of the parameters ?
    virtual SReal getSolutionIntegrationFactor(int /*outputDerivative*/) const { msg_error() << "getSolutionIntegrationFactor not implemented !"; return 0; }


    /// Given the solution dx of the linear system inversion, how much will it affect the velocity
    ///
    /// This method is used to compute the compliance for contact corrections
    virtual SReal getVelocityIntegrationFactor() const
    {
        return getSolutionIntegrationFactor(1);
    }

    /// Given the solution dx of the linear system inversion, how much will it affect the position
    ///
    /// This method is used to compute the compliance for contact corrections
    virtual SReal getPositionIntegrationFactor() const
    {
        return getSolutionIntegrationFactor(0);
    }


    bool insertInNode( objectmodel::BaseNode* node ) override;
    bool removeInNode( objectmodel::BaseNode* node ) override;

};

} // namespace sofa::core::behavior
