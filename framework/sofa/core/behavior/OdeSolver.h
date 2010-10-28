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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_BEHAVIOR_ODESOLVER_H
#define SOFA_CORE_BEHAVIOR_ODESOLVER_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/ExecParams.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/MultiVec.h>
#include <sofa/core/behavior/MultiMatrix.h>
#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>

namespace sofa
{

namespace core
{

namespace behavior
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
    SOFA_CLASS(OdeSolver, objectmodel::BaseObject);

    OdeSolver();

    virtual ~OdeSolver();

    /// Main computation method.
    ///
    /// Specify and execute all computation for timestep integration, i.e.
    /// advancing the state from time t to t+dt, putting the resulting position and velocity in the provided vectors.
    virtual void solve(double /*dt*/, MultiVecCoordId /*xResult*/, MultiVecDerivId /*vResult*/, const core::ExecParams* /*params*/) = 0; // { serr << "ERROR: " << getClassName() << " don't implement solve on custom x and v" << sendl; }

    /// Main computation method.
    ///
    /// Specify and execute all computation for timestep integration, i.e.
    /// advancing the state from time t to t+dt.
    virtual void solve (double dt, const core::ExecParams* params) { solve(dt, VecCoordId::position(), VecDerivId::velocity(), params); }


    /** Find all the Constraint present in the scene graph, build the constraint equation system, solve and apply the correction
     **/
    virtual void solveConstraint(double /*dt*/, MultiVecId, ConstraintParams::ConstOrder) {};

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
    virtual double getIntegrationFactor(int inputDerivative, int outputDerivative) const = 0;

    /// Given a solution of the linear system,
    /// how much will it affect the output derivative of the given order.
    ///
    virtual double getSolutionIntegrationFactor(int outputDerivative) const = 0;


    /// Given the solution dx of the linear system inversion, how much will it affect the velocity
    ///
    /// This method is used to compute the compliance for contact corrections
    virtual double getVelocityIntegrationFactor() const
    {
        return getSolutionIntegrationFactor(1);
    }

    /// Given the solution dx of the linear system inversion, how much will it affect the position
    ///
    /// This method is used to compute the compliance for contact corrections
    virtual double getPositionIntegrationFactor() const
    {
        return getSolutionIntegrationFactor(0);
    }

};

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
