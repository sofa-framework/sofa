/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
 *                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_ODESOLVER_EULERSOLVER_H
#define SOFA_COMPONENT_ODESOLVER_EULERSOLVER_H

#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/OdeSolverImpl.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/helper/map.h>

namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace sofa::component::linearsolver;
/** The simplest time integration.
Two variants are available, depending on the value of field "symplectic".
If true (the default), the symplectic variant of Euler's method is applied:
If false, the basic Euler's method is applied (less robust)
*/
class EulerSolver : public sofa::simulation::OdeSolverImpl
{
protected:
    typedef sofa::simulation::MechanicalAccumulateLMConstraint::ConstraintData ConstraintData;
public:

    EulerSolver();
    void solve (double dt);


    //Constraint resolution using Lapack
#ifdef SOFA_HAVE_LAPACK
    /** Find all the LMConstraint present in the scene graph and solve a part of them
     * @param Id nature of the constraint to be solved
     * @param propagateVelocityToPosition need to update the position once the velocity has been constrained
     **/
    void solveConstraint(VecId Id, bool propagateVelocityToPosition=false);

    Data<bool> constraintAcc;
    Data<bool> constraintVel;
    Data<bool> constraintPos;

    Data<bool> constraintResolution;
    Data<unsigned int> numIterations;
    Data<double> maxError;
#endif
    Data<bool> symplectic;

    /// Given an input derivative order (0 for position, 1 for velocity, 2 for acceleration),
    /// how much will it affect the output derivative of the given order.
    virtual double getIntegrationFactor(int inputDerivative, int outputDerivative) const
    {
        const double dt = getContext()->getDt();
        double matrix[3][3] =
        {
            { 1, dt, ((symplectic.getValue())?dt*dt:0.0)},
            { 0, 1, dt},
            { 0, 0, 0}
        };
        if (inputDerivative >= 3 || outputDerivative >= 3)
            return 0;
        else
            return matrix[outputDerivative][inputDerivative];
    }

    /// Given a solution of the linear system,
    /// how much will it affect the output derivative of the given order.
    ///
    virtual double getSolutionIntegrationFactor(int outputDerivative) const
    {
        const double dt = getContext()->getDt();
        double vect[3] = { ((symplectic.getValue())?dt*dt:0.0), dt, 1};
        if (outputDerivative >= 3)
            return 0;
        else
            return vect[outputDerivative];
    }
    void init()
    {
        OdeSolverImpl::init();
        reinit();
    }

#ifdef SOFA_HAVE_LAPACK
    void reinit()
    {
        numIterations.setDisplayed(constraintResolution.getValue());
        maxError.setDisplayed(constraintResolution.getValue());
    }

protected:
    /// Construct the Right hand term of the system
    void buildRightHandTerm(VecId &Id, sofa::simulation::MechanicalAccumulateLMConstraint &LMConstraintVisitor, FullVector<double>  &c);
    /** Apply the correction to the state corresponding
     * @param id nature of the constraint, and correction to apply
     * @param dof MechanicalState to correct
     * @param invM_Jtrans matrix M^-1.J^T to apply the correction from the independant dofs through the mapping
     * @param c correction vector
     * @param propageVelocityChange need to propagate the correction done to the velocity for the position
     **/
    void constraintStateCorrection(VecId &id, sofa::core::componentmodel::behavior::BaseMechanicalState* dof,
            FullMatrix<double>  &invM_Jtrans, FullVector<double>  &c,  bool propageVelocityChange=false);
#endif
};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
