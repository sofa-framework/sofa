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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_BEHAVIOR_CONSTRAINTSOLVER_H
#define SOFA_CORE_BEHAVIOR_CONSTRAINTSOLVER_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/BaseConstraintSet.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/MechanicalParams.h>

namespace sofa
{

namespace core
{

namespace behavior
{

class BaseConstraintCorrection;

/**
 *  \brief Component responsible for the expression and solution of system of equations related to constraints.
 The main method is solveConstraint(const ConstraintParams *, MultiVecId , MultiVecId );
 The default implementation successively calls: prepareStates, buildSystem, solveSystem, applyCorrection.
 The parameters are defined in class ConstraintParams.
 *
 */
class SOFA_CORE_API ConstraintSolver : public virtual objectmodel::BaseObject
{
public:

    SOFA_ABSTRACT_CLASS(ConstraintSolver, objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(ConstraintSolver)
protected:
    ConstraintSolver();

    virtual ~ConstraintSolver();
	
private:
	ConstraintSolver(const ConstraintSolver& n) ;
	ConstraintSolver& operator=(const ConstraintSolver& n) ;

	
public:
    /**
     * Launch the sequence of operations in order to solve the constraints
     */
    virtual void solveConstraint(const ConstraintParams *, MultiVecId res1, MultiVecId res2=MultiVecId::null());

    /**
     * Do the precomputation: compute free state, or propagate the states to the mapped mechanical states, where the constraint can be expressed
     */
    virtual bool prepareStates(const ConstraintParams *, MultiVecId res1, MultiVecId res2=MultiVecId::null())=0;

    /**
     * Create the system corresponding to the constraints
     */
    virtual bool buildSystem(const ConstraintParams *, MultiVecId res1, MultiVecId res2=MultiVecId::null())=0;

    /**
     * Rebuild the system using a mass and force factor.
     * Experimental API used to investigate convergence issues.
     */
    virtual void rebuildSystem(double /*massfactor*/, double /*forceFactor*/){}

    /**
     * Use the system previously built and solve it with the appropriate algorithm
     */
    virtual bool solveSystem(const ConstraintParams *, MultiVecId res1, MultiVecId res2=MultiVecId::null())=0;

    /**
     * Correct the Mechanical State with the solution found
     */
    virtual bool applyCorrection(const ConstraintParams *, MultiVecId res1, MultiVecId res2=MultiVecId::null())=0;


    /// Compute the residual in the newton iterations due to the constraints forces
    /// i.e. compute Vecid::force() += J^t lambda
    /// the result is accumulated in Vecid::force()
    virtual void computeResidual(const core::ExecParams* /*params*/) { std::cerr << "ComputeResidual is not implemented in " << this->getName() << std::endl; }


    /// @name Resolution DOFs vectors API
    /// @{

    VecDerivId getForce() const
    {
        return m_fId;
    }

    void setForce(VecDerivId id)
    {
        m_fId = id;
    }

    VecDerivId getDx() const
    {
        return m_dxId;
    }

    void setDx(VecDerivId id)
    {
        m_dxId = id;
    }

    /// @}

    /// Remove reference to ConstraintCorrection
    ///
    /// @param c is the ConstraintCorrection
    virtual void removeConstraintCorrection(BaseConstraintCorrection *s) = 0;


protected:

    VecDerivId m_fId;
    VecDerivId m_dxId;

public:

    virtual bool insertInNode( objectmodel::BaseNode* node );
    virtual bool removeInNode( objectmodel::BaseNode* node );
};

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
