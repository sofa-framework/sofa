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
#include <sofa/core/behavior/MechanicalState.h>

namespace sofa::core
{
class GeometricStiffnessMatrix;

class MappingMatrixAccumulator;

/** An interface to convert a model state to an other model state.
The model states are positions and velocities or generalizations of these (class sofa::core::BaseState).
The source is denoted using various names: from, input, master, parent…
The target is denoted using various names: to, output, slave, child…
The mapping must be located somewhere between the master and the slave, so that the visitors traverse it after the master and before the slave during the top-down traversals, and the other way round during the bottom-up traversals.
It is typically located in the same graph node as the slave, with the master in the parent node, but this is not a must.
Mappings typically store constant local coordinates of the output points, and update the output points by applying input displacements to the local coordinates.
 */
class SOFA_CORE_API BaseMapping : public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(BaseMapping, objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(BaseMapping)
protected:

    /// Constructor
    BaseMapping();

    /// Destructor
    ~BaseMapping() override;
	
private:
    BaseMapping(const BaseMapping& n) = delete;
    BaseMapping& operator=(const BaseMapping& n) = delete;
	
public:
    Data<bool> f_mapForces; ///< Are forces mapped ?
    Data<bool> f_mapConstraints; ///< Are constraints mapped ?
    Data<bool> f_mapMasses; ///< Are masses mapped ?
    Data<bool> f_mapMatrices; ///< Are matrix explicit mapped?

    /// Apply the transformation from the input model to the output model (like apply displacement from BehaviorModel to VisualModel)
    virtual void apply (const MechanicalParams* mparams = mechanicalparams::defaultInstance(), MultiVecCoordId outPos = VecCoordId::position(), ConstMultiVecCoordId inPos = ConstVecCoordId::position() ) = 0;
    /// Compute output velocity based on input velocity, using the linearized transformation (tangent operator). Also used to propagate small displacements.
    virtual void applyJ(const MechanicalParams* mparams = mechanicalparams::defaultInstance(), MultiVecDerivId outVel = VecDerivId::velocity(), ConstMultiVecDerivId inVel = ConstVecDerivId::velocity() ) = 0;

    /// Accessor to the input model of this mapping
    virtual type::vector<BaseState*> getFrom() = 0;
    /// If the type is compatible set the input model and return true, otherwise do nothing and return false.
    virtual bool setFrom( BaseState* from );

    /// Accessor to the output model of this mapping
    virtual type::vector<BaseState*> getTo() = 0;
    /// If the type is compatible set the output model and return true, otherwise do nothing and return false.
    virtual bool setTo( BaseState* to );

    /** @name Mechanical mapping API
     *  Methods related to the transmission of forces
     */
    ///@{
    /// Accumulate child force in the parent force. In implicit methods, this is also used to accumulate a change of child force to a change of parent force.
    virtual void applyJT(const MechanicalParams* mparams, MultiVecDerivId inForce, ConstMultiVecDerivId outForce) = 0;
    /// Accumulate a change of parent force due to the change of the mapping, for a constant child force. Null for linear mappings.
    virtual void applyDJT(const MechanicalParams* mparams, MultiVecDerivId inForce, ConstMultiVecDerivId outForce) = 0;
    /// Propagate constraint Jacobians upward
    virtual void applyJT(const ConstraintParams* mparams, MultiMatrixDerivId inConst, ConstMultiMatrixDerivId outConst) = 0;
    virtual void computeAccFromMapping(const MechanicalParams* mparams, MultiVecDerivId outAcc, ConstMultiVecDerivId inVel, ConstMultiVecDerivId inAcc) = 0;

    virtual bool areForcesMapped() const;
    virtual bool areConstraintsMapped() const;
    virtual bool areMassesMapped() const;
    virtual bool areMatricesMapped() const;

    virtual void setForcesMapped(bool b);
    virtual void setConstraintsMapped(bool b);
    virtual void setMassesMapped(bool b);
    virtual void setMatricesMapped(bool b);

    virtual void setNonMechanical();
    ///@}

    /// Return true if this mapping should be used as a mechanical mapping.
    virtual bool isMechanical() const;

    /// Return true if the destination model has the same topology as the source model.
    ///
    /// This is the case for mapping keeping a one-to-one correspondance between
    /// input and output DOFs (mostly identity or data-conversion mappings).
    virtual bool sameTopology() const { return false; }

    /// Get the (sparse) jacobian matrix of this mapping, as used in applyJ/applyJT.
    /// This matrix should have as many columns as DOFs in the input mechanical states
    /// (one after the other in case of multi-mappings), and as many lines as DOFs in
    /// the output mechanical states.
    ///
    /// applyJ(out, in) should be equivalent to $out = J in$.
    /// applyJT(out, in) should be equivalent to $out = J^T in$.
    ///
    /// @todo Note that if the mapping provides this matrix, then a default implementation
    /// of all other related methods could be provided, or optionally used to verify the
    /// provided implementations for debugging.
    virtual const sofa::linearalgebra::BaseMatrix* getJ(const MechanicalParams* /*mparams*/);

    /// @deprecated
    virtual const sofa::linearalgebra::BaseMatrix* getJ();


    typedef sofa::linearalgebra::BaseMatrix* (*func_createMappedMatrix)(const behavior::BaseMechanicalState* , const behavior::BaseMechanicalState* );
    /// Create a matrix for mapped mechanical objects
    /// If the two mechanical objects is identical, create a new stiffness matrix for this mapped objects
    /// If the two mechanical objects is different, create a new interaction matrix
    virtual sofa::linearalgebra::BaseMatrix* createMappedMatrix(const behavior::BaseMechanicalState* state1, const behavior::BaseMechanicalState* state2, func_createMappedMatrix);

    /// Get the source (upper) mechanical state.
    virtual type::vector<behavior::BaseMechanicalState*> getMechFrom() = 0;
    /// Get the destination (lower, mapped) mechanical state.
    virtual type::vector<behavior::BaseMechanicalState*> getMechTo() = 0;

    /// Disable the mapping to get the original coordinates of the mapped model.
    virtual void disable()=0;

    /// @name API for global matrix assembly (used in the Compliant plugin)
    /// @{

    /// Returns pointers to Jacobian matrices associated with parent states, consistently with getFrom(). Most mappings have only one parent, however Multimappings have several parents.
    /// For efficiency concerns, please return pointers to defaulttype::EigenBaseSparseMatrix
    virtual const type::vector<sofa::linearalgebra::BaseMatrix*>* getJs() { dmsg_error() << "Calling a virtual method not implemented."; return nullptr; }

    /// Compute the geometric stiffness matrix based on given child forces
    /// K = dJ^T * outForce
    /// Default implementation does nothing, corresponding to a linear mapping.
    virtual void updateK( const MechanicalParams* /*mparams*/, ConstMultiVecDerivId /*outForce*/ ) {}

    /// Returns a pointer to the geometric stiffness matrix.
    /// This is the equivalent of applyDJT, for matrix assembly instead of matrix-vector product.
    /// This matrix is associated with the parent DOFs. It is a square matrix with a size of the total number of parent DOFs.
    /// For efficiency concerns, please return a pointer to a defaulttype::EigenBaseSparseMatrix
    virtual const linearalgebra::BaseMatrix* getK() { return nullptr; }

    /**
     * \brief Assembles the geometric stiffness matrix of the mapping in the
     *        provided matrix object.
     *
     * The geometric stiffness is defined as dJ^T/dx * outForce, where outForce
     * is the forces applying on the output of the mapping, J is the jacobian
     * matrix of the mapping and x is the position of the input of the mapping.
     *
     * Inside the method, outForce is retrieved, while dJ^T/dx and its product
     * with outForce must be computed.
     *
     * Note that dJ^T/dx is null for linear mappings, meaning that the method
     * can remain empty.
     *
     * The geometric stiffness is a term that must be added into the global
     * mechanical matrix.
     *
     * \param matrices The matrix to fill in with the geometric stiffness of the mapping
     */
    virtual void buildGeometricStiffnessMatrix(sofa::core::GeometricStiffnessMatrix* matrices);

    /// @}

protected:
    bool testMechanicalState(BaseState* state);

public:
    bool insertInNode( objectmodel::BaseNode* node ) override;
    bool removeInNode( objectmodel::BaseNode* node ) override;
};

} // namespace sofa::core

