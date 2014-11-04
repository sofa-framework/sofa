/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_BASEMAPPING_H
#define SOFA_CORE_BASEMAPPING_H

#include <stdlib.h>
#include <string>
#include <iostream>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/BehaviorModel.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/VecId.h>
#include <sofa/helper/vector.h>

namespace sofa
{

namespace core
{

/** An interface to convert a model state to tan other model state.
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
protected:
    /// Constructor
    BaseMapping();

    /// Destructor
    virtual ~BaseMapping();
public:
    Data<bool> f_mapForces;
    Data<bool> f_mapConstraints;
    Data<bool> f_mapMasses;
    Data<bool> f_mapMatrices;

    /// Apply the transformation from the input model to the output model (like apply displacement from BehaviorModel to VisualModel)
    virtual void apply (const MechanicalParams* mparams = MechanicalParams::defaultInstance(), MultiVecCoordId outPos = VecCoordId::position(), ConstMultiVecCoordId inPos = ConstVecCoordId::position() ) = 0;
    /// Compute output velocity based on input velocity, using the linearized transformation (tangent operator). Also used to propagate small displacements.
    virtual void applyJ(const MechanicalParams* mparams = MechanicalParams::defaultInstance(), MultiVecDerivId outVel = VecDerivId::velocity(), ConstMultiVecDerivId inVel = ConstVecDerivId::velocity() ) = 0;

    /// Accessor to the input model of this mapping
    virtual helper::vector<BaseState*> getFrom() = 0;

    /// Accessor to the output model of this mapping
    virtual helper::vector<BaseState*> getTo() = 0;
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
    /// Store data used to apply geometric stiffness. The child force can be read using mparams->readF(this->toModel). Default implementation does nothing, corresponding to a linear mapping.
//    virtual void computeGeometricStiffness(const MechanicalParams* /*mparams*/) {}
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
    virtual const sofa::defaulttype::BaseMatrix* getJ(const MechanicalParams* /*mparams*/);

    virtual const sofa::defaulttype::BaseMatrix* getJ();

    typedef sofa::defaulttype::BaseMatrix* (*func_createMappedMatrix)(const behavior::BaseMechanicalState* , const behavior::BaseMechanicalState* );


    //Create a matrix for mapped mechanical objects
    //If the two mechanical objects is identical, create a new stiffness matrix for this mapped objects
    //If the two mechanical objects is different, create a new interaction matrix
    virtual sofa::defaulttype::BaseMatrix* createMappedMatrix(const behavior::BaseMechanicalState* state1, const behavior::BaseMechanicalState* state2, func_createMappedMatrix);

    ///<TO REMOVE>
    ///Necessary ?
    /// Get the source (upper) mechanical state.
    virtual helper::vector<behavior::BaseMechanicalState*> getMechFrom() = 0;
    /// Get the destination (lower, mapped) mechanical state.
    virtual helper::vector<behavior::BaseMechanicalState*> getMechTo() = 0;
    /// Return false if this mapping should only be used as a regular

    /// Disable the mapping to get the original coordinates of the mapped model.
    virtual void disable()=0;

    /// @name Experimental API used in the Compliant solver to perform global matrix assembly (François Faure, 2012)
    /// @{

    /// Returns pointers to Jacobian matrices associated with parent states, consistently with getFrom(). Most mappings have only one parent, however Multimappings have several parents.
    virtual const helper::vector<sofa::defaulttype::BaseMatrix*>* getJs()  { serr<<"getJs not implemented"<<sendl; return 0; }

    /** Return pointers to the geometric stiffness matrices. 
	This is the equivalent of applyDJT, for matrix assembly instead of matrix-vector product. 
	These matrices are associated with the parent DOFs. 
    @todo Are there cross-dependencies between the parents, resulting in off-diagonal geometric stiffness blocks? I do not think so but I am not totally sure (FF, June 2013)
    Answer (M Nesme, oct. 2014): there can be off-diagonal blocks and only specific non-linear multimappings can be implemented. A workaround is to decompose the problematic
        non-linear multimappings in a small graph including only linear multimappings + non-linear simple mappings (where geometric stiffness can be implemented).
        e.g.: A DistanceMultimapping (non-linear) is not implementable with this API, but can be decomposed in a SubsetMultiMapping (linear) + a DistanceMapping (non-linear)
    */
    virtual const helper::vector<sofa::defaulttype::BaseMatrix*>* getKs() { return NULL; }

    /// @}

protected:
    bool testMechanicalState(BaseState* state);
};

} // namespace core

} // namespace sofa

#endif
