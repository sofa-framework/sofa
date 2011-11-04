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

/**
 *  \brief An interface to convert a model to an other model
 *
 *  This Interface is used for the Mappings. A Mapping can convert one model to an other.
 *  For example, we can have a mapping from a BehaviorModel to a VisualModel.
 *
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
    virtual void apply (const MechanicalParams* mparams /* PARAMS FIRST  = MechanicalParams::defaultInstance()*/, MultiVecCoordId outPos = VecCoordId::position(), ConstMultiVecCoordId inPos = ConstVecCoordId::position() ) = 0;
    virtual void applyJ(const MechanicalParams* mparams /* PARAMS FIRST  = MechanicalParams::defaultInstance()*/, MultiVecDerivId outVel = VecDerivId::velocity(), ConstMultiVecDerivId inVel = ConstVecDerivId::velocity() ) = 0;

    /// Accessor to the input model of this mapping
    virtual helper::vector<BaseState*> getFrom() = 0;

    /// Accessor to the output model of this mapping
    virtual helper::vector<BaseState*> getTo() = 0;

    // BaseMechanicalMapping
    virtual void applyJT(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecDerivId inForce, ConstMultiVecDerivId outForce) = 0;
    virtual void applyDJT(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecDerivId inForce, ConstMultiVecDerivId outForce) = 0;
    virtual void applyJT(const ConstraintParams* mparams /* PARAMS FIRST */, MultiMatrixDerivId inConst, ConstMultiMatrixDerivId outConst) = 0;
    virtual void computeAccFromMapping(const MechanicalParams* mparams /* PARAMS FIRST */, MultiVecDerivId outAcc, ConstMultiVecDerivId inVel, ConstMultiVecDerivId inAcc) = 0;

    virtual bool areForcesMapped() const;
    virtual bool areConstraintsMapped() const;
    virtual bool areMassesMapped() const;
    virtual bool areMatricesMapped() const;

    virtual void setForcesMapped(bool b);
    virtual void setConstraintsMapped(bool b);
    virtual void setMassesMapped(bool b);
    virtual void setMatricesMapped(bool b);

    virtual void setNonMechanical();

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
    /// @TODO Note that if the mapping provides this matrix, then a default implementation
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

protected:
    bool testMechanicalState(BaseState* state);
};

} // namespace core

} // namespace sofa

#endif
