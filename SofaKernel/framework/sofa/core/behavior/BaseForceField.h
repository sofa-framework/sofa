/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_BEHAVIOR_BASEFORCEFIELD_H
#define SOFA_CORE_BEHAVIOR_BASEFORCEFIELD_H

#include <sofa/core/core.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace core
{

namespace behavior
{

/**
 *  \brief Component computing forces within simulated bodies.
 *
 *  This class define the abstract API common to all force fields.
 *  A force field computes forces applied to one or more simulated body
 *  given its current position and velocity.
 *
 *  Forces can be internal to a given body (attached to one MechanicalState,
 *  see the ForceField class), or link several bodies together (such as contact
 *  forces, see the InteractionForceField class).
 *
 *  For implicit integration schemes, it must also compute the derivative
 *  ( df, given a displacement dx ).
 *
 */
class SOFA_CORE_API BaseForceField : public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(BaseForceField, objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(BaseForceField)
protected:
    BaseForceField();
    virtual ~BaseForceField() {}
	
private:
	BaseForceField(const BaseForceField& n) ;
	BaseForceField& operator=(const BaseForceField& n) ;	

	
public:
    /// @name Vector operations
    /// @{

    /// \brief Given the current position and velocity states, update the current force
    /// vector by computing and adding the forces associated with this
    /// ForceField.
    ///
    /// If the ForceField can be represented as a matrix, this method computes
    ///
    ///                          $ f += B v + K x $
    ///
    /// where K is the stiffness matrix (associated with forces which derive from a potential),
    /// and B is the damping matrix (associated with viscous forces).
    /// Very often, at least one of these matrices is null.
    ///
    /// \param mparams
    /// - \a mparams->bFactor() is the coefficient for damping contributions (i.e. first derivatives term in the ODE)
    /// - \a mparams->kFactor() is the coefficient for stiffness contributions (i.e. DOFs term in the ODE)
    /// - \a mparams->readX() is the input vector of position
    /// - \a mparams->readV() is the input vector of velocity
    /// - \a mparams->readF() is the input vector of force
    /// - if \a mparams->energy() is true, the method computes and internally stores the potential energy,
    /// which will be subsequently returned by method getPotentialEnergy()
    /// \param fId the output vector of forces
    virtual void addForce(const MechanicalParams* mparams, MultiVecDerivId fId )=0;

    /// \brief Compute the force derivative given a small displacement from the
    /// position and velocity used in the previous call to addForce().
    ///
    /// The derivative should be directly derived from the computations
    /// done by addForce. Any forces neglected in addDForce will be integrated
    /// explicitly (i.e. using its value at the beginning of the timestep).
    ///
    /// If the ForceField can be represented as a matrix, this method computes
    /// 
    ///                    $ df += kFactor K dx + bFactor B dx $
    ///
    /// where K is the stiffness matrix (associated with forces which derive from a potential),
    /// and B is the damping matrix (associated with viscous forces).
    ///
    /// \param mparams
    /// - \a mparams->mFactor() is the  coefficient for mass contributions (i.e. second-order derivatives term in the ODE)
    /// - \a mparams->kFactor() is the coefficient for stiffness contributions (i.e. DOFs term in the ODE)
    /// - \a mparams->readDx() input vector
    /// \param dfId the output vector
    virtual void addDForce(const MechanicalParams* mparams, MultiVecDerivId dfId )=0;

    /// \brief Accumulate the contribution of M, B, and/or K matrices multiplied
    /// by the dx vector with the given coefficients.
    ///
    /// This method computes
    ///
    ///            $ df += mFactor M dx + bFactor B dx + kFactor K dx $
    ///
    /// where M is the mass matrix (associated with inertial forces),
    /// K is the stiffness matrix (associated with forces which derive from a potential),
    /// and B is the damping matrix (associated with viscous forces).
    ///
    /// Very often, at least one of these matrices is null.
    /// In most cases only one of these matrices will be non-null for a given
    /// component. For forcefields without mass it simply calls addDForce.
    ///
    /// \param mparams
    /// - \a mparams->readDx() is the input vector
    /// - \a mparams->mFactor() is the coefficient for mass contributions (i.e. second-order derivatives term in the ODE)
    /// - \a mparams->bFactor() is the coefficient for damping contributions (i.e. first derivatives term in the ODE)
    /// - \a mparams->kFactor() is the coefficient for stiffness contributions (i.e. DOFs term in the ODE)
    /// \param dfId the output vector
    virtual void addMBKdx(const MechanicalParams* mparams, MultiVecDerivId dfId);

    /// \brief Get the potential energy associated to this ForceField during the
    /// last call of addForce( const MechanicalParams* mparams );
    ///
    /// Used to extimate the total energy of the system by some
    /// post-stabilization techniques.
    virtual SReal getPotentialEnergy( const MechanicalParams* mparams = MechanicalParams::defaultInstance() ) const=0;
    /// @}


    /// @name Matrix operations
    /// @{

    /// \brief Compute the system matrix corresponding to k K
    ///
    /// \param mparams \a mparams->kFactor() is the coefficient for stiffness contributions (i.e. DOFs term in the ODE)
    /// \param matrix the matrix to add the result to
    virtual void addKToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix ) = 0;
    //virtual void addKToMatrix(sofa::defaulttype::BaseMatrix * matrix, SReal kFact, unsigned int &offset);

    virtual void addSubKToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix, const helper::vector<unsigned> & subMatrixIndex);

    /// \brief Compute the system matrix corresponding to b B
    ///
    /// \param mparams \a mparams->bFactor() is the coefficient for damping contributions (i.e. first derivatives term in the ODE)
    /// \param matrix the matrix to add the result to
    virtual void addBToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix );
    //virtual void addBToMatrix(sofa::defaulttype::BaseMatrix * matrix, SReal bFact, unsigned int &offset);

    virtual void addSubBToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix, const helper::vector<unsigned> & vecIds);

    /// \brief Compute the system matrix corresponding to m M + b B + k K
    ///
    /// \param mparams
    /// - \a mparams->mFactor() is the coefficient for mass contributions (i.e. second-order derivatives term in the ODE)
    /// - \a mparams->bFactor() is the coefficient for damping contributions (i.e. first derivatives term in the ODE)
    /// - \a mparams->kFactor() is the coefficient for stiffness contributions (i.e. DOFs term in the ODE)
    /// \param matrix the matrix to add the result to
    virtual void addMBKToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix );
    ////virtual void addMBKToMatrix(sofa::defaulttype::BaseMatrix * matrix, SReal mFact, SReal bFact, SReal kFact, unsigned int &offset);

    /// \brief addMBKToMatrix only on the subMatrixIndex
    virtual void addSubMBKToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix, const helper::vector<unsigned> subMatrixIndex);

    /// @}



    /** @name API to consider the ForceField as a constraint as in the "Compliant formulation"
     * see [M Tournier, M Nesme, B Gilles, F Faure, Stable Constrained Dynamics, Siggraph 2015] for more details
     * Each ForceField may be processed either as a traditional force function, or a as a compliance (provided that its stiffness matrix is invertible).
     * If isCompliance==false then the ForceField is handled as a traditional force function.
     * In this case, the stiffness matrix is used to set up the implicit equation matrix, while addForce is used to set up the right-hand term as usual.
     * If isCompliance==true, the ForceField is handled as a compliance and getComplianceMatrix must return a non-null pointer for assembled solver and/or
     * must implement addClambda for a graph-scattered (unassembled) implementation.
     */
    /// @{

    /// Considered as compliance, else considered as stiffness (default to false)
    Data< bool > isCompliance;

    /// Return a pointer to the compliance matrix C
    /// $ C = K^{-1} $
    virtual const sofa::defaulttype::BaseMatrix* getComplianceMatrix(const MechanicalParams*) { return NULL; }

    /// \brief Accumulate the contribution of the C compliant matrix multiplied
    /// by the given Lagrange multipliers lambda vector with the given cFactor coefficient.
    ///
    /// This method computes
    ///
    ///            $ res += cFactor C lambda $
    ///
    /// where C is the Compliant matrix (inverse of the Stiffness matrix K
    /// $ C = K^{-1} $
    ///
    virtual void addClambda(const MechanicalParams* /*mparams*/, MultiVecDerivId /*resId*/, MultiVecDerivId /*lambdaId*/, SReal /*cFactor*/ ){}

    /// @}



    /** @name Rayleigh Damping (stiffness contribution)
     */
    /// @{

    /// Rayleigh Damping stiffness matrix coefficient
    Data< SReal > rayleighStiffness;

    /// @}


    /// Useful when the forcefield is applied only on a subset of dofs.
    /// It is automatically called by addForce.
    ///
    /// That way, we can optimize the time spent to transfer quantities through the mechanical mappings.
    /// Every Dofs are inserted by default. The forcefields using only a subset of dofs should only insert these dofs in the mask.
    virtual void updateForceMask() = 0;



    virtual bool insertInNode( objectmodel::BaseNode* node );
    virtual bool removeInNode( objectmodel::BaseNode* node );
};

} // namespace behavior

} // namespace core

} // namespace sofa

#endif  /* SOFA_CORE_BEHAVIOR_BASEFORCEFIELD_H */
