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

#include <sofa/core/fwd.h>
#include <sofa/core/BaseState.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Quat.h>
#include <sofa/linearalgebra/fwd.h> /// For BaseMatrix

namespace sofa::core::behavior
{

/**
 *  \brief Component storing all state vectors of a simulated body (position, velocity, etc).
 *
 *  This class only contains the data of the body and not any of its
 *  <i>active</i> computations, which are handled by the Mass, ForceField, and
 *  Constraint components.
 *
 *  Two types of vectors are used :
 *  \li \code VecCoord \endcode : containing positions.
 *  \li \code VecDeriv \endcode : derivative values, i.e. velocity, forces, displacements.
 *  In most cases they are the same (i.e. 3D/2D point particles), but they can
 *  be different (rigid frames for instance).
 *
 *  Several pre-defined vectors are stored :
 *  \li \code position \endcode
 *  \li \code velocity \endcode
 *  \li \code force \endcode
 *  \li \code dx \endcode (displacement)
 *
 *  Other vectors can be allocated to store other temporary values.
 *  Vectors can be assigned efficiently by just swapping pointers.
 *
 *  In addition to state vectors, the current constraint system matrix is also
 *  stored, containing the coefficient of each constraint defined over the DOFs
 *  in this body.
 *
 */
class SOFA_CORE_API BaseMechanicalState : public virtual BaseState
{
public:
    SOFA_ABSTRACT_CLASS(BaseMechanicalState, BaseState);
    SOFA_BASE_CAST_IMPLEMENTATION(BaseMechanicalState);

    using Index = sofa::Index;
protected:
    BaseMechanicalState();

    ~BaseMechanicalState() override;
private:
    BaseMechanicalState(const BaseMechanicalState& n) = delete;
    BaseMechanicalState& operator=(const BaseMechanicalState& n) = delete;

public:
    /// @name Methods allowing to have access to the geometry without a template class (generic but not efficient)
    /// @{
    virtual SReal getPX(Index /*i*/) const { return 0.0; }
    virtual SReal getPY(Index /*i*/) const { return 0.0; }
    virtual SReal getPZ(Index /*i*/) const { return 0.0; }

    /// @}

    /// @name Vectors allocation and generic operations (based on VecId)
    /// @{
    /// Increment the index of the given VecCoordId, so that all 'allocated' vectors in this state have a lower index
    virtual void vAvail(const ExecParams* params, VecCoordId& v) = 0;
    /// Increment the index of the given VecDerivId, so that all 'allocated' vectors in this state have a lower index
    virtual void vAvail(const ExecParams* params, VecDerivId& v) = 0;
    /// Increment the index of the given MatrixDerivId, so that all 'allocated' vectors in this state have a lower index
    //virtual void vAvail(MatrixDerivId& v) = 0;

    /// Allocate a new temporary vector
    virtual void vAlloc(const ExecParams* params, VecCoordId v, const core::VecIdProperties& properties = {}) = 0;
    /// Allocate a new temporary vector
    virtual void vAlloc(const ExecParams* params, VecDerivId v, const core::VecIdProperties& properties = {}) = 0;
    /// Allocate a new temporary vector
    //virtual void vAlloc(MatrixDerivId v) = 0;

    /// Reallocate a new temporary vector
    virtual void vRealloc(const ExecParams* params, VecCoordId v, const core::VecIdProperties& properties = {}) = 0;
    /// Reallocate a new temporary vector
    virtual void vRealloc(const ExecParams* params, VecDerivId v, const core::VecIdProperties& properties = {}) = 0;


    /// Free a temporary vector
    virtual void vFree(const ExecParams* params, VecCoordId v) = 0;
    /// Free a temporary vector
    virtual void vFree(const ExecParams* params, VecDerivId v) = 0;
    /// Free a temporary vector
    //virtual void vFree(MatrixDerivId v) = 0;

    /// Initialize an unset vector
    virtual void vInit(const ExecParams* params, VecCoordId v, ConstVecCoordId vSrc) = 0;
    /// Initialize an unset vector
    virtual void vInit(const ExecParams* params, VecDerivId v, ConstVecDerivId vSrc) = 0;
    /// Initialize an unset vector
    //virtual void vInit(const ExecParams* params, MatrixDerivId v, ConstMatrixDerivId vSrc) = 0;

    /// \brief Compute a linear operation on vectors : v = a + b * f.
    ///
    /// This generic operation can be used for many simpler cases :
    /// \li v = 0
    /// \li v = a
    /// \li v = a + b
    /// \li v = b * f
    virtual void vOp(const ExecParams* params, VecId v,
                     ConstVecId a = ConstVecId::null(),
                     ConstVecId b = ConstVecId::null(),
                     SReal f = 1.0 ) = 0;

    /// Data structure describing a set of linear operation on vectors
    /// \see vMultiOp
    class VMultiOpEntry : public std::pair< MultiVecId, type::vector< std::pair< ConstMultiVecId, SReal > > >
    {
    public:
        typedef std::pair< ConstMultiVecId, SReal > Fact;
        typedef type::vector< Fact > VecFact;
        typedef std::pair< MultiVecId, VecFact > Inherit;
        VMultiOpEntry() : Inherit(MultiVecId::null(), VecFact()) {}
        VMultiOpEntry(MultiVecId v) : Inherit(v, VecFact()) {}
        VMultiOpEntry(MultiVecId v, ConstMultiVecId a, SReal af = 1.0) : Inherit(v, VecFact())
        { this->second.push_back(Fact(a, af)); }
        VMultiOpEntry(MultiVecId v, ConstMultiVecId a, ConstMultiVecId b, SReal bf = 1.0) : Inherit(v, VecFact())
        { this->second.push_back(Fact(a,1.0));  this->second.push_back(Fact(b, bf)); }
        VMultiOpEntry(MultiVecId v, ConstMultiVecId a, SReal af, ConstMultiVecId b, SReal bf = 1.0) : Inherit(v, VecFact())
        { this->second.push_back(Fact(a, af));  this->second.push_back(Fact(b, bf)); }
    };

    typedef type::vector< VMultiOpEntry > VMultiOp;

    /// \brief Perform a sequence of linear vector accumulation operation $r_i = sum_j (v_j*f_{ij})$
    ///
    /// This is used to compute in on steps operations such as $v = v + a*dt, x = x + v*dt$.
    /// Note that if the result vector appears inside the expression, it must be the first operand.
    /// By default this method decompose the computation into multiple vOp calls.
    virtual void vMultiOp(const ExecParams* params, const VMultiOp& ops);

    /// Compute the scalar products between two vectors.
    virtual SReal vDot(const ExecParams* params, ConstVecId a, ConstVecId b) = 0;

    /// Sum of the entries of state vector a at the power of l>0. This is used to compute the l-norm of the vector.
    virtual SReal vSum(const ExecParams* params, ConstVecId a, unsigned l) = 0;

    /// Maximum of the absolute values of the entries of state vector a. This is used to compute the infinite-norm of the vector.
    virtual SReal vMax(const ExecParams* params, ConstVecId a) = 0;

    /// Get vector size
    virtual Size vSize( const ExecParams* params, ConstVecId v ) = 0;


    /// Apply a threshold (lower bound) to all entries
    virtual void vThreshold( VecId a, SReal threshold ) = 0;

    /// @}

    /// @name Mechanical integration related methods
    /// Note: all these methods can now be implemented generically using VecId-based operations
    /// @{

    /// Called at the beginning of each integration step.
    virtual void beginIntegration(SReal /*dt*/)
    {
        // it is no longer necessary to switch forceId to internalForce here...
    }

    /// Called at the end of each integration step.
    virtual void endIntegration(const ExecParams* params, SReal /*dt*/)
    {
        vOp(params, sofa::core::vec_id::write_access::externalForce, ConstVecId::null(), ConstVecId::null(), 1.0); // externalForce = 0
    }

    /// Set F = 0
    virtual void resetForce( const ExecParams* params, VecDerivId f = vec_id::write_access::force)
    { vOp( params, f, ConstVecId::null(), ConstVecId::null(), 1.0 ); }

    /// Set Acc =0
    virtual void resetAcc( const ExecParams* params, VecDerivId a = vec_id::write_access::dx )
    { vOp( params, a, ConstVecId::null(), ConstVecId::null(), 1.0 ); }

    /// Add stored external forces to F
    virtual void accumulateForce( const ExecParams* params, VecDerivId f = vec_id::write_access::force )
    {
        vOp( params, f, f, sofa::core::vec_id::read_access::externalForce, 1.0 ); // f += externalForce
    }

    /// @}

    /// @name Constraints related methods
    /// @{

    /// Reset the constraint matrix
    virtual void resetConstraint(const ConstraintParams* params) = 0;

    /// build the jacobian of the constraint in a baseMatrix
    virtual void getConstraintJacobian(const ConstraintParams* params, sofa::linearalgebra::BaseMatrix* J,unsigned int & off) = 0;

    /// fill the jacobian matrix (of the constraints) with identity blocks on the provided list of nodes(dofs)
    virtual void buildIdentityBlocksInJacobian(const sofa::type::vector<unsigned int>& list_n, core::MatrixDerivId &mID) = 0;

    class ConstraintBlock
    {
    public:
        ConstraintBlock( unsigned int c, linearalgebra::BaseMatrix *m):column(c),matrix(m) {}

        unsigned int getColumn() const {return column;}
        const linearalgebra::BaseMatrix &getMatrix() const {return *matrix;}
        linearalgebra::BaseMatrix *getMatrix() {return matrix;}
    protected:
        unsigned int column;
        linearalgebra::BaseMatrix *matrix;
    };

    /// Express the matrix L in term of block of matrices, using the indices of the lines in the MatrixDeriv container
    virtual std::list<ConstraintBlock> constraintBlocks( const std::list<unsigned int> &/* indices */) const
    {  return std::list<ConstraintBlock>();  }

    /// Compute the error given a state vector and a line of the Jacobian (line in vector C)
    virtual SReal getConstraintJacobianTimesVecDeriv( unsigned int /*line*/, ConstVecId /*id*/)
    {
        msg_error() << "getConstraintJacobianTimesVecDeriv not implemented yet.";
        return SReal(0.0);
    }

    /// @}

    /// @name events
    ///   Methods related to Event processing
    /// @{

    /// Handle state Changes
    /// @deprecated topological changes now rely on TopologyHandler
    virtual void handleStateChange() {}

    /// Handle state Changes from a given Topology
    /// @deprecated topological changes now rely on TopologyHandler
    virtual void handleStateChange(core::topology::Topology* t);

    ///@}

    /// @name Misc properties and actions
    /// @{

    /// Write current state to the given output stream
    virtual void writeState( std::ostream& out );

    virtual Size getCoordDimension() const { return 0; }
    virtual Size getDerivDimension() const { return 0; }

    /// Translate the current state
    virtual void applyTranslation(const SReal dx, const SReal dy, const SReal dz)=0;

    /// \brief Rotate the current state
    ///
    /// This method is optional, it is used when the user want to interactively change the position of an object using Euler angles
    virtual void applyRotation (const SReal /*rx*/, const SReal /*ry*/, const SReal /*rz*/) {}

    /// Rotate the current state
    virtual void applyRotation(const type::Quat<SReal> q)=0;

    /// Scale the current state
    virtual void applyScale(const SReal /*sx*/,const SReal /*sy*/,const SReal /*sz*/)=0;

    virtual type::Vec3 getScale() const { return type::Vec3(1.0_sreal,1.0_sreal,1.0_sreal); }

    virtual bool addBBox(SReal* /*minBBox*/, SReal* /*maxBBox*/)
    {
        return false;
    }

    /// \brief Find mechanical particles hit by the given ray.
    ///
    /// A mechanical particle is defined as a 2D or 3D, position or rigid DOF
    /// Returns false if this object does not support picking
    virtual bool pickParticles(const ExecParams* /* params */, double /*rayOx*/, double /*rayOy*/, double /*rayOz*/,
            double /*rayDx*/, double /*rayDy*/, double /*rayDz*/,
            double /*radius0*/, double /*dRadius*/,
            std::multimap< double, std::pair<sofa::core::behavior::BaseMechanicalState*, int> >& /*particles*/)
    {
        return false;
    }

    /// @}

    /// @name Interface with BaseMatrix / BaseVector
    /// @{

    /// \brief Get the number of scalars per Deriv value, as necessary to build mechanical matrices and vectors.
    ///
    /// If not all Derivs have the same number of scalars, then return 1 here and overload the getMatrixSize() method.
    virtual Size getMatrixBlockSize() const { return getDerivDimension(); }

    /// \brief Get the number of rows necessary to build mechanical matrices and vectors.
    ///
    /// In most cases this is equivalent to getSize() * getMatrixBlockSize().
    virtual Size getMatrixSize() const { return getSize() * getMatrixBlockSize(); }

    /// \brief Copy data to a global BaseVector from the state stored in a local vector.
    /// @param offset the offset in the BaseVector where the scalar values will be used. It will be updated to the first scalar value after the ones used by this operation when this method returns
    virtual void copyToBaseVector(linearalgebra::BaseVector* dest, ConstVecId src, unsigned int &offset) = 0;

    /// \brief Copy data to a local vector from the state stored in a global BaseVector.
    /// @param offset the offset in the BaseVector where the scalar values will be used. It will be updated to the first scalar value after the ones used by this operation when this method returns
    virtual void copyFromBaseVector(VecId dest, const linearalgebra::BaseVector* src, unsigned int &offset) = 0;

    /// \brief Copy data to a global BaseMatrix from the state stored in a local vector.
    virtual void copyToBaseMatrix(linearalgebra::BaseMatrix* dest, ConstMatrixDerivId src, unsigned int& offset) = 0;

    /// \brief Copy data to an external, user-allocated buffer.
    ///
    /// *Exact* element count must be provided for consistency checks.
    virtual void copyToBuffer(SReal* dst, ConstVecId src, unsigned int n) const = 0;

    /// \brief Copy data from an external, user-allocated buffer.
    ///
    /// *Exact* element count must be provided for consistency checks.
    virtual void copyFromBuffer(VecId dst, const SReal* src, unsigned int n) = 0;

    /// \brief Add data from an external, user-allocated buffer.
    ///
    /// *Exact* element count must be provided for consistency checks.
    virtual void addFromBuffer(VecId dst, const SReal* src, unsigned int n) = 0;

    /// \brief Add data to a global BaseVector from the state stored in a local vector.
    /// @param offset the offset in the BaseVector where the scalar values will be used. It will be updated to the first scalar value after the ones used by this operation when this method returns
    virtual void addToBaseVector(linearalgebra::BaseVector* dest, ConstVecId src, unsigned int &offset) = 0;

    /// \brief
    ///
    /// Perform dest[i][j] += src[offset + i][j] 0<= i < src_entries 0<= j < 3 (for 3D objects) 0 <= j < 2 (for 2D objects)
    /// @param offset the offset in the BaseVector where the scalar values will be used. It will be updated to the first scalar value after the ones used by this operation when this method returns
    virtual void addFromBaseVectorSameSize(VecId dest, const linearalgebra::BaseVector* src, unsigned int &offset) = 0;


    /// \brief
    ///
    /// Perform dest[ offset + i ][j] += src[i][j]  0<= i < src_entries  0<= j < 3 (for 3D objects) 0 <= j < 2 (for 2D objects)
    /// @param offset the offset in the MechanicalObject local vector specified by VecId dest. It will be updated to the first scalar value after the ones used by this operation when this method returns.
    virtual void addFromBaseVectorDifferentSize(VecId dest, const linearalgebra::BaseVector* src, unsigned int &offset ) = 0;
    /// @}

    /// @name Data output
    /// @{

    virtual void printDOF( ConstVecId v, std::ostream& out = std::cerr, int firstIndex = 0, int range = -1 ) const = 0;
    virtual unsigned printDOFWithElapsedTime(ConstVecId /*v*/, unsigned /*count*/ = 0, unsigned /*time*/ = 0, std::ostream& /*out*/ = std::cerr ) { return 0; }
    virtual void initGnuplot(const std::string /*filepath*/) {}
    virtual void exportGnuplot(SReal /*time*/) {}

    virtual void writeVec(ConstVecId v, std::ostream &out) = 0;
    virtual void readVec(VecId v, std::istream &in) = 0;
    virtual SReal compareVec(ConstVecId v, std::istream &in) = 0;

    /// @}getPotent

    bool insertInNode( objectmodel::BaseNode* node ) override;
    bool removeInNode( objectmodel::BaseNode* node ) override;

};

} // namespace sofa::core::behavior
