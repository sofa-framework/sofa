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
#ifndef SOFA_CORE_BEHAVIOR_BASEMECHANICALSTATE_H
#define SOFA_CORE_BEHAVIOR_BASEMECHANICALSTATE_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/VecId.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/ParticleMask.h>

#include <sstream>
#include <iostream>


namespace sofa
{

namespace core
{

namespace behavior
{

class BaseMechanicalMapping;

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
class BaseMechanicalState : public virtual objectmodel::BaseObject
{
public:
    SOFA_CLASS(BaseMechanicalState, objectmodel::BaseObject);

    BaseMechanicalState():useMask(initData(&useMask, true, "useMask", "Usage of a mask to optimize the computation of the system, highly reducing the passage through the mappings")), forceMask(helper::ParticleMask(useMask))
    {
        useMask.endEdit();
    }
    virtual ~BaseMechanicalState()
    { }

    /// Resize all stored vector
    virtual void resize(int vsize) = 0;

    /// functions that allows to have access to the geometry without a template class : not efficient
    virtual int getSize() const { return 0; }
    virtual unsigned int getCoordDimension() const { return 0; }
    virtual unsigned int getDerivDimension() const { return 0; }
    virtual double getPX(int /*i*/) const { return 0.0; }
    virtual double getPY(int /*i*/) const { return 0.0; }
    virtual double getPZ(int /*i*/) const { return 0.0; }
    virtual defaulttype::Vector3 getScale() const { return defaulttype::Vector3(1.0,1.0,1.0); }

    /// @name Integration related methods
    /// @{

    /// Called at the beginning of each integration step.
    virtual void beginIntegration(double /*dt*/) { }

    /// Called at the end of each iteration step.
    virtual void endIntegration(double /*dt*/) { }

    /// Set F = 0
    virtual void resetForce() =0;//{ vOp( VecId::force() ); }

    /// Set Acc =0
    virtual void resetAcc() =0; //{ vOp( VecId::accFromFrame() ); }

    /// Reset the constraint matrix
    virtual void resetConstraint() =0;

    /// Add stored external forces to F
    virtual void accumulateForce() { }

    /// Add external forces derivatives to F
    virtual void accumulateDf() { }

    /// Translate the current state
    virtual void applyTranslation(const double dx, const double dy, const double dz)=0;


    /// Rotate the current state
    /// This method is optional, it is used when the user want to interactively change the position of an object using Euler angles
    virtual void applyRotation (const double /*rx*/, const double /*ry*/, const double /*rz*/) {};

    /// Rotate the current state
    virtual void applyRotation(const defaulttype::Quat q)=0;

    /// Scale the current state
    virtual void applyScale(const double /*sx*/,const double /*sy*/,const double /*sz*/)=0;

    /// Renumber the constraint ids with the given permutation vector
    virtual void renumberConstraintId(const sofa::helper::vector<unsigned>& renumbering) = 0;


    virtual bool addBBox(double* /*minBBox*/, double* /*maxBBox*/)
    {
        return false;
    }

    /// Identify one vector stored in MechanicalState
    typedef sofa::core::VecId VecId;

    class ConstraintBlock
    {
    public:
        ConstraintBlock( unsigned int c, defaulttype::BaseMatrix *m):column(c),matrix(m) {}

        unsigned int getColumn() const {return column;}
        const defaulttype::BaseMatrix &getMatrix() const {return *matrix;};
        defaulttype::BaseMatrix *getMatrix() {return matrix;};
    protected:
        unsigned int column;
        defaulttype::BaseMatrix *matrix;
    };


    /// Express the matrix L in term of block of matrices, using the indices of the lines in the VecConst container
    virtual std::list<ConstraintBlock> constraintBlocks( const std::list<unsigned int> &/* indices */) const {return std::list<ConstraintBlock>();};


    /// Compute the error given a state vector and a line of the Jacobian (line in vector C)
    virtual SReal getConstraintJacobianTimesVecDeriv( unsigned int /*line*/, VecId /*id*/) {this->serr << "NOT IMPLEMENTED YET" << this->sendl; return (SReal)0;};

    Data<bool> useMask;
    /// Mask to filter the particles. Used inside MechanicalMappings inside applyJ and applyJT methods.
    helper::ParticleMask forceMask;


    /// Increment the index of the given VecId, so that all 'allocated' vectors in this state have a lower index
    virtual void vAvail(VecId& v) = 0;

    /// Allocate a new temporary vector
    virtual void vAlloc(VecId v) = 0;

    /// Free a temporary vector
    virtual void vFree(VecId v) = 0;

    /// Compute a linear operation on vectors : v = a + b * f.
    ///
    /// This generic operation can be used for many simpler cases :
    /// \li v = 0
    /// \li v = a
    /// \li v = a + b
    /// \li v = b * f
    virtual void vOp(VecId v, VecId a = VecId::null(), VecId b = VecId::null(), double f=1.0) = 0; // {}
#ifdef SOFA_SMP
    virtual void vOp(VecId v, VecId a, VecId b, double f, a1::Shared<double> * fSh) = 0; // {}
    virtual void vOpMEq(VecId v, VecId a = VecId::null(), a1::Shared<double> * fSh=NULL) = 0; // {}
    virtual void vDot(a1::Shared<double> *result,VecId a, VecId b) = 0; //{ return 0; }
#endif
    /// Data structure describing a set of linear operation on vectors
    /// \see vMultiOp
    typedef helper::vector< std::pair< VecId, helper::vector< std::pair< VecId, double > > > > VMultiOp;

    /// Perform a sequence of linear vector accumulation operation $r_i = sum_j (v_j*f_{ij})$
    ///
    /// This is used to compute in on steps operations such as $v = v + a*dt, x = x + v*dt$.
    /// Note that if the result vector appears inside the expression, it must be the first operand.
    /// By default this method decompose the computation into multiple vOp calls.
    virtual void vMultiOp(const VMultiOp& ops)
    {
        for(VMultiOp::const_iterator it = ops.begin(), itend = ops.end(); it != itend; ++it)
        {
            VecId r = it->first;
            const helper::vector< std::pair< VecId, double > >& operands = it->second;
            int nop = operands.size();
            if (nop==0)
            {
                vOp(r);
            }
            else if (nop==1)
            {
                if (operands[0].second == 1.0)
                    vOp(r, operands[0].first);
                else
                    vOp(r, VecId::null(), operands[0].first, operands[0].second);
            }
            else
            {
                int i;
                if (operands[0].second == 1.0)
                {
                    vOp(r, operands[0].first, operands[1].first, operands[1].second);
                    i = 2;
                }
                else
                {
                    vOp(r, VecId::null(), operands[0].first, operands[0].second);
                    i = 1;
                }
                for (; i<nop; ++i)
                    vOp(r, r, operands[i].first, operands[i].second);
            }
        }
    }

    /// Compute the scalar products between two vectors.
    virtual double vDot(VecId a, VecId b) = 0; //{ return 0; }

    /// Apply a threshold to all entries
    virtual void vThreshold( VecId a, double threshold )=0;

    /// Make the position vector point to the identified vector.
    ///
    /// To reset it to the default storage use \code setX(VecId::position()) \endcode
    virtual void setX(VecId v) = 0; //{}

    /// Make the free-motion position vector point to the identified vector.
    ///
    /// To reset it to the default storage use \code setV(VecId::freePosition()) \endcode
    virtual void setXfree(VecId v) = 0; //{}

    /// Make the free-motion velocity vector point to the identified vector.
    ///
    /// To reset it to the default storage use \code setV(VecId::freeVelocity()) \endcode
    virtual void setVfree(VecId v) = 0; //{}

    /// Make the velocity vector point to the identified vector.
    ///
    /// To reset it to the default storage use \code setV(VecId::velocity()) \endcode
    virtual void setV(VecId v) = 0; //{}

    /// Make the force vector point to the identified vector.
    ///
    /// To reset it to the default storage use \code setF(VecId::force()) \endcode
    virtual void setF(VecId v) = 0; //{}

    /// Make the displacement vector point to the identified vector.
    ///
    /// To reset it to the default storage use \code setDx(VecId::dx()) \endcode
    /// to make it point to accFromFrame use \code setDx(VecId::accFromFrame()) \endcode
    virtual void setDx(VecId v) = 0; //{}

    /// Make the holonomic constraint system matrix point to either holonomic Constraints or nonHolonomic Constraints.
    ///
    /// To reset it to the default storage or to make it point to holonomicConstraints use \code setDx(VecId::holonomicC()) \endcode
    /// To make it point to nonNolonomicConstraints use \code setC(VecId::NonHolonomicC()) \endcode
    virtual void setC(VecId v) = 0;


    /// new: get the size of the vecconst (used when mapping constraints)
    virtual unsigned int getCSize() const = 0;

    /// new : get compliance on the constraints
    virtual void getCompliance(double ** /*w*/) { }
    /// apply contact force AND compute the subsequent dX
    virtual void applyContactForce(double * /*f*/) { }

    virtual void resetContactForce(void) {}

    virtual void addDxToCollisionModel(void) = 0; //{}

    /// Get the number of scalars per Deriv value, as necessary to build mechanical matrices and vectors.
    /// If not all Derivs have the same number of scalars, then return 1 here and overload the getMatrixSize() method.
    virtual unsigned int getMatrixBlockSize() const { return getDerivDimension(); }

    /// Get the number of rows necessary to build mechanical matrices and vectors.
    /// In most cases this is equivalent to getSize() * getMatrixBlockSize().
    virtual unsigned int getMatrixSize() const { return getSize() * getMatrixBlockSize(); }

    /// Load local mechanical data stored in the state in a global BaseVector basically stored in solvers
    virtual void loadInBaseVector(defaulttype::BaseVector *, VecId , unsigned int &) = 0;

    /// Load local mechanical data stored in the state in a (possibly smaller) BaseVector
    virtual void loadInVector(defaulttype::BaseVector *, VecId , unsigned int) = 0;

    /// Add data stored in a BaseVector to a local mechanical vector of the MechanicalState
    virtual void addBaseVectorToState(VecId , defaulttype::BaseVector *, unsigned int &) = 0;

    /// Add data stored in a Vector (whose size is smaller or equal to the State vector)  to a local mechanical vector of the MechanicalState
    virtual void addVectorToState(VecId , defaulttype::BaseVector *, unsigned int &) = 0;

    /// @}

    /// @name Data output
    /// @{
    virtual void printDOF( VecId, std::ostream& =std::cerr, int firstIndex=0, int range=-1 ) const = 0;
    virtual void initGnuplot(const std::string) {}
    virtual void exportGnuplot(double) {}
    virtual unsigned printDOFWithElapsedTime(VecId, unsigned =0, unsigned =0, std::ostream& =std::cerr ) {return 0;};

    virtual void writeX(std::ostream &out)=0;
    virtual void readX(std::istream &in)=0;
    virtual double compareX(std::istream &in)=0;

    virtual void writeV(std::ostream &out)=0;
    virtual void readV(std::istream &in)=0;
    virtual double compareV(std::istream &in)=0;

    virtual void writeF(std::ostream &out)=0;
    virtual void writeDx(std::ostream &out)=0;
    /// @}

    /// Find mechanical particles hit by the given ray.
    /// A mechanical particle is defined as a 2D or 3D, position or rigid DOF
    /// Returns false if this object does not support picking
    virtual bool pickParticles(double /*rayOx*/, double /*rayOy*/, double /*rayOz*/, double /*rayDx*/, double /*rayDy*/, double /*rayDz*/, double /*radius0*/, double /*dRadius*/,
            std::multimap< double, std::pair<sofa::core::behavior::BaseMechanicalState*, int> >& /*particles*/)
    {
        return false;
    }
};

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
