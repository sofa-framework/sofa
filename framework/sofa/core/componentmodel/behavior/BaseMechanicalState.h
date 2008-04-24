/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASEMECHANICALSTATE_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASEMECHANICALSTATE_H

#include <sofa/defaulttype/Quat.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/defaulttype/Vec.h>
#include <iostream>


namespace sofa
{

namespace core
{

namespace componentmodel
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
    typedef sofa::defaulttype::Vector3::value_type Real_Sofa;
    BaseMechanicalState()
    {}
    virtual ~BaseMechanicalState()
    { }

    /// Resize all stored vector
    virtual void resize(int vsize) = 0;

    /// functions that allows to have access to the geometry without a template class : not efficient
    virtual int getSize() const { return 0; }
    virtual Real_Sofa getPX(int /*i*/) const { return 0.0; }
    virtual Real_Sofa getPY(int /*i*/) const { return 0.0; }
    virtual Real_Sofa getPZ(int /*i*/) const { return 0.0; }

    /// @name Integration related methods
    /// @{

    /// Called at the beginning of each integration step.
    virtual void beginIntegration(Real_Sofa /*dt*/) { }

    /// Called at the end of each iteration step.
    virtual void endIntegration(Real_Sofa /*dt*/) { }

    /// Set F = 0
    virtual void resetForce() =0;//{ vOp( VecId::force() ); }

    /// Reset the constraint matrix
    virtual void resetConstraint() =0;

    /// Add stored external forces to F
    virtual void accumulateForce() { }

    /// Add external forces derivatives to F
    virtual void accumulateDf() { }

    /// Translate the MechanicalObject
    virtual void applyTranslation(const double dx, const double dy, const double dz)=0;

    /// Translate the MechanicalObject
    virtual void applyRotation(const defaulttype::Quat q)=0;

    /// Scale the MechanicalObject
    virtual void applyScale(const double s)=0;


    virtual bool addBBox(Real_Sofa* /*minBBox*/, Real_Sofa* /*maxBBox*/)
    {
        return false;
    }

    /// Identify one vector stored in MechanicalState
    class VecId
    {
    public:
        enum { V_FIRST_DYNAMIC_INDEX = 8 }; ///< This is the first index used for dynamically allocated vectors
        enum Type
        {
            V_NULL=0,
            V_COORD,
            V_DERIV
        };
        Type type;
        unsigned int index;
        VecId(Type t, unsigned int i) : type(t), index(i) { }
        VecId() : type(V_NULL), index(0) { }
        bool isNull() const { return type==V_NULL; }
        static VecId null()     { return VecId(V_NULL,0); }
        static VecId position() { return VecId(V_COORD,0); }
        static VecId restPosition() { return VecId(V_COORD,1); }
        static VecId velocity() { return VecId(V_DERIV,0); }
        static VecId restVelocity() { return VecId(V_DERIV,1); }
        static VecId force() { return VecId(V_DERIV,3); }
        static VecId dx() { return VecId(V_DERIV,4); }
        static VecId freePosition() { return VecId(V_COORD,2); }
        static VecId freeVelocity() { return VecId(V_DERIV,2); }
        /// Test if two VecId identify the same vector
        bool operator==(const VecId& v)
        {
            return type == v.type && index == v.index;
        }
    };

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
    virtual void vOp(VecId v, VecId a = VecId::null(), VecId b = VecId::null(), Real_Sofa f=1.0) = 0; // {}

    /// Compute the scalar products between two vectors.
    virtual Real_Sofa vDot(VecId a, VecId b) = 0; //{ return 0; }

    /// Apply a threshold to all entries
    virtual void vThreshold( VecId a, Real_Sofa threshold )=0;

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
    virtual void setDx(VecId v) = 0; //{}

    /// new : get compliance on the constraints
    virtual void getCompliance(Real_Sofa ** /*w*/) { }
    /// apply contact force AND compute the subsequent dX
    virtual void applyContactForce(Real_Sofa * /*f*/) { }

    virtual void resetContactForce(void) {}

    virtual void addDxToCollisionModel(void) = 0; //{}

    /// Add the Mechanical State Dimension [DOF number * DOF dimension] to the global matrix dimension
    virtual void contributeToMatrixDimension(unsigned int * const, unsigned int * const) = 0;

    /// Load local mechanical data stored in the state in a global BaseVector basically stored in solvers
    virtual void loadInBaseVector(defaulttype::BaseVector *, VecId , unsigned int &) = 0;

    /// Add data stored in a BaseVector to a local mechanical vector of the MechanicalState
    virtual void addBaseVectorToState(VecId , defaulttype::BaseVector *, unsigned int &) = 0;

    /// Update offset index during the subgraph traversal
    virtual void setOffset(unsigned int &) = 0;

    /// @}

    /// @name Data output
    /// @{
    virtual void printDOF( VecId, std::ostream& =std::cerr ) = 0;
    virtual void initGnuplot(const std::string) {}
    virtual void exportGnuplot(Real_Sofa) {}
    virtual unsigned printDOFWithElapsedTime(VecId, unsigned =0, unsigned =0, std::ostream& =std::cerr ) {return 0;};
    /// @}

};

inline std::ostream& operator<<(std::ostream& o, const BaseMechanicalState::VecId& v)
{
    switch (v.type)
    {
    case BaseMechanicalState::VecId::V_NULL: o << "vNull"; break;
    case BaseMechanicalState::VecId::V_COORD: o << "vCoord"; break;
    case BaseMechanicalState::VecId::V_DERIV: o << "vDeriv"; break;
    default: o << "vUNKNOWN"; break;
    }
    o << '[' << v.index << ']';
    return o;
}

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
