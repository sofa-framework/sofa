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
#ifndef SOFA_CORE_BEHAVIOR_MECHANICALSTATE_H
#define SOFA_CORE_BEHAVIOR_MECHANICALSTATE_H

#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/State.h>
#include <sofa/defaulttype/DataTypeInfo.h>

namespace sofa
{

namespace core
{

namespace behavior
{

/**
 *  \brief Component storing all state vectors of a simulated body (position,
 *  velocity, etc), using the datatype specified in the templace.
 *
 *  The given DataTypes class should define the following internal types:
 *  \li \code Real \endcode : scalar values (float or double).
 *  \li \code Coord \endcode : position values.
 *  \li \code Deriv \endcode : derivative values (velocity, forces, displacements).
 *  \li \code VecReal \endcode : container of scalar values with the same API as sofa::helper::vector.
 *  \li \code VecCoord \endcode : container of Coord values with the same API as sofa::helper::vector.
 *  \li \code VecDeriv \endcode : container of Deriv values with the same API as sofa::helper::vector.
 *  \li \code SparseVecDeriv \endcode : sparse vector of Deriv values (defining coefficient of a constraint).
 *  \li \code VecConst \endcode : vector of constraints (i.e. of SparseVecDeriv).
 *
 *  Other vectors can be allocated to store other temporary values.
 *  Vectors can be assigned efficiently by just swapping pointers.
 *
 *  In addition to state vectors, the current constraint system matrix is also
 *  stored, containing the coefficient of each constraint defined over the DOFs
 *  in this body.
 *
 */
template<class TDataTypes>
class MechanicalState : public BaseMechanicalState, public State<TDataTypes>
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE(MechanicalState,TDataTypes), BaseMechanicalState, SOFA_TEMPLATE(State,TDataTypes));

    typedef TDataTypes DataTypes;
    /// Scalar values (float or double).
    typedef typename DataTypes::Real Real;
    /// Position values.
    typedef typename DataTypes::Coord Coord;
    /// Derivative values (velocity, forces, displacements).
    typedef typename DataTypes::Deriv Deriv;
    /// Container of scalar values with the same API as sofa::helper::vector.
    typedef typename DataTypes::VecReal VecReal;
    /// Container of Coord values with the same API as sofa::helper::vector.
    typedef typename DataTypes::VecCoord VecCoord;
    /// Container of Deriv values with the same API as sofa::helper::vector.
    typedef typename DataTypes::VecDeriv VecDeriv;
    ///// Sparse vector of Deriv values (defining coefficient of a constraint).
    //typedef typename DataTypes::SparseVecDeriv SparseVecDeriv;
    ///// Vector of constraints (i.e. of SparseVecDeriv).
    //typedef typename DataTypes::VecConst VecConst;
    /// Sparse matrix containing derivative values (constraints)
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;

    virtual ~MechanicalState() { }

    /// Resize all stored vector
    virtual void resize(int vsize) = 0;

    virtual unsigned int getCoordDimension() const { return defaulttype::DataTypeInfo<Coord>::size(); }
    virtual unsigned int getDerivDimension() const { return defaulttype::DataTypeInfo<Deriv>::size(); }
    /// Return the free-motion velocity vector (read-write access).
    virtual VecDeriv* getVfree() = 0;
    /// Return the current velocity vector (read-write access).
    virtual VecDeriv* getV() = 0;
    /// Return the force vector (read-write access).
    virtual VecDeriv* getF() = 0;
    /// Return the external forces vector (read-write access).
    virtual VecDeriv* getExternalForces() = 0;
    /// Return the displacement vector (read-write access).
    virtual VecDeriv* getDx() = 0;
    /// Return the constraints system matrix (read-write access).
    virtual MatrixDeriv* getC() = 0;
    /// Return the free-motion position vector (read-write access).
    virtual VecCoord* getXfree() = 0;
    /// Return the current position vector (read-write access).
    virtual VecCoord* getX() = 0;
    /// Return the current position vector (read-write access).
    virtual VecCoord* getX0() = 0;
    /// Return the current reset position vector (read-write access)
    /// (return NULL if the state does not store rest position .
    virtual VecCoord* getXReset() = 0;
    // Mechanical State does not store any normal
    virtual VecCoord* getN() { return NULL; };

    /// Return the current position vector (read-only access).
    virtual const VecCoord* getX()  const = 0;
    /// Return the current velocity vector (read-only access).
    virtual const VecDeriv* getV()  const = 0;
    /// Return the force vector (read-only access).
    virtual const VecDeriv* getF()  const = 0;
    /// Return the external forces vector (read-write access).
    virtual const VecDeriv* getExternalForces() const = 0;
    /// Return the displacement vector (read-only access).
    virtual const VecDeriv* getDx() const = 0;
    /// Return the constraints system matrix (read-only access).
    virtual const MatrixDeriv* getC() const = 0;
    /// Return the free-motion position vector (read-only access).
    virtual const VecCoord* getXfree()  const = 0;
    /// Return the free-motion velocity vector (read-only access).
    virtual const VecDeriv* getVfree() const = 0;
    /// Return the current position vector (read-only access).
    virtual const VecCoord* getX0()  const = 0;
    /// Return the current reset position vector (read-write access)
    /// (return NULL if the state does not store rest position .
    virtual const VecCoord* getXReset() const = 0;
    /// Return the initial velocity vector (read-only access).
    virtual const VecDeriv* getV0()  const = 0;
    // Mechanical State does not store any normal
    virtual const VecCoord* getN() const { return NULL; };

    /// Return a VecCoord given its index
    virtual VecCoord* getVecCoord(unsigned int index) = 0;

    /// Return a VecCoord given its index, or NULL if it does not exists
    virtual const VecCoord* getVecCoord(unsigned int index) const = 0;

    /// Return a VecDeriv given its index
    virtual VecDeriv* getVecDeriv(unsigned int index) = 0;

    /// Return a VecDeriv given its index, or NULL if it does not exists
    virtual const VecDeriv* getVecDeriv(unsigned int index) const = 0;

    /// Return a VecConst given its index
    virtual MatrixDeriv* getMatrixDeriv(unsigned int index) = 0;

    /// Return a VecConst given its index, or NULL if it does not exists
    virtual const MatrixDeriv* getMatrixDeriv(unsigned int index) const = 0;

    // virtual unsigned int getCSize() const { return getC()->size(); } // unused
    virtual unsigned int getCSize() const { return 0; } // unused

    /// Get the indices of the particles located in the given bounding box
    virtual void getIndicesInSpace(sofa::helper::vector<unsigned>& /*indices*/, Real /*xmin*/, Real /*xmax*/,Real /*ymin*/, Real /*ymax*/, Real /*zmin*/, Real /*zmax*/) const=0;

    /// Add a constraint ID
    virtual void setConstraintId(unsigned int ) = 0;
    /// Return the constraint IDs corresponding to the entries in the constraints matrix returned by getC()
    virtual sofa::helper::vector<unsigned int>& getConstraintId() = 0;

    virtual VecId getForceId() const = 0;
    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const MechanicalState<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    //static std::string Name(const MechanicalState<DataTypes>* = NULL)
    //{
    //  return std::string("MechanicalState");
    //}

};

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
