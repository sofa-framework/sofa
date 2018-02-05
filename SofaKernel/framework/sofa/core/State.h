/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_CORE_STATE_H
#define SOFA_CORE_STATE_H

#include <sofa/core/core.h>
#include <sofa/core/BaseState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <limits>

namespace sofa
{

namespace core
{

/**
 *  \brief Component storing position and velocity vectors.
 *
 *  This class define the interface of components used as source and
 *  destination of regular (non mechanical) mapping. It is then specialized as
 *  MechanicalState (storing other mechanical data) or MappedModel (if no
 *  mechanical data is used, such as for VisualModel).
 *
 *  The given DataTypes class should define the following internal types:
 *  \li \code Real \endcode : scalar values (float or double).
 *  \li \code Coord \endcode : position values.
 *  \li \code Deriv \endcode : derivative values (velocity).
 *  \li \code VecReal \endcode : container of scalar values with the same API as sofa::helper::vector.
 *  \li \code VecCoord \endcode : container of Coord values with the same API as sofa::helper::vector.
 *  \li \code VecDeriv \endcode : container of Deriv values with the same API as sofa::helper::vector.
 *  \li \code MatrixDeriv \endcode : vector of Jacobians (sparse constraint matrices).
 *
 */
template<class TDataTypes>
class State : public virtual BaseState
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(State,TDataTypes), BaseState);

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
    /// Vector of Jacobians (sparse constraint matrices).
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;

    /** @name Accessors
     *  Types and functions to ease data access
     */
    //@{
    typedef helper::ReadAccessor <Data<Real> >    ReadReal;
    typedef helper::WriteAccessor<Data<Real> >    WriteReal;
    typedef helper::ReadAccessor <Data<VecReal> > ReadVecReal;
    typedef helper::WriteAccessor<Data<VecReal> > WriteVecReal;

    typedef helper::ReadAccessor     <Data<Coord> >    ReadCoord;
    typedef helper::WriteAccessor    <Data<Coord> >    WriteCoord;
    typedef helper::WriteOnlyAccessor<Data<Coord> >    WriteOnlyCoord;
    typedef helper::ReadAccessor     <Data<VecCoord> > ReadVecCoord;
    typedef helper::WriteAccessor    <Data<VecCoord> > WriteVecCoord;
    typedef helper::WriteOnlyAccessor<Data<VecCoord> > WriteOnlyVecCoord;
    ReadVecCoord  readPositions() const     { return ReadVecCoord (*this->read (core::ConstVecCoordId::position())); }
    WriteVecCoord writePositions()          { return WriteVecCoord(*this->write(core::VecCoordId::position())); }
    WriteOnlyVecCoord writeOnlyPositions()      { return WriteOnlyVecCoord(*this->write(core::VecCoordId::position())); }
    ReadVecCoord  readRestPositions() const { return ReadVecCoord (*this->read (core::ConstVecCoordId::restPosition())); }
    WriteVecCoord writeRestPositions()      { return WriteVecCoord(*this->write(core::VecCoordId::restPosition())); }
    WriteOnlyVecCoord writeOnlyRestPositions()  { return WriteOnlyVecCoord(*this->write(core::VecCoordId::restPosition())); }

    typedef helper::ReadAccessor     <Data<Deriv> >    ReadDeriv;
    typedef helper::WriteAccessor    <Data<Deriv> >    WriteDeriv;
    typedef helper::WriteOnlyAccessor<Data<Deriv> >    WriteOnlyDeriv;
    typedef helper::ReadAccessor     <Data<VecDeriv> > ReadVecDeriv;
    typedef helper::WriteAccessor    <Data<VecDeriv> > WriteVecDeriv;
    typedef helper::WriteOnlyAccessor<Data<VecDeriv> > WriteOnlyVecDeriv;
    ReadVecDeriv  readVelocities() const    { return ReadVecDeriv (*this->read (core::ConstVecDerivId::velocity())); }
    WriteVecDeriv writeVelocities()         { return WriteVecDeriv(*this->write(core::VecDerivId::velocity())); }
    WriteOnlyVecDeriv writeOnlyVelocities() { return WriteOnlyVecDeriv(*this->write(core::VecDerivId::velocity())); }
    ReadVecDeriv  readForces() const        { return ReadVecDeriv (*this->read (core::ConstVecDerivId::force())); }
    WriteVecDeriv writeForces()             { return WriteVecDeriv(*this->write(core::VecDerivId::force())); }
    WriteOnlyVecDeriv writeOnlyForces()     { return WriteOnlyVecDeriv(*this->write(core::VecDerivId::force())); }
    ReadVecDeriv  readDx() const            { return ReadVecDeriv (*this->read (core::ConstVecDerivId::dx())); }
    WriteVecDeriv writeDx()                 { return WriteVecDeriv(*this->write(core::VecDerivId::dx())); }
    WriteOnlyVecDeriv writeOnlyDx()         { return WriteOnlyVecDeriv(*this->write(core::VecDerivId::dx())); }
    ReadVecDeriv  readNormals() const       { return ReadVecDeriv (*this->read (core::ConstVecDerivId::normal())); }
    //@}


protected:
    State() {}
    virtual ~State() { }
	
private:
	State(const State& n) ;
	State& operator=(const State& n) ;
	
public:
    /// @name New vectors access API based on VecId
    /// @{

    virtual Data< VecCoord >* write(VecCoordId v) = 0;
    virtual const Data< VecCoord >* read(ConstVecCoordId v) const = 0;

    virtual Data< VecDeriv >* write(VecDerivId v) = 0;
    virtual const Data< VecDeriv >* read(ConstVecDerivId v) const = 0;

    virtual Data< MatrixDeriv >* write(MatrixDerivId v) = 0;
    virtual const Data< MatrixDeriv >* read(ConstMatrixDerivId v) const = 0;

    /// @}

    /// @name BaseData vectors access API based on VecId
    /// @{

    virtual objectmodel::BaseData* baseWrite(VecId v) override;
    virtual const objectmodel::BaseData* baseRead(ConstVecId v) const override;

    /// @}


    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const State<DataTypes>* = NULL)
    {
        return TDataTypes::Name();
    }

    virtual void computeBBox(const core::ExecParams* params, bool onlyVisible=false) override;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_CORE_STATE_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_CORE_API State<defaulttype::Vec3dTypes>;
extern template class SOFA_CORE_API State<defaulttype::Vec2dTypes>;
extern template class SOFA_CORE_API State<defaulttype::Vec1dTypes>;
extern template class SOFA_CORE_API State<defaulttype::Vec6dTypes>;
extern template class SOFA_CORE_API State<defaulttype::Rigid3dTypes>;
extern template class SOFA_CORE_API State<defaulttype::Rigid2dTypes>;
extern template class SOFA_CORE_API State<defaulttype::ExtVec3dTypes>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_CORE_API State<defaulttype::Vec3fTypes>;
extern template class SOFA_CORE_API State<defaulttype::Vec2fTypes>;
extern template class SOFA_CORE_API State<defaulttype::Vec1fTypes>;
extern template class SOFA_CORE_API State<defaulttype::Vec6fTypes>;
extern template class SOFA_CORE_API State<defaulttype::Rigid2fTypes>;
extern template class SOFA_CORE_API State<defaulttype::Rigid3fTypes>;
#endif

extern template class SOFA_CORE_API State<defaulttype::ExtVec3fTypes>;

#endif
} // namespace core

} // namespace sofa


#endif
