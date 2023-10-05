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

#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_RigidTypes.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_VecTypes.h>
#include <sofa/core/State.h>
#include <sofa/helper/StringUtils.h>

namespace sofa::core::behavior
{

/**
 *  \brief Component storing all state vectors of a simulated body (position,
 *  velocity, etc), using the datatype specified in the templace.
 *
 *  The given DataTypes class should define the following internal types:
 *  \li \code Real \endcode : scalar values (float or double).
 *  \li \code Coord \endcode : position values.
 *  \li \code Deriv \endcode : derivative values (velocity, forces, displacements).
 *  \li \code VecReal \endcode : container of scalar values with the same API as sofa::type::vector.
 *  \li \code VecCoord \endcode : container of Coord values with the same API as sofa::type::vector.
 *  \li \code VecDeriv \endcode : container of Deriv values with the same API as sofa::type::vector.
 *  \li \code MatrixDeriv \endcode : vector of constraints.
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
    /// Container of scalar values with the same API as sofa::type::vector.
    typedef typename DataTypes::VecReal VecReal;
    /// Container of Coord values with the same API as sofa::type::vector.
    typedef typename DataTypes::VecCoord VecCoord;
    /// Container of Deriv values with the same API as sofa::type::vector.
    typedef typename DataTypes::VecDeriv VecDeriv;
    /// Sparse matrix containing derivative values (constraints)
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;

    using Index = sofa::Index;

public:
    Size getCoordDimension() const override { return defaulttype::DataTypeInfo<Coord>::size(); }
    Size getDerivDimension() const override { return defaulttype::DataTypeInfo<Deriv>::size(); }

    /// Get the indices of the particles located in the given bounding box
    virtual void getIndicesInSpace(sofa::type::vector<Index>& /*indices*/, Real /*xmin*/, Real /*xmax*/,Real /*ymin*/, Real /*ymax*/, Real /*zmin*/, Real /*zmax*/) const=0;

    template<class T>
    static std::string shortName(const T* ptr = nullptr, objectmodel::BaseObjectDescription* arg = nullptr)
    {
        std::string name = Inherit1::shortName(ptr, arg);
        sofa::helper::replaceAll(name, "Mechanical", "M");
        sofa::helper::replaceAll(name, "mechanical", "m");
        return name;
    }

    void copyToBuffer(SReal* dst, ConstVecId src, unsigned n) const override;
    void copyFromBuffer(VecId dst, const SReal* src, unsigned n) override;
    void addFromBuffer(VecId dst, const SReal* src, unsigned n) override;

protected:
    ~MechanicalState() override {}
};

#if !defined(SOFA_CORE_BEHAVIOR_MECHANICALSTATE_CPP)
extern template class SOFA_CORE_API MechanicalState<defaulttype::Vec1Types>;
extern template class SOFA_CORE_API MechanicalState<defaulttype::Vec2Types>;
extern template class SOFA_CORE_API MechanicalState<defaulttype::Vec3Types>;
extern template class SOFA_CORE_API MechanicalState<defaulttype::Vec6Types>;
extern template class SOFA_CORE_API MechanicalState<defaulttype::Rigid2Types>;
extern template class SOFA_CORE_API MechanicalState<defaulttype::Rigid3Types>;
#endif

} /// namespace sofa::core::behavior

