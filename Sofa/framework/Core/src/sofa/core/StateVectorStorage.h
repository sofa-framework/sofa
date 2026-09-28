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

#include <sofa/core/config.h>
#include <sofa/core/State.h>

namespace sofa::core
{

template<class TDataTypes>
class StateVectorStorage : public virtual State<TDataTypes>
{
public:
    SOFA_CLASS(
        StateVectorStorage<TDataTypes>,
        State<TDataTypes>);


    using DataTypes = TDataTypes;
    using Real = Real_t<TDataTypes>;
    using Coord = Coord_t<TDataTypes>;
    using Deriv = Deriv_t<TDataTypes>;
    using VecCoord = VecCoord_t<TDataTypes>;
    using VecDeriv = VecDeriv_t<TDataTypes>;
    using MatrixDeriv = MatrixDeriv_t<TDataTypes>;

    ~StateVectorStorage() override;

    void init() override;

    Data< int > d_size; ///< Size of the vectors
    Data< int > f_reserve; ///< Size to reserve when creating vectors. (default=0)


    Size getSize() const override;
    void resize( Size vsize) override;
    virtual void reserve(Size vsize);

    /// @name Vectors access API based on VecId
    /// @{

    virtual Data< VecCoord >* write(VecCoordId v);
    virtual const Data< VecCoord >* read(ConstVecCoordId v) const;

    virtual Data< VecDeriv >* write(VecDerivId v);
    virtual const Data< VecDeriv >* read(ConstVecDerivId v) const;

    virtual Data< MatrixDeriv >* write(MatrixDerivId v);
    virtual const Data< MatrixDeriv >* read(ConstMatrixDerivId v) const;

    /// @}


protected:

    sofa::type::vector< Data< VecCoord >    * > vectorsCoord; ///< Coordinates DOFs vectors table (static and dynamic allocated)
    sofa::type::vector< Data< VecDeriv >    * > vectorsDeriv; ///< Derivates DOFs vectors table (static and dynamic allocated)
    sofa::type::vector< Data< MatrixDeriv > * > vectorsMatrixDeriv; ///< Constraint vectors table

    /**
     * @brief Inserts VecCoord DOF coordinates vector at index in the vectorsCoord container.
     */
    void setVecCoord(core::ConstVecCoordId vecId, Data<VecCoord>* vecData);

    /**
     * @brief Inserts VecDeriv DOF derivates vector at index in the vectorsDeriv container.
     */
    void setVecDeriv(core::ConstVecDerivId vecId, Data<VecDeriv>* vecData);

    /**
     * @brief Inserts MatrixDeriv DOF  at index in the MatrixDeriv container.
     */
    void setVecMatrixDeriv(core::ConstMatrixDerivId vecId, Data< MatrixDeriv>* vecData);


    StateVectorStorage();
};

#if !defined(SOFA_CORE_STATEVECTORSTORAGE_CPP)
extern template class SOFA_CORE_API StateVectorStorage<defaulttype::Vec3dTypes>;
extern template class SOFA_CORE_API StateVectorStorage<defaulttype::Vec2Types>;
extern template class SOFA_CORE_API StateVectorStorage<defaulttype::Vec1Types>;
extern template class SOFA_CORE_API StateVectorStorage<defaulttype::Vec6Types>;
extern template class SOFA_CORE_API StateVectorStorage<defaulttype::Rigid3Types>;
extern template class SOFA_CORE_API StateVectorStorage<defaulttype::Rigid2Types>;
extern template class SOFA_CORE_API StateVectorStorage<defaulttype::Vec3fTypes>;
#endif

}
