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

/**
 * @class StateVectorStorage
 * @brief A storage class for managing state vectors (coordinates, derivatives, and matrix derivatives).
 *
 * This class provides a centralized storage system for vectors associated with degrees of freedom (DOFs).
 * It maps vector identifiers (e.g., `VecCoordId`, `VecDerivId`, `MatrixDerivId`) to `Data<>` containers,
 * enabling dynamic access, resizing, and reservation of vectors.
 *
 * @tparam TDataTypes The data types used for the state (e.g., coordinates, derivatives).
 *
 * @section Features
 * - **Vector Management**: Stores and manages vectors for coordinates (`VecCoord`), derivatives (`VecDeriv`), and matrix derivatives (`MatrixDeriv`).
 * - **Dynamic Access**: Provides methods to read/write vectors using their identifiers (e.g., `write(VecCoordId)`, `read(ConstVecCoordId)`).
 * - **Automatic Creation**: If a `VecId` does not exist in the storage, a new `Data<>` container is dynamically created and associated with the identifier.
 * - **Resizing and Reservation**: Supports resizing vectors to match the state size and reserving memory for performance optimization.
 * - **Automatic Initialization**: Initializes vectors with a default size of 1 and optionally reserves memory based on `f_reserve`.
 *
 * @section Usage
 * Derived classes can associate vector identifiers with `Data<>` containers using the protected methods:
 * - `setVecCoord(core::ConstVecCoordId, Data<VecCoord>*)`
 * - `setVecDeriv(core::ConstVecDerivId, Data<VecDeriv>*)`
 * - `setVecMatrixDeriv(core::ConstMatrixDerivId, Data<MatrixDeriv>*)`
 *
 * Example:
 * @code
 * Data<VecCoord> x;
 * setVecCoord(core::vec_id::write_access::position, &x);
 * @endcode
 *
 * Alternatively, writing to a non-existent `VecId` will automatically create a new `Data<>` container:
 * @code
 * Data<VecCoord>* coordData = write(core::vec_id::write_access::position);
 * @endcode
 *
 * @section Vector Containers
 * - `vectorsCoord`: Table of coordinate vectors (static and dynamically allocated).
 * - `vectorsDeriv`: Table of derivative vectors (static and dynamically allocated).
 * - `vectorsMatrixDeriv`: Table of matrix derivative vectors.
 *
 * @note Debug checks are enabled in non-release builds to ensure vector sizes match the state size.
 */
template <class TDataTypes>
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
