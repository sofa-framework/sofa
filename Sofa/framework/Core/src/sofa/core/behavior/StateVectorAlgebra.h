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
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/StateVectorStorage.h>
#include <sofa/core/VecId.h>
#include <sofa/helper/accessor.h>

namespace sofa::core::behavior
{

/**
 * @class StateVectorAlgebra
 * @brief Implements the vector lifecycle and vector arithmetic API of BaseMechanicalState
 * on top of the vectors stored in a StateVectorStorage.
 *
 * Provided operations:
 * - **Lifecycle**: vAvail, vAlloc, vRealloc, vFree, vInit
 * - **Arithmetic**: vOp, vMultiOp, vThreshold, resetVec
 * - **Reductions**: vDot, vSum, vMax, vSize
 *
 * @tparam TDataTypes The data types used for the state (e.g., coordinates, derivatives).
 */
template <class TDataTypes>
class StateVectorAlgebra : public BaseMechanicalState, public StateVectorStorage<TDataTypes>
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE(StateVectorAlgebra, TDataTypes), BaseMechanicalState, SOFA_TEMPLATE(StateVectorStorage, TDataTypes));

    using DataTypes = TDataTypes;
    using Real = Real_t<TDataTypes>;
    using Coord = Coord_t<TDataTypes>;
    using Deriv = Deriv_t<TDataTypes>;
    using VecCoord = VecCoord_t<TDataTypes>;
    using VecDeriv = VecDeriv_t<TDataTypes>;
    using MatrixDeriv = MatrixDeriv_t<TDataTypes>;
    using VMultiOp = typename BaseMechanicalState::VMultiOp;

    using StateVectorStorage<TDataTypes>::write;
    using StateVectorStorage<TDataTypes>::read;

    /// @name Vector lifecycle
    /// @{

    /// Increment the index of the given VecCoordId, so that all 'allocated' vectors in this state have a lower index
    void vAvail(const ExecParams* params, VecCoordId& vecId) override;
    /// Increment the index of the given VecDerivId, so that all 'allocated' vectors in this state have a lower index
    void vAvail(const ExecParams* params, VecDerivId& vecId) override;

    /// Allocate a new temporary vector
    void vAlloc(const ExecParams* params, VecCoordId vecId, const VecIdProperties& properties = {}) override;
    /// Allocate a new temporary vector
    void vAlloc(const ExecParams* params, VecDerivId vecId, const VecIdProperties& properties = {}) override;

    /// Reallocate a new temporary vector
    void vRealloc(const ExecParams* params, VecCoordId vecId, const VecIdProperties& properties = {}) override;
    /// Reallocate a new temporary vector
    void vRealloc(const ExecParams* params, VecDerivId vecId, const VecIdProperties& properties = {}) override;

    /// Free a temporary vector
    void vFree(const ExecParams* params, VecCoordId v) override;
    /// Free a temporary vector
    void vFree(const ExecParams* params, VecDerivId v) override;

    /// Initialize an unset vector
    void vInit(const ExecParams* params, VecCoordId v, ConstVecCoordId vSrc) override;
    /// Initialize an unset vector
    void vInit(const ExecParams* params, VecDerivId v, ConstVecDerivId vSrc) override;

    /// @}

    /// @name Vector arithmetic
    /// @{

    void vOp(const ExecParams* params, VecId r,
             ConstVecId a = ConstVecId::null(),
             ConstVecId b = ConstVecId::null(), SReal f = 1.0) override;

    void vMultiOp(const ExecParams* params, const VMultiOp& ops) override;

    void vThreshold(VecId a, SReal threshold) override;

    SReal vDot(const ExecParams* params, ConstVecId a, ConstVecId b) override;

    /// Sum of the entries of state vector a at the power of l>0. This is used to compute the l-norm of the vector.
    SReal vSum(const ExecParams* params, ConstVecId a, unsigned l) override;

    /// Maximum of the absolute values of the entries of state vector a. This is used to compute the infinite-norm of the vector.
    SReal vMax(const ExecParams* params, ConstVecId a) override;

    Size vSize(const ExecParams* params, ConstVecId vecId) override;

    void resetVec(const ExecParams* params, VecId vId) override;

    /// @}

protected:
    StateVectorAlgebra() = default;
    ~StateVectorAlgebra() override = default;

    using StateVectorStorage<TDataTypes>::vectorsCoord;
    using StateVectorStorage<TDataTypes>::vectorsDeriv;
    using StateVectorStorage<TDataTypes>::vectorsMatrixDeriv;

    /// Generic implementation of the method vAvail
    template<VecType vtype>
    void vAvailImpl(
        TVecId<vtype, V_WRITE>& v,
        sofa::type::vector< objectmodel::Data< StateVecType_t<DataTypes, vtype> >* >& dataContainer);

    /// Generic implementation of the method vAlloc
    template<VecType vtype>
    void vAllocImpl(TVecId<vtype, V_WRITE> v, const VecIdProperties& properties);

    /// Generic implementation of the method vRealloc
    template<VecType vtype>
    void vReallocImpl(TVecId<vtype, V_WRITE> v, const VecIdProperties& properties);

    /// Generic implementation of the method vFree
    template<VecType vtype>
    void vFreeImpl(TVecId<vtype, V_WRITE> v);

    /// Generic implementation of the method vInit
    template<VecType vtype>
    void vInitImpl(const ExecParams* params,
                   TVecId<vtype, V_WRITE> vId,
                   TVecId<vtype, V_READ> vSrcId);

    /// Shortcut to get a write-only accessor corresponding to the provided VecType from a VecId
    template<VecType vtype>
    helper::WriteOnlyAccessor<objectmodel::Data<StateVecType_t<DataTypes, vtype> > >
        getWriteOnlyAccessor(VecId v);

    /// Shortcut to get a write accessor corresponding to the provided VecType from a VecId
    template<VecType vtype>
    helper::WriteAccessor<objectmodel::Data<StateVecType_t<DataTypes, vtype> > >
        getWriteAccessor(VecId v);

    /// Shortcut to get a read accessor corresponding to the provided VecType from a VecId
    template<VecType vtype>
    helper::ReadAccessor<objectmodel::Data<StateVecType_t<DataTypes, vtype> > >
        getReadAccessor(ConstVecId v);

    template <VecType vtype, VecAccess vaccess>
    static void setVecIdProperties(TVecId<vtype, vaccess> v, const VecIdProperties& properties, BaseData* vec_d);
};

#if !defined(SOFA_CORE_BEHAVIOR_STATEVECTORALGEBRA_CPP)
extern template class SOFA_CORE_API StateVectorAlgebra<defaulttype::Vec1Types>;
extern template class SOFA_CORE_API StateVectorAlgebra<defaulttype::Vec2Types>;
extern template class SOFA_CORE_API StateVectorAlgebra<defaulttype::Vec3Types>;
extern template class SOFA_CORE_API StateVectorAlgebra<defaulttype::Vec6Types>;
extern template class SOFA_CORE_API StateVectorAlgebra<defaulttype::Rigid2Types>;
extern template class SOFA_CORE_API StateVectorAlgebra<defaulttype::Rigid3Types>;
#endif

} // namespace sofa::core::behavior
