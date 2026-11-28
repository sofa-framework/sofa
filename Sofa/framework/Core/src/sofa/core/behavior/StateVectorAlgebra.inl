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
#include <sofa/core/behavior/StateVectorAlgebra.h>
#include <sofa/core/StateVectorStorage.inl>
#include <sofa/defaulttype/DataTypeOperations.h>
#include <sofa/helper/accessor.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace sofa::core::behavior::statevectoralgebra
{

/**
 * Helper function to apply a predicate based on the value of a VecType. The predicate is applied
 * only for V_COORD and V_DERIV values. The predicate must be compatible with a std::integral_constant.
 * The usual case is a generic lambda.
 * Example:
 * core::VecId v;
 * applyPredicateIfCoordOrDeriv(v.type, [](auto vtype) {  });
 */
template<class Function>
bool applyPredicateIfCoordOrDeriv(sofa::core::VecType vtype, Function&& f)
{
    if (vtype == sofa::core::V_COORD)
    {
        std::forward<Function>(f)(std::integral_constant<sofa::core::VecType, sofa::core::V_COORD>{});
    }
    else if (vtype == sofa::core::V_DERIV)
    {
        std::forward<Function>(f)(std::integral_constant<sofa::core::VecType, sofa::core::V_DERIV>{});
    }
    else
    {
        return false;
    }
    return true;
}

/**
 * Overload of the previous function accepting two VecType. The predicate must accept two parameters.
 * Warning: the case where vtype_a is V_DERIV and vtype_b is V_COORD is not applied
 */
template<class Function>
bool applyPredicateIfCoordOrDeriv(sofa::core::VecType vtype_a, sofa::core::VecType vtype_b, Function&& f)
{
    if (vtype_a == sofa::core::V_COORD)
    {
        return applyPredicateIfCoordOrDeriv(vtype_b, [&f](auto b)
        {
            std::forward<Function>(f)(std::integral_constant<sofa::core::VecType, sofa::core::V_COORD>{},
                                      std::integral_constant<sofa::core::VecType, b>{});
        });
    }
    else if (vtype_a == sofa::core::V_DERIV)
    {
        if (vtype_b == sofa::core::V_COORD)
        {
            // predicate cannot be applied because vtype_a and vtype_b are incompatible:
            // it is not permitted to perform operations such as [DERIV] += [COORD]
            return false;
        }
        else if (vtype_b == sofa::core::V_DERIV)
        {
            std::forward<Function>(f)(std::integral_constant<sofa::core::VecType, sofa::core::V_DERIV>{},
                                      std::integral_constant<sofa::core::VecType, sofa::core::V_DERIV>{});
            return true;
        }
    }
    return false;
}

template <class T> inline void clear( T& t )
{
    t.clear();
}

template<> inline void clear( float& t )
{
    t=0;
}

template<> inline void clear( double& t )
{
    t=0;
}

} // namespace sofa::core::behavior::statevectoralgebra

namespace sofa::core::behavior
{

template <class DataTypes>
template <core::VecType vtype>
void StateVectorAlgebra<DataTypes>::vAvailImpl(core::TVecId<vtype, core::V_WRITE>& v,
    sofa::type::vector<Data<core::StateVecType_t<DataTypes, vtype> >* >& dataContainer)
{
    for (unsigned int i = v.index; i < dataContainer.size(); ++i)
    {
        if (dataContainer[i] && dataContainer[i]->isSet())
        {
            v.index = i+1;
        }
    }
}

template <class DataTypes>
void StateVectorAlgebra<DataTypes>::vAvail(const core::ExecParams* params, core::VecCoordId& vecId)
{
    SOFA_UNUSED(params);
    vAvailImpl(vecId, vectorsCoord);
}

template <class DataTypes>
void StateVectorAlgebra<DataTypes>::vAvail(const core::ExecParams* params, core::VecDerivId& vecId)
{
    SOFA_UNUSED(params);
    vAvailImpl(vecId, vectorsDeriv);
}

template <class DataTypes>
template <core::VecType vtype>
void StateVectorAlgebra<DataTypes>::vAllocImpl(
    core::TVecId<vtype, core::V_WRITE> v,
    const core::VecIdProperties& properties)
{
    if (v.index >= core::TVecId<vtype, core::V_WRITE>::V_FIRST_DYNAMIC_INDEX)
    {
        auto* vec_d = this->write(v);
        vec_d->beginEdit()->resize(this->d_size.getValue());
        vec_d->endEdit();

        setVecIdProperties(v, properties, vec_d);
    }

    //vOp(v); // clear vector
}

template <class DataTypes>
void StateVectorAlgebra<DataTypes>::vAlloc(const core::ExecParams* params, core::VecCoordId vecId, const core::VecIdProperties& properties)
{
    SOFA_UNUSED(params);
    vAllocImpl(vecId, properties);
}

template <class DataTypes>
void StateVectorAlgebra<DataTypes>::vAlloc(const core::ExecParams* params, core::VecDerivId vecId, const core::VecIdProperties& properties)
{
    SOFA_UNUSED(params);
    vAllocImpl(vecId, properties);
}

template <class DataTypes>
template <core::VecType vtype>
void StateVectorAlgebra<DataTypes>::vReallocImpl(
    core::TVecId<vtype, core::V_WRITE> v,
    const core::VecIdProperties& properties)
{
    auto* vec_d = this->write(v);

    if ( !vec_d->isSet() /*&& v.index >= core::TVecId<vtype, core::V_WRITE>::V_FIRST_DYNAMIC_INDEX*/ )
    {
        vec_d->beginEdit()->resize(this->d_size.getValue());
        vec_d->endEdit();
    }

    setVecIdProperties(v, properties, vec_d);
}

template <class DataTypes>
void StateVectorAlgebra<DataTypes>::vRealloc(const core::ExecParams* params, core::VecCoordId vecId, const core::VecIdProperties& properties)
{
    SOFA_UNUSED(params);
    vReallocImpl(vecId, properties);
}

template <class DataTypes>
void StateVectorAlgebra<DataTypes>::vRealloc(const core::ExecParams* params, core::VecDerivId vecId, const core::VecIdProperties& properties)
{
    SOFA_UNUSED(params);
    vReallocImpl(vecId, properties);
}

template <class DataTypes>
template <core::VecType vtype, core::VecAccess vaccess>
void StateVectorAlgebra<DataTypes>::setVecIdProperties(core::TVecId<vtype, vaccess> v, const core::VecIdProperties& properties, core::BaseData* vec_d)
{
    if (!properties.label.empty())
    {
        const std::string newname = properties.label;
        const std::string oldname = properties.label + core::VecTypeLabels.at(vtype);
        const auto base = vec_d->getOwner();
        if(base && !base->findData(oldname))
        {
            base->addAlias(vec_d, oldname.c_str());

            // This allows to find the data field in getData() using its new name eg: constraint_dx
            if(base->findData(newname))
                msg_error(base) << "Unable to expose a dynamic data field named '" << newname << "' as that name already exists. Please report this issue to https://github.com/sofa-framework/sofa/issues (for more infos see PR: https://github.com/sofa-framework/sofa/pull/3783)";
            else
                base->addAlias(vec_d, newname.c_str());

        }
        vec_d->setName(newname);
        vec_d->setHelp("VecId: " + oldname);
    }
    if (!properties.group.empty())
    {
        vec_d->setGroup(properties.group);
    }
    else
    {
        const auto group = v.getGroup();
        if (!group.empty())
        {
            vec_d->setGroup(group);
        }
    }
}

template <class DataTypes>
template <core::VecType vtype>
void StateVectorAlgebra<DataTypes>::vFreeImpl(core::TVecId<vtype, core::V_WRITE> v)
{
    if (v.index >= core::TVecId<vtype, core::V_WRITE>::V_FIRST_DYNAMIC_INDEX)
    {
        auto* vec_d = this->write(v);

        auto* vec = vec_d->beginEdit();
        vec->resize(0);
        vec_d->endEdit();

        vec_d->unset();
    }
}

template <class DataTypes>
void StateVectorAlgebra<DataTypes>::vFree(const core::ExecParams* params, core::VecCoordId vId)
{
    SOFA_UNUSED(params);
    vFreeImpl(vId);
}

template <class DataTypes>
void StateVectorAlgebra<DataTypes>::vFree(const core::ExecParams* params, core::VecDerivId vId)
{
    SOFA_UNUSED(params);
    vFreeImpl(vId);
}

template <class DataTypes>
template <core::VecType vtype>
void StateVectorAlgebra<DataTypes>::vInitImpl(const core::ExecParams* params,
    core::TVecId<vtype, core::V_WRITE> vId,
    core::TVecId<vtype, core::V_READ> vSrcId)
{
    auto* vec_d = this->write(vId);

    if (!vec_d->isSet() || vec_d->getValue().empty())
    {
        vec_d->forceSet();
        vOp(params, vId, vSrcId);
    }
}

template <class DataTypes>
void StateVectorAlgebra<DataTypes>::vInit(const core::ExecParams* params
                                        , core::VecCoordId vId
                                        , core::ConstVecCoordId vSrcId)
{
    vInitImpl(params, vId, vSrcId);
}

template <class DataTypes>
void StateVectorAlgebra<DataTypes>::vInit(const core::ExecParams* params,
                                        core::VecDerivId vId,
                                        core::ConstVecDerivId vSrcId)
{
    vInitImpl(params, vId, vSrcId);
}

template <class DataTypes>
template <core::VecType vtype>
helper::WriteOnlyAccessor<core::objectmodel::Data<core::StateVecType_t<DataTypes, vtype>>>
StateVectorAlgebra<DataTypes>::getWriteOnlyAccessor(core::VecId v)
{
    return sofa::helper::getWriteOnlyAccessor(
        *this->write(core::TVecId<vtype, core::V_WRITE>(v)));
}

template <class DataTypes>
template <core::VecType vtype>
helper::WriteAccessor<core::objectmodel::Data<core::StateVecType_t<DataTypes, vtype>>> StateVectorAlgebra<
DataTypes>::getWriteAccessor(core::VecId v)
{
    return sofa::helper::getWriteAccessor(
        *this->write(core::TVecId<vtype, core::V_WRITE>(v)));
}

template <class DataTypes>
template <core::VecType vtype>
helper::ReadAccessor<core::objectmodel::Data<core::StateVecType_t<DataTypes, vtype>>> StateVectorAlgebra<
DataTypes>::getReadAccessor(core::ConstVecId v)
{
    return sofa::helper::getReadAccessor(
        *this->read(core::TVecId<vtype, core::V_READ>(v)));
}

#define APPLY_PREDICATE_ONEPARAMS(PRED, v)\
    bool canApplyPredicate = false;\
    if (v.type == core::V_COORD)\
    {\
        canApplyPredicate = PRED<core::V_COORD>(*this, v, k);\
    }\
    else if (v.type == core::V_DERIV)\
    {\
        canApplyPredicate = PRED<core::V_DERIV>(*this, v, k);\
    }

#define APPLY_PREDICATE_TWOPARAMS(PRED, v, b)\
    bool canApplyPredicate = false;\
    if (v.type == core::V_COORD)\
    {\
        if (b.type == core::V_COORD)\
        {\
            canApplyPredicate = PRED<core::V_COORD, core::V_COORD>(*this, v, b, k);\
        }\
        else if (b.type == core::V_DERIV)\
        {\
            canApplyPredicate = PRED<core::V_COORD, core::V_DERIV>(*this, v, b, k);\
        }\
    }\
    else if (v.type == core::V_DERIV)\
    {\
        if (b.type == core::V_COORD)\
        {\
            canApplyPredicate = PRED<core::V_DERIV, core::V_COORD>(*this, v, b, k);\
        }\
        else if (b.type == core::V_DERIV)\
        {\
            canApplyPredicate = PRED<core::V_DERIV, core::V_DERIV>(*this, v, b, k);\
        }\
    }

#define APPLY_PREDICATE_THREEPARAMS(PRED, v, a, b)\
    bool canApplyPredicate = false;\
    if (v.type == core::V_COORD)\
    {\
        if (a.type == core::V_COORD)\
        {\
            if (b.type == core::V_COORD)\
            {\
                canApplyPredicate = PRED<core::V_COORD, core::V_COORD, core::V_COORD>(*this, v, a, b, k);\
            }\
            else if (b.type == core::V_DERIV)\
            {\
                canApplyPredicate = PRED<core::V_COORD, core::V_COORD, core::V_DERIV>(*this, v, a, b, k);\
            }\
        }\
        else if (a.type == core::V_DERIV)\
        {\
            if (b.type == core::V_COORD)\
            {\
                canApplyPredicate = PRED<core::V_COORD, core::V_DERIV, core::V_COORD>(*this, v, a, b, k);\
            }\
            else if (b.type == core::V_DERIV)\
            {\
                canApplyPredicate = PRED<core::V_COORD, core::V_DERIV, core::V_DERIV>(*this, v, a, b, k);\
            }\
        }\
    }\
    else if (v.type == core::V_DERIV)\
    {\
        if (a.type == core::V_COORD)\
        {\
            if (b.type == core::V_COORD)\
            {\
                canApplyPredicate = PRED<core::V_DERIV, core::V_COORD, core::V_COORD>(*this, v, a, b, k);\
            }\
            else if (b.type == core::V_DERIV)\
            {\
                canApplyPredicate = PRED<core::V_DERIV, core::V_COORD, core::V_DERIV>(*this, v, a, b, k);\
            }\
        }\
        else if (a.type == core::V_DERIV)\
        {\
            if (b.type == core::V_COORD)\
            {\
                canApplyPredicate = PRED<core::V_DERIV, core::V_DERIV, core::V_COORD>(*this, v, a, b, k);\
            }\
            else if (b.type == core::V_DERIV)\
            {\
                canApplyPredicate = PRED<core::V_DERIV, core::V_DERIV, core::V_DERIV>(*this, v, a, b, k);\
            }\
        }\
    }

namespace statevectoralgebra
{

template <core::VecType vtype, class DataTypes>
bool vOp_vf(StateVectorAlgebra<DataTypes>& self, core::VecId v, Real_t<DataTypes> k)
{
    if constexpr (requires(core::StateVecType_t<DataTypes, vtype> vc, Real_t<DataTypes> fc){vc[0] *= fc;})
    {
        // write accessor (instead of write-only accessor) because content must be updated if dirty before the multiplication
        auto vv = helper::getWriteAccessor(*self.write(core::TVecId<vtype, core::V_WRITE>(v)));
        for (unsigned int i = 0; i < vv.size(); ++i)
        {
            vv[i] *= k;
        }
        return true;
    }
    else
    {
        return false;
    }
}

template <core::VecType vtype, core::VecType btype, class DataTypes>
bool vOp_vbf(StateVectorAlgebra<DataTypes>& self, core::VecId v, core::ConstVecId b, Real_t<DataTypes> k)
{
    if constexpr (requires(core::StateVecType_t<DataTypes, vtype> vc, core::StateVecType_t<DataTypes, btype> bc, Real_t<DataTypes> fc){vc[0] = bc[0] * fc;})
    {
        // Read accessors are created before the write-only accessor on purpose: see vOp_va.
        auto vb = helper::getReadAccessor(*self.read(core::TVecId<btype, core::V_READ>(b)));
        if (vb.empty())
            return true;

        // write-only accessor (instead of write-accessor) because content is irrelevant and will be overwritten
        auto vv = helper::getWriteOnlyAccessor(*self.write(core::TVecId<vtype, core::V_WRITE>(v)));
        if (vv.size() < vb.size())
            vv.resize(vb.size());
        for (unsigned int i = 0; i < vb.size(); ++i)
        {
            vv[i] = vb[i] * k;
        }
        return true;
    }
    else
    {
        return false;
    }
}

template <core::VecType vtype, core::VecType btype, class DataTypes>
bool vOp_va(StateVectorAlgebra<DataTypes>& self, core::VecId v, core::ConstVecId b, Real_t<DataTypes> k)
{
    SOFA_UNUSED(k);
    if constexpr (requires(core::StateVecType_t<DataTypes, vtype> vc, core::StateVecType_t<DataTypes, btype> bc){vc[0] = bc[0];})
    {
        // The read accessor must be created *before* the write-only accessor: reading b may
        // trigger a Data update (e.g. b is linked to an engine output), which reassigns the
        // copy-on-write shared_ptr and would free the buffer captured by a write accessor
        // opened earlier. This matters in particular when v == b (self-copy of a linked Data).
        auto vb = helper::getReadAccessor(*self.read(core::TVecId<btype, core::V_READ>(b)));

        // write-only accessor (instead of write-accessor) because content is irrelevant and will be overwritten
        auto vv = helper::getWriteOnlyAccessor(*self.write(core::TVecId<vtype, core::V_WRITE>(v)));
        vv.wref() = vb.ref();
        return true;
    }
    else
    {
        return false;
    }
}

template <core::VecType vtype, core::VecType btype, class DataTypes>
bool vOp_vb(StateVectorAlgebra<DataTypes>& self, core::VecId v, core::ConstVecId b, Real_t<DataTypes> k)
{
    SOFA_UNUSED(k);
    if constexpr (requires(core::StateVecType_t<DataTypes, vtype> vc, core::StateVecType_t<DataTypes, btype> bc){vc[0] += bc[0];})
    {
        // Read accessors are created before the write-only accessor on purpose: see vOp_va.
        auto vb = helper::getReadAccessor(*self.read(core::TVecId<btype, core::V_READ>(b)));
        if (vb.empty())
            return true;

        // write accessor (instead of write-only accessor) because content must be updated if dirty before the accumulation
        auto vv = helper::getWriteAccessor(*self.write(core::TVecId<vtype, core::V_WRITE>(v)));
        if (vv.size() < vb.size())
            vv.resize(vb.size());
        for (unsigned int i = 0; i < vb.size(); ++i)
        {
            vv[i] += vb[i];
        }
        return true;
    }
    else
    {
        return false;
    }
}

template <core::VecType vtype, core::VecType btype, class DataTypes>
bool vOp_avf(StateVectorAlgebra<DataTypes>& self, core::VecId v, core::ConstVecId b, Real_t<DataTypes> k)
{
    if constexpr (requires(core::StateVecType_t<DataTypes, vtype> vc, core::StateVecType_t<DataTypes, btype> bc){vc[0] *= k; vc[0] += bc[0];})
    {
        // Read accessors are created before the write-only accessor on purpose: see vOp_va.
        auto vb = helper::getReadAccessor(*self.read(core::TVecId<btype, core::V_READ>(b)));

        // write accessor (instead of write-only accessor) because content must be updated if dirty before the accumulation
        auto vv = helper::getWriteAccessor(*self.write(core::TVecId<vtype, core::V_WRITE>(v)));
        if (vv.size() < vb.size())
            vv.resize(vb.size());
        for (unsigned int i = 0; i < vb.size(); ++i)
        {
            vv[i] *= k;
            vv[i] += vb[i];
        }
        return true;
    }
    else
    {
        return false;
    }
}

template <core::VecType vtype, core::VecType btype, class DataTypes>
bool vOp_v_inc_bf(StateVectorAlgebra<DataTypes>& self, core::VecId v, core::ConstVecId b, Real_t<DataTypes> k)
{
    if constexpr (requires(core::StateVecType_t<DataTypes, vtype> vc, core::StateVecType_t<DataTypes, btype> bc, Real_t<DataTypes> fc){vc[0] += bc[0] * fc;})
    {
        // Read accessors are created before the write-only accessor on purpose: see vOp_va.
        auto vb = helper::getReadAccessor(*self.read(core::TVecId<btype, core::V_READ>(b)));
        if (vb.empty())
            return true;

        // write accessor (instead of write-only accessor) because content must be updated if dirty before the accumulation
        auto vv = helper::getWriteAccessor(*self.write(core::TVecId<vtype, core::V_WRITE>(v)));
        if (vv.size() < vb.size())
            vv.resize(vb.size());
        for (unsigned int i = 0; i < vb.size(); ++i)
        {
            vv[i] += vb[i] * k;
        }
        return true;
    }
    else
    {
        return false;
    }
}

template <core::VecType vtype, core::VecType atype, core::VecType btype, class DataTypes>
bool vOp_vab(StateVectorAlgebra<DataTypes>& self, core::VecId v, core::ConstVecId a, core::ConstVecId b, Real_t<DataTypes> k)
{
    SOFA_UNUSED(k);

    if constexpr (requires(core::StateVecType_t<DataTypes, vtype> vc, core::StateVecType_t<DataTypes, atype> ac, core::StateVecType_t<DataTypes, btype> bc){vc[0] = ac[0] + bc[0];})
    {
        // Read accessors are created before the write-only accessor on purpose: see vOp_va.
        auto va = helper::getReadAccessor(*self.read(core::TVecId<atype, core::V_READ>(a)));
        if (va.empty())
            return true;

        auto vb = helper::getReadAccessor(*self.read(core::TVecId<btype, core::V_READ>(b)));
        if (vb.empty())
            return true;

        // write-only accessor (instead of write-accessor) because content is irrelevant and will be overwritten
        auto vv = helper::getWriteOnlyAccessor(*self.write(core::TVecId<vtype, core::V_WRITE>(v)));
        const auto size = std::min(va.size(), vb.size());
        if (vv.size() < size)
            vv.resize(size);
        for (unsigned int i = 0; i < size; ++i)
        {
            vv[i] = va[i] + vb[i];
        }
        return true;
    }
    else
    {
        return false;
    }
}

template <core::VecType vtype, core::VecType atype, core::VecType btype, class DataTypes>
bool vOp_vabf(StateVectorAlgebra<DataTypes>& self, core::VecId v, core::ConstVecId a, core::ConstVecId b, Real_t<DataTypes> k)
{
    if constexpr (requires(core::StateVecType_t<DataTypes, vtype> vc, core::StateVecType_t<DataTypes, atype> ac, core::StateVecType_t<DataTypes, btype> bc, Real_t<DataTypes> fc){vc[0] = ac[0] + bc[0] * fc;})
    {
        // Read accessors are created before the write-only accessor on purpose: see vOp_va.
        auto va = helper::getReadAccessor(*self.read(core::TVecId<atype, core::V_READ>(a)));
        if (va.empty())
            return true;

        auto vb = helper::getReadAccessor(*self.read(core::TVecId<btype, core::V_READ>(b)));
        if (vb.empty())
            return true;

        // write-only accessor (instead of write-accessor) because content is irrelevant and will be overwritten
        auto vv = helper::getWriteOnlyAccessor(*self.write(core::TVecId<vtype, core::V_WRITE>(v)));
        const auto size = std::min(va.size(), vb.size());
        if (vv.size() < size)
            vv.resize(size);
        for (unsigned int i = 0; i < size; ++i)
        {
            vv[i] = va[i] + vb[i] * k;
        }
        return true;
    }
    else
    {
        return false;
    }
}

} // namespace statevectoralgebra

template <class DataTypes>
void StateVectorAlgebra<DataTypes>::vOp(const core::ExecParams* params, core::VecId r,
                                      core::ConstVecId a, core::ConstVecId b, SReal k)
{
    SOFA_UNUSED(params);

    if(r.isNull())
    {
        // ERROR
        msg_error() << "Invalid vOp operation 1 (" << r << ',' << a << ',' << b << ',' << k << ")";
        return;
    }

    if (a.isNull())
    {
        if (b.isNull())
        {
            // r = 0
            statevectoralgebra::applyPredicateIfCoordOrDeriv(r.type, [this, &r](auto vec)
            {
                auto vv = getWriteOnlyAccessor<vec>(r);
                vv.resize(this->d_size.getValue());
                sofa::defaulttype::resetDataTypeVec(vv.wref());
            });
        }
        else
        {
            if (r == b)
            {
                // r *= k
                APPLY_PREDICATE_ONEPARAMS(statevectoralgebra::vOp_vf, r)
                msg_error_when(!canApplyPredicate) << "Cannot apply vector operation r *= k (" << r << ',' << a << ',' << b << ',' << k << ")";
            }
            else
            {
                // r = b*k
                APPLY_PREDICATE_TWOPARAMS(statevectoralgebra::vOp_vbf, r, b)
                msg_error_when(!canApplyPredicate) << "Cannot apply vector operation r = b*k (" << r << ',' << a << ',' << b << ',' << k << ")";
            }
        }
    }
    else
    {
        if (b.isNull())
        {
            // r = a
            APPLY_PREDICATE_TWOPARAMS(statevectoralgebra::vOp_va, r, a)
            msg_error_when(!canApplyPredicate) << "Cannot apply vector operation r = a (" << r << ',' << a << ',' << b << ',' << k << ")";
        }
        else
        {
            if (r == a)
            {
                if (k == 1._sreal)
                {
                    // r += b
                    APPLY_PREDICATE_TWOPARAMS(statevectoralgebra::vOp_vb, r, b)
                    msg_error_when(!canApplyPredicate) << "Cannot apply vector operation r += b (" << r << ',' << a << ',' << b << ',' << k << ")";
                }
                else
                {
                    // r += b*k
                    APPLY_PREDICATE_TWOPARAMS(statevectoralgebra::vOp_v_inc_bf, r, b)
                    msg_error_when(!canApplyPredicate) << "Cannot apply vector operation r += b*k (" << r << ',' << a << ',' << b << ',' << k << ")";
                }
            }
            else if (r == b)
            {
                if (k == 1._sreal)
                {
                    // r += a
                    APPLY_PREDICATE_TWOPARAMS(statevectoralgebra::vOp_vb, r, a)
                    msg_error_when(!canApplyPredicate) << "Cannot apply vector operation r += a (" << r << ',' << a << ',' << b << ',' << k << ")";
                }
                else
                {
                    // r = a+r*k
                    APPLY_PREDICATE_TWOPARAMS(statevectoralgebra::vOp_avf, r, a)
                    msg_error_when(!canApplyPredicate) << "Cannot apply vector operation r = a+r*k (" << r << ',' << a << ',' << b << ',' << k << ")";
                }
            }
            else
            {
                if (k == 1._sreal)
                {
                    // r = a+b
                    APPLY_PREDICATE_THREEPARAMS(statevectoralgebra::vOp_vab, r, a, b)
                    msg_error_when(!canApplyPredicate) << "Cannot apply vector operation r = a+b (" << r << ',' << a << ',' << b << ',' << k << ")";
                }
                else
                {
                    const auto generalCase = [&]()
                    {
                        // r = a+b*k
                        APPLY_PREDICATE_THREEPARAMS(statevectoralgebra::vOp_vabf, r, a, b)
                        msg_error_when(!canApplyPredicate) << "Cannot apply vector operation r = a+b*k (" << r << ',' << a << ',' << b << ',' << k << ")";
                    };
                    
                    if constexpr (requires (Coord ca, Coord cb) {DataTypes::coordDifference(ca, cb);})
                    {
                        if (k == -1._sreal &&
                           (r.type == core::V_DERIV && a.type == core::V_COORD && b.type == core::V_COORD))
                        {
                            // r = a-b
                            // Read accessors are created before the write-only accessor on purpose: see vOp_va.
                            auto va = this->getReadAccessor<core::V_COORD>(a);
                            auto vb = this->getReadAccessor<core::V_COORD>(b);
                            auto vv = this->getWriteOnlyAccessor<core::V_DERIV>(r);
                            vv.resize(vb.size());
                            for (unsigned int i = 0; i < vv.size(); ++i)
                            {
                                vv[i] = DataTypes::coordDifference(va[i], vb[i]);
                            }
                        }
                        else
                        {
                            generalCase();
                        }
                    }
                    else
                    {
                        generalCase();
                    }
                }
            }
        }
    }

}

template <class DataTypes>
void StateVectorAlgebra<DataTypes>::vMultiOp(const core::ExecParams* params, const VMultiOp& ops)
{
    // optimize common integration case: v += a*dt, x += v*dt
    if (ops.size() == 2
            && ops[0].second.size() == 2
            && ops[0].first.getId(this) == ops[0].second[0].first.getId(this)
            && ops[0].first.getId(this).type == sofa::core::V_DERIV
            && ops[0].second[1].first.getId(this).type == sofa::core::V_DERIV
            && ops[1].second.size() == 2
            && ops[1].first.getId(this) == ops[1].second[0].first.getId(this)
            && ops[0].first.getId(this) == ops[1].second[1].first.getId(this)
            && ops[1].first.getId(this).type == sofa::core::V_COORD)
    {
        auto va = getReadAccessor<core::V_DERIV>(ops[0].second[1].first.getId(this));
        auto vv = getWriteAccessor<core::V_DERIV>(ops[0].first.getId(this));
        auto vx = getWriteAccessor<core::V_COORD>(ops[1].first.getId(this));

        const auto n = vx.size();
        const Real f_v_v = (Real)(ops[0].second[0].second);
        const Real f_v_a = (Real)(ops[0].second[1].second);
        const Real f_x_x = (Real)(ops[1].second[0].second);
        const Real f_x_v = (Real)(ops[1].second[1].second);

        if (f_v_v == 1.0 && f_x_x == 1.0) // very common case
        {
            if (f_v_a == 1.0) // used by euler implicit and other integrators that directly computes a*dt
            {
                for (unsigned int i=0; i<n; ++i)
                {
                    vv[i] += va[i];
                    vx[i] += vv[i]*f_x_v;
                }
            }
            else
            {
                for (unsigned int i=0; i<n; ++i)
                {
                    vv[i] += va[i]*f_v_a;
                    vx[i] += vv[i]*f_x_v;
                }
            }
        }
        else if (f_x_x == 1.0) // some damping is applied to v
        {
            for (unsigned int i=0; i<n; ++i)
            {
                vv[i] *= f_v_v;
                vv[i] += va[i];
                vx[i] += vv[i]*f_x_v;
            }
        }
        else // general case
        {
            for (unsigned int i=0; i<n; ++i)
            {
                vv[i] *= f_v_v;
                vv[i] += va[i]*f_v_a;
                vx[i] *= f_x_x;
                vx[i] += vv[i]*f_x_v;
            }
        }
    }
    else if(ops.size()==2 //used in the ExplicitBDF solver only (Electrophysiology)
            && ops[0].second.size()==1
            && ops[0].second[0].second == 1.0
            && ops[1].second.size()==3
            )
    {
        auto v11 = getReadAccessor<core::V_COORD>(ops[0].second[0].first.getId(this));
        auto v21 = getReadAccessor<core::V_COORD>(ops[1].second[0].first.getId(this));
        auto v22 = getReadAccessor<core::V_COORD>(ops[1].second[1].first.getId(this));
        auto v23 = getReadAccessor<core::V_DERIV>(ops[1].second[2].first.getId(this));

        auto previousPos = getWriteAccessor<core::V_COORD>(ops[0].first.getId(this));
        auto newPos = getWriteAccessor<core::V_COORD>(ops[1].first.getId(this));

        const auto n = v11.size();
        const Real f_1 = (Real)(ops[1].second[0].second);
        const Real f_2 = (Real)(ops[1].second[1].second);
        const Real f_3 = (Real)(ops[1].second[2].second);

        for (unsigned int i=0; i<n; ++i)
        {
            previousPos[i] = v11[i];
            newPos[i]  = v21[i]*f_1;
            newPos[i] += v22[i]*f_2;
            newPos[i] += v23[i]*f_3;
        }
    }
    else // no optimization for now for other cases
        BaseMechanicalState::vMultiOp(params, ops);
}

template <class DataTypes>
void StateVectorAlgebra<DataTypes>::vThreshold(core::VecId a, SReal threshold)
{
    if( a.type==sofa::core::V_DERIV)
    {
        auto vv = getWriteAccessor<core::V_DERIV>(a);
        const Real t2 = static_cast<Real>(threshold*threshold);
        for (unsigned int i=0; i<vv.size(); i++)
        {
            if( vv[i]*vv[i] < t2 )
                statevectoralgebra::clear(vv[i]);
        }
    }
    else
    {
        msg_error()<<"vThreshold does not apply to coordinate vectors";
    }
}

template <class DataTypes>
SReal StateVectorAlgebra<DataTypes>::vDot(const core::ExecParams*, core::ConstVecId a, core::ConstVecId b)
{
    Real r = 0.0;

    bool error = a.type != b.type;

    if (!error)
    {
        error = !statevectoralgebra::applyPredicateIfCoordOrDeriv(a.type, [this, &r, &a, &b](auto vtype)
        {
            auto va = this->getReadAccessor<vtype>(a);
            auto vb = this->getReadAccessor<vtype>(b);
            for (unsigned int i = 0; i < va.size(); ++i)
            {
                r += va[i] * vb[i];
            }
        });
    }

    msg_error_when(error) << "Invalid dot operation (" << a << ',' << b << ")";

    return r;
}


template <class DataTypes>
SReal StateVectorAlgebra<DataTypes>::vSum(const core::ExecParams* params, core::ConstVecId a, unsigned l)
{
    SOFA_UNUSED(params);

    Real r = 0.0;

    if (a.type == sofa::core::V_COORD )
    {
        msg_error() << "Invalid vSum operation: cannot compute the sum of V_Coord terms in vector "<< a;
    }
    else if (a.type == sofa::core::V_DERIV)
    {
        const VecDeriv &va = this->read(core::ConstVecDerivId(a))->getValue();

        if( l==0 ) for (std::size_t i=0; i<va.size(); i++)
        {
            for(unsigned j=0; j<DataTypes::deriv_total_size; j++)
                if ( fabs(va[i][j])>r) r=fabs(va[i][j]);
        }
        else for (unsigned int i=0; i<va.size(); i++)
        {
            for(unsigned j=0; j<DataTypes::deriv_total_size; j++)
                r += (Real) exp(va[i][j]/l);
        }
    }
    else
    {
        msg_error() << "Invalid vSum operation ("<<a<<")";
    }

    return r;
}

template <class DataTypes>
SReal StateVectorAlgebra<DataTypes>::vMax(const core::ExecParams* params, core::ConstVecId a )
{
    SOFA_UNUSED(params);

    Real r = 0.0;

    const bool isApplied = statevectoralgebra::applyPredicateIfCoordOrDeriv(a.type, [this, &r, &a](auto vtype)
    {
        auto va = this->getReadAccessor<vtype>(a);
        for (unsigned int i = 0; i < va.size(); ++i)
        {
            for(unsigned j = 0; j < core::StateTypeSize_v<DataTypes, vtype>; ++j)
            {
                if (fabs(va[i][j]) > r)
                    r = fabs(va[i][j]);
            }
        }
    });

    msg_error_when(!isApplied) << "Invalid vMax operation (" << a << ")";

    return r;
}

template <class DataTypes>
Size StateVectorAlgebra<DataTypes>::vSize(const core::ExecParams* params, core::ConstVecId vecId)
{
    SOFA_UNUSED(params);

    Size size = 0;

    const bool isApplied = statevectoralgebra::applyPredicateIfCoordOrDeriv(vecId.type, [this, &size, &vecId](auto vtype)
    {
        auto va = this->getReadAccessor<vtype>(vecId);
        size = static_cast<Size>(va.size() * core::StateTypeSize_v<DataTypes, vtype>);
    });

    msg_error_when(!isApplied) << "Invalid size operation (" << vecId << ")";
    return size;
}

template <class DataTypes>
void StateVectorAlgebra<DataTypes>::resetVec(const core::ExecParams* params, core::VecId fid)
{
    SOFA_UNUSED(params);

    if (fid.type == sofa::core::V_COORD)
    {
        helper::WriteOnlyAccessor< Data<VecCoord> > f_wa( *this->write(core::VecCoordId(fid)) );
        sofa::defaulttype::resetDataTypeVec(f_wa.wref());
    }
    else if (fid.type == sofa::core::V_DERIV)
    {
        helper::WriteOnlyAccessor< Data<VecDeriv> > f_wa( *this->write(core::VecDerivId(fid)) );
        sofa::defaulttype::resetDataTypeVec(f_wa.wref());
    }
}

#undef APPLY_PREDICATE_ONEPARAMS
#undef APPLY_PREDICATE_TWOPARAMS
#undef APPLY_PREDICATE_THREEPARAMS

} // namespace sofa::core::behavior
