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
#include <sofa/core/StateVectorStorage.h>
#include <sofa/core/State.inl>

namespace sofa::core
{

template <class TDataTypes>
StateVectorStorage<TDataTypes>::StateVectorStorage()
    : d_size(initData(&d_size, 0, "size", "Size of the vectors"))
    , f_reserve(initData(&f_reserve, 0, "reserve", "Size to reserve when creating vectors. (default=0)"))
{
    // default size is 1
    StateVectorStorage::resize(1);
}

template <class TDataTypes>
StateVectorStorage<TDataTypes>::~StateVectorStorage()
{
    for (unsigned i = core::VecCoordId::V_FIRST_DYNAMIC_INDEX; i < vectorsCoord.size(); i++)
        if (vectorsCoord[i] != nullptr)
        {
            delete vectorsCoord[i];
            vectorsCoord[i] = nullptr;
        }

    if (vectorsCoord[core::VecCoordId::null().getIndex()] != nullptr)
    {
        delete vectorsCoord[core::VecCoordId::null().getIndex()];
        vectorsCoord[core::VecCoordId::null().getIndex()] = nullptr;
    }

    for (unsigned i = core::VecDerivId::V_FIRST_DYNAMIC_INDEX; i < vectorsDeriv.size(); i++)
        if (vectorsDeriv[i] != nullptr)
        {
            delete vectorsDeriv[i];
            vectorsDeriv[i] = nullptr;
        }

    if (vectorsDeriv[core::VecDerivId::null().getIndex()] != nullptr)
    {
        delete vectorsDeriv[core::VecDerivId::null().getIndex()];
        vectorsDeriv[core::VecDerivId::null().getIndex()] = nullptr;
    }

    if (core::vec_id::write_access::dforce.getIndex() < vectorsDeriv.size() &&
        vectorsDeriv[core::vec_id::write_access::dforce.getIndex()] != nullptr)
    {
        delete vectorsDeriv[core::vec_id::write_access::dforce.getIndex()];
        vectorsDeriv[core::vec_id::write_access::dforce.getIndex()] = nullptr;
    }

    for (unsigned i = core::MatrixDerivId::V_FIRST_DYNAMIC_INDEX; i < vectorsMatrixDeriv.size(); i++)
        if (vectorsMatrixDeriv[i] != nullptr)
        {
            delete vectorsMatrixDeriv[i];
            vectorsMatrixDeriv[i] = nullptr;
        }
}
template <class TDataTypes>
void StateVectorStorage<TDataTypes>::init()
{
    Inherit1::init();

    if (f_reserve.getValue() > 0)
        reserve(f_reserve.getValue());
}

template <class TDataTypes>
Size StateVectorStorage<TDataTypes>::getSize() const
{
    return d_size.getValue();
}
template <class TDataTypes>
void StateVectorStorage<TDataTypes>::resize(Size vsize)
{
    if(vsize>0)
    {
        if (d_size.getValue() != static_cast<int>(vsize))
            d_size.setValue(static_cast<int>(vsize));

        const auto resizeFunction = [&vsize](auto& dataList)
        {
            for (auto* vec : dataList)
            {
                if (vec != nullptr && vec->isSet())
                {
                    auto wa = helper::getWriteAccessor(*vec);
                    if (wa.size() != vsize)
                    {
                        wa.resize(vsize);
                    }
                }
            }
        };
        resizeFunction(vectorsCoord);
        resizeFunction(vectorsDeriv);
    }
    else // clear
    {
        d_size.setValue(0);

        const auto resizeFunction = [](auto& dataList)
        {
            for (auto* vec : dataList)
            {
                if (vec != nullptr && vec->isSet())
                {
                    helper::getWriteAccessor(*vec).clear();
                }
            }
        };
        resizeFunction(vectorsCoord);
        resizeFunction(vectorsDeriv);
    }
}
template <class TDataTypes>
void StateVectorStorage<TDataTypes>::reserve(Size vsize)
{
    if (vsize == 0) return;

    const auto reserveFunction = [vsize](auto& dataList)
    {
        for (auto* vec : dataList)
        {
            if (vec != nullptr && vec->isSet())
            {
                helper::getWriteAccessor(*vec).reserve(vsize);
            }
        }
    };
    reserveFunction(vectorsCoord);
    reserveFunction(vectorsDeriv);
}
template <class TDataTypes>
auto StateVectorStorage<TDataTypes>::write(VecCoordId v) -> Data<VecCoord>*
{
    if (v.index >= vectorsCoord.size())
    {
        vectorsCoord.resize(v.index + 1, 0);
    }

    if (vectorsCoord[v.index] == nullptr)
    {
        vectorsCoord[v.index] = new Data< VecCoord >;
        vectorsCoord[v.index]->setName(v.getName());
        const auto group = v.getGroup();
        if (!group.empty())
        {
            vectorsCoord[v.index]->setGroup(group);
        }
        else
        {
            vectorsCoord[v.index]->setGroup("Vector");
        }
        this->addData(vectorsCoord[v.index]);
        if (f_reserve.getValue() > 0)
        {
            vectorsCoord[v.index]->beginWriteOnly()->reserve(f_reserve.getValue());
            vectorsCoord[v.index]->endEdit();
        }
        if (vectorsCoord[v.index]->getValue().size() != getSize())
        {
            vectorsCoord[v.index]->beginWriteOnly()->resize( getSize() );
            vectorsCoord[v.index]->endEdit();
        }
    }
    Data<VecCoord>* d = vectorsCoord[v.index];
#if !defined(NDEBUG)
    const VecCoord& val = d->getValue();
    if (!val.empty() && val.size() != (unsigned int)this->getSize())
    {
        msg_error() << "Writing to State vector " << v << " with incorrect size : " << val.size() << " != " << this->getSize();
    }
#endif
    return d;
}

template <class TDataTypes>
auto StateVectorStorage<TDataTypes>::read(ConstVecCoordId v) const -> const Data<VecCoord>*
{
    if (v.isNull())
    {
        msg_error() << "Accessing null VecCoord";
    }

    if (v.index < vectorsCoord.size() && vectorsCoord[v.index] != nullptr)
    {
        const Data<VecCoord>* d = vectorsCoord[v.index];
#if !defined(NDEBUG)
        const VecCoord& val = d->getValue();
        if (!val.empty() && val.size() != (unsigned int)this->getSize())
        {
            msg_error() << "Accessing State vector " << v << " with incorrect size : " << val.size() << " != " << this->getSize();
        }
#endif
        return d;
    }
    else
    {
        msg_error() << "Vector " << v << " does not exist";
        return nullptr;
    }
}

template <class TDataTypes>
auto StateVectorStorage<TDataTypes>::write(VecDerivId v) -> Data<VecDeriv>*
{
    if (v.index >= vectorsDeriv.size())
    {
        vectorsDeriv.resize(v.index + 1, 0);
    }

    if (vectorsDeriv[v.index] == nullptr)
    {
        vectorsDeriv[v.index] = new Data< VecDeriv >;
        vectorsDeriv[v.index]->setName(v.getName());
        const auto group = v.getGroup();
        if (!group.empty())
        {
            vectorsDeriv[v.index]->setGroup(group);
        }
        else
        {
            vectorsDeriv[v.index]->setGroup("Vector");
        }
        this->addData(vectorsDeriv[v.index]);
        if (f_reserve.getValue() > 0)
        {
            vectorsDeriv[v.index]->beginWriteOnly()->reserve(f_reserve.getValue());
            vectorsDeriv[v.index]->endEdit();
        }
        if (vectorsDeriv[v.index]->getValue().size() != getSize())
        {
            vectorsDeriv[v.index]->beginWriteOnly()->resize( getSize() );
            vectorsDeriv[v.index]->endEdit();
        }
    }
    Data<VecDeriv>* d = vectorsDeriv[v.index];

#if !defined(NDEBUG)
    const VecDeriv& val = d->getValue();
    if (!val.empty() && val.size() != (unsigned int)this->getSize())
    {
        msg_error() << "Writing to State vector " << v << " with incorrect size : " << val.size() << " != " << this->getSize();
    }
#endif
    return d;
}

template <class TDataTypes>
auto StateVectorStorage<TDataTypes>::read(ConstVecDerivId v) const -> const Data<VecDeriv>*
{
    if (v.index < vectorsDeriv.size())
    {
        const Data<VecDeriv>* d = vectorsDeriv[v.index];

#if !defined(NDEBUG)
        if(d!=NULL)
        {
            const VecDeriv& val = d->getValue();
            if (!val.empty() && val.size() != (unsigned int)this->getSize())
            {
                msg_error() << "Accessing State vector " << v << " with incorrect size : " << val.size() << " != " << this->getSize();
            }
        }
#endif // !defined(NDEBUG)

        return d;
    }
    else
    {
        msg_error() << "Vector " << v << "does not exist";
        return nullptr;
    }
}

template <class TDataTypes>
auto StateVectorStorage<TDataTypes>::write(MatrixDerivId v) -> Data<MatrixDeriv>*
{
    if (v.index >= vectorsMatrixDeriv.size())
    {
        vectorsMatrixDeriv.resize(v.index + 1, 0);
    }

    if (vectorsMatrixDeriv[v.index] == nullptr)
    {
        vectorsMatrixDeriv[v.index] = new Data< MatrixDeriv >;
        vectorsMatrixDeriv[v.index]->setName(v.getName());
        const auto group = v.getGroup();
        if (!group.empty())
        {
            vectorsMatrixDeriv[v.index]->setGroup(group);
        }
        else
        {
            vectorsMatrixDeriv[v.index]->setGroup("Vector");
        }
        this->addData(vectorsMatrixDeriv[v.index]);
    }

    return vectorsMatrixDeriv[v.index];
}

template <class TDataTypes>
auto StateVectorStorage<TDataTypes>::read(ConstMatrixDerivId v) const -> const Data<MatrixDeriv>*
{
    if (v.index < vectorsMatrixDeriv.size())
        return vectorsMatrixDeriv[v.index];
    else
    {
        msg_error() << "Vector " << v << "does not exist";
        return nullptr;
    }
}

template <class TDataTypes>
void StateVectorStorage<TDataTypes>::setVecCoord(core::ConstVecCoordId vecId, Data<VecCoord>* vecData)
{
    const auto index = vecId.getIndex();
    if (index >= vectorsCoord.size())
    {
        vectorsCoord.resize(index + 1, 0);
    }

    vectorsCoord[index] = vecData;

    const auto group = vecId.getGroup();
    if (!group.empty())
    {
        vecData->setGroup(group);
    }
}
template <class TDataTypes>
void StateVectorStorage<TDataTypes>::setVecDeriv(core::ConstVecDerivId vecId, Data<VecDeriv>* vecData)
{
    const auto index = vecId.getIndex();
    if (index >= vectorsDeriv.size())
    {
        vectorsDeriv.resize(index + 1, 0);
    }

    vectorsDeriv[index] = vecData;

    const auto group = vecId.getGroup();
    if (!group.empty())
    {
        vecData->setGroup(group);
    }
}

template <class TDataTypes>
void StateVectorStorage<TDataTypes>::setVecMatrixDeriv(core::ConstMatrixDerivId vecId, Data<MatrixDeriv>* vecData)
{
    const auto index = vecId.getIndex();
    if (index >= vectorsMatrixDeriv.size())
    {
        vectorsMatrixDeriv.resize(index + 1, 0);
    }

    vectorsMatrixDeriv[index] = vecData;

    const auto group = vecId.getGroup();
    if (!group.empty())
    {
        vecData->setGroup(group);
    }
}

}  // namespace sofa::core
