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
#include <sofa/core/Mapping.h>
#include <sofa/core/MultiMapping.h>
#include <sofa/core/Multi2Mapping.h>

namespace sofa::component::mapping::linear
{

namespace crtp
{

template<class TMapping>
class CRTPLinearMapping : public TMapping
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(CRTPLinearMapping, TMapping), TMapping);
    using TMapping::TMapping;

    virtual bool isLinear() const override { return true; }
};

}

template <class TIn, class TOut>
using LinearMapping = crtp::CRTPLinearMapping<core::Mapping<TIn, TOut>>;

template <class TIn, class TOut>
using LinearMultiMapping = crtp::CRTPLinearMapping<core::MultiMapping<TIn, TOut>>;

template <class TIn1, class TIn2, class TOut>
using LinearMulti2Mapping = crtp::CRTPLinearMapping<core::Multi2Mapping<TIn1, TIn2, TOut>>;

using LinearBaseMapping = crtp::CRTPLinearMapping<sofa::core::BaseMapping>;

}
