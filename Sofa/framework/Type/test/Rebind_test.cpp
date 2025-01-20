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
#include <sofa/type/trait/Rebind.h>
#include <sofa/type/vector.h>

static_assert(sofa::type::CanTypeRebind<sofa::type::vector<float>, int>);
static_assert(sofa::type::CanTypeRebind<sofa::type::vector<int>, int>);

static_assert(
    std::is_same_v<
        sofa::type::rebind_to<sofa::type::vector<float>, int>,
        sofa::type::vector<int>
    >);
static_assert(
    std::is_same_v<
        sofa::type::rebind_to<sofa::type::vector<int>, int>,
        sofa::type::vector<int>
    >);

template<class T>
struct DummyNoRebind{};

static_assert(!sofa::type::CanTypeRebind<DummyNoRebind<float>, int>);

static_assert(
    std::is_same_v<
        sofa::type::rebind_to<DummyNoRebind<float>, int>,
        DummyNoRebind<int>
    >);

template<class T>
struct DummyWithConstraintRebind
{
    template<class U>
    requires std::is_integral_v<U>
    using rebind_to = U;
};

static_assert(sofa::type::CanTypeRebind<DummyWithConstraintRebind<float>, int>);
static_assert(!sofa::type::CanTypeRebind<DummyWithConstraintRebind<float>, std::string>);
