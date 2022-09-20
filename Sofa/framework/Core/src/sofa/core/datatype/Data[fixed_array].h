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

#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/datatype/DataRegistrationMacro.h>
#include <sofa/type/fixed_array.h>

#ifndef SOFA_CORE_DATATYPE_DATAFIXEDARRAY_DEFINITION
DATATYPEINFO_DECLARE(sofa::type::FixedArray1i);
DATATYPEINFO_DECLARE(sofa::type::FixedArray2i);
DATATYPEINFO_DECLARE(sofa::type::FixedArray3i);
DATATYPEINFO_DECLARE(sofa::type::FixedArray4i);
DATATYPEINFO_DECLARE(sofa::type::FixedArray5i);
DATATYPEINFO_DECLARE(sofa::type::FixedArray6i);
DATATYPEINFO_DECLARE(sofa::type::FixedArray7i);
DATATYPEINFO_DECLARE(sofa::type::FixedArray8i);

DATATYPEINFO_DECLARE(sofa::type::FixedArray1I);
DATATYPEINFO_DECLARE(sofa::type::FixedArray2I);
DATATYPEINFO_DECLARE(sofa::type::FixedArray3I);
DATATYPEINFO_DECLARE(sofa::type::FixedArray4I);
DATATYPEINFO_DECLARE(sofa::type::FixedArray5I);
DATATYPEINFO_DECLARE(sofa::type::FixedArray6I);
DATATYPEINFO_DECLARE(sofa::type::FixedArray7I);
DATATYPEINFO_DECLARE(sofa::type::FixedArray8I);

DATATYPEINFO_DECLARE(sofa::type::FixedArray1f);
DATATYPEINFO_DECLARE(sofa::type::FixedArray2f);
DATATYPEINFO_DECLARE(sofa::type::FixedArray3f);
DATATYPEINFO_DECLARE(sofa::type::FixedArray4f);
DATATYPEINFO_DECLARE(sofa::type::FixedArray5f);
DATATYPEINFO_DECLARE(sofa::type::FixedArray6f);
DATATYPEINFO_DECLARE(sofa::type::FixedArray7f);
DATATYPEINFO_DECLARE(sofa::type::FixedArray8f);

DATATYPEINFO_DECLARE(sofa::type::FixedArray1d);
DATATYPEINFO_DECLARE(sofa::type::FixedArray2d);
DATATYPEINFO_DECLARE(sofa::type::FixedArray3d);
DATATYPEINFO_DECLARE(sofa::type::FixedArray4d);
DATATYPEINFO_DECLARE(sofa::type::FixedArray5d);
DATATYPEINFO_DECLARE(sofa::type::FixedArray6d);
DATATYPEINFO_DECLARE(sofa::type::FixedArray7d);
DATATYPEINFO_DECLARE(sofa::type::FixedArray8d);

namespace 
{
    using FixedArray2str = sofa::type::fixed_array<std::string, 2>;
}
DATATYPEINFO_DECLARE(FixedArray2str);
#endif
