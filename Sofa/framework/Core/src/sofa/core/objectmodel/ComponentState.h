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
#include <iosfwd>

namespace sofa::core::objectmodel
{

/// enum class is a C++ x11 feature (http://en.cppreference.com/w/cpp/language/enum),
/// Indicate the state of a sofa object.
enum class ComponentState {
    Undefined,                        ///< component that does not make use of this field have this one
    Loading,                          ///< the component is loading but never passed successfully its init() function
    Valid,                            ///< the component has passed successfully its init function
    Dirty,                            ///< the component is ready to be used but requires a call to reinit
    Busy,                             ///< the component is doing "something", don't trust its values for doing your computation
    Invalid                           ///< the component reached an error and is thus unable to behave normally.
};

/// Defining the in/ou operator for use of component status with Data<>
SOFA_CORE_API std::ostream& operator<<(std::ostream& o, const ComponentState& s);
SOFA_CORE_API std::istream& operator>>(std::istream& i, ComponentState& s);


}  /// namespace sofa::core::objectmodel



