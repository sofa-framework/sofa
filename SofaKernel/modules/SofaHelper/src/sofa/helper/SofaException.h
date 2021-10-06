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

#include <exception>
#include <string>
#include <functional>

namespace sofa::helper
{

///
/// \brief Exception to use when a sofa component encounter an error that need to stop the simulation loop (if any).
///
/// A SofaException wraps around an existing std::exception and attach the stacktrace at the throwing point.
class SofaException : public std::exception
{
    std::vector<std::string> stacktrace;  /// Hold the stacktrace
    std::exception nested_exception;      /// The underlying exception

public:
    SofaException(std::exception e);

    const std::vector<std::string>& getTrace() const;
    const char* what() const noexcept override;
};

/// Execute a lambda function within a try/catch block.
void executeWithException(const std::string& src, bool withException, std::function<void()> cb);

}
