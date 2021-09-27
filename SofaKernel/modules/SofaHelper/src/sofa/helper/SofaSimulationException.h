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
#include <sstream>
#include <sofa/helper/logging/Messaging.h>

#include <sofa/helper/BackTrace.h>
namespace sofa::helper
{

class SofaSimulationException : public std::exception
{
    std::vector<std::string> stacktrace;
    std::exception nested_exception;
public:
    SofaSimulationException(std::exception e)
    {
        stacktrace = sofa::helper::BackTrace::getTrace();
        nested_exception = e;
    }

    const std::vector<std::string>& getTrace() const { return stacktrace; }

    const char* what() const noexcept override
    {
        return nested_exception.what();
    }
};

void executeWithException(const std::string& src, bool withException, std::function<void()> cb)
{
    if(!withException)
        return cb();
    try
    {
        cb();
    } catch (const sofa::helper::SofaSimulationException& e)
    {
        std::stringstream tmp;
        tmp << "At:" << msgendl;
        for(auto& name : e.getTrace())
        {
            tmp << "  " << name << msgendl;
        }
        msg_error(src)    << "Exception received." << msgendl
                          << "c++ exception:" << msgendl
                          << "  " << e.what()
                          << msgendl << msgendl
                          << tmp.str();
    }
}
}
