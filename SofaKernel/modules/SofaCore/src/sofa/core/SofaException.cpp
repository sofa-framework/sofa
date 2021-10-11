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
#include <sstream>
#include <sofa/core/SofaException.h>
#include <sofa/core/objectmodel/Base.h>
using sofa::core::objectmodel::Base;
using sofa::core::objectmodel::ComponentState;

#include <sofa/helper/BackTrace.h>

namespace sofa::core
{

SofaException::SofaException(sofa::core::objectmodel::Base* emitter_, std::exception e)
{
    emitter = emitter_;
    stacktrace = sofa::helper::BackTrace::getTrace();
    nested_exception = e;
    emitter->d_componentState.setValue(ComponentState::Invalid) ;
}

const std::vector<std::string>& SofaException::getTrace() const
{
    return stacktrace;
}

sofa::core::objectmodel::Base* SofaException::getEmitter() const
{
    return emitter;
}

const char* SofaException::what() const noexcept
{
    return nested_exception.what();
}

std::ostream& operator<<(std::ostream& out, const SofaException& e)
{
    out    << "c++ exception:" << std::endl
           << "  " << e.what()
           << std::endl << std::endl;
    for(auto& name : e.getTrace())
    {
        out << "  " << name << std::endl;
    }
    return out;
}

}
