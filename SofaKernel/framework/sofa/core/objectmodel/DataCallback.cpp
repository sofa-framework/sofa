/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/core/objectmodel/DataCallback.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

DataCallback::DataCallback(BaseData* data)
{
    addInput(data);
}

DataCallback::DataCallback(std::initializer_list<BaseData*> datas)
{
    for(BaseData* data : datas)
    {
        addInput(data);
    }
}

void DataCallback::addCallback(std::function<void(void)> f)
{
    m_callbacks.push_back(f);
}

void DataCallback::notifyEndEdit(const core::ExecParams* params)
{
    if (!m_updating)
    {
        m_updating = true;
        for (auto& callback : m_callbacks)
            callback();

        sofa::core::objectmodel::DDGNode::notifyEndEdit(params);
        m_updating = false;
    }
}

const std::string& DataCallback::getName() const
{
    static std::string s="";
    return s;
}

sofa::core::objectmodel::Base* DataCallback::getOwner() const
{
    return nullptr;
}

sofa::core::objectmodel::BaseData* DataCallback::getData() const
{
    return nullptr;
}

void DataCallback::update()
{

}

} /// namespace objectmodel

} /// namespace core

} /// namespace sofa


