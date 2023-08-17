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
#include <sofa/core/objectmodel/DataCallback.h>
#include <sofa/core/objectmodel/BaseData.h>


namespace sofa::core::objectmodel::_datacallback_
{

void DataCallback::addInputs(std::initializer_list<BaseData*> data)
{
    for(BaseData* d : data)
    {
        addInput(d);
    }
}

void DataCallback::addCallback(std::function<void(void)> f)
{
    m_callbacks.push_back(f);
}

void DataCallback::notifyEndEdit()
{
    if (!m_updating)
    {
        m_updating = true;
        for (auto& callback : m_callbacks)
            callback();

        sofa::core::objectmodel::DDGNode::notifyEndEdit();
        m_updating = false;
    }
    else
    {
        msg_warning("DataCallback") << "A DataCallback seems to have a circular dependency, please fix it to remove this warning.";
    }
}

void DataCallback::update()
{

}

} /// namespace sofa::core::objectmodel::_datacallback_


