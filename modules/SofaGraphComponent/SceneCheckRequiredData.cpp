/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "SceneCheckRequiredData.h"

#include <sofa/version.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseObjectDescription.h>
#include <sofa/simulation/Node.h>


namespace sofa
{
namespace simulation
{
namespace _scenechecking_
{

using sofa::core::objectmodel::Base;
using sofa::core::objectmodel::BaseObject;
using sofa::core::objectmodel::BaseObjectDescription;
using sofa::core::ObjectFactory;


SceneCheckRequiredData::SceneCheckRequiredData()
{

}

SceneCheckRequiredData::~SceneCheckRequiredData()
{

}

const std::string SceneCheckRequiredData::getName()
{
    return "SceneCheckRequiredData";
}

const std::string SceneCheckRequiredData::getDesc()
{
    return "Check if a Component has required Datas that are not set.";
}

void SceneCheckRequiredData::doInit(Node* node)
{
    m_missingDatas.clear();
}

void SceneCheckRequiredData::doCheckOn(Node* node)
{
    for (auto& object : node->object )
    {
        Base::VecData vecData = object->getDataFields();
        for(Base::VecData::const_iterator iData = vecData.begin(); iData != vecData.end(); ++iData)
        {
            if ((*iData)->isRequired() && !(*iData)->isSet())
            {
                std::vector<sofa::core::objectmodel::BaseData*> v = m_missingDatas[object.get()];
                if ( v.empty() || std::find(v.begin(), v.end(), *iData) == v.end() )
                {
                    m_missingDatas[object.get()].push_back(*iData);
                }
            }
        }
    }
}

void SceneCheckRequiredData::doPrintSummary()
{
    if(m_missingDatas.empty())
    {
        return;
    }

    for (auto &i : m_missingDatas)
    {
        std::stringstream errorStr;
        errorStr << "Required datas have not been set: ";

        bool first = true;
        for (auto &data : i.second)
        {
             if (first) first = false;
             else errorStr << ", ";
             errorStr << "\"" << data->getName() << "\" (current value is " << data->getValueString() << ")";
        }

        msg_warning(i.first) << this->getName() << ": " << errorStr.str();
    }
}

} // namespace _scenechecking_
} // namespace simulation
} // namespace sofa
