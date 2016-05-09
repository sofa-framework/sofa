/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <sofa/core/objectmodel/BaseNode.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

ConfigurationSetting::ConfigurationSetting()
{
    this->name.setDisplayed(false);        this->name.setPersistent(false);
    this->f_printLog.setDisplayed(false);  this->f_printLog.setPersistent(false);
    this->f_tags.setDisplayed(false);      this->f_tags.setPersistent(false);
    this->f_listening.setDisplayed(false); this->f_listening.setPersistent(false);
}

ConfigurationSetting::~ConfigurationSetting()
{
}

void ConfigurationSetting::init()
{
    //Set all the Data in read only mode:
    for (VecData::const_iterator it = m_vecData.begin(); it!=m_vecData.end(); ++it)
    {
        BaseData* data=*it;
        data->setReadOnly(true);
    }
}


bool ConfigurationSetting::insertInNode( objectmodel::BaseNode* node )
{
    node->addConfigurationSetting(this);
    Inherit1::insertInNode(node);
    return true;
}

bool ConfigurationSetting::removeInNode( objectmodel::BaseNode* node )
{
    node->removeConfigurationSetting(this);
    Inherit1::removeInNode(node);
    return true;
}


} // namespace objectmodel

} // namespace core

} // namespace sofa
