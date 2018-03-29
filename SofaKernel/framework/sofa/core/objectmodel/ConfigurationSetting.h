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
#ifndef SOFA_CORE_OBJECTMODEL_CONFIGURATIONSETTING_H
#define SOFA_CORE_OBJECTMODEL_CONFIGURATIONSETTING_H

#include <sofa/core/core.h>
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

/**
 *  \brief Base class for all the configuration settings of SOFA
 *
 */
class SOFA_CORE_API ConfigurationSetting: public BaseObject
{
public:
    SOFA_CLASS(ConfigurationSetting, BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(ConfigurationSetting)
protected:
    ConfigurationSetting(); ///< Default constructor.

    virtual ~ConfigurationSetting();
public:
    virtual void init() override;

    virtual bool insertInNode( objectmodel::BaseNode* node ) override;
    virtual bool removeInNode( objectmodel::BaseNode* node ) override;

};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
