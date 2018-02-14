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
#ifndef SOFA_COMPONENT_MISC_DATAMONITOR_H
#define SOFA_COMPONENT_MISC_DATAMONITOR_H
#include "config.h"

#include <sofa/core/BaseState.h>

namespace sofa
{

namespace component
{

namespace misc
{

/**
 * @brief  DataMonitor Class
 */
class SOFA_VALIDATION_API DataMonitor : public virtual sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(DataMonitor, core::objectmodel::BaseObject);

    /// Get the value of the associated variable
    const char* getValue();

protected:
    DataMonitor();
    ~DataMonitor() {}

    sofa::core::objectmodel::Data<std::string> data; ///< Monitored data
};

} // namespace misc

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MISC_DATAMONITOR_H
