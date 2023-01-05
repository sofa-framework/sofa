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

#include <sofa/core/objectmodel/Event.h>
#include <string>


namespace sofa::core::objectmodel
{

/**
 * @brief This event notifies about GUI interaction.
 */
class SOFA_CORE_API GUIEvent : public sofa::core::objectmodel::Event
{
public:

    SOFA_EVENT_H( GUIEvent )

    /**
     * @brief Constructor.
     */
    GUIEvent(const char* controlID, const char* valueName, const char* value);

    /**
     * @brief Destructor.
     */
    ~GUIEvent() override;

    /**
     * @brief Get the emitter control ID
     */
    const std::string getControlID(void) const {return m_controlID;}

    /**
     * @brief Get the value name
     */
    const std::string getValueName(void) const {return m_valueName;}

    /**
     * @brief Get the value
     */
    const std::string getValue(void) const {return m_value;}


    inline static const char* GetClassName() { return "GUIEvent"; }
private:

    std::string     m_controlID;
    std::string     m_valueName;
    std::string     m_value;

};
} // namespace sofa::core::objectmodel
