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

#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa::core::objectmodel
{

/**
 *  \brief Base class for simulation objects that modify the shared context (such as gravity, local coordinate system, ...).
 *
 */
class SOFA_CORE_API ContextObject : public virtual BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(ContextObject, BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(ContextObject)
protected:
    ContextObject()
    {}

    ~ContextObject() override
    {}

public:
    /**
     * !!! WARNING since v25.12 !!!
     * 
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doApply" internally,
     * which is the method to override from now on.
     *
     * Modify the context.
     * 
     **/
    virtual void apply() final
    {
        //TODO (SPRINT SED 2025): Component state mechamism
        this->doApply();
    }


    bool insertInNode( objectmodel::BaseNode* node ) override;
    bool removeInNode( objectmodel::BaseNode* node ) override;

protected:
    // Modify the context.
    virtual void doApply() = 0 ;

};
} // namespace sofa::core::objectmodel
