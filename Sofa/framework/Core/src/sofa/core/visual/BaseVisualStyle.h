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
#include <sofa/core/fwd.h>
#include <sofa/core/visual/VisualModel.h>
#include <string>
#include <iostream>


namespace sofa::core::visual
{

class SOFA_CORE_API BaseVisualStyle : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(BaseVisualStyle,sofa::core::objectmodel::BaseObject);

    typedef sofa::core::visual::VisualParams VisualParams;
    typedef sofa::core::visual::DisplayFlags DisplayFlags;

protected:
    BaseVisualStyle() { }
    ~BaseVisualStyle() override { }

    virtual void doUpdateVisualFlags(VisualParams* ) { };
    virtual void doApplyBackupFlags(VisualParams* ) { };

public:
    /**
     * !!! WARNING since v25.12 !!! 
     * 
     * The template method pattern has been applied to this part of the API. 
     * This method calls the newly introduced method "doUpdateVisualFlags" internally,
     * which is the method to override from now on.
     * 
     **/  
    virtual void updateVisualFlags(VisualParams* vparams) final { 
        //TODO (SPRINT SED 2025): Component state mechamism
        return this->doUpdateVisualFlags(vparams); 
    };

    /**
     * !!! WARNING since v25.12 !!! 
     * 
     * The template method pattern has been applied to this part of the API. 
     * This method calls the newly introduced method "doApplyBackupFlags" internally,
     * which is the method to override from now on.
     * 
     **/  
    virtual void applyBackupFlags(VisualParams* vparams) final { 
        //TODO (SPRINT SED 2025): Component state mechamism
        return this->doApplyBackupFlags(vparams); 
    };

};

} // namespace sofa::simulation::graph

