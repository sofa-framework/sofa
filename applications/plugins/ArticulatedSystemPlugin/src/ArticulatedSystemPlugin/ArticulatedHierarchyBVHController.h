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
//
// C++ Implementation: ArticulatedHierarchyBVHController
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#pragma once
#include <ArticulatedSystemPlugin/config.h>

#include <ArticulatedSystemPlugin/ArticulatedHierarchyController.h>
#include <ArticulatedSystemPlugin/bvh/BVHLoader.h>

namespace sofa::component::controller
{

/**
 * @brief ArticulatedHierarchyController Class.
 *
 * Implements a handler that controls the values of the
 * articulations of an articulated hierarchy container.
 * .bvh files are controlling the value.
 */
class SOFA_ARTICULATEDSYSTEMPLUGIN_API ArticulatedHierarchyBVHController : public ArticulatedHierarchyController
{
public:
    SOFA_CLASS(ArticulatedHierarchyBVHController,ArticulatedHierarchyController);
protected:
    /**
    * @brief Default Constructor.
     */
    ArticulatedHierarchyBVHController()
        : useExternalTime( initData(&useExternalTime, false, "useExternalTime", "use the external time line"))
        , externalTime( initData(&externalTime, 0.0, "externalTime", " value of the External Time") )
    {
        this->f_listening.setValue(true);
    };

    /**
     * @brief Default Destructor.
     */
    virtual ~ArticulatedHierarchyBVHController() {};
public:
    /**
     * @brief Init method called during the scene graph initialization.
     */
    virtual void init() override;

    /**
     * @brief Reset to initial state
     */
    virtual void reset() override;

    /**
     * @brief Apply the controller current modifications to its controlled component.
     */
    virtual void applyController(void) override;

protected:
    Data< bool > useExternalTime; ///< use the external time line
    Data< double > externalTime; ///<  value of the External Time
    ArtCenterVec m_artCenterVec; ///< List of ArticulationCenters controlled by the controller.
    ArticulatedHierarchyContainer* ahc;
    int frame;
    int n;
};

} // namespace sofa::component::controller
