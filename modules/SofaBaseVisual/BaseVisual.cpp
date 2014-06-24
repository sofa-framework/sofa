/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <SofaBaseVisual/BaseVisual.h>
#include <sofa/core/Plugin.h>
#include <SofaBaseVisual/InteractiveCamera.h>
#include <SofaBaseVisual/RecordedCamera.h>
#include <SofaBaseVisual/VisualModelImpl.h>
#include <SofaBaseVisual/VisualStyle.h>
#include <SofaBaseVisual/VisualTransform.h>

using namespace sofa::component::visualmodel;

class BaseVisualPlugin: public sofa::core::Plugin {
public:
    BaseVisualPlugin(): Plugin("BaseVisual") {
        setDescription("");
        setVersion("");
        setLicense("LGPL");
        setAuthors("The SOFA Team");

        addComponent< InteractiveCamera >("InteractiveCamera");
        addAlias("InteractiveCamera","Camera");

        addComponent< RecordedCamera >("Camera moving along a predetermined path (currently only a rotation)");

        addComponent< VisualModelImpl >("Generic visual model. If a viewer is active it will replace the VisualModel alias, otherwise nothing will be displayed.");
        addAlias("VisualModelImpl","VisualModel");

        addComponent< VisualStyle >("Edit the visual style.\n Allowed values for displayFlags data are a combination of the following:\n\
        showAll, hideAll,\n\
            showVisual, hideVisual,\n\
                showVisualModels, hideVisualModels,\n\
            showBehavior, hideBehavior,\n\
                showBehaviorModels, hideBehaviorModels,\n\
                showForceFields, hideForceFields,\n\
                showInteractionForceFields, hideInteractionForceFields\n\
            showMapping, hideMapping\n\
                showMappings, hideMappings\n\
                showMechanicalMappings, hideMechanicalMappings\n\
            showCollision, hideCollision\n\
                showCollisionModels, hideCollisionModels\n\
                showBoundingCollisionModels, hideBoundingCollisionModels\n\
            showOptions hideOptions\n\
                showRendering hideRendering\n\
                showNormals hideNormals\n\
                showWireframe hideWireframe");

        addComponent< VisualTransform >("TODO");

    }
};

SOFA_PLUGIN(BaseVisualPlugin);
