/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef OGRESOFAVIEWER_H
#define OGRESOFAVIEWER_H

#include <sofa/gui/qt/viewer/VisualModelPolicy.h>
#include <sofa/gui/qt/viewer/SofaViewer.h>
#include "DrawToolOGRE.h"
namespace sofa
{
namespace gui
{
namespace qt
{
namespace viewer
{

class OgreVisualModelPolicy : public VisualModelPolicy
{
protected:
    sofa::core::ObjectFactory::ClassEntry::SPtr classVisualModel;
    sofa::core::ObjectFactory::ClassEntry::SPtr classOglModel;
    sofa::core::visual::DrawToolOGRE drawToolOGRE;
public:
    void load()
    {
        // Replace OpenGL visual models with OgreVisualModel
        sofa::core::ObjectFactory::AddAlias("OglModel", "OgreVisualModel", true, &classOglModel);
        sofa::core::ObjectFactory::AddAlias("VisualModel", "OgreVisualModel", true, &classVisualModel);
        vparams->drawTool() = &drawToolOGRE;
//    vparams->setSupported(sofa::core::visual::API_OGRE); // ?
    }

    void unload()
    {
        sofa::core::ObjectFactory::ResetAlias("OglModel", classOglModel);
        sofa::core::ObjectFactory::ResetAlias("VisualModel", classVisualModel);
        vparams->drawTool() = NULL;
    }

};

typedef sofa::gui::qt::viewer::CustomPolicySofaViewer< OgreVisualModelPolicy > OgreSofaViewer;

}
}
}
}


#endif // OGRESOFAVIEWER_H
