/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "SofaGUI.h"
#include <sofa/component/init.h>
#include <sofa/simulation/tree/xml/initXml.h>

namespace sofa
{

namespace gui
{

SOFA_LINK_CLASS(BatchGUI)

#ifdef SOFA_GUI_GLUT
SOFA_LINK_CLASS(SimpleGUI)
#endif

#ifdef SOFA_GUI_QGLVIEWER
SOFA_LINK_CLASS(QGLViewerGUI)
#endif
#ifdef SOFA_GUI_QTVIEWER
SOFA_LINK_CLASS(QTGUI)
#endif
#ifdef SOFA_GUI_QTOGREVIEWER
SOFA_LINK_CLASS(OgreGUI)
#endif


// ligne 1 et 3
//#ifdef __GNUC__ 4
//    #ifdef __GNUC__MINOR__ 2
//        int fish
//    #endif
//#endif
//
//#ifndef SOFA_DEV
//    int fish
//#endif
//
//#ifdef __APPLE__
//    int fish
//#endif

// ligne 2
#ifdef __GNUC__ 4
#ifdef __GNUC__MINOR__ 3
int fish
#endif
#endif

#ifdef __GNUC__ 4
#ifdef __GNUC__MINOR__ 1
int fish
#endif
#endif

#ifdef SOFA_GPU_CUDA
int fish
#endif



int SofaGUI::Init()
{
    sofa::component::init();
    sofa::simulation::tree::xml::initXml();
    if (guiCreators().empty())
    {
        std::cerr << "ERROR(SofaGUI): No GUI registered."<<std::endl;
        return 1;
    }
    const char* name = GetGUIName();
    if (currentGUI)
        return 0; // already initialized

    GUICreator* creator = GetGUICreator(name);
    if (!creator)
    {
        return 1;
    }
    if (creator->init)
        return (*creator->init)(name, guiOptions);
    else
        return 0;
}

int SofaGUI::createGUI(sofa::simulation::Node* groot, const char* filename)
{
    const char* name = GetGUIName();
    if (!currentGUI)
    {
        GUICreator* creator = GetGUICreator(name);
        if (!creator)
        {
            return 1;
        }
        currentGUI = (*creator->creator)(name, guiOptions, groot, filename);
        if (!currentGUI)
        {
            std::cerr << "ERROR(SofaGUI): GUI "<<name<<" creation failed."<<std::endl;
            return 1;
        }
    }
    return 0;
}


int SofaGUI::MainLoop(sofa::simulation::Node* groot, const char* filename)
{
    int ret = 0;
    const char* name = GetGUIName();
    if (!currentGUI)
    {
        createGUI(groot, filename);
    }
    ret = currentGUI->mainLoop();
    if (ret)
    {
        std::cerr << "ERROR(SofaGUI): GUI "<<name<<" main loop failed (code "<<ret<<")."<<std::endl;
        return ret;
    }
    return ret;
}

} // namespace gui

} // namespace sofa
