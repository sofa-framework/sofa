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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef INITSimpleGUI_H
#define INITSimpleGUI_H


#include <sofa/helper/system/config.h>

#ifdef SOFA_BUILD_SOFASIMPLEGUI
#define SOFA_SOFASIMPLEGUI_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_SOFASIMPLEGUI_API  SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

namespace sofa {
namespace newgui {

/** \mainpage SimpleGUI - a simplified GUI for Sofa
 * This plugin proposes a simple API for including Sofa in a graphics interface:
 * - class SofaScene provides a callback-level API: init, draw, animate
 * - class SofaGL allows drawing and picking in OpenGL viewers
 * - a choice of Interactor objects allow the user to manipulate the simulation.
 * - for single-viewer applications, class SofaGLScene combines SofaScene with SofaGL
 *
 * The main differences with the standard Sofa GUI are:
 * - The main loop is controled by the user application
 * - No viewer nor camera are provided, but helpers are available in SofaGL
 * - Picking simply returns basic information about the particle under the cursor: PickedPoint. Based on this, it is the application's job to create, manage and delete Interactor objects.
 *
 * Examples:
 * - glutOnePick: A Sofa simulation within a basic Glut interface. The user can click and drag one point at a time to interact with the simulaton
 * - qtSofa: A Sofa simulation within a basic Qt interface. The user can click and drag multiple points.
 * - qtQuickSofa: A Sofa simulation within a QtQuick interface.
 * - SofaQtExternal.zip: A graphics application including a Sofa simulation, in a project which is not a sub-project of Sofa.
 *
 * Dependencies:
 * - Qt > 5.0 is necessary for qtQuickSofa
 *
 * Issues:
 * - Currently there is no way to create interactors in plugins. They are created by the user application.
 *
  This is a the starting page of the plugin documentation, defined in file initSimpleGUI.h
  */

}
}


#endif // INITSimpleGUI_H
