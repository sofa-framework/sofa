/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This program is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU General Public License as published by the Free   *
* Software Foundation; either version 2 of the License, or (at your option)    *
* any later version.                                                           *
*                                                                              *
* This program is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for     *
* more details.                                                                *
*                                                                              *
* You should have received a copy of the GNU General Public License along with *
* this program; if not, write to the Free Software Foundation, Inc., 51        *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                    *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_VIEWER_H
#define SOFA_VIEWER_H

#include <qstring.h>
#include <qwidget.h>

#include <sofa/simulation/automatescheduler/Automate.h>
#include <sofa/helper/gl/Capture.h>

namespace sofa
{

namespace gui
{

namespace viewer
{

using namespace sofa::simulation::automatescheduler;

class SofaViewer : public Automate::DrawCB
{

public:
    SofaViewer () {};
    ~SofaViewer() {};

    virtual sofa::simulation::tree::GNode* getScene()        {  return groot;}
    virtual const std::string&             getSceneFileName() {  return sceneFileName;}

    virtual QString helpString()=0;

protected:
    sofa::simulation::tree::GNode* groot;
    std::string sceneFileName;
    sofa::helper::gl::Capture capture;

    //Pure Virtual
    virtual void setScene(sofa::simulation::tree::GNode* scene, const char* filename=NULL)=0;
    virtual void SwitchToPresetView()=0;

    //*************************************************************
    // QT
    //*************************************************************
    //SLOTS
    virtual void resetView()=0;
    virtual void saveView()=0;
    virtual void screenshot()=0;
    virtual void setSizeW(int)=0;
    virtual void setSizeH(int)=0;


    //SIGNALS
    virtual void redrawn()=0;
    virtual void resizeW( int )=0;
    virtual void resizeH( int )=0;

};

}
}
}

#endif
