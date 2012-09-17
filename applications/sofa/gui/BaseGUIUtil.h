/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#ifndef SOFA_GUI_BASEGUIUTIL_H
#define SOFA_GUI_BASEGUIUTIL_H

#include "BaseGUI.h"
#include "PickHandler.h"

namespace sofa
{

namespace gui
{

class SOFA_SOFAGUI_API BaseGUIUtil : public BaseGUI
{
public:
    BaseGUIUtil() : mCreateViewersOpt(false), frameCounter(0)
    {}
    virtual ~BaseGUIUtil()
    {}

    virtual sofa::simulation::Node* currentSimulation()
    {
        return mViewer ? mViewer->getScene() : NULL;
    }

    virtual void registerViewer(BaseViewer* viewer)
    {
        mViewer = viewer;
    }

    virtual void removeViewer()
    {
        if(mCreateViewersOpt && mViewer != NULL)
        {
            delete mViewer;
            mViewer = NULL;
        }
    }

//    virtual void createViewers(const char* viewerName, viewer::SofaViewerArgument arg);
//    virtual void initViewer();
//    virtual void changeViewer();

    virtual void unloadScene()
    {
        if(mViewer != NULL)
        {
            mViewer->getPickHandler()->reset();
            mViewer->getPickHandler()->unload();
            mViewer->unload();
        }

        simulation::getSimulation()->unload ( currentSimulation() );

        if(mViewer != NULL)
            mViewer->setScene(NULL);
    }

    virtual void fileOpen(std::string filename, bool temporaryFile)
    {
        if ( sofa::helper::system::DataRepository.findFile (filename) )
            filename = sofa::helper::system::DataRepository.getFile ( filename );
        else
            return;

        frameCounter = 0;
        sofa::simulation::xml::numDefault = 0;

        if( this->currentSimulation() ) this->unloadScene();
        simulation::Node::SPtr root = simulation::getSimulation()->load ( filename.c_str() );
        simulation::getSimulation()->init ( root.get() );
        if ( root == NULL )
        {
            std::cerr<<"Failed to load "<<filename.c_str()<<std::endl;
            return;
        }
        setScene ( root, filename.c_str(), temporaryFile );
        configureGUI(root.get());
    }
protected:
    bool mCreateViewersOpt;// to deal with from RealGUI
    BaseViewer* mViewer;
    const char* mViewerName;

    int frameCounter;

    // remonte createViewer
    // remonte changeViewer => change unloadSceneView with unloadVisualScene
    // remonte SofaMouseManager from initViewer
};


}
}
#endif
