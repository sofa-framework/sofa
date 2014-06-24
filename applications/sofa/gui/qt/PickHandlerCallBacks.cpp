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
#include "PickHandlerCallBacks.h"
#include "RealGUI.h"
#include "viewer/SofaViewer.h"
#include "QSofaListView.h"
#include <sofa/core/objectmodel/BaseObject.h>
#include <SofaUserInteraction/MouseInteractor.h>

#ifdef SOFA_QT4
#   include <QStatusBar>
#else
#   include <qstatusbar.h>
#endif
namespace sofa
{
namespace gui
{
namespace qt
{

InformationOnPickCallBack::InformationOnPickCallBack()
    :gui(NULL)
{
}

InformationOnPickCallBack::InformationOnPickCallBack(RealGUI *g)
    :gui(g)
{
}

void InformationOnPickCallBack::execute(const sofa::component::collision::BodyPicked &body)
{
    if(!gui) return;
    core::objectmodel::BaseObject *objectPicked=NULL;
    if (body.body)
    {
        Q3ListViewItem* item=gui->simulationGraph->getListener()->items[body.body];
        gui->simulationGraph->ensureItemVisible(item);
        gui->simulationGraph->clearSelection();
        gui->simulationGraph->setSelected(item,true);
        objectPicked=body.body;
    }
    else if (body.mstate)
    {
        Q3ListViewItem* item=gui->simulationGraph->getListener()->items[body.mstate];
        gui->simulationGraph->ensureItemVisible(item);
        gui->simulationGraph->clearSelection();
        gui->simulationGraph->setSelected(item,true);
        objectPicked=body.mstate;
    }
    else
        gui->simulationGraph->clearSelection();

    if (objectPicked)
    {
        QString messagePicking;
        simulation::Node *n=static_cast<simulation::Node*>(objectPicked->getContext());
        messagePicking=QString("Index ") + QString::number(body.indexCollisionElement)
                + QString(" of  ")
                + QString(n->getPathName().c_str())
                + QString("/") + QString(objectPicked->getName().c_str())
                + QString(" : ") + QString(objectPicked->getClassName().c_str());
        if (!objectPicked->getTemplateName().empty())
            messagePicking += QString("<") + QString(objectPicked->getTemplateName().c_str()) + QString(">");
        gui->statusBar()->message(messagePicking,3000); //display message during 3 seconds
    }
}

ColourPickingRenderCallBack::ColourPickingRenderCallBack()
    :_viewer(NULL)
{
}

ColourPickingRenderCallBack::ColourPickingRenderCallBack(viewer::SofaViewer* viewer)
    :_viewer(viewer)
{
}

void ColourPickingRenderCallBack::render(ColourPickingVisitor::ColourCode code)
{
    if(_viewer)
    {
        _viewer->drawColourPicking(code);
    }

}

}
}
}
