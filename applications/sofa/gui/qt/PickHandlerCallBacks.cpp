#include "PickHandlerCallBacks.h"
#include "RealGUI.h"
#include "viewer/SofaViewer.h"
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/component/collision/MouseInteractor.h>

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

void ColourPickingRenderCallBack::render(core::CollisionModel::ColourCode code)
{
    if(_viewer)
    {
        _viewer->drawColourPicking(code);
    }

}

}
}
}
