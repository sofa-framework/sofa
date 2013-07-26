#include "labelimagetoolboxaction.h"
#include "labelimagetoolbox.h"

namespace sofa
{
namespace gui
{
namespace qt
{

LabelImageToolBoxAction::LabelImageToolBoxAction(sofa::component::engine::LabelImageToolBox* lba,QObject *parent) : QObject(parent),
    p_label(lba)
{
}

void LabelImageToolBoxAction::setVisible(bool v)
{
    for(int i=0;i<l_actions.size();i++)
    {
        l_actions[i]->setVisible(v);
    }
    
    for(int i=0;i<l_widgets.size();i++)
    {
        l_widgets[i]->setEnabled(v);
    }
}


void LabelImageToolBoxAction::buttonSelectedOff()
{
    for(int i=0;i<l_actions.size();i++)
    {
        l_actions[i]->setChecked(false);
    }
}

void LabelImageToolBoxAction::setGraphScene(QGraphicsScene *XY,QGraphicsScene *XZ,QGraphicsScene *ZY)
{
    GraphXY = XY;
    GraphXZ = XZ;
    GraphZY = ZY;
    
    addOnGraphs();
    updateGraphs();
}

}
}
}

