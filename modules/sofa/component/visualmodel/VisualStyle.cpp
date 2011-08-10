#include <sofa/component/visualmodel/VisualStyle.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/common/UpdateContextVisitor.h>

namespace sofa
{
namespace component
{
namespace visualmodel
{

using namespace sofa::core::visual;
using namespace sofa::core::objectmodel;
using namespace sofa::simulation;

int VisualStyleClass = core::RegisterObject("Edit the visual style.").add<VisualStyle>();

VisualStyle::VisualStyle()
    :displayFlags(initData(&displayFlags,"displayFlags","Display Flags"))
{
    displayFlags.setWidget("widget_displayFlags");
    displayFlags.setGroup("Display Flags");
}

void VisualStyle::fwdDraw(VisualParams* vparams)
{
    Node* node = dynamic_cast<Node*>(this->getContext() );
    backupFlags = node->displayFlags();
    vparams->displayflags() = displayFlags.getValue(vparams);
    node->displayFlags() = displayFlags.getValue(vparams);
    //launch update visual flags visitor to propagate the changes to
    //the subgraph of this node;
    UpdateVisualContextVisitor act(vparams);
    node->executeVisitor(&act);



}

void VisualStyle::bwdDraw(VisualParams* vparams)
{
    Node* node = dynamic_cast<Node*>(this->getContext() );
    node->displayFlags()    = backupFlags;
    vparams->displayflags() = backupFlags;
    //launch update visual flags visitor to propagate the changes to
    //the subgraph of this node;
    UpdateVisualContextVisitor act(vparams);
    node->executeVisitor(&act);

}


}
}
}

