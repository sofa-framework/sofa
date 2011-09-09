#include <sofa/component/visualmodel/VisualStyle.h>
#include <sofa/core/visual/VisualParams.h>
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
    backupFlags = vparams->displayFlags();
    vparams->displayFlags() = sofa::core::visual::merge_displayFlags(backupFlags, displayFlags.getValue(vparams));
}

void VisualStyle::bwdDraw(VisualParams* vparams)
{
    vparams->displayFlags() = backupFlags;

}


}
}
}

