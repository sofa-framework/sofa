#ifndef SOFA_COMPONENT_VISUALMODEL_VISUALSTYLE_H
#define SOFA_COMPONENT_VISUALMODEL_VISUALSTYLE_H

#include <sofa/component/component.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/visual/DisplayFlags.h>

namespace sofa
{
namespace component
{
namespace visualmodel
{

class SOFA_COMPONENT_VISUALMODEL_API VisualStyle : public sofa::core::visual::VisualModel
{
public:
    SOFA_CLASS(VisualStyle,sofa::core::visual::VisualModel);

    typedef sofa::core::visual::VisualParams VisualParams;
    typedef sofa::core::visual::DisplayFlags DisplayFlags;
    VisualStyle();

    void fwdDraw(VisualParams* );
    void bwdDraw(VisualParams* );

    Data<DisplayFlags> displayFlags;

protected:
    DisplayFlags backupFlags;
};


} // visual

} // component

} // sofa

#endif // SOFA_COMPONENT_VISUALMODEL_VISUALSTYLE_H
