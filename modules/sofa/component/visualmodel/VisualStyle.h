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
/** \brief VisualStyle component controls the DisplayFlags state
* embedded in the VisualParams for the current subgraph.
* It merges the DisplayFlags conveyed by the VisualParams with
* its own DisplayFlags.
*
* example:
* <VisualStyle displayFlags="hideVisual showCollision showWireframe" />
*
* allowed values for displayFlags data are a combination of the following:
* showAll, hideAll,
*   showVisual, hideVisual,
*     showVisualModels, hideVisualModels,
*   showBehavior, hideBehavior,
*     showBehaviorModels, hideBehaviorModels,
*     showForceFields, hideForceFields,
*     showInteractionForceFields, hideInteractionForceFields
*   showMapping, hideMapping
*     showMappings, hideMappings
*     showMechanicalMappings, hideMechanicalMappings
*   showCollision, hideCollision
*      showCollisionModels, hideCollisionModels
*      showBoundingCollisionModels, hideBoundingCollisionModels
* showOptions hideOptions
*   showNormals hideNormals
*   showWireframe hideWireframe
*/
class SOFA_OPENGL_VISUAL_API VisualStyle : public sofa::core::visual::VisualModel
{
public:
    SOFA_CLASS(VisualStyle,sofa::core::visual::VisualModel);

    typedef sofa::core::visual::VisualParams VisualParams;
    typedef sofa::core::visual::DisplayFlags DisplayFlags;
protected:
    VisualStyle();
public:
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
