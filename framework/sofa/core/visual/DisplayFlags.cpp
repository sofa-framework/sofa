#include <sofa/core/visual/DisplayFlags.h>

namespace sofa
{
namespace core
{
namespace visual
{

//const std::string DisplayFlags::valid_entries[] =
//{
//       "showAll",
//       "showVisualModels",
//       "showBehavior",
//       "showBehaviorModels",
//       "showForceFields",
//       "showInteractionForceFields",
//       "showCollision",
//       "showCollisionModels",
//       "showBoundingCollisionModels",
//       "showMapping",
//       "showMappings",
//       "showMechanicalMappings",
//       "showNormals",
//       "showWireFrame",
//       "showProcessorColor"
//};

//const size_t DisplayFlags::num_entries = 15;

DisplayFlags::DisplayFlags()
    :m_showAll(false)
    ,m_showVisual(false)
    ,m_showVisualModels(true)
    ,m_showBehavior(false)
    ,m_showBehaviorModels(false)
    ,m_showForceFields(false)
    ,m_showInteractionForceFields(false)
    ,m_showCollision(false)
    ,m_showCollisionModels(false)
    ,m_showBoundingCollisionModels(false)
    ,m_showMapping(false)
    ,m_showMappings(false)
    ,m_showMechanicalMappings(false)
    ,m_showNormals(false)
    ,m_showWireFrame(false)
#ifdef SOFA_SMP
    ,m_showProcessorColor(false)
#endif
{
}

}

}

}
