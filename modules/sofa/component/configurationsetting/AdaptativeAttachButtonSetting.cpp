#include <sofa/component/configurationsetting/AdaptativeAttachButtonSetting.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace configurationsetting
{

SOFA_DECL_CLASS(AdaptativeAttachButtonSetting)
int AdaptativeAttachButtonSettingClass = core::RegisterObject("Adaptative Attach Button configuration")
        .add< AdaptativeAttachButtonSetting >()
        .addAlias("AdaptativeAttachButton")
        ;


AdaptativeAttachButtonSetting::AdaptativeAttachButtonSetting():
    stiffness(initData(&stiffness, (SReal)1000.0, "stiffness", "Stiffness of the spring to attach a particule"))
    , arrowSize(initData(&arrowSize, (SReal)0.0, "arrowSize", "Size of the drawn spring: if >0 an arrow will be drawn"))
{
}

} //configurationsetting

} //component

} //sofa

