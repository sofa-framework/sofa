#ifndef SOFA_COMPONENT_CONFIGURATIONSETTING_ADAPTATIVEATTACHBUTTON_H
#define SOFA_COMPONENT_CONFIGURATIONSETTING_ADAPTATIVEATTACHBUTTON_H

#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <sofa/component/configurationsetting/MouseButtonSetting.h>
#include <sofa/component/component.h>

namespace sofa
{

namespace component
{

namespace configurationsetting
{

class SOFA_GRAPH_COMPONENT_API AdaptativeAttachButtonSetting: public MouseButtonSetting
{
public:
    SOFA_CLASS(AdaptativeAttachButtonSetting,MouseButtonSetting);
protected:
    AdaptativeAttachButtonSetting();
public:
    std::string getOperationType() {return "AdaptativeAttach";}
    Data<SReal> stiffness;
    Data<SReal> arrowSize;
};

}

}

}
#endif
