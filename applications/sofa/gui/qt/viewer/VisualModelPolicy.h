#ifndef SOFA_GUI_QT_VIEWER_VISUALMODELPOLICY_H
#define SOFA_GUI_QT_VIEWER_VISUALMODELPOLICY_H

#include <sofa/gui/qt/SofaGUIQt.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/visual/DrawToolGL.h>

namespace sofa
{
namespace gui
{
namespace qt
{
namespace viewer
{

class SOFA_SOFAGUIQT_API VisualModelPolicy
{
public:
    VisualModelPolicy(core::visual::VisualParams* vparams = core::visual::VisualParams::defaultInstance())
        :vparams(vparams)
    {}
    virtual ~VisualModelPolicy() {};
    virtual void load() = 0;
    virtual void unload() = 0;
protected:
    sofa::core::visual::VisualParams* vparams;

};


class OglModelPolicy : public VisualModelPolicy
{
protected:
    sofa::core::ObjectFactory::ClassEntry* classVisualModel;
    sofa::core::visual::DrawToolGL drawTool;
public:
    void load()
    {
        // Replace generic visual models with OglModel
        sofa::core::ObjectFactory::AddAlias("VisualModel", "OglModel", true,
                &classVisualModel);
        vparams->drawTool() = &drawTool;
    }
    void unload()
    {
        sofa::core::ObjectFactory::ResetAlias("VisualModel", classVisualModel);
        vparams->drawTool() = NULL;
    }
};

}
}
}
}



#endif // SOFA_GUI_QT_VIEWER_VISUALMODELPOLICY_H
