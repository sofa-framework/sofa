#ifndef SOFA_GUI_QT_INFORMATIONONPICKCALLBACK
#define SOFA_GUI_QT_INFORMATIONONPICKCALLBACK

#include <sofa/gui/PickHandler.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/gui/ColourPickingVisitor.h>

namespace sofa
{
namespace component
{
namespace collision
{
struct BodyPicked;
}
}
namespace gui
{
namespace qt
{
namespace viewer
{
class SofaViewer;
}

class RealGUI;

class InformationOnPickCallBack: public CallBackPicker
{
public:
    InformationOnPickCallBack();
    InformationOnPickCallBack(RealGUI *g);
    void execute(const sofa::component::collision::BodyPicked &body);
protected:
    RealGUI *gui;
};


class ColourPickingRenderCallBack : public sofa::gui::CallBackRender
{
public:
    ColourPickingRenderCallBack();
    ColourPickingRenderCallBack(viewer::SofaViewer* viewer);
    void render(ColourPickingVisitor::ColourCode code);
protected:
    viewer::SofaViewer* _viewer;

};
}
}
}

#endif // SOFA_GUI_QT_INFORMATIONONPICKCALLBACK
