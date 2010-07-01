#ifndef SOFA_GUI_QT_VIEWERFACTORY_H
#define SOFA_GUI_QT_VIEWERFACTORY_H

#include <sofa/helper/Factory.h>
#include <sofa/gui/qt/viewer/SofaViewer.h>

namespace sofa
{
namespace gui
{
namespace qt
{
namespace viewer
{

struct CreatorArgument
{
    QWidget* parent;
    std::string name;
};





typedef sofa::helper::Factory< std::string, sofa::gui::qt::viewer::SofaViewer, sofa::gui::qt::viewer::CreatorArgument > SofaViewerFactory;
}
}
}
}

#endif //SOFA_GUI_QT_VIEWERFACTORY_H

