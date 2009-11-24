#ifndef SOFA_GUI_QT_DATAWIDGET_CPP
#define SOFA_GUI_QT_DATAWIDGET_CPP

#include "DataWidget.h"
#include "ModifyObject.h"
#include <sofa/helper/Factory.inl>



namespace sofa
{
namespace gui
{
namespace qt
{
using namespace sofa::helper;
template class Factory<std::string, DataWidget, DataWidget::CreatorArgument>;

bool DefaultDataWidget::createWidgets(QWidget *parent)
{
    w = new QLineEdit(parent);
    if (w == NULL) return false;
    std::string s = data->getValueString();
    w->setText(QString(s.c_str()));
    counter = data->getCounter();
    if (this->readOnly)
        w->setEnabled(false);
    else
        dialog->connect(w, SIGNAL( textChanged(const QString&) ), dialog, SLOT( changeValue() ));
    return true;
}

}//qt
}//gui
}//sofa

#endif //SOFA_GUI_QT_DATAWIDGET_CPP

