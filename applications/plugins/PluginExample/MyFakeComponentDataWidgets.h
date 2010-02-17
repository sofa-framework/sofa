#ifndef SOFA_GUI_QT_MYFAKECOMPONENTDATAWIDGET_H
#define SOFA_GUI_QT_MYFAKECOMPONENTDATAWIDGET_H

#include <sofa/gui/qt/DataWidget.h>

#ifdef SOFA_QT4
#include <QLabel>
#include <QVBoxLayout>
#include <QSlider>
#include <QString>
#else
#include <qlabel.h>
#include <qlayout.h>
#include <qslider.h>
#include <qstring.h>
#endif



namespace sofa
{
namespace gui
{
namespace qt
{
/* class used to override the widget representation of myData for the component MyFakeComponent */
class WidgetmyData : public TDataWidget<unsigned>
{
    Q_OBJECT
public :
    WidgetmyData(QWidget* parent, const char* name, core::objectmodel::TData<unsigned>* data):
        TDataWidget<unsigned>(parent,name,data) {};
    virtual bool createWidgets();
protected slots:
    void change();
protected:
    virtual void readFromData();
    virtual void writeToData();
    QSlider* qslider;
    QLabel*  label1;
    QLabel*  label2;
};

}

}

}
#endif
