#ifndef SOFA_GUI_QT_MyDataWidgetUnsigned_H
#define SOFA_GUI_QT_MyDataWidgetUnsigned_H

#include "initPlugin.h"
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
/**
*\brief Customization of the representation of Data<unsigned> types
* in the gui. In the .cpp file this widget is registered to represent
* myData from MyBehaviorModel in the gui.
**/
class SOFA_MyPluginExample_API MyDataWidgetUnsigned : public TDataWidget<unsigned>
{
    Q_OBJECT
public :
    ///The class constructor takes a TData<unsigned> since it creates
    ///a widget for a that particular data type.
    MyDataWidgetUnsigned(QWidget* parent, const char* name, core::objectmodel::Data<unsigned>* data):
        TDataWidget<unsigned>(parent,name,data) {};
    ///In this method we  create the widgets and perform the signal / slots
    ///connections.
    virtual bool createWidgets();
protected slots:
    void change();
protected:
    ///Implements how update the widgets knowing the data value.
    virtual void readFromData();
    ///Implements how to update the data, knowing the widget value.
    virtual void writeToData();
    QSlider* qslider;
    QLabel*  label1;
    QLabel*  label2;
};

}

}

}
#endif
