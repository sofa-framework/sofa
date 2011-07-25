#ifndef SOFA_GUI_QT_DISPLAYDATAWIDGET_H
#define SOFA_GUI_QT_DISPLAYDATAWIDGET_H

#include "DataWidget.h"
#ifdef SOFA_QT4
#include <QWidget>
#include <QLineEdit>
#include <QTextEdit>
#include <Q3GroupBox>
#include <QSlider>
#else
#include <qslider.h>
#include <qwidget.h>
#include <qtextedit.h>
#include <qlineedit.h>
#include <qgroupbox.h>
#endif

#ifndef SOFA_QT4
typedef QGroupBox Q3GroupBox;
typedef QTextEdit   Q3TextEdit;
#endif

namespace sofa
{
namespace core
{
namespace objectmodel
{
class BaseData;
}
}
namespace gui
{
namespace qt
{

class DataWidget;
class QDisplayDataInfoWidget;
struct ModifyObjectFlags;

class QDisplayDataWidget : public Q3GroupBox
{
    Q_OBJECT
public:
    QDisplayDataWidget(QWidget* parent,
            core::objectmodel::BaseData* data,
            const ModifyObjectFlags& flags);
    unsigned int getNumWidgets() const { return numWidgets_;};

public slots:
    void UpdateData();              //QWidgets ---> BaseData
    void UpdateWidgets();           //BaseData ---> QWidget
signals:
    void WidgetDirty(bool);
    void WidgetUpdate();
    void DataUpdate();
    void DataOwnerDirty(bool);
protected:
    core::objectmodel::BaseData* data_;
    QDisplayDataInfoWidget*  datainfowidget_;
    DataWidget* datawidget_;
    unsigned int numWidgets_;

};



class QDataSimpleEdit : public DataWidget
{
    Q_OBJECT
    typedef enum QEditType { TEXTEDIT, LINEEDIT } QEditType;
    typedef union QEditWidgetPtr
    {
        QLineEdit* lineEdit;
        QTextEdit* textEdit;
    } QEditWidgetPtr;

    typedef struct QSimpleEdit
    {
        QEditType type;
        QEditWidgetPtr widget;
    } QSimpleEdit;
public :
    QDataSimpleEdit(QWidget*, const char* name, core::objectmodel::BaseData*);
    virtual unsigned int numColumnWidget() {return 3;}
    virtual unsigned int sizeWidget() {return 1;}
    virtual bool createWidgets();
protected:
    virtual void readFromData();
    virtual void writeToData();
    QSimpleEdit innerWidget_;
};

class QPoissonRatioWidget : public TDataWidget<double>
{
    Q_OBJECT
public :
    QPoissonRatioWidget(QWidget*, const char*, core::objectmodel::Data<double>*);
    virtual bool createWidgets();

protected slots :
    void changeLineEditValue();
    void changeSliderValue();

protected:
    virtual void readFromData();
    virtual void writeToData();
    QSlider* slider;
    QLineEdit* lineEdit;

};





} // qt
} // gui
} //sofa

#endif // SOFA_GUI_QT_DISPLAYDATAWIDGET_H

