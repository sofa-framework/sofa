#ifndef SOFA_GUI_QT_DISPLAYDATAWIDGET_H
#define SOFA_GUI_QT_DISPLAYDATAWIDGET_H

#ifdef SOFA_QT4
#include <QWidget>
#include <QTextEdit>
#include <Q3GroupBox>
#else
#include <qwidget.h>
#include <qtextedit.h>
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
    void WidgetHasChanged(bool);
    void WidgetUpdate();
    void DataUpdate();
    void DataParentNameChanged();
protected:
    core::objectmodel::BaseData* data_;
    QDisplayDataInfoWidget*  datainfowidget_;
    DataWidget* datawidget_;
    unsigned int numWidgets_;

};



} // qt
} // gui
} //sofa

#endif // SOFA_GUI_QT_DISPLAYDATAWIDGET_H

