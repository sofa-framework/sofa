#ifndef SOFA_GUI_QT_DISPLAYLINKWIDGET_H
#define SOFA_GUI_QT_DISPLAYLINKWIDGET_H

#include "LinkWidget.h"
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

namespace gui
{

namespace qt
{

class LinkWidget;
class QDisplayLinkInfoWidget;
struct ModifyObjectFlags;

class QDisplayLinkWidget : public Q3GroupBox
{
    Q_OBJECT
public:
    QDisplayLinkWidget(QWidget* parent,
            core::objectmodel::BaseLink* link,
            const ModifyObjectFlags& flags);
    unsigned int getNumWidgets() const { return numWidgets_;};

public slots:
    void UpdateLink();              //QWidgets ---> BaseLink
    void UpdateWidgets();           //BaseLink ---> QWidget
signals:
    void WidgetDirty(bool);
    void WidgetUpdate();
    void LinkUpdate();
    void LinkOwnerDirty(bool);
protected:
    core::objectmodel::BaseLink* link_;
    QDisplayLinkInfoWidget*  linkinfowidget_;
    LinkWidget* linkwidget_;
    unsigned int numWidgets_;

};



class QLinkSimpleEdit : public LinkWidget
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
    QLinkSimpleEdit(QWidget*, const char* name, core::objectmodel::BaseLink*);
    virtual unsigned int numColumnWidget() {return 3;}
    virtual unsigned int sizeWidget() {return 1;}
    virtual bool createWidgets();
protected:
    virtual void readFromLink();
    virtual void writeToLink();
    QSimpleEdit innerWidget_;
};

} // namespace qt

} // namespace gui

} // namespace sofa

#endif // SOFA_GUI_QT_DISPLAYLINKWIDGET_H
