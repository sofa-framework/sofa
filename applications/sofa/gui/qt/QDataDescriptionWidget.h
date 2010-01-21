#ifndef SOFA_GUI_QT_QDATADESCRIPTIONWIDGET_H
#define SOFA_GUI_QT_QDATADESCRIPTIONWIDGET_H

#include <sofa/core/objectmodel/Base.h>

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
namespace gui
{
namespace qt
{

struct ModifyObjectFlags;
class QDataDescriptionWidget : public QWidget
{
    Q_OBJECT
public:
    QDataDescriptionWidget(QWidget* parent, core::objectmodel::Base* object);

};


} // qt
} // gui
} //sofa

#endif // SOFA_GUI_QT_QDATADESCRIPTIONWIDGET_H

