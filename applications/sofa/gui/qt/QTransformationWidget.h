#ifndef SOFA_GUI_QT_TRANSFORMATIONWIDGET_H
#define SOFA_GUI_QT_TRANSFORMATIONWIDGET_H

#include <sofa/simulation/common/Node.h>
#include "WDoubleLineEdit.h"

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
class QTransformationWidget : public Q3GroupBox
{
    Q_OBJECT
public:
    QTransformationWidget(QWidget* parent, QString name);
    unsigned int getNumWidgets() const { return numWidgets_;};

    void setDefaultValues();
    bool isDefaultValues() const;
    void applyTransformation(simulation::Node *node);
public slots:
    void changeValue() {emit TransformationDirty(true);}
signals:
    void TransformationDirty(bool);
protected:
    const unsigned int numWidgets_;

    WDoubleLineEdit* translation[3];
    WDoubleLineEdit* rotation[3];
    WDoubleLineEdit* scale[3];
};


} // qt
} // gui
} //sofa

#endif // SOFA_GUI_QT_TRANSFORMATIONWIDGET_H

