#ifndef SOFA_GUI_QT_SOFA_GUI_QT_QVISITORCONTROLPANEL_H
#define SOFA_GUI_QT_SOFA_GUI_QT_QVISITORCONTROLPANEL_H

#include <sofa/simulation/common/Node.h>

#ifdef SOFA_QT4
#include <QWidget>
#else
#include <qwidget.h>
#endif

#include "WDoubleLineEdit.h"


namespace sofa
{
namespace gui
{
namespace qt
{

class QVisitorControlPanel : public QWidget
{
    Q_OBJECT
public:
    QVisitorControlPanel(QWidget* parent);

    void changeFirstIndex(int);
    void changeRange(int);
public slots:
    void activateTraceStateVectors(bool);
    void changeFirstIndex();
    void changeRange();
    void filterResults();
signals:
    void focusOn(QString);
    void clearGraph();
protected:
    QLineEdit *textFilter;
    WDoubleLineEdit *spinIndex;
    WDoubleLineEdit *spinRange;
};


} // qt
} // gui
} //sofa

#endif // SOFA_GUI_QT_QVISITORCONTROLPANEL_H

