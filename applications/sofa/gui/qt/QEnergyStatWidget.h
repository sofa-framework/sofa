#ifndef SOFA_GUI_QT_QENERGYSTATWIDGET_H
#define SOFA_GUI_QT_QENERGYSTATWIDGET_H

#include <sofa/simulation/common/Node.h>
#include <sofa/core/componentmodel/behavior/BaseForceField.h>
#include <sofa/core/componentmodel/behavior/BaseMass.h>

#ifdef SOFA_QT4
#include <QWidget>
#include <QTextEdit>
#include <Q3GroupBox>
#else
#include <qwidget.h>
#include <qtextedit.h>
#include <qgroupbox.h>
#endif


#include <qwt_plot.h>
#include <qwt_plot_curve.h>

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

class QEnergyStatWidget : public QWidget
{
    Q_OBJECT
public:
    QEnergyStatWidget(QWidget* parent, simulation::Node* node);
    void step();
    void updateVisualization();

protected:
    simulation::Node *node;

    std::vector< double > history;
    std::vector< double > energy_history[3];


    QwtPlot *graphEnergy;
    QwtPlotCurve *energy_curve[3];
};


} // qt
} // gui
} //sofa

#endif // SOFA_GUI_QT_QDATADESCRIPTIONWIDGET_H

