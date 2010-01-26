
#include <sofa/gui/qt/QVisitorControlPanel.h>
#include <sofa/simulation/common/Visitor.h>

#ifdef SOFA_QT4
#include <QPushButton>
#include <QCheckBox>
#include <QSpinBox>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#else
#include <qpushbutton.h>
#include <qcheckbox.h>
#include <qspinbox.h>
#include <qlabel.h>
#include <qlayout.h>
#endif



namespace sofa
{
namespace gui
{
namespace qt
{

QVisitorControlPanel::QVisitorControlPanel(QWidget* parent): QWidget(parent)
{
    QVBoxLayout *vbox=new QVBoxLayout(this);

    //Parameters to configure the export of the state vectors
    QWidget *exportStateParameters = new QWidget(this);

    QHBoxLayout *hboxParameters=new QHBoxLayout(exportStateParameters);

    QCheckBox *activation=new QCheckBox(QString("Trace State Vector"), exportStateParameters);

    QSpinBox *spinIndex = new QSpinBox(exportStateParameters);
    spinIndex->setMinValue(0); spinIndex->setValue(sofa::simulation::Visitor::GetFirstIndexStateVector());
    QSpinBox *spinRange = new QSpinBox(exportStateParameters);
    spinRange->setValue(sofa::simulation::Visitor::GetRangeStateVector());


    connect(activation, SIGNAL(toggled(bool)), this, SLOT(activateTraceStateVectors(bool)));
    connect(spinIndex, SIGNAL(valueChanged(int)), this, SLOT(changeFirstIndex(int)));
    connect(spinRange, SIGNAL(valueChanged(int)), this, SLOT(changeRange(int)));


    hboxParameters->addWidget(activation);

    hboxParameters->addWidget(spinIndex);
    hboxParameters->addWidget(new QLabel(QString("First Index"), exportStateParameters));

    hboxParameters->addWidget(spinRange);
    hboxParameters->addWidget(new QLabel(QString("Range"), exportStateParameters));

    hboxParameters->addStretch();



    activateTraceStateVectors(sofa::simulation::Visitor::IsExportStateVectorEnabled());


    //Filter the results to quickly find a visitor
    QWidget *filterResult = new QWidget(this);
    QHBoxLayout *hboxFilers=new QHBoxLayout(filterResult);

    textFilter = new QLineEdit(filterResult);
    QPushButton *findFilter = new QPushButton(QString("Find"), filterResult);
    findFilter->setAutoDefault(true);

    hboxFilers->addWidget(new QLabel(QString("Focus on:"), filterResult));
    hboxFilers->addWidget(textFilter);
    hboxFilers->addWidget(findFilter);

    connect(findFilter, SIGNAL(clicked()), this, SLOT(filterResults()));
    connect(textFilter, SIGNAL(returnPressed()), this, SLOT(filterResults()));

    //Configure the main window
    vbox->addWidget(exportStateParameters);
    vbox->addWidget(filterResult);
}

void QVisitorControlPanel::activateTraceStateVectors(bool active)
{
    sofa::simulation::Visitor::EnableExportStateVector(active);
}
void QVisitorControlPanel::changeFirstIndex(int idx)
{
    sofa::simulation::Visitor::SetFirstIndexStateVector(idx);
}
void QVisitorControlPanel::changeRange(int range)
{
    sofa::simulation::Visitor::SetRangeStateVector(range);
}
void QVisitorControlPanel::filterResults()
{
    emit focusOn(textFilter->text());
}



} // qt
} // gui
} //sofa


