
#include <sofa/gui/qt/QVisitorControlPanel.h>
#include <sofa/simulation/common/Visitor.h>

#ifdef SOFA_QT4
#include <QPushButton>
#include <QCheckBox>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#else
#include <qpushbutton.h>
#include <qcheckbox.h>
#include <qlabel.h>
#include <qlayout.h>
#endif

#include "WFloatLineEdit.h"


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

    WFloatLineEdit *spinIndex = new WFloatLineEdit(exportStateParameters, "index");
    spinIndex->setMinFloatValue( (float)-INFINITY );
    spinIndex->setMaxFloatValue( (float)INFINITY );
    spinIndex->setIntValue(sofa::simulation::Visitor::GetFirstIndexStateVector());
    WFloatLineEdit *spinRange = new WFloatLineEdit(exportStateParameters, "range");
    spinRange->setMinFloatValue( (float)-INFINITY );
    spinRange->setMaxFloatValue( (float)INFINITY );
    spinRange->setIntValue(sofa::simulation::Visitor::GetRangeStateVector());


    connect(activation, SIGNAL(toggled(bool)), this, SLOT(activateTraceStateVectors(bool)));
    connect(spinIndex, SIGNAL(lostFocus()), this, SLOT(changeFirstIndex()));
    connect(spinRange, SIGNAL(lostFocus()), this, SLOT(changeRange()));


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
void QVisitorControlPanel::changeFirstIndex()
{
    WFloatLineEdit *w=(WFloatLineEdit *) sender();
    int value=w->getIntValue();
    changeFirstIndex(value);
}
void QVisitorControlPanel::changeRange()
{
    WFloatLineEdit *w=(WFloatLineEdit *) sender();
    int value=w->getIntValue();
    changeRange(value);
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


