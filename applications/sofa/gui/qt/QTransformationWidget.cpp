
#include <sofa/gui/qt/QTransformationWidget.h>
#include <sofa/simulation/common/TransformationVisitor.h>

#ifdef SOFA_QT4
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <Q3GroupBox>
#include <QLabel>
#else
#include <qlayout.h>
#include <qlabel.h>
#include <qgroupbox.h>
#endif



namespace sofa
{

namespace gui
{
namespace qt
{
QTransformationWidget::QTransformationWidget(QWidget* parent, QString name):Q3GroupBox(parent), numWidgets_(2)

{

    this->setColumns(4);
    this->setTitle(name);
    //********************************************************************************
    //Translation
    new QLabel(QString("Translation"), this);
    translation[0] = new WFloatLineEdit( this, "translation[0]" );
    translation[0]->setMinFloatValue( (float)-INFINITY );
    translation[0]->setMaxFloatValue( (float)INFINITY );

    translation[1] = new WFloatLineEdit( this, "translation[1]" );
    translation[1]->setMinFloatValue( (float)-INFINITY );
    translation[1]->setMaxFloatValue( (float)INFINITY );

    translation[2] = new WFloatLineEdit( this, "translation[2]" );
    translation[2]->setMinFloatValue( (float)-INFINITY );
    translation[2]->setMaxFloatValue( (float)INFINITY );


    //********************************************************************************
    //Rotation
    new QLabel(QString("Rotation"), this);
    rotation[0] = new WFloatLineEdit( this, "rotation[0]" );
    rotation[0]->setMinFloatValue( (float)-INFINITY );
    rotation[0]->setMaxFloatValue( (float)INFINITY );

    rotation[1] = new WFloatLineEdit( this, "rotation[1]" );
    rotation[1]->setMinFloatValue( (float)-INFINITY );
    rotation[1]->setMaxFloatValue( (float)INFINITY );

    rotation[2] = new WFloatLineEdit( this, "rotation[2]" );
    rotation[2]->setMinFloatValue( (float)-INFINITY );
    rotation[2]->setMaxFloatValue( (float)INFINITY );


    //********************************************************************************
    //Scale
    QLabel *textScale = new QLabel(QString("Scale"), this);
    scale[0] = new WFloatLineEdit( this, "scale[0]" );
    scale[0]->setMinFloatValue( (float)-INFINITY );
    scale[0]->setMaxFloatValue( (float)INFINITY );

    scale[1] = new WFloatLineEdit( this, "scale[1]" );
    scale[1]->setMinFloatValue( (float)-INFINITY );
    scale[1]->setMaxFloatValue( (float)INFINITY );

    scale[2] = new WFloatLineEdit( this, "scale[2]" );
    scale[2]->setMinFloatValue( (float)-INFINITY );
    scale[2]->setMaxFloatValue( (float)INFINITY );

    setDefaultValues();

    connect( translation[0], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    connect( translation[1], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    connect( translation[2], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    connect( rotation[0], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    connect( rotation[1], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    connect( rotation[2], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    connect( scale[0], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    connect( scale[1], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
    connect( scale[2], SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );


    //Option still experimental : disabled !!!!
    textScale->hide();
    scale[0]->hide();
    scale[1]->hide();
    scale[2]->hide();
}

void QTransformationWidget::setDefaultValues()
{
    //********************************************************************************
    //Default values
    translation[0]->setFloatValue(0);
    translation[1]->setFloatValue(0);
    translation[2]->setFloatValue(0);

    rotation[0]->setFloatValue(0);
    rotation[1]->setFloatValue(0);
    rotation[2]->setFloatValue(0);

    scale[0]->setFloatValue(1);
    scale[1]->setFloatValue(1);
    scale[2]->setFloatValue(1);
}

bool QTransformationWidget::isDefaultValues() const
{
    return ( (translation[0]->getFloatValue() == 0 && translation[1]->getFloatValue() == 0 && translation[2]->getFloatValue() == 0 ) &&
            (rotation[0]->getFloatValue() == 0    && rotation[1]->getFloatValue() == 0    && rotation[2]->getFloatValue() == 0 ) &&
            (scale[0]->getFloatValue() == 1       && scale[1]->getFloatValue() == 1       && scale[2]->getFloatValue() == 1 ) );
}

void QTransformationWidget::applyTransformation(simulation::Node *node)
{
    sofa::simulation::TransformationVisitor transform;
    transform.setTranslation(translation[0]->getFloatValue(),translation[1]->getFloatValue(),translation[2]->getFloatValue());
    transform.setRotation(rotation[0]->getFloatValue(),rotation[1]->getFloatValue(),rotation[2]->getFloatValue());
    transform.setScale(scale[0]->getFloatValue(),scale[1]->getFloatValue(),scale[2]->getFloatValue());
    transform.execute(node);
}


} // qt
} //gui
} //sofa

