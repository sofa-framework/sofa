#include "MyDataWidgetUnsigned.h"
#include <sofa/helper/Factory.h>

namespace sofa
{

namespace gui
{

namespace qt
{

/*
register this new class in the DataWidgetFactory.
The factory key is the Data widget property
(see MyBehaviorModel constructor)
*/
helper::Creator<DataWidgetFactory,MyDataWidgetUnsigned> DW_myData("widget_myData",false);

bool MyDataWidgetUnsigned::createWidgets()
{
    unsigned myData_value = getData()->virtualGetValue();



    qslider = new QSlider(Qt::Horizontal, this);
    qslider->setTickmarks(QSlider::Below);
    qslider->setRange(0,100);
    qslider->setValue((int)myData_value);

    QString label1_text("Data current value = ");
    label1_text.append(getData()->getValueString().c_str());
    label1 = new QLabel(this);
    label1->setText( label1_text );

    QString label2_text = "Data value after updating = ";
    label2_text.append( QString().setNum(qslider->value()) );
    label2 = new QLabel(this);
    label2->setText( label2_text );


    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->add(label1);
    layout->add(label2);
    layout->add(qslider);

    connect(qslider,SIGNAL( sliderReleased() ), this, SLOT( setWidgetDirty() ));
    connect(qslider,SIGNAL( valueChanged(int) ),  this, SLOT( setWidgetDirty() ));
    connect(qslider,SIGNAL( sliderReleased() ), this, SLOT( change() ) );
    connect(qslider,SIGNAL( valueChanged(int) ), this, SLOT( change() ) );



    return true;
}

void MyDataWidgetUnsigned::readFromData()
{
    qslider->setValue( (int)getData()->virtualGetValue() );

    QString label1_text("myData current value = ");
    label1_text.append(getData()->getValueString().c_str());

    QString label2_text = "myData value after updating = ";
    label2_text.append( QString().setNum(qslider->value()) );

    label1->setText(label1_text);
    label2->setText(label2_text);

}

void MyDataWidgetUnsigned::writeToData()
{
    unsigned widget_value = (unsigned)qslider->value();
    getData()->virtualSetValue(widget_value);

    QString label1_text("myData current value = ");
    label1_text.append(getData()->getValueString().c_str());
    QString label2_text = "myData value after updating = ";
    label2_text.append( QString().setNum(qslider->value()) );

    label1->setText(label1_text);
    label2->setText(label2_text);

}

void MyDataWidgetUnsigned::change()
{
    QString label1_text("myData current value = ");
    label1_text.append(getData()->getValueString().c_str());
    QString label2_text = "myData value after updating = ";
    label2_text.append( QString().setNum(qslider->value()) );

    label1->setText(label1_text);
    label2->setText(label2_text);


}




} // qt
} // gui
} // sofa

