/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "QDisplayDataWidget.h"

#include <QTableWidget>

#include <QPalette>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include <QValidator>

#define TEXTSIZE_THRESHOLD 45

namespace sofa
{

using namespace core::objectmodel;
using namespace sofa::component::misc;

namespace gui
{
namespace qt
{

QDisplayDataWidget::QDisplayDataWidget(QWidget* parent,
        BaseData* data,
        const ModifyObjectFlags& flags):QGroupBox(),
    data_(data),
    flags_(flags),
    datainfowidget_(NULL),
    datawidget_(NULL),
    numWidgets_(0)

{
    gridLayout_ = new QGridLayout();
    this->setLayout(gridLayout_);

    parent->layout()->addWidget(this);

    setAutoFillBackground(true);

    if(data_ == NULL)
        return;

    const char* help_text = data_->getHelp();
    const std::string label_text = help_text == NULL ? "" : help_text;

    if (label_text != "TODO")
    {
        datainfowidget_ = new QDisplayDataInfoWidget(this,label_text,data_,flags.LINKPATH_MODIFIABLE_FLAG, flags_);
        datainfowidget_->setContentsMargins(0, 0, 0, 0);
        datainfowidget_->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
        gridLayout_->addWidget(datainfowidget_, 0,0);
        numWidgets_ += 1;
    }

    setToolTip(data_->getHelp());

    DataWidget::CreatorArgument dwarg;
    dwarg.name =  data_->getName();
    dwarg.data = data_;
    dwarg.parent = this;
    dwarg.readOnly = (data_->isReadOnly() && flags.READONLY_FLAG);

    if( dynamic_cast<core::objectmodel::DataFileName*>(data_) != NULL )
    {
        // a bit of a hack for DataFileName widgets.
        // A custom widget is used by default if we run this code from the Modeler

        std::string widgetName=data_->getWidget();
        if( widgetName.empty() && flags.MODELER_FLAG )
        {
            data_->setWidget("widget_filename");
        }
    }

    datawidget_= DataWidget::CreateDataWidget(dwarg);

    if (datawidget_ == 0)
    {
        datawidget_ = new QDataSimpleEdit(this,dwarg.data->getName().c_str(), dwarg.data);
        datawidget_->createWidgets();
        datawidget_->setDataReadOnly(dwarg.readOnly);
        assert(datawidget_ != NULL);
    }

    if(datawidget_->layout())
    {
        datawidget_->layout()->setAlignment(Qt::AlignCenter);
        datawidget_->layout()->setContentsMargins(0, 0, 0, 0);
    }

    datawidget_->setContentsMargins(0, 16, 0, 0);
    datawidget_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    const std::string valuetype = data_->getValueTypeString();
    if (!valuetype.empty())
        datawidget_->setToolTip(valuetype.c_str());

    numWidgets_ += datawidget_->sizeWidget();
    connect(datawidget_,SIGNAL(WidgetDirty(bool)), this, SIGNAL ( WidgetDirty(bool) ) );
    connect(this, SIGNAL( WidgetUpdate() ), datawidget_, SLOT( updateWidgetValue() ) );
    connect(this, SIGNAL( DataUpdate() ), datawidget_, SLOT(updateDataValue() ) );
    connect(datawidget_,SIGNAL(DataOwnerDirty(bool)),this,SIGNAL(DataOwnerDirty(bool)) );
    connect(datawidget_,SIGNAL(dataValueChanged(QString)),this,SIGNAL(dataValueChanged(QString)) );

    if(flags.PROPERTY_WIDGET_FLAG)
    {
        QWidget* refreshWidget = new QWidget(this);
        refreshWidget->setFixedSize(QSize(16, 16));
        QPushButton *refresh = new QPushButton(RefreshIcon(), "", refreshWidget);
        refresh->setHidden(true);
        refresh->setFixedSize(QSize(16, 16));
        refresh->setContentsMargins(0, 0, 0, 0);

        ++numWidgets_;

        {
            connect(datawidget_,SIGNAL(WidgetDirty(bool)), refresh, SLOT ( setVisible(bool) ) );

            if(datainfowidget_ && flags.LINKPATH_MODIFIABLE_FLAG)
            {
                connect(datainfowidget_,SIGNAL(WidgetDirty()), refresh, SLOT ( show() ) );
                connect(refresh, SIGNAL(clicked()), datainfowidget_, SLOT(linkEdited()));
            }

            connect(refresh, SIGNAL(clicked()), this, SLOT(UpdateData()));
            connect(refresh, SIGNAL(clicked(bool)), refresh, SLOT(setVisible(bool)));
        }

        setStyleSheet("QGroupBox{border:0;}");
        setContentsMargins(0, 0, 0, 0);
    }
    else
    {
        setTitle(data_->getName().c_str());
        setContentsMargins(0,0,0,0);
    }
    gridLayout_->setContentsMargins(10,10,10,10);
    gridLayout_->addWidget(datawidget_, 0, 1);
    gridLayout_->setAlignment(datawidget_, Qt::AlignVCenter);
}

void QDisplayDataWidget::UpdateData()
{
    emit DataUpdate();
}

void QDisplayDataWidget::UpdateWidgets()
{
    emit WidgetUpdate();
}

QDataSimpleEdit::QDataSimpleEdit(QWidget* parent, const char* name, BaseData* data):
    DataWidget(parent,name,data)
{
}
bool QDataSimpleEdit::createWidgets()
{
    QString str  = QString( getBaseData()->getValueString().c_str() );
    QLayout* layout = new QHBoxLayout(this);
    if( str.length() > TEXTSIZE_THRESHOLD )
    {
        innerWidget_.type = TEXTEDIT;
        innerWidget_.widget.textEdit = new QTextEdit(this); innerWidget_.widget.textEdit->setText(str);
        connect(innerWidget_.widget.textEdit , SIGNAL( textChanged() ), this, SLOT ( setWidgetDirty() ) );
        layout->addWidget(innerWidget_.widget.textEdit);
    }
    else
    {
        innerWidget_.type = LINEEDIT;
        innerWidget_.widget.lineEdit  = new QLineEdit(this);
        innerWidget_.widget.lineEdit->setText(str);
        connect( innerWidget_.widget.lineEdit, SIGNAL(textChanged(const QString&)), this, SLOT( setWidgetDirty() ) );
        layout->addWidget(innerWidget_.widget.lineEdit);
    }




    return true;
}

void QDataSimpleEdit::setDataReadOnly(bool readOnly)
{
    if(innerWidget_.type == TEXTEDIT)
    {
        innerWidget_.widget.textEdit->setReadOnly(readOnly);
    }
    else if(innerWidget_.type == LINEEDIT)
    {
        innerWidget_.widget.lineEdit->setReadOnly(readOnly);
    }
}

void QDataSimpleEdit::readFromData()
{
    QString str = QString( getBaseData()->getValueString().c_str() );
    if(innerWidget_.type == TEXTEDIT)
    {
        innerWidget_.widget.textEdit->setText(str);
    }
    else if(innerWidget_.type == LINEEDIT)
    {
        innerWidget_.widget.lineEdit->setText(str);
    }
}

void QDataSimpleEdit::writeToData()
{
    if(!getBaseData())
        return ;

    if(getBaseData()->isReadOnly())
        return ;
    std::string value;
    if( innerWidget_.type == TEXTEDIT)
    {
        value = innerWidget_.widget.textEdit->toPlainText().toStdString();
    }
    else if( innerWidget_.type == LINEEDIT)
    {
        value = innerWidget_.widget.lineEdit->text().toStdString();
    }
    getBaseData()->read(value);
}

/* QPoissonRatioWidget */
QPoissonRatioWidget::QPoissonRatioWidget(QWidget * parent, const char * name, sofa::core::objectmodel::Data<double> *data)
    :TDataWidget<double>(parent,name,data)
{

}


bool QPoissonRatioWidget::createWidgets()
{
    QGridLayout* layout = new QGridLayout(this /*,2,3 */);

    lineEdit = new QLineEdit(this);
    lineEdit->setText(QString("-1.0"));
    lineEdit->setMaximumSize(lineEdit->size());
    lineEdit->setAlignment(Qt::AlignHCenter);

    lineEdit->setValidator(new QDoubleValidator(0.0,0.5,2,this));

    layout->addWidget(lineEdit,0,1,Qt::AlignHCenter);
    QLabel* min = new QLabel(this);
    min->setText(QString("0.0"));
    min->setMaximumSize( min->sizeHint() );
    layout->addWidget(min,1,0,Qt::AlignHCenter);

    slider = new QSlider(Qt::Horizontal, this);
    slider->setRange(0,50); //max times 10 at the power 2 (2 digits after ".")
    slider->setTickPosition(QSlider::TicksBelow);
    slider->setTickInterval(5);
    layout->addWidget(slider,1,1,Qt::AlignHCenter);

    QLabel* max = new QLabel(this);
    max->setText(QString("0.5"));
    max->setMaximumSize ( max->sizeHint() );

    layout->addWidget(max,1,2,Qt::AlignHCenter);

    // synchronization between qslider and qlineedit
    connect(slider, SIGNAL( valueChanged(int) ), this, SLOT ( changeLineEditValue() ) );
    connect(slider, SIGNAL( sliderReleased()   ), this, SLOT ( changeLineEditValue() ) );
    connect(lineEdit, SIGNAL( textChanged(const QString&) ), this, SLOT (changeSliderValue() ) );

    // synchronization between the widgets and the modify object dialog box
    connect(lineEdit, SIGNAL( textChanged(const QString&) ), this, SLOT( setWidgetDirty() ) );
    connect(slider, SIGNAL( sliderReleased()  ), this, SLOT ( setWidgetDirty() ) );
    connect(slider, SIGNAL( valueChanged(int) ), this, SLOT ( setWidgetDirty() ) );


    return true;
}

void QPoissonRatioWidget::setDataReadOnly(bool readOnly)
{
    lineEdit->setReadOnly(readOnly);
    slider->setEnabled(!readOnly);
}

void QPoissonRatioWidget::readFromData()
{
    double value = this->getData()->virtualGetValue();
    QString str;
    str.setNum(value);
    lineEdit->setText(str);
    changeSliderValue();
}

void QPoissonRatioWidget::writeToData()
{
    bool ok;
    double d = lineEdit->text().toDouble(&ok);
    if(ok)
    {
        this->getData()->virtualSetValue(d);
    }
}

void QPoissonRatioWidget::changeLineEditValue()
{
    int v = slider->value();
    double db = (double)v / 100.;
    QString str;
    str.setNum(db);
    lineEdit->setText(str);
}

void QPoissonRatioWidget::changeSliderValue()
{
    bool ok;
    double v = lineEdit->text().toDouble(&ok);
    if(ok)
    {
        slider->setValue( (int)(v*100.) );
    }
}

helper::Creator<DataWidgetFactory, QPoissonRatioWidget> DWClass_Poissonratio("poissonRatio",false);

} // qt
} //gui
} //sofa

