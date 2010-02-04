#include "QDisplayDataWidget.h"
#include "DataWidget.h"
#include "ModifyObject.h"
#include "QMonitorTableWidget.h"


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

#define TEXTSIZE_THRESHOLD 45

namespace sofa
{

using namespace core::objectmodel;
using namespace sofa::component::misc;
SOFA_LINK_CLASS(GraphDataWidget);
SOFA_LINK_CLASS(SimpleDataWidget);
SOFA_LINK_CLASS(StructDataWidget);
SOFA_LINK_CLASS(TableDataWidget);

namespace gui
{
namespace qt
{
QDisplayDataWidget::QDisplayDataWidget(QWidget* parent,
        BaseData* data,
        const ModifyObjectFlags& flags):Q3GroupBox(parent),
    data_(data),
    datainfowidget_(NULL),
    datawidget_(NULL),
    numWidgets_(0)

{
    if(data_ == NULL)
    {
        return;
    }

    setTitle(data_->getName().c_str());

    const std::string label_text = data_->getHelp();
    if (label_text != "TODO")
    {
        datainfowidget_ = new QDisplayDataInfoWidget(this,label_text,data_,flags.LINKPATH_MODIFIABLE_FLAG);
        numWidgets_ += datainfowidget_->getNumLines()/3;
    }

    DataWidget::CreatorArgument dwarg;
    dwarg.name =  data_->getName();
    dwarg.data = data_;
    dwarg.parent = this;
    dwarg.readOnly = (data_->isReadOnly() && flags.READONLY_FLAG);

    std::string widget = data_->getWidget();
    if (widget.empty())
        datawidget_ = DataWidgetFactory::CreateAnyObject(dwarg);
    else
        datawidget_ = DataWidgetFactory::CreateObject(dwarg.data->getWidget(), dwarg);
    if (datawidget_ == NULL)
    {
        Data<Monitor< defaulttype::Vec3Types >::MonitorData > *  ff;
        ff = dynamic_cast < Data<Monitor< defaulttype::Vec3Types >::MonitorData > *> (data_);
        if (ff )
        {
            QMonitorTableWidget<defaulttype::Vec3Types>* tableWidget = new QMonitorTableWidget<defaulttype::Vec3Types>(ff,flags,this);
            connect(this, SIGNAL(WidgetUpdate()), tableWidget, SLOT(UpdateWidget()) ) ;
            connect(this, SIGNAL( DataUpdate() ), tableWidget, SLOT(UpdateData() ) );

            setColumns(tableWidget->numColumnWidget());
            numWidgets_ += tableWidget->sizeWidget();
        }
        else
        {
            QDataSimpleEdit* dataSimpleEdit = new QDataSimpleEdit(this,data_,data_->isReadOnly() && flags.READONLY_FLAG);
            connect( dataSimpleEdit, SIGNAL( WidgetDirty(bool) ), this, SIGNAL (WidgetHasChanged(bool) ) );
            connect( this, SIGNAL (WidgetUpdate() ), dataSimpleEdit, SLOT( UpdateWidget() ) );
            connect( this, SIGNAL( DataUpdate() ), dataSimpleEdit, SLOT( UpdateData() ) );
            numWidgets_ += dataSimpleEdit->sizeWidget();
            setColumns(dataSimpleEdit->numColumnWidget());
        }

    }
    else
    {
        setColumns(datawidget_->numColumnWidget());
        //std::cout << "WIDGET created for data " << dwarg.data << " : " << dwarg.name << " : " << dwarg.data->getValueTypeString() << std::endl;
        numWidgets_+=datawidget_->sizeWidget();
        connect(datawidget_,SIGNAL(requestChange(bool)), this, SIGNAL ( WidgetHasChanged(bool) ) );
        connect(this, SIGNAL( WidgetUpdate() ), datawidget_, SLOT( updateWidget() ) );
        connect(this, SIGNAL( DataUpdate() ), datawidget_, SLOT(updateData() ) );
        connect(datawidget_,SIGNAL(dataParentNameChanged()),this,SIGNAL(DataParentNameChanged()) );
    }
}

void QDisplayDataWidget::UpdateData()
{
    emit DataUpdate();
}

void QDisplayDataWidget::UpdateWidgets()
{
    emit WidgetUpdate();
}

QDataSimpleEdit::QDataSimpleEdit(QWidget* parent, BaseData* data, bool readOnly):
    QWidget(parent),
    data_(data)
{
    if( data_ )
    {
        QString str  = QString( data_->getValueString().c_str() );
        if( str.size() > TEXTSIZE_THRESHOLD )
        {
            innerWidget_.type = TEXTEDIT;
            innerWidget_.widget.textEdit = new QTextEdit(parent);
            connect(innerWidget_.widget.textEdit , SIGNAL( textChanged() ), this, SLOT ( setWidgetDirty() ) );
            innerWidget_.widget.textEdit->setText(str);
            innerWidget_.widget.textEdit->setReadOnly(readOnly);
        }
        else
        {
            innerWidget_.type = LINEEDIT;
            innerWidget_.widget.lineEdit  = new QLineEdit(parent);
            connect( innerWidget_.widget.lineEdit, SIGNAL(textChanged(const QString&)), this, SLOT( setWidgetDirty() ) );
            innerWidget_.widget.lineEdit->setText(str);
            innerWidget_.widget.lineEdit->setReadOnly(readOnly);
        }
    }
}
void QDataSimpleEdit::UpdateWidget()
{
    if(data_)
    {
        QString str = QString( data_->getValueString().c_str() );
        if(innerWidget_.type == TEXTEDIT)
        {
            innerWidget_.widget.textEdit->setText(str);
        }
        else if(innerWidget_.type == LINEEDIT)
        {
            innerWidget_.widget.lineEdit->setText(str);
        }
    }
}

void QDataSimpleEdit::UpdateData()
{
    if(data_)
    {
        std::string value;
        if( innerWidget_.type == TEXTEDIT)
        {
            value = innerWidget_.widget.textEdit->text().ascii();
        }
        else if( innerWidget_.type == LINEEDIT)
        {
            value = innerWidget_.widget.lineEdit->text().ascii();
        }
        data_->read(value);
    }
}

void QDataSimpleEdit::setWidgetDirty(bool value)
{
    emit WidgetDirty(value);
}






} // qt
} //gui
} //sofa

