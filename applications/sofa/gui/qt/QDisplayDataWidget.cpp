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



namespace sofa
{

using namespace core::objectmodel;
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
        const ModifyObjectFlags& flags):
    data_(data),
    datainfowidget_(NULL),
    datawidget_(NULL),
    numWidgets_(0),
    Q3GroupBox(parent)
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
        setColumns(4);
        Data<sofa::component::misc::Monitor< defaulttype::Vec3Types >::MonitorData > *  ff;
        if ( ff = dynamic_cast < Data<sofa::component::misc::Monitor< defaulttype::Vec3Types >::MonitorData > *> (data_) )
        {
            QMonitorTableWidget<defaulttype::Vec3Types>* tableWidget = new QMonitorTableWidget<defaulttype::Vec3Types>(ff,flags,this);
            connect(this,SIGNAL(WidgetUpdate()),tableWidget,SLOT(UpdateWidget()) ) ;
            connect(this, SIGNAL( DataUpdate() ), tableWidget, SLOT(UpdateData() ) );
            numWidgets_ += 3;
        }
        else
        {
            QDataTextEdit* textedit = new QDataTextEdit(this,data_,flags);
            connect( textedit, SIGNAL( textChanged() ), this, SLOT( TextChange() ) );
            connect( this, SIGNAL (WidgetUpdate() ), textedit, SLOT( UpdateWidget() ) );
            connect( this, SIGNAL( DataUpdate() ), textedit, SLOT( UpdateData() ) );
            numWidgets_ += 1;
        }
    }
    else
    {
        setColumns(2);
        //std::cout << "WIDGET created for data " << dwarg.data << " : " << dwarg.name << " : " << dwarg.data->getValueTypeString() << std::endl;
        numWidgets_+=datawidget_->sizeWidget();
        connect(datawidget_,SIGNAL(requestChange(bool)), this, SIGNAL ( WidgetHasChanged(bool) ) );
        connect(this, SIGNAL( WidgetUpdate() ), datawidget_, SLOT( updateWidget() ) );
        connect(this, SIGNAL( DataUpdate() ), datawidget_, SLOT(updateData() ) );
        connect(datawidget_,SIGNAL(dataParentNameChanged()),this,SIGNAL(DataParentNameChanged()) );
    }
}

void QDisplayDataWidget::TextChange()
{
    emit WidgetHasChanged(true);
}

void QDisplayDataWidget::UpdateData()
{
    emit DataUpdate();
}

void QDisplayDataWidget::UpdateWidgets()
{
    emit WidgetUpdate();
}

QDataTextEdit::QDataTextEdit(QWidget* parent, BaseData* data, const ModifyObjectFlags& flags):
    data_(data),
    QTextEdit(parent)
{
    if( data_ )
    {
        setText( QString( data_->getValueString().c_str() ) );
        if( data_->getValueString().empty() && !flags.EMPTY_FLAG )
        {
            hide();
            std::cerr << data_->getValueTypeString() << " Not added because empty \n";
        }
    }
}
void QDataTextEdit::UpdateWidget()
{
    if(data_) setText( data_->getValueString().c_str() );
}

void QDataTextEdit::UpdateData()
{
    if(data_)
    {
        std::string value = text().ascii();
        data_->read(value);
    }
}





} // qt
} //gui
} //sofa

