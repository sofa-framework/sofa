#include "QDisplayLinkWidget.h"
#include "ModifyObject.h"


#ifdef SOFA_QT4
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <Q3GroupBox>
#include <QLabel>
#include <QValidator>
#else
#include <qlayout.h>
#include <qlabel.h>
#include <qgroupbox.h>
#include <qvalidator.h>
#endif

#define TEXTSIZE_THRESHOLD 45

namespace sofa
{

using namespace core::objectmodel;
using namespace sofa::component::misc;

namespace gui
{

namespace qt
{

QDisplayLinkWidget::QDisplayLinkWidget(QWidget* parent,
        BaseLink* link,
        const ModifyObjectFlags& flags)
    : Q3GroupBox(parent),
      link_(link),
      linkinfowidget_(NULL),
      linkwidget_(NULL),
      numWidgets_(0)
{
    if(link_ == NULL)
    {
        return;
    }

    setTitle(link_->getName().c_str());
    setInsideMargin(4);
    setInsideSpacing(2);

    const std::string label_text = link_->getHelp();

    if (label_text != "TODO")
    {
        linkinfowidget_ = new QDisplayLinkInfoWidget(this,label_text,link_,flags.LINKPATH_MODIFIABLE_FLAG);
        numWidgets_ += linkinfowidget_->getNumLines()/3;

    }

    const std::string valuetype = link_->getValueTypeString();
    if (!valuetype.empty())
        setToolTip(valuetype.c_str());

    LinkWidget::CreatorArgument dwarg;
    dwarg.name =  link_->getName();
    dwarg.link = link_;
    dwarg.parent = this;
    dwarg.readOnly = (!link_->storePath() && flags.READONLY_FLAG);

    linkwidget_= LinkWidget::CreateLinkWidget(dwarg);

    if (linkwidget_ == 0)
    {
        linkwidget_ = new QLinkSimpleEdit(this,dwarg.link->getName().c_str(), dwarg.link);
        linkwidget_->createWidgets();
        linkwidget_->setEnabled( !(dwarg.readOnly) );
        assert(linkwidget_ != NULL);
    }

    setColumns(linkwidget_->numColumnWidget());
    //std::cout << "WIDGET created for link " << dwarg.link << " : " << dwarg.name << " : " << dwarg.link->getValueTypeString() << std::endl;
    numWidgets_+=linkwidget_->sizeWidget();
    connect(linkwidget_,SIGNAL(WidgetDirty(bool)), this, SIGNAL ( WidgetDirty(bool) ) );
    connect(this, SIGNAL( WidgetUpdate() ), linkwidget_, SLOT( updateWidgetValue() ) );
    connect(this, SIGNAL( LinkUpdate() ), linkwidget_, SLOT(updateLinkValue() ) );
    connect(linkwidget_,SIGNAL(LinkOwnerDirty(bool)),this,SIGNAL(LinkOwnerDirty(bool)) );

}

void QDisplayLinkWidget::UpdateLink()
{
    emit LinkUpdate();
}

void QDisplayLinkWidget::UpdateWidgets()
{
    emit WidgetUpdate();
}

QLinkSimpleEdit::QLinkSimpleEdit(QWidget* parent, const char* name, BaseLink* link)
    : LinkWidget(parent,name,link)
{
}

bool QLinkSimpleEdit::createWidgets()
{
    QString str  = QString( getBaseLink()->getValueString().c_str() );
    QLayout* layout = new QHBoxLayout(this);
    if( str.length() > TEXTSIZE_THRESHOLD )
    {
        innerWidget_.type = TEXTEDIT;
        innerWidget_.widget.textEdit = new QTextEdit(this); innerWidget_.widget.textEdit->setText(str);
        connect(innerWidget_.widget.textEdit , SIGNAL( textChanged() ), this, SLOT ( update() ) );
        layout->add(innerWidget_.widget.textEdit);
    }
    else
    {
        innerWidget_.type = LINEEDIT;
        innerWidget_.widget.lineEdit  = new QLineEdit(this);
        innerWidget_.widget.lineEdit->setText(str);
        connect( innerWidget_.widget.lineEdit, SIGNAL(textChanged(const QString&)), this, SLOT( update() ) );
        layout->add(innerWidget_.widget.lineEdit);
    }

    return true;
}

void QLinkSimpleEdit::readFromLink()
{
    QString str = QString( getBaseLink()->getValueString().c_str() );
    if(innerWidget_.type == TEXTEDIT)
    {
        innerWidget_.widget.textEdit->setText(str);
    }
    else if(innerWidget_.type == LINEEDIT)
    {
        innerWidget_.widget.lineEdit->setText(str);
    }
}

void QLinkSimpleEdit::writeToLink()
{
    if(getBaseLink())
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
        getBaseLink()->read(value);
    }
}

} // namespace qt

} // namespace gui

} // namespace sofa
