
#include "QDataDescriptionWidget.h"

#include <sofa/core/ObjectFactory.h>

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
QDataDescriptionWidget::QDataDescriptionWidget(QWidget* parent, core::objectmodel::Base* object):QWidget(parent)
{

    QVBoxLayout* tabLayout = new QVBoxLayout( this, 0, 1, QString("tabInfoLayout"));
    //Instance
    {
        Q3GroupBox *box = new Q3GroupBox(this, QString("Instance"));
        box->setColumns(2);
        box->setTitle(QString("Instance"));
        new QLabel(QString("Name"), box);
        new QLabel(QString(object->getName().c_str()), box);
        new QLabel(QString("Class"), box);
        new QLabel(QString(object->getClassName().c_str()), box);
        std::string namespacename = object->decodeNamespaceName(typeid(*object));
        if (!namespacename.empty())
        {
            new QLabel(QString("Namespace"), box);
            (new QLabel(QString(namespacename.c_str()), box))->setMinimumWidth(20);
        }
        if (!object->getTemplateName().empty())
        {
            new QLabel(QString("Template"), box);
            (new QLabel(QString(object->getTemplateName().c_str()), box))->setMinimumWidth(20);
        }

        tabLayout->addWidget( box );
    }


    //Class description
    core::ObjectFactory::ClassEntry* entry = core::ObjectFactory::getInstance()->getEntry(object->getClassName());
    if (entry != NULL && ! entry->creatorList.empty())
    {
        Q3GroupBox *box = new Q3GroupBox(this, QString("Class"));
        box->setColumns(2);
        box->setTitle(QString("Class"));
        if (!entry->description.empty() && entry->description != std::string("TODO"))
        {
            new QLabel(QString("Description"), box);
            (new QLabel(QString(entry->description.c_str()), box))->setMinimumWidth(20);
        }
        core::ObjectFactory::CreatorMap::iterator it = entry->creatorMap.find(object->getTemplateName());
        if (it != entry->creatorMap.end() && *it->second->getTarget())
        {
            new QLabel(QString("Provided by"), box);
            (new QLabel(QString(it->second->getTarget()), box))->setMinimumWidth(20);
        }

        if (!entry->authors.empty() && entry->authors != std::string("TODO"))
        {
            new QLabel(QString("Authors"), box);
            (new QLabel(QString(entry->authors.c_str()), box))->setMinimumWidth(20);
        }
        if (!entry->license.empty() && entry->license != std::string("TODO"))
        {
            new QLabel(QString("License"), box);
            (new QLabel(QString(entry->license.c_str()), box))->setMinimumWidth(20);
        }
        tabLayout->addWidget( box );
    }

    tabLayout->addStretch();
}



} // qt
} //gui
} //sofa

