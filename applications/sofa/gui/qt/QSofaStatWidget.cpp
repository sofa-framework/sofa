/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include "QSofaStatWidget.h"
#include "GraphListenerQListView.h"
#include <sofa/core/CollisionModel.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/simulation/Node.h>


#include <QApplication>
#include <QPixmap>
#include <QVBoxLayout>

using namespace sofa::simulation;
using namespace sofa::core::objectmodel;
namespace sofa
{
namespace gui
{
namespace qt
{

QSofaStatWidget::QSofaStatWidget(QWidget* parent)
    : QWidget(parent)
{
    QVBoxLayout* layout = new QVBoxLayout(this);
    statsLabel = new QLabel(this);
    statsLabel->setText(QString("Collision Elements present :"));
    statsLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
//        statsLabel->setObjectName(QString("statsLabel"));
    statsLabel->setWordWrap(false);

    layout->addWidget(statsLabel);

    statsCounter = new QTreeWidget(this);
    statsCounter->setHeaderLabels(QStringList() << "Name" << "Type" << "Size" << "Groups");

//    QTreeWidgetItem* item = new QTreeWidgetItem();

//    statsCounter->addColumn(QString("Name"));
//    statsCounter->header()->setClickEnabled(true, statsCounter->header()->count() - 1);
//    statsCounter->header()->setResizeEnabled(true, statsCounter->header()->count() - 1);
//    statsCounter->addColumn(QString("Type"));
//    statsCounter->header()->setClickEnabled(true, statsCounter->header()->count() - 1);
//    statsCounter->header()->setResizeEnabled(true, statsCounter->header()->count() - 1);
//    statsCounter->addColumn(QString("Size"));
//    statsCounter->header()->setClickEnabled(true, statsCounter->header()->count() - 1);
//    statsCounter->header()->setResizeEnabled(true, statsCounter->header()->count() - 1);
//    statsCounter->addColumn(QString("Groups"));
//    statsCounter->header()->setClickEnabled(true, statsCounter->header()->count() - 1);
//    statsCounter->header()->setResizeEnabled(true, statsCounter->header()->count() - 1);
//    statsCounter->setResizeMode(QTreeWidget::LastColumn);
//    statsCounter->header()->setLabel(0, QString("Name"));
//    statsCounter->header()->setLabel(1, QString("Type"));
//    statsCounter->header()->setLabel(2, QString("Size"));
//    statsCounter->header()->setLabel(3, QString("Groups"));
    layout->addWidget(statsCounter);



}

void QSofaStatWidget::CreateStats(Node* root)
{
    sofa::helper::vector< sofa::core::CollisionModel* > list_collisionModels;
    root->get< sofa::core::CollisionModel >( &list_collisionModels, BaseContext::SearchDown);

    if (items_stats.size() != 0)
    {
        delete items_stats[0].second;
        items_stats.clear();
    }

    for(int i=0 ; i<statsCounter->topLevelItemCount() ; i++)
        delete statsCounter->takeTopLevelItem(i);


    addCollisionModelsStat(list_collisionModels);

    addSummary();


}

void QSofaStatWidget::addCollisionModelsStat(const sofa::helper::vector< sofa::core::CollisionModel* >& v)
{
    std::map< BaseContext*, QTreeWidgetItem* > listStats;
    for (unsigned int i=0; i<v.size(); i++)
    {
        if ( !v[i]->isActive()) continue;
        std::map< BaseContext*, QTreeWidgetItem* >::iterator it = listStats.find(v[i]->getContext());
        QTreeWidgetItem *item;
        if (it != listStats.end())
        {
            item = new QTreeWidgetItem((*it).second);
        }
        else
        {
            QTreeWidgetItem *node = new QTreeWidgetItem(statsCounter);
            node->setText(0,QString(v[i]->getContext()->getName().c_str()));
            QPixmap* pix = getPixmap(v[i]->getContext(), false,false,false);
            if (pix) node->setIcon(0, QIcon(*pix));
            listStats.insert(std::make_pair(v[i]->getContext(), node));
            item = new QTreeWidgetItem(node);
            node->setExpanded(true);
        }
        assert(item);
        item->setText(0,v[i]->getName().c_str());
        item->setText(1,QString(v[i]->getClassName().c_str()));
        item->setText(0,v[i]->getName().c_str());
        item->setText(2,QString::number(v[i]->getSize()));
        {
        const std::set<int>& groups = v[i]->getGroups();
        QString groupString;
        std::set<int>::const_iterator it = groups.begin(), itend = groups.end();
        for( ; it != itend ; ++it ) groupString += QString::number(*it) + ", ";
        item->setText(3,groupString);
        }
        items_stats.push_back(std::make_pair(v[i], item));
    }
}

void QSofaStatWidget::addSummary()
{
    std::set< std::string > nameElement;
    std::map< std::string, int > mapElement;
    for (unsigned int i=0; i < items_stats.size(); i++)
        nameElement.insert(items_stats[i].first->getClassName());


    for (unsigned int i=0; i < items_stats.size(); i++)
        mapElement[items_stats[i].first->getClassName()] += (items_stats[i].second->text(2).toInt());


    std::string textStats("<hr>Collision Elements present: <ul>");
    std::map< std::string, int >::const_iterator it;

    for (it=mapElement.begin(); it!=mapElement.end(); ++it)
    {
        if (it->second)
        {
            char buf[100];
            sprintf ( buf, "<li><b>%s:</b> %d</li>", it->first.c_str(), it->second );
            textStats += buf;
        }
    }
    textStats += "</ul><br>";
    statsLabel->setText( textStats.c_str());
    statsLabel->update();
}

}//qt
}//gui
}//sofa
