/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "QSofaStatWidget.h"
#include "GraphListenerQListView.h"
#include <sofa/core/CollisionModel.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/simulation/common/Node.h>



#ifdef SOFA_QT4
#include <QApplication>
#include <QPixmap>
#include <QVBoxLayout>
#else
#include "qapplication.h"
#include "qpixmap.h"
#include "qlabel.h"
#include "qlayout.h"
#endif

using namespace sofa::simulation;
using namespace sofa::core::objectmodel;
namespace sofa
{
namespace gui
{
namespace qt
{

QSofaStatWidget::QSofaStatWidget(QWidget* parent):QWidget(parent)
{
    QVBoxLayout* layout = new QVBoxLayout(this);
    statsLabel = new QLabel(this);
    statsLabel->setText(QString("Collision Elements present :"));
//        statsLabel->setObjectName(QString("statsLabel"));

#ifdef SOFA_QT4
    statsLabel->setWordWrap(false);
#endif
    layout->addWidget(statsLabel);
    statsCounter = new Q3ListView(this);
    statsCounter->addColumn(QString("Name"));
    statsCounter->header()->setClickEnabled(true, statsCounter->header()->count() - 1);
    statsCounter->header()->setResizeEnabled(true, statsCounter->header()->count() - 1);
    statsCounter->addColumn(QString("Type"));
    statsCounter->header()->setClickEnabled(true, statsCounter->header()->count() - 1);
    statsCounter->header()->setResizeEnabled(true, statsCounter->header()->count() - 1);
    statsCounter->addColumn(QString("Size"));
    statsCounter->header()->setClickEnabled(true, statsCounter->header()->count() - 1);
    statsCounter->header()->setResizeEnabled(true, statsCounter->header()->count() - 1);
    statsCounter->addColumn(QString("Groups"));
    statsCounter->header()->setClickEnabled(true, statsCounter->header()->count() - 1);
    statsCounter->header()->setResizeEnabled(true, statsCounter->header()->count() - 1);
    statsCounter->setResizeMode(Q3ListView::LastColumn);
    statsCounter->header()->setLabel(0, QString("Name"));
    statsCounter->header()->setLabel(1, QString("Type"));
    statsCounter->header()->setLabel(2, QString("Size"));
    statsCounter->header()->setLabel(3, QString("Groups"));
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
    statsCounter->clear();

    addCollisionModelsStat(list_collisionModels);

    addSummary();


}

void QSofaStatWidget::addCollisionModelsStat(const sofa::helper::vector< sofa::core::CollisionModel* >& v)
{
    std::map< BaseContext*, Q3ListViewItem* > listStats;
    for (unsigned int i=0; i<v.size(); i++)
    {
        if ( !v[i]->isActive()) continue;
        std::map< BaseContext*, Q3ListViewItem* >::iterator it = listStats.find(v[i]->getContext());
        Q3ListViewItem *item;
        if (it != listStats.end())
        {
            item = new Q3ListViewItem((*it).second);
        }
        else
        {
            Q3ListViewItem *node = new Q3ListViewItem(statsCounter);
            node->setText(0,QString(v[i]->getContext()->getName().c_str()));
            QPixmap* pix = getPixmap(v[i]->getContext());
            if (pix) node->setPixmap(0,*pix);
            listStats.insert(std::make_pair(v[i]->getContext(), node));
            item = new Q3ListViewItem(node);
            node->setOpen(true);
        }
        assert(item);
        item->setText(0,v[i]->getName().c_str());
        item->setText(1,QString(v[i]->getClassName().c_str()));
        item->setText(0,v[i]->getName().c_str());
        item->setText(2,QString::number(v[i]->getSize()));
        {
        const helper::set<int>& groups = v[i]->getGroups();
        QString groupString;
        helper::set<int>::const_iterator it = groups.begin(), itend = groups.end();
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
        mapElement[items_stats[i].first->getClassName()] += atoi(items_stats[i].second->text(2));


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
