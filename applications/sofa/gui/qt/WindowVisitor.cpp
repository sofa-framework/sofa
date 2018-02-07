/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "WindowVisitor.h"

#include <QHeaderView>
#include <QMenu>
#include <QMessageBox>
#include <QPainter>
#include <QGridLayout>

namespace sofa
{

namespace gui
{

namespace qt
{
QPixmap *WindowVisitor::icons[WindowVisitor::OTHER+1];

WindowVisitor::WindowVisitor()
{
    setupUi(this);

    this->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(graphView, SIGNAL(customContextMenuRequested(const QPoint&)),  this, SLOT( rightClick(const QPoint&)));

    QImage * img[OTHER+1];
    img[NODE] = new QImage(10,10,QImage::Format_ARGB32);

    img[NODE]->fill(qRgba(0,0,0,0));
    // Workaround for qt 3.x where fill() does not set the alpha channel
    for (int y=0 ; y < 10 ; y++)
        for (int x=0 ; x < 10 ; x++)
            img[NODE]->setPixel(x,y,qRgba(0,0,0,0));

    for (int y=0 ; y < 10 ; y++)
        img[NODE]->setPixel(0,y,qRgba(0,0,0,255));


    img[OTHER] = new QImage(img[NODE]->copy());

    //BORDER!!!!-------------------------------------
    for (int x=1; x <10 ; x++)
    {
        img[NODE]   ->setPixel(x,0,qRgba(0,0,0,255));
        img[NODE]   ->setPixel(x,9,qRgba(0,0,0,255));
    }
    for (int y=0 ; y < 10 ; y++)
    {
        img[NODE]   ->setPixel(0,y,qRgba(0,0,0,255));
        img[NODE]   ->setPixel(9,y,qRgba(0,0,0,255));
    }
    //-----------------------------------------------

    img[COMMENT] = new QImage(img[NODE]->copy());
    img[COMPONENT] = new QImage(img[NODE]->copy());

    for (int y=0 ; y < 9 ; y++)
        for (int x=0 ; x < 9 ; x++)
        {
            img[NODE]   ->setPixel(x,y,qRgba(125,125,125,255));
            img[COMMENT]->setPixel(x,y,qRgba(255,255,0,255));
            img[COMPONENT]->setPixel(x,y,qRgba(255,0,255,255));
        }

    for (int y=0 ; y < 5 ; y++)
        for (int x=0 ; x < 2*y+1 ; x++)
        {
            img[OTHER]->setPixel(x,y,qRgba(0,0,255,255));
        }
    for (int y=5 ; y < 10 ; y++)
        for (int x=0 ; x < 2*(10-y) ; x++)
        {
            img[OTHER]   ->setPixel(x,y,qRgba(0,0,255,255));
        }
    icons[NODE]    = new QPixmap(QPixmap::fromImage(*img[NODE]));
    icons[COMMENT]    = new QPixmap(QPixmap::fromImage(*img[COMMENT]));
    icons[COMPONENT]    = new QPixmap(QPixmap::fromImage(*img[COMPONENT]));
    icons[OTHER]    = new QPixmap(QPixmap::fromImage(*img[OTHER]));

    statsWidget=new QWidget(splitterStats);

    QGridLayout *statsLayout=new QGridLayout(statsWidget);

    typeOfCharts = new QComboBox(statsWidget);
    QStringList list;
    list << "Latest execution"
            << "Most time-consuming step"
            << "Total execution time";


    splitterStats->addWidget(statsWidget);
    typeOfCharts->insertItems(0,list);

    chartsComponent=new ChartsWidget("Component (name)", statsWidget);
    chartsVisitor  =new ChartsWidget("Visitor", statsWidget);

    statsLayout->addWidget(typeOfCharts,0,0);
    statsLayout->addWidget(chartsComponent,1,0);
    statsLayout->addWidget(chartsVisitor,2,0);
    connect(typeOfCharts, SIGNAL(activated(int)), this, SLOT(setCurrentCharts(int)));


    //Add Control Panel
    controlPanel = new QVisitorControlPanel(splitterWindow);
    connect( controlPanel, SIGNAL(focusOn(QString)), this, SLOT(focusOn(QString)));
    connect( controlPanel, SIGNAL(clearGraph()), this, SLOT(clearGraph()));
    controlPanel->setMaximumHeight(110);

}


void WindowVisitor::setCharts(std::vector< dataTime >&latestC, std::vector< dataTime >&maxTC, std::vector< dataTime >&totalC,
        std::vector< dataTime >&latestV, std::vector< dataTime >&maxTV, std::vector< dataTime >&totalV)
{
    componentsTime=latestC;
    componentsTimeMax=maxTC;
    componentsTimeTotal=totalC;
    visitorsTime=latestV;
    visitorsTimeMax=maxTV;
    visitorsTimeTotal=totalV;
    setCurrentCharts(typeOfCharts->currentIndex());
}



void WindowVisitor::setCurrentCharts(int type)
{
    switch(type)
    {
    case 0:
        chartsComponent->setChart(componentsTime, componentsTime.size());
        chartsVisitor->setChart(visitorsTime, visitorsTime.size());
        break;
    case 1:
        chartsComponent->setChart(componentsTimeMax, componentsTimeMax.size());
        chartsVisitor->setChart(visitorsTimeMax, visitorsTimeMax.size());
        break;
    case 2:
        chartsComponent->setChart(componentsTimeTotal, componentsTimeTotal.size());
        chartsVisitor->setChart(visitorsTimeTotal, visitorsTimeTotal.size());
        break;
    }
}

void WindowVisitor::rightClick( const QPoint& point)
{
    QTreeWidgetItem *item = graphView->itemAt( point );

    if (!item) return;

    QMenu *contextMenu = new QMenu ( this  );
    contextMenu->setObjectName( "ContextMenu");

    if(item->childCount())
    {
        contextMenu->addAction("Collapse", this,SLOT(collapseNode()));
        contextMenu->addAction("Expand", this,SLOT(expandNode()));

        contextMenu->exec ( this->mapToGlobal(point));
    }
}

void WindowVisitor::focusOn(QString text)
{
    if(graphView->topLevelItemCount() < 1) return;

    bool found = false;
    for(int i=0 ; i<graphView->topLevelItemCount() && !found; i++)
    {
        QTreeWidgetItem *item = graphView->topLevelItem(i);
        found = setFocusOn(item, text);
    }

    graphView->clearSelection();

}

bool WindowVisitor::setFocusOn(QTreeWidgetItem *item, QString text)
{
    for ( int c=0; c<graphView->columnCount(); ++c)
    {
        if (item->text(c).contains(text, Qt::CaseInsensitive))
        {
            if ( !graphView->currentItem() ||
                    graphView->visualItemRect(graphView->currentItem()).topLeft().y() < graphView->visualItemRect(item).topLeft().y() )
            {
//                graphView->ensureItemVisible(item);
                graphView->scrollToItem(item);
                graphView->clearSelection();
                graphView->setCurrentItem(item);
                item->setExpanded(true);
                return true;
            }
        }
    }

    bool found = false;
    for(int i=0 ; i<item->childCount() && !found; i++)
    {
        QTreeWidgetItem *child = item->child(i);
        found = setFocusOn(child, text);

    }

    return found;
}

void WindowVisitor::expandNode()
{
    expandNode(graphView->currentItem());
}

void WindowVisitor::expandNode(QTreeWidgetItem* item)
{
    if (!item) return;

    item->setExpanded( true );
    if ( item != NULL )
    {
        QTreeWidgetItem* child;

        for(int i=0 ; i<item->childCount() ; i++)
        {
            child = item->child(i);
            child->setExpanded( true );
            expandNode(item);
        }
    }
}

void WindowVisitor::collapseNode()
{
    collapseNode(graphView->currentItem());
    QTreeWidgetItem* item = graphView->currentItem();
    QTreeWidgetItem* child;

    for(int i=0 ; i<item->childCount() ; i++)
    {
        child = item->child(i);
        collapseNode(child);
    }

    graphView->currentItem()->setExpanded(true);
}
void WindowVisitor::collapseNode(QTreeWidgetItem* item)
{
    if (!item) return;

    item->setExpanded(false);

    QTreeWidgetItem* child;

    for(int i=0 ; i<item->childCount() ; i++)
    {
        child = item->child(i);
        collapseNode(child);
    }
}


}
}
}
