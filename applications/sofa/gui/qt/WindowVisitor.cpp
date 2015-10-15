/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "WindowVisitor.h"

#ifdef SOFA_QT4
#include <Q3Header>
#include <Q3PopupMenu>
#include <QMessageBox>
#include <QPainter>
#include <QGridLayout>
#else
#include <qheader.h>
#include <qpopupmenu.h>
#include <qmessagebox.h>
#include <qpainter.h>
#include <qlayout.h>
#include <qsplitter.h>
#endif


#ifndef SOFA_QT4
typedef QPopupMenu Q3PopupMenu;
typedef QListViewItem Q3ListViewItem;
typedef QTable QTableWidget;
#endif
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
#ifdef SOFA_QT4
    connect(graphView, SIGNAL(rightButtonClicked ( Q3ListViewItem *, const QPoint &, int )),  this, SLOT( rightClick(Q3ListViewItem *, const QPoint &, int )));
#else
    connect(graphView, SIGNAL(rightButtonClicked ( QListViewItem *, const QPoint &, int )),  this, SLOT( rightClick(QListViewItem *, const QPoint &, int )));
#endif


    QImage * img[OTHER+1];
    img[NODE] = new QImage(10,10,32);


    img[NODE]->setAlphaBuffer(true);
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
#ifdef SOFA_QT4
    icons[NODE]    = new QPixmap(QPixmap::fromImage(*img[NODE]));
    icons[COMMENT]    = new QPixmap(QPixmap::fromImage(*img[COMMENT]));
    icons[COMPONENT]    = new QPixmap(QPixmap::fromImage(*img[COMPONENT]));
    icons[OTHER]    = new QPixmap(QPixmap::fromImage(*img[OTHER]));
#else

    icons[NODE]    = new QPixmap(*img[NODE]   );
    icons[COMMENT] = new QPixmap(*img[COMMENT]);
    icons[COMPONENT] = new QPixmap(*img[COMPONENT]);
    icons[OTHER]   = new QPixmap(*img[OTHER]  );
#endif
    statsWidget=new QWidget(splitterStats);

    QGridLayout *statsLayout=new QGridLayout(statsWidget);

    typeOfCharts = new QComboBox(statsWidget);
    QStringList list;
    list << "Latest execution"
            << "Most time-consuming step"
            << "Total execution time";


#ifdef SOFA_QT4
    splitterStats->addWidget(statsWidget);
    typeOfCharts->insertItems(0,list);
#else
    typeOfCharts->insertStringList(list,0);
#endif

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
    setCurrentCharts(typeOfCharts->currentItem());
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

void WindowVisitor::rightClick(Q3ListViewItem *item, const QPoint &point, int index)
{
    if (!item) return;

    Q3PopupMenu *contextMenu = new Q3PopupMenu ( this, "ContextMenu" );
    if(item->childCount())
    {
        contextMenu->insertItem("Collapse", this, SLOT( collapseNode()));
        contextMenu->insertItem("Expand"  , this, SLOT( expandNode()));

        contextMenu->popup ( point, index );
    }
}

void WindowVisitor::focusOn(QString text)
{
    Q3ListViewItem *item = graphView->firstChild();

    while (item)
    {
        bool found=setFocusOn(item, text);
        if (found) return;
        item = item->nextSibling();
    }
    graphView->clearSelection();

}

bool WindowVisitor::setFocusOn(Q3ListViewItem *item, QString text)
{
    for ( int c=0; c<graphView->columns(); ++c)
    {
        if (item->text(c).contains(text, false))
        {
            if ( !graphView->selectedItem() ||
                    graphView->itemPos(graphView->selectedItem()) < graphView->itemPos(item) )
            {
                graphView->ensureItemVisible(item);
                graphView->clearSelection();
                graphView->setSelected(item,true);
                item->setOpen(true);
                return true;
            }
        }
    }

    item = item->firstChild();
    while (item)
    {
        bool found=setFocusOn(item, text);
        if (found) return true;
        item = item->nextSibling();
    }
    return false;
}

void WindowVisitor::expandNode()
{
    expandNode(graphView->currentItem());
}

void WindowVisitor::expandNode(Q3ListViewItem* item)
{
    if (!item) return;

    item->setOpen ( true );
    if ( item != NULL )
    {
        Q3ListViewItem* child;
        child = item->firstChild();
        while ( child != NULL )
        {
            item = child;
            child->setOpen ( true );
            expandNode(item);
            child = child->nextSibling();
        }
    }
}

void WindowVisitor::collapseNode()
{
    collapseNode(graphView->currentItem());
    Q3ListViewItem* item = graphView->currentItem();
    item = item->firstChild();
    while (item)
    {
        collapseNode(item);
        item = item->nextSibling();
    }
    graphView->currentItem()->setOpen(true);
}
void WindowVisitor::collapseNode(Q3ListViewItem* item)
{
    if (!item) return;

    item->setOpen(false);
    item = item->firstChild();
    while ( item )
    {
        collapseNode(item);
        item = item->nextSibling();
    }
}


}
}
}
