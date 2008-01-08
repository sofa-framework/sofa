/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This program is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU General Public License as published by the Free   *
* Software Foundation; either version 2 of the License, or (at your option)    *
* any later version.                                                           *
*                                                                              *
* This program is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for     *
* more details.                                                                *
*                                                                              *
* You should have received a copy of the GNU General Public License along with *
* this program; if not, write to the Free Software Foundation, Inc., 51        *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                    *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/

#ifndef GRAPHLISTENERQLISTVIEW_H
#define GRAPHLISTENERQLISTVIEW_H




#ifdef QT_MODULE_QT3SUPPORT
#include <Q3ListViewItem>
#include <Q3ListView>
#include <QWidget>
#include <Q3PopupMenu>

#else
#include <qlistview.h>
#include <qwidget.h>
#include <qpopupmenu.h>


#include <qlabel.h>
#include <qcheckbox.h>
#include <qpushbutton.h>
#include <qimage.h>
#include <qspinbox.h>

#endif


#include "RealGUI.h"

#include "iconnode.xpm"
#include <sofa/simulation/tree/Simulation.h>
#include <sofa/simulation/tree/MutationListener.h>

#include <sofa/simulation/tree/Colors.h>
#include "WFloatLineEdit.h"


namespace sofa
{

namespace gui
{

namespace qt
{
using sofa::simulation::tree::GNode;
using sofa::simulation::tree::Simulation;
using sofa::simulation::tree::MutationListener;

#ifdef QT_MODULE_QT3SUPPORT
typedef Q3ListView QListView;
typedef Q3PopupMenu QPopupMenu;
#else
typedef QListViewItem Q3ListViewItem;
typedef QListView Q3ListView;
typedef QPopupMenu Q3PopupMenu;
#endif

static const int iconWidth=8;
static const int iconHeight=10;
static const int iconMargin=6;

static int hexval(char c)
{
    if (c>='0' && c<='9') return c-'0';
    else if (c>='a' && c<='f') return (c-'a')+10;
    else if (c>='A' && c<='F') return (c-'A')+10;
    else return 0;
}

/*
static QImage* classIcons = NULL;
static const int iconMargin=7;
static const int iconHeight=20;

// mapping; mmapping; constraint; iff; ffield; topology; mass; mstate; solver; bmodel; vmodel; cmodel; pipeline; context; object; gnode;

static const int iconPos[16]=
{
0, // node
1, // object
2, // context
6, // bmodel
4, // cmodel
8, // mstate
13, // constraint
12, // iff
11, // ff
7, // solver
3, // pipeline
14, // mmap
15, // map
9, // mass
10, // topo
5, // vmodel
};

static QImage icons[16];
     */
static QPixmap* getPixmap(core::objectmodel::Base* obj)
{
    using namespace sofa::simulation::tree::Colors;
    /*
      if (classIcons == NULL)
      {
      classIcons = new QImage("classicons.png");
      std::cout << "classicons.png: "<<classIcons->width()<<"x"<<classIcons->height()<<std::endl;
      if (classIcons->height() < 16) return NULL;
      // Find each icon
      QRgb bg = classIcons->pixel(0,0);
      }
      if (classIcons->height() < 16) return NULL;
    */
    unsigned int flags=0;

    if (dynamic_cast<core::objectmodel::BaseNode*>(obj))
    {
        static QPixmap pixNode((const char**)iconnode_xpm);
        return &pixNode;
        //flags |= 1 << NODE;
    }
    else if (dynamic_cast<core::objectmodel::BaseObject*>(obj))
    {
        if (dynamic_cast<core::objectmodel::ContextObject*>(obj))
            flags |= 1 << CONTEXT;
        if (dynamic_cast<core::BehaviorModel*>(obj))
            flags |= 1 << BMODEL;
        if (dynamic_cast<core::CollisionModel*>(obj))
            flags |= 1 << CMODEL;
        if (dynamic_cast<core::componentmodel::behavior::BaseMechanicalState*>(obj))
            flags |= 1 << MMODEL;
        if (dynamic_cast<core::componentmodel::behavior::BaseConstraint*>(obj))
            flags |= 1 << CONSTRAINT;
        if (dynamic_cast<core::componentmodel::behavior::InteractionForceField*>(obj) &&
            dynamic_cast<core::componentmodel::behavior::InteractionForceField*>(obj)->getMechModel1()!=dynamic_cast<core::componentmodel::behavior::InteractionForceField*>(obj)->getMechModel2())
            flags |= 1 << IFFIELD;
        else if (dynamic_cast<core::componentmodel::behavior::BaseForceField*>(obj))
            flags |= 1 << FFIELD;
        if (dynamic_cast<core::componentmodel::behavior::MasterSolver*>(obj)
            || dynamic_cast<core::componentmodel::behavior::OdeSolver*>(obj))
            flags |= 1 << SOLVER;
        if (dynamic_cast<core::componentmodel::collision::Pipeline*>(obj)
            || dynamic_cast<core::componentmodel::collision::Intersection*>(obj)
            || dynamic_cast<core::componentmodel::collision::Detection*>(obj)
            || dynamic_cast<core::componentmodel::collision::ContactManager*>(obj)
            || dynamic_cast<core::componentmodel::collision::CollisionGroupManager*>(obj))
            flags |= 1 << COLLISION;
        if (dynamic_cast<core::componentmodel::behavior::BaseMechanicalMapping*>(obj))
            flags |= 1 << MMAPPING;
        else if (dynamic_cast<core::BaseMapping*>(obj))
            flags |= 1 << MAPPING;
        if (dynamic_cast<core::componentmodel::behavior::BaseMass*>(obj))
            flags |= 1 << MASS;
        if (dynamic_cast<core::componentmodel::topology::Topology *>(obj))
            flags |= 1 << TOPOLOGY;
        if (dynamic_cast<core::VisualModel*>(obj) && !flags)
            flags |= 1 << VMODEL;
        if (!flags)
            flags |= 1 << OBJECT;
    }
    else return NULL;

    static std::map<unsigned int, QPixmap*> pixmaps;
    if (!pixmaps.count(flags))
    {
        int nc = 0;
        for (int i=0; i<16; i++)
            if (flags & (1<<i))
                ++nc;
        int nx = 2+iconWidth*nc+iconMargin;
        QImage * img = new QImage(nx,iconHeight,32);
        img->setAlphaBuffer(true);
        img->fill(qRgba(0,0,0,0));
        // Workaround for qt 3.x where fill() does not set the alpha channel
        for (int y=0 ; y < iconHeight ; y++)
            for (int x=0 ; x < nx ; x++)
                img->setPixel(x,y,qRgba(0,0,0,0));

        for (int y=0 ; y < iconHeight ; y++)
            img->setPixel(0,y,qRgba(0,0,0,255));
        nc = 0;
        for (int i=0; i<16; i++)
            if (flags & (1<<i))
            {
                int x0 = 1+iconWidth*nc;
                int x1 = x0+iconWidth-1;
                //QColor c(COLOR[i]);
                const char* color = COLOR[i];
                //c.setAlpha(255);
                int r = (hexval(color[1])*16+hexval(color[2]));
                int g = (hexval(color[3])*16+hexval(color[4]));
                int b = (hexval(color[5])*16+hexval(color[6]));
                int a = 255;
                for (int x=x0; x <=x1 ; x++)
                {
                    img->setPixel(x,0,qRgba(0,0,0,255));
                    img->setPixel(x,iconHeight-1,qRgba(0,0,0,255));
                    for (int y=1 ; y < iconHeight-1 ; y++)
                        //img->setPixel(x,y,c.value());
                        img->setPixel(x,y,qRgba(r,g,b,a));
                }
                //bitBlt(img,nimg*(iconWidth+2),0,classIcons,iconMargin,iconPos[i],iconWidth,iconHeight);
                ++nc;
            }
        for (int y=0 ; y < iconHeight ; y++)
            img->setPixel(2+iconWidth*nc-1,y,qRgba(0,0,0,255));
        pixmaps[flags] = new QPixmap(*img);
        delete img;
    }
    return pixmaps[flags];
}



class GraphListenerQListView : public MutationListener
{
public:
    Q3ListView* widget;
    bool frozen;
    std::map<core::objectmodel::Base*, Q3ListViewItem* > items;
    GraphListenerQListView(Q3ListView* w)
        : widget(w), frozen(false)
    {
    }


    /*****************************************************************************************************************/
    Q3ListViewItem* createItem(Q3ListViewItem* parent);
    void addChild(GNode* parent, GNode* child);
    void removeChild(GNode* parent, GNode* child);
    void moveChild(GNode* previous, GNode* parent, GNode* child);
    void addObject(GNode* parent, core::objectmodel::BaseObject* object);
    void removeObject(GNode* /*parent*/, core::objectmodel::BaseObject* object);
    void moveObject(GNode* previous, GNode* parent, core::objectmodel::BaseObject* object);
    void freeze(GNode* groot);
    void unfreeze(GNode* groot);
};

}
}
}
#endif
