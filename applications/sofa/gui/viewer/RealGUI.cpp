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

#ifdef SOFA_GUI_QTOGREVIEWER
#include "QtOgreViewer/QtOgreViewer.h"
#endif

#ifdef SOFA_GUI_QTVIEWER
#include "QtViewer/QtViewer.h"
#endif

#ifdef SOFA_GUI_QGLVIEWER
#include "QtGLViewer/QtGLViewer.h"
#endif
#include "RealGUI.h"

#include <sofa/simulation/tree/Simulation.h>
#include <sofa/simulation/tree/MutationListener.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/simulation/automatescheduler/ThreadSimulation.h>
#include <sofa/simulation/automatescheduler/ExecBus.h>
#include <sofa/simulation/automatescheduler/Node.h>

namespace sofa
{
namespace simulation
{
namespace automatescheduler
{
extern simulation::tree::GNode* groot;
}
}
}


#ifdef QT_MODULE_QT3SUPPORT
#include <Q3FileDialog>
#include <QStatusBar>
#include <Q3DockWindow>
#include <Q3ListViewItem>
#include <Q3ListView>
#include <QStackedWidget>
#include <QRadioButton>
#include <QSplitter>
#include <Q3TextEdit>
#include <QCursor>
#include <QWidget>
#include <QLayout>
#include <QTimer>
#include <QAction>
#include <QMessageBox>
#define WIDTH_OFFSET 2
#define HEIGHT_OFFSET 2
#define INFINITY 9.0e10
#else
#include <qlistview.h>
#include <qcheckbox.h>
#include <qpushbutton.h>
#include <qlineedit.h>
#include <qlabel.h>
#include <qstatusbar.h>
#include <qfiledialog.h>
#include <qheader.h>
#include <qimage.h>
#include <qdockwindow.h>
#include <qspinbox.h>
#include <qradiobutton.h>
#include <qsplitter.h>
#include <qtextedit.h>
#include <qcursor.h>
#include <qwidget.h>
#include <qwidgetstack.h>
#include <qlayout.h>
#include <qtimer.h>
#include <qapplication.h>
#include <qaction.h>
#include <qmessagebox.h>

#define WIDTH_OFFSET 0
#define HEIGHT_OFFSET 0
#endif

#include <GenGraphForm.h>
#include "GUIField.h"

#include <sofa/simulation/tree/Colors.h>
#include "WFloatLineEdit.h"
#include <sofa/core/objectmodel/BaseObject.h>
#include <limits.h>


namespace sofa
{

namespace gui
{

namespace guiviewer
{

#ifdef QT_MODULE_QT3SUPPORT
typedef Q3ListView QListView;
typedef Q3FileDialog QFileDialog;
typedef Q3DockWindow QDockWindow;
typedef QStackedWidget QWidgetStack;
typedef Q3TextEdit QTextEdit;
#else
typedef QListViewItem Q3ListViewItem;
typedef QListView Q3ListView;
typedef QFileDialog Q3FileDialog;
//typedef QWidgetStack QStackedWidget;
typedef QTextEdit Q3TextEdit;
#endif


#include "iconnode.xpm"

using sofa::core::objectmodel::BaseObject;

using namespace sofa::helper::system::thread;
using namespace sofa::simulation::tree;
using namespace sofa::simulation::automatescheduler;


///////////////////////////////////////////////////////////
//////////////////// SofaGUI Interface ////////////////////
///////////////////////////////////////////////////////////

#ifdef SOFA_GUI_QGLVIEWER
SOFA_DECL_CLASS(QGLViewerGUI)
int QGLViewerGUIClass = SofaGUI::RegisterGUI("qglviewer", &RealGUI::CreateGUI, &RealGUI::InitGUI, 2);
#endif
#ifdef SOFA_GUI_QTVIEWER
SOFA_DECL_CLASS(QTGUI)
int QtGUIClass = SofaGUI::RegisterGUI("qt", &RealGUI::CreateGUI, &RealGUI::InitGUI, 1);
#endif
#ifdef SOFA_GUI_QTOGREVIEWER
SOFA_DECL_CLASS(OgreGUI)
int QtOGREGUIClass = SofaGUI::RegisterGUI("ogre", &RealGUI::CreateGUI, &RealGUI::InitGUI, 0);
#endif

int RealGUI::InitGUI(const char* name, const std::vector<std::string>& /* options */)
{
#ifdef SOFA_GUI_QGLVIEWER
    if (!name[0] || !strcmp(name,"qglviewer"))
    {
        return sofa::gui::guiqglviewer::QtGLViewer::EnableViewer();
    }
    else
#endif
#ifdef SOFA_GUI_QTVIEWER
        if (!name[0] || !strcmp(name,"qt"))
        {
            return sofa::gui::qt::QtViewer::EnableViewer();
        }
        else
#endif
#ifdef SOFA_GUI_QTOGREVIEWER
            if (!name[0] || !strcmp(name,"ogre"))
            {
                return sofa::gui::qtogreviewer::QtOgreViewer::EnableViewer();
            }
            else
#endif
            {
                std::cerr << "ERROR(QtGUI): unknown or disabled gui name "<<name<<std::endl;
                return 1;
            }
}

extern QApplication* application; // = NULL;
extern RealGUI* gui;

using sofa::simulation::tree::GNode;

SofaGUI* RealGUI::CreateGUI(const char* name, const std::vector<std::string>& options, sofa::simulation::tree::GNode* groot, const char* filename)
{
    {
        int argc=1;
        char* argv[1];
        argv[0] = strdup(SofaGUI::GetProgramName());
        application = new QApplication(argc,argv);
        free(argv[0]);
    }
    // create interface
    gui = new RealGUI( name, options );
    if (groot)
        gui->setScene(groot, filename);

    //gui->viewer->SwitchToPresetView();

    application->setMainWidget( gui );

    // Threads Management
    if (ThreadSimulation::initialized())
    {
        ThreadSimulation::getInstance()->computeVModelsList(groot);
        groot->setMultiThreadSimulation(true);
        sofa::simulation::automatescheduler::groot = groot;

        Automate::setDrawCB(gui->viewer);

        gui->viewer->getQWidget()->update();
        ThreadSimulation::getInstance()->start();
    }
    // show the gui
    gui->show();
    return gui;
}

int RealGUI::mainLoop()
{
    return application->exec();
}

void RealGUI::redraw()
{
    viewer->getQWidget()->update();
}

int RealGUI::closeGUI()
{
    delete this;
    return 0;
}

sofa::simulation::tree::GNode* RealGUI::currentSimulation()
{
    return viewer->getScene();
}

///////////////////////////////////////////////////////////

static const int iconWidth=8;
static const int iconHeight=10;
static const int iconMargin=6;

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
static int hexval(char c)
{
    if (c>='0' && c<='9') return c-'0';
    else if (c>='a' && c<='f') return (c-'a')+10;
    else if (c>='A' && c<='F') return (c-'A')+10;
    else return 0;
}

static QPixmap* getPixmap(core::objectmodel::Base* obj)
{
    using namespace sofa::simulation::tree::Colors;
    /*
    if (classIcons == NULL)
    {
    classIcons = new QImage("../examples/classicons.png");
    std::cout << "../examples/classicons.png: "<<classIcons->width()<<"x"<<classIcons->height()<<std::endl;
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


using sofa::simulation::tree::Simulation;
using sofa::simulation::tree::MutationListener;

// TODO: Move this code somewhere else

class GraphListenerQListView : public MutationListener
{
public:
    Q3ListView* widget;
    std::map<core::objectmodel::Base*, Q3ListViewItem* > items;
    GraphListenerQListView(Q3ListView* w)
        : widget(w)
    {
    }

    Q3ListViewItem* createItem(Q3ListViewItem* parent)
    {
        Q3ListViewItem* last = parent->firstChild();
        if (last == NULL)
            return new Q3ListViewItem(parent);
        while (last->nextSibling()!=NULL)
            last = last->nextSibling();
        return new Q3ListViewItem(parent, last);
    }

    void addChild(GNode* parent, GNode* child)
    {
        if (items.count(child)) return;
        Q3ListViewItem* item;
        if (parent == NULL)
            item = new Q3ListViewItem(widget);
        else if (items.count(parent))
            item = createItem(items[parent]);
        else
        {
            std::cerr << "Graph -> QT ERROR: Unknown parent node "<<parent->getName()<<std::endl;
            return;
        }
        if (std::string(child->getName(),0,7) != "default")
            item->setText(0, child->getName().c_str());
        QPixmap* pix = getPixmap(child);
        if (pix)
            item->setPixmap(0, *pix);
        item->setOpen(true);
        items[child] = item;
        // Add all objects and grand-children
        MutationListener::addChild(parent, child);
    }

    void removeChild(GNode* parent, GNode* child)
    {
        MutationListener::removeChild(parent, child);
        if (items.count(child))
        {
            delete items[child];
            items.erase(child);
        }
    }

    void moveChild(GNode* previous, GNode* parent, GNode* child)
    {
        if (!items.count(child) || !items.count(previous))
        {
            addChild(parent, child);
        }
        else if (!items.count(parent))
        {
            removeChild(previous, child);
        }
        else
        {
            Q3ListViewItem* itemChild = items[child];
            Q3ListViewItem* itemPrevious = items[previous];
            Q3ListViewItem* itemParent = items[parent];
            itemPrevious->takeItem(itemChild);
            itemParent->insertItem(itemChild);
        }
    }

    void addObject(GNode* parent, core::objectmodel::BaseObject* object)
    {
        if (items.count(object)) return;
        Q3ListViewItem* item;
        if (items.count(parent))
            item = createItem(items[parent]);
        else
        {
            std::cerr << "Graph -> QT ERROR: Unknown parent node "<<parent->getName()<<std::endl;
            return;
        }
        std::string name = sofa::helper::gettypename(typeid(*object));
        std::string::size_type pos = name.find('<');
        if (pos != std::string::npos)
            name.erase(pos);
        if (std::string(object->getName(),0,7) != "default")
        {
            name += "  ";
            name += object->getName();
        }
        item->setText(0, name.c_str());
        QPixmap* pix = getPixmap(object);
        if (pix)
            item->setPixmap(0, *pix);
        items[object] = item;
    }

    void removeObject(GNode* /*parent*/, core::objectmodel::BaseObject* object)
    {
        if (items.count(object))
        {
            delete items[object];
            items.erase(object);
        }
    }

    void moveObject(GNode* previous, GNode* parent, core::objectmodel::BaseObject* object)
    {
        if (!items.count(object) || !items.count(previous))
        {
            addObject(parent, object);
        }
        else if (!items.count(parent))
        {
            removeObject(previous, object);
        }
        else
        {
            Q3ListViewItem* itemObject = items[object];
            Q3ListViewItem* itemPrevious = items[previous];
            Q3ListViewItem* itemParent = items[parent];
            itemPrevious->takeItem(itemObject);
            itemParent->insertItem(itemObject);
        }
    }
};

RealGUI::RealGUI( const char* viewername, const std::vector<std::string>& /*options*/)
    : viewerName(viewername), graphListener(NULL), id_timer(-1)
{
    left_stack = new QWidgetStack(splitter2);
#ifndef QT_MODULE_QT3SUPPORT
    GUILayout->addWidget(left_stack);
#endif

    connect( startButton, SIGNAL( toggled(bool) ), this , SLOT( playpauseGUI(bool) ) );

    fpsLabel = new QLabel("9999.9 FPS", statusBar());
    fpsLabel->setAlignment(Qt::AlignRight);
    fpsLabel->setMinimumSize(fpsLabel->sizeHint());
    fpsLabel->clear();

    timeLabel = new QLabel("T: 999.9999 s", statusBar());
    timeLabel->setAlignment(Qt::AlignLeft);
    timeLabel->setMinimumSize(timeLabel->sizeHint());
    timeLabel->clear();


    statusBar()->addWidget(fpsLabel);
    statusBar()->addWidget(timeLabel);

    timerStep = new QTimer(this);
    connect( timerStep, SIGNAL(timeout()), this, SLOT(step()) );
    connect( ResetSceneButton, SIGNAL( clicked() ), this, SLOT( resetScene() ) );
    connect( dtEdit, SIGNAL( textChanged(const QString&) ), this, SLOT( setDt(const QString&) ) );
    //connect( ResetViewButton, SIGNAL( clicked() ), viewer->getQWidget(), SLOT( resetView() ) );
    //connect( SaveViewButton, SIGNAL( clicked() ), viewer->getQWidget(), SLOT( saveView() ) );
    connect( showVisual, SIGNAL( toggled(bool) ), this, SLOT( slot_showVisual(bool) ) );
    connect( showBehavior, SIGNAL( toggled(bool) ), this, SLOT( slot_showBehavior(bool) ) );
    connect( showCollision, SIGNAL( toggled(bool) ), this, SLOT( slot_showCollision(bool) ) );
    connect( showBoundingCollision, SIGNAL( toggled(bool) ), this, SLOT( slot_showBoundingCollision(bool) ) );
    connect( showMapping, SIGNAL( toggled(bool) ), this, SLOT( slot_showMapping(bool) ) );
    connect( showMechanicalMapping, SIGNAL( toggled(bool) ), this, SLOT( slot_showMechanicalMapping(bool) ) );
    connect( showForceField, SIGNAL( toggled(bool) ), this, SLOT( slot_showForceField(bool) ) );
    connect( showInteractionForceField, SIGNAL( toggled(bool) ), this, SLOT( slot_showInteractionForceField(bool) ) );
    connect( showWireFrame, SIGNAL( toggled(bool) ), this, SLOT( slot_showWireFrame(bool) ) );
    connect( showNormals, SIGNAL( toggled(bool) ), this, SLOT( slot_showNormals(bool) ) );
    connect( stepButton, SIGNAL( clicked() ), this, SLOT( step() ) );
    //connect( screenshotButton, SIGNAL( clicked() ), viewer->getQWidget(), SLOT( screenshot() ) );
    //connect( xmlSave_pushButton, SIGNAL( pressed() ), this, SLOT( saveXML() ) );
    connect( ExportGraphButton, SIGNAL( clicked() ), this, SLOT( exportGraph() ) );
    //connect( exportGraphAction, SIGNAL( activated() ), viewer, SLOT( exportGraph() ) );
    //connect( sizeW, SIGNAL( valueChanged(int) ), viewer->getQWidget(), SLOT( setSizeW(int) ) );
    //connect( sizeH, SIGNAL( valueChanged(int) ), viewer->getQWidget(), SLOT( setSizeH(int) ) );
    connect( dumpStateCheckBox, SIGNAL( toggled(bool) ), this, SLOT( dumpState(bool) ) );
    connect( exportGnuplotFilesCheckbox, SIGNAL(toggled(bool)), this, SLOT(setExportGnuplot(bool)) );
    connect( displayComputationTimeCheckBox, SIGNAL( toggled(bool) ), this, SLOT( displayComputationTime(bool) ) );

    addViewer();

#ifdef SOFA_PML
    pmlreader = NULL;
    lmlreader = NULL;
#endif
}

RealGUI::~RealGUI()
{
#ifdef SOFA_PML
    if(pmlreader)
    {
        delete pmlreader;
        pmlreader = NULL;
    }
    if(lmlreader)
    {
        delete lmlreader;
        lmlreader = NULL;
    }
#endif
}

void RealGUI::init()
{
    id_timer = -1;
    _animationOBJ = false;
    _animationOBJcounter = 0;
    m_dumpState = false;
    m_dumpStateStream = 0;
    m_displayComputationTime = false;
    m_exportGnuplot = false;
}

void RealGUI::addViewer()
{
    init();
    const char* name = viewerName;

    // set menu state
#ifdef SOFA_GUI_QTVIEWER
    viewerOpenGLAction->setEnabled(true);
#else
    viewerOpenGLAction->setEnabled(false);
    viewerOpenGLAction->setToolTip("enable SOFA_GUI_QTVIEWER in sofa-local.cfg to activate");
#endif
    viewerOpenGLAction->setOn(false);
#ifdef SOFA_GUI_QGLVIEWER
    viewerQGLViewerAction->setEnabled(true);
#else
    viewerQGLViewerAction->setEnabled(false);
    viewerQGLViewerAction->setToolTip("enable SOFA_GUI_QGLVIEWER in sofa-local.cfg to activate");
#endif
    viewerQGLViewerAction->setOn(false);
#ifdef SOFA_GUI_QTOGREVIEWER
    viewerOGREAction->setEnabled(true);
#else
    viewerOGREAction->setEnabled(false);
    viewerOGREAction->setToolTip("enable SOFA_GUI_QTOGREVIEWER in sofa-local.cfg to activate");
#endif
    viewerOGREAction->setOn(false);

#ifdef SOFA_GUI_QGLVIEWER
    if (!name[0] || !strcmp(name,"qglviewer"))
    {
        viewer = new sofa::gui::guiqglviewer::QtGLViewer( left_stack, "viewer" );
        viewerQGLViewerAction->setOn(true);
    }
    else
#endif
#ifdef SOFA_GUI_QTVIEWER
        if (!name[0] || !strcmp(name,"qt"))
        {
            viewer = new sofa::gui::qt::QtViewer( left_stack, "viewer" );
            viewerOpenGLAction->setOn(true);
        }
        else
#endif
#ifdef SOFA_GUI_QTOGREVIEWER
            if (!name[0] || !strcmp(name,"ogre"))
            {
                viewer = new sofa::gui::qtogreviewer::QtOgreViewer( left_stack , "viewer" );
                viewerOGREAction->setOn(true);
            }
            else
#endif
            {
                std::cerr << "ERROR(QtGUI): unknown or disabled viewer name "<<name<<std::endl;
                application->exit();
            }

#ifdef QT_MODULE_QT3SUPPORT
    left_stack->addWidget ( viewer->getQWidget() );
    left_stack->setCurrentWidget ( viewer->getQWidget() );
#else
    int id_viewer = left_stack->addWidget(viewer->getQWidget());
    left_stack->raiseWidget(id_viewer);
#endif
    viewer->getQWidget()->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)7, (QSizePolicy::SizeType)7, 100, 1,
            viewer->getQWidget()->sizePolicy().hasHeightForWidth() ) );
    viewer->getQWidget()->setMinimumSize( QSize( 0, 0 ) );
#ifndef QT_MODULE_QT3SUPPORT
    viewer->getQWidget()->setCursor( QCursor( 2 ) );
#endif
    viewer->getQWidget()->setMouseTracking( TRUE );

#ifdef QT_MODULE_QT3SUPPORT
    viewer->getQWidget()->setFocusPolicy( Qt::StrongFocus );
#else
    viewer->getQWidget()->setFocusPolicy( QWidget::StrongFocus );
#endif

    viewer->setup();

    //#ifdef SOFA_GUI_QTOGREVIEWER
    //	id_timer = startTimer(100);
    //#endif

    /*

    //connect( xmlSave_pushButton, SIGNAL( pressed() ), this, SLOT( saveXML() ) );
    if (filename != "")
    {

    switch(TYPE)
    {
    case NORMAL:
    groot = Simulation::load(filename);
    setScene(groot, filename);
    break;
    case PML:
    #ifdef SOFA_PML
    groot = Simulation::load(scene.c_str());
    if (groot){
    if (!pmlreader) pmlreader = new PMLReader;
    pmlreader->BuildStructure(filename, groot);
    setScene(groot, filename);
    }
    #endif
    break;
    default:break;
    }
    }
    */

    connect( ResetViewButton, SIGNAL( clicked() ), viewer->getQWidget(), SLOT( resetView() ) );
    connect( SaveViewButton, SIGNAL( clicked() ), viewer->getQWidget(), SLOT( saveView() ) );
    connect( screenshotButton, SIGNAL( clicked() ), viewer->getQWidget(), SLOT( screenshot() ) );
    connect( sizeW, SIGNAL( valueChanged(int) ), viewer->getQWidget(), SLOT( setSizeW(int) ) );
    connect( sizeH, SIGNAL( valueChanged(int) ), viewer->getQWidget(), SLOT( setSizeH(int) ) );
    connect(viewer->getQWidget(), SIGNAL(resizeW(int)), sizeW, SLOT(setValue(int)));
    connect(viewer->getQWidget(), SIGNAL(resizeH(int)), sizeH, SLOT(setValue(int)));

    //startTimer(50);

    QSplitter *splitter_ptr = dynamic_cast<QSplitter *> ( splitter2 );
    splitter_ptr->moveToLast(left_stack);

#ifdef QT_MODULE_QT3SUPPORT
    QList<int> list;
#else
    QValueList<int> list;
#endif
    list.push_back(75);
    list.push_back(500);
    splitter_ptr->setSizes(list);
    viewer->getQWidget()->setFocus();
    viewer->getQWidget()->show();

    setGUI();
}

void RealGUI::viewerOpenGL()
{
    viewerOpenGLAction->setOn(setViewer("qt"));
}

void RealGUI::viewerQGLViewer()
{
    viewerOpenGLAction->setOn(setViewer("qglviewer"));
}

void RealGUI::viewerOGRE()
{
    viewerOpenGLAction->setOn(setViewer("ogre"));
}

bool RealGUI::setViewer(const char* name)
{
    if (!strcmp(name,viewerName))
        return true; // nothing to do
    if (!strcmp(name,"qt"))
    {
#ifndef SOFA_GUI_QTVIEWER
        std::cerr << "OpenGL viewer not activated. Enable SOFA_GUI_QGLVIEWER in sofa-local.cfg to activate.\n";
        return false;
#endif
    }
    else if (!strcmp(name,"qglviewer"))
    {
#ifndef SOFA_GUI_QGLVIEWER
        std::cerr << "QGLViewer viewer not activated. Enable SOFA_GUI_QGLVIEWER in sofa-local.cfg to activate.\n";
        return false;
#endif
    }
    else if (!strcmp(name,"ogre"))
    {
#ifndef SOFA_GUI_QTOGREVIEWER
        std::cerr << "OGRE viewer not activated. Enable SOFA_GUI_QTOGREVIEWER in sofa-local.cfg to activate.\n";
        return false;
#endif
    }
    else
    {
        std::cerr << "Unknown viewer.\n";
        return false;
    }
    if ( QMessageBox::warning ( this, "Changing Viewer", "Changing viewer requires to reload the current scene.\nAre you sure you want to do that ?", QMessageBox::Yes | QMessageBox::Default, QMessageBox::No ) != QMessageBox::Yes )
        return false;

    std::string filename = viewer->getSceneFileName();
    GNode* groot = new GNode; // empty scene to do the transition
    setScene(groot,filename.c_str());
    left_stack->removeWidget(viewer->getQWidget());
    delete viewer;
    viewer = NULL;
    // Disable Viewer-specific classes
#ifdef SOFA_GUI_QTVIEWER
    if (!strcmp(viewerName,"qt"))
    {
        sofa::gui::qt::QtViewer::DisableViewer();
    }
    else
#endif
#ifdef SOFA_GUI_QGLVIEWER
        if (!strcmp(viewerName,"qglviewer"))
        {
            sofa::gui::guiqglviewer::QtGLViewer::DisableViewer();
        }
        else
#endif
#ifdef SOFA_GUI_QTOGREVIEWER
            if (!strcmp(viewerName,"ogre"))
            {
                sofa::gui::qtogreviewer::QtOgreViewer::DisableViewer();
            }
            else
#endif
            {}
    // Enable Viewer-specific classes
#ifdef SOFA_GUI_QTVIEWER
    if (!strcmp(name,"qt"))
    {
        sofa::gui::qt::QtViewer::EnableViewer();
    }
    else
#endif
#ifdef SOFA_GUI_QGLVIEWER
        if (!strcmp(name,"qglviewer"))
        {
            sofa::gui::guiqglviewer::QtGLViewer::EnableViewer();
        }
        else
#endif
#ifdef SOFA_GUI_QTOGREVIEWER
            if (!strcmp(name,"ogre"))
            {
                sofa::gui::qtogreviewer::QtOgreViewer::EnableViewer();
            }
#endif
    std::cerr<<"Test Viewer change!\n";
    viewerName = name;
    std::cerr<<"Add Viewer!\n";
    addViewer();
    std::cerr<<"fileOpen!\n";
    fileOpen(filename.c_str());
    return true;
}

void RealGUI::fileOpen(const char* filename)
{
    //left_stack->removeWidget(viewer->getQWidget());
    //graphListener->removeChild(NULL, groot);
    //delete viewer;
    //viewer = NULL;
    //addViewer(filename);
    GNode* groot = Simulation::load(filename);
    if (groot == NULL)
    {
        qFatal("Failed to load %s",filename);
        return;
    }
    setScene(groot, filename);
}

#ifdef SOFA_PML
void RealGUI::pmlOpen(const char* filename, bool resetView)
{
    std::string scene = "PML/default.scn";
    if (!sofa::helper::system::DataRepository.findFile(scene))
    {
        std::cerr << "File " << scene << " not found " << std::endl;
        return;
    }
    groot = Simulation::load(scene.c_str());
    if (groot)
    {
        if (!pmlreader) pmlreader = new PMLReader;
        pmlreader->BuildStructure(filename, groot);
        setScene(groot, filename);
    }
}

void RealGUI::lmlOpen(const char* filename)
{
    if (pmlreader)
    {
        if (!lmlreader) lmlreader = new LMLReader;
        lmlreader->BuildStructure(filename, pmlreader);
        groot = viewer->getScene();
        Simulation::init(groot);
    }
    else
        std::cerr<<"You must load the pml file before the lml file"<<endl;
}
#endif

void RealGUI::setScene(GNode* groot, const char* filename)
{
    if (viewer->getScene()!=NULL)
    {
        Simulation::unload(viewer->getScene());
        if (graphListener!=NULL)
            delete graphListener;
        graphView->clear();
    }

    setTitle(filename);
    //this->groot = groot;
    //sceneFileName = filename;

    viewer->setScene(groot, filename);
    eventNewTime();


    showVisual->setChecked(groot->getShowVisualModels());
    showBehavior->setChecked(groot->getShowBehaviorModels());
    showCollision->setChecked(groot->getShowCollisionModels());
    showBoundingCollision->setChecked(groot->getShowBoundingCollisionModels());
    showForceField->setChecked(groot->getShowForceFields());
    showInteractionForceField->setChecked(groot->getShowInteractionForceFields());
    showMapping->setChecked(groot->getShowMappings());
    showMechanicalMapping->setChecked(groot->getShowMechanicalMappings());
    showWireFrame->setChecked(groot->getShowWireFrame());
    showNormals->setChecked(groot->getShowNormals());
    startButton->setOn(groot->getContext()->getAnimate());
    dtEdit->setText(QString::number(groot->getDt()));

    graphView->setSorting(-1);
    //graphView->setTreeStepSize(10);
    graphView->header()->hide();
    //dumpGraph(groot, new Q3ListViewItem(graphView));
    graphListener = new GraphListenerQListView(graphView);
    graphListener->addChild(NULL, groot);
}

void RealGUI::fileOpen()
{
    std::string filename = viewer->getSceneFileName();

#ifdef SOFA_PML
    QString s = Q3FileDialog::getOpenFileName(filename.empty()?NULL:filename.c_str(), "Scenes (*.scn *.pml *.lml)",  this, "open file dialog",  "Choose a file to open" );

    if (s.length()>0)
    {
        if(s.endsWith(".pml"))
            pmlOpen(s);
        else if(s.endsWith(".lml"))
            lmlOpen(s);
        else
            fileOpen(s);
    }
#else
    QString s = Q3FileDialog::getOpenFileName(filename.empty()?NULL:filename.c_str(), "Scenes (*.scn)", this, "open file dialog", "Choose a file to open" );

    if (s.length()>0)
        fileOpen(s);
#endif
}

void RealGUI::fileReload()
{
    std::string filename = viewer->getSceneFileName();

    if (!filename.empty())
    {
        fileOpen(filename.c_str());
    }
    else std::cerr << "Reload failed: no file loaded." << filename << "\n";
}

void RealGUI::fileSaveAs()
{
    std::string filename = viewer->getSceneFileName();
    QString s = Q3FileDialog::getSaveFileName( filename.empty()?NULL:filename.c_str(), "Scenes (*.scn)", this, "save file dialog", "Choose where the scene will be saved" );
    if (s.length()>0)
        fileSaveAs(s);
}

void RealGUI::fileSaveAs(const char* filename)
{
    Simulation::printXML( viewer->getScene(), filename);
}

void RealGUI::fileExit()
{
    close();
}

void RealGUI::saveXML()
{
    Simulation::printXML( viewer->getScene(), "scene.scn");
}

void RealGUI::setTitle( const char* windowTitle )
{
    std::string str = "Sofa";
    if (windowTitle && *windowTitle)
    {
        str += " - ";
        str += windowTitle;
    }
#ifdef _WIN32
    setWindowTitle(str.c_str());
#else
    setCaption(str.c_str());
#endif
}


void RealGUI::DoubleClickeItemInSceneView(QListViewItem *item)
{
    // This happens because the clicked() signal also calls the select callback with
    // NULL as a parameter.
    if(item == NULL)
        return;

    // cancel the visibility action
    item->setOpen( !item->isOpen() );


    core::objectmodel::Base* node;
    for( std::map<core::objectmodel::Base*, Q3ListViewItem* >::iterator it = graphListener->items.begin() ; it != graphListener->items.end() ; ++ it)
    {
        if(  (*it).second == item )
        {
            node = (*it).first;
            break;
        }
    }



    // displayWidget
    if (node)
    {
        // if the gui is already opene, do nothing
        if( _alreadyOpen[ node ] )
        {
            _alreadyOpen[ node ]->raise();
            _alreadyOpen[ node ]->show();
            return;
        }

        // else Create the associated QWidget

        QWidget *qwidget = new QWidget(NULL,node->getName().data());

        const std::map< std::string, core::objectmodel::FieldBase* >& fields = node->getFields();

        int i=0;
        for( std::map< std::string, core::objectmodel::FieldBase* >::const_iterator it = fields.begin(); it!=fields.end(); ++it)
        {
            // The label
            QLabel *label = new QLabel(QString((*it).first.c_str()), qwidget,0);
            label->setGeometry( 10, i*25+5, 200, 20 );

            if( strcmp((*it).second->help,"TODO") )
            {
                QLabel *help = new QLabel((*it).second->help, qwidget);
                help->setGeometry( 380, i*25+5, 1000, 20 );
            }

            const std::string& fieldname = (*it).second->getValueTypeString();
            if( fieldname=="int")
            {
                QSpinBox* spinBox = new QSpinBox((int)INT_MIN,(int)INT_MAX,1,qwidget);
                spinBox->setGeometry( 205, i*25+5, 170, 20 );

                if( DataField<int> * ff = dynamic_cast< DataField<int> * >( (*it).second )  )
                {
                    spinBox->setValue(ff->getValue());
                    connect( spinBox, SIGNAL( valueChanged(int) ), new GUIFieldInt(ff), SLOT( changeValue(int) ) );
                }
            }
            else if( fieldname=="unsigned int")
            {
                QSpinBox* spinBox = new QSpinBox((int)0,(int)INT_MAX,1,qwidget);
                spinBox->setGeometry( 205, i*25+5, 170, 20 );

                if( DataField<unsigned int> * ff = dynamic_cast< DataField<unsigned int> * >( (*it).second )  )
                {
                    spinBox->setValue(ff->getValue());
                    connect( spinBox, SIGNAL( valueChanged(int) ), new GUIFieldUnsignedInt(ff), SLOT( changeValue(int) ) );
                }
            }
            else if( fieldname=="float" || fieldname=="double" )
            {

                WFloatLineEdit* editSFFloat = new WFloatLineEdit( qwidget, "editSFFloat" );
                editSFFloat->setMinFloatValue( (float)-INFINITY );
                editSFFloat->setMaxFloatValue( (float)INFINITY );
                editSFFloat->setGeometry( 205, i*25+5, 170, 20 );


                if( DataField<float> * ff = dynamic_cast< DataField<float> * >( (*it).second )  )
                {
                    editSFFloat->setFloatValue(ff->getValue());
                    connect( editSFFloat, SIGNAL( floatValueChanged(float) ), new GUIFieldFloat(ff), SLOT( changeValue(float) ) );
                }
                else if(DataField<double> * ff = dynamic_cast< DataField<double> * >( (*it).second )  )
                {
                    editSFFloat->setFloatValue(ff->getValue());
                    connect( editSFFloat, SIGNAL( floatValueChanged(float) ), new GUIFieldDouble(ff), SLOT( changeValue(float) ) );
                }

            }
            else if( fieldname=="bool" )
            {

                // the bool line edit
                QRadioButton* radioButton = new QRadioButton(qwidget);
                radioButton->setGeometry( 205, i*25+5, 170, 20 );


                if( DataField<bool> * ff = dynamic_cast< DataField<bool> * >( (*it).second )  )
                {
                    radioButton->setChecked(ff->getValue());
                    connect( radioButton, SIGNAL( toggled(bool) ), new GUIFieldBool(ff), SLOT( changeValue(bool) ) );
                }

            }
            else if( fieldname=="string" )
            {
                QLineEdit* lineEdit = new QLineEdit(qwidget);
                lineEdit->setGeometry( 205, i*25+5, 170, 20 );

                if( DataField<std::string> * ff = dynamic_cast< DataField<std::string> * >( (*it).second )  )
                {
                    lineEdit->setText(QString(ff->getValue().c_str()));
                    connect( lineEdit, SIGNAL( textChanged(const QString&) ), new GUIFieldString(ff), SLOT( changeValue(const QString&) ) );
                }

            }
            else if( fieldname=="Vec3f" || fieldname=="Vec3d" )
            {

                WFloatLineEdit* editSFFloatX = new WFloatLineEdit( qwidget, "editSFFloatX" );
                editSFFloatX->setMinFloatValue( (float)-INFINITY );
                editSFFloatX->setMaxFloatValue( (float)INFINITY );
                editSFFloatX->setGeometry( 205, i*25+5, 52, 20 );
                WFloatLineEdit* editSFFloatY = new WFloatLineEdit( qwidget, "editSFFloatY" );
                editSFFloatY->setMinFloatValue( (float)-INFINITY );
                editSFFloatY->setMaxFloatValue( (float)INFINITY );
                editSFFloatY->setGeometry( 262, i*25+5, 52, 20 );
                WFloatLineEdit* editSFFloatZ = new WFloatLineEdit( qwidget, "editSFFloatZ" );
                editSFFloatZ->setMinFloatValue( (float)-INFINITY );
                editSFFloatZ->setMaxFloatValue( (float)INFINITY );
                editSFFloatZ->setGeometry( 319, i*25+5, 52, 20 );


                if( DataField<Vec3f> * ff = dynamic_cast< DataField<Vec3f> * >( (*it).second )  )
                {
                    editSFFloatX->setFloatValue(ff->getValue()[0]);
                    editSFFloatY->setFloatValue(ff->getValue()[1]);
                    editSFFloatZ->setFloatValue(ff->getValue()[2]);

                    connect( editSFFloatX, SIGNAL( floatValueChanged(float) ), new GUIFieldVec3f(ff,0), SLOT( changeValue(float) ) );
                    connect( editSFFloatY, SIGNAL( floatValueChanged(float) ), new GUIFieldVec3f(ff,1), SLOT( changeValue(float) ) );
                    connect( editSFFloatZ, SIGNAL( floatValueChanged(float) ), new GUIFieldVec3f(ff,2), SLOT( changeValue(float) ) );
                }
                else if(DataField<Vec3d> * ff = dynamic_cast< DataField<Vec3d> * >( (*it).second )  )
                {
                    editSFFloatX->setFloatValue(ff->getValue()[0]);
                    editSFFloatY->setFloatValue(ff->getValue()[1]);
                    editSFFloatZ->setFloatValue(ff->getValue()[2]);

                    connect( editSFFloatX, SIGNAL( floatValueChanged(float) ), new GUIFieldVec3d(ff,0), SLOT( changeValue(float) ) );
                    connect( editSFFloatY, SIGNAL( floatValueChanged(float) ), new GUIFieldVec3d(ff,1), SLOT( changeValue(float) ) );
                    connect( editSFFloatZ, SIGNAL( floatValueChanged(float) ), new GUIFieldVec3d(ff,2), SLOT( changeValue(float) ) );
                }

            }
            else if( fieldname=="Vec2f" || fieldname=="Vec2d" )
            {

                WFloatLineEdit* editSFFloatX = new WFloatLineEdit( qwidget, "editSFFloatX" );
                editSFFloatX->setMinFloatValue( (float)-INFINITY );
                editSFFloatX->setMaxFloatValue( (float)INFINITY );
                editSFFloatX->setGeometry( 205, i*25+5, 52, 20 );
                WFloatLineEdit* editSFFloatY = new WFloatLineEdit( qwidget, "editSFFloatY" );
                editSFFloatY->setMinFloatValue( (float)-INFINITY );
                editSFFloatY->setMaxFloatValue( (float)INFINITY );
                editSFFloatY->setGeometry( 262, i*25+5, 52, 20 );


                if( DataField<Vec2f> * ff = dynamic_cast< DataField<Vec2f> * >( (*it).second )  )
                {
                    editSFFloatX->setFloatValue(ff->getValue()[0]);
                    editSFFloatY->setFloatValue(ff->getValue()[1]);

                    connect( editSFFloatX, SIGNAL( floatValueChanged(float) ), new GUIFieldVec2f(ff,0), SLOT( changeValue(float) ) );
                    connect( editSFFloatY, SIGNAL( floatValueChanged(float) ), new GUIFieldVec2f(ff,1), SLOT( changeValue(float) ) );
                }
                else if(DataField<Vec2d> * ff = dynamic_cast< DataField<Vec2d> * >( (*it).second )  )
                {
                    editSFFloatX->setFloatValue(ff->getValue()[0]);
                    editSFFloatY->setFloatValue(ff->getValue()[1]);

                    connect( editSFFloatX, SIGNAL( floatValueChanged(float) ), new GUIFieldVec2d(ff,0), SLOT( changeValue(float) ) );
                    connect( editSFFloatY, SIGNAL( floatValueChanged(float) ), new GUIFieldVec2d(ff,1), SLOT( changeValue(float) ) );
                }

            }
            else
                std::cerr<<"RealGUI.cpp: UNKNOWN GUI FIELD TYPE : "<<fieldname<<"   --> add a new GUIField"<<std::endl;



            // 					std::cerr<<(*it).first<<"     "<<(*it).second->getValueString()<<"   "<<(*it).second->getValueTypeString()<<"    "<<std::endl;

            ++i;
        }


        if(BaseObject*bo=dynamic_cast<BaseObject*>(node))
        {
            QPushButton*button=new QPushButton(qwidget,"update");
            button->setGeometry(5,i*25+10,150,20);
            button->setText("update");
            connect( button, SIGNAL( pressed() ), new GUIButton(bo), SLOT( reinit() ) );
        }


        if (qwidget)
        {
            qwidget->raise();
            qwidget->show();

            qwidget->setFixedSize(700,(i+1)*25+15);
            qwidget->setMinimumSize(200,100);
            qwidget->setMaximumSize(2000,(i+1)*25+15);
            qwidget->setCaption((node->getTypeName()+"::"+node->getName()).data());
            _alreadyOpen[ node ] = qwidget;
        }

        // 			std::cerr<<"\n\n\n";

    }




}

void RealGUI::playpauseGUI(bool value)
{
    if (value)
    {
        /*
        if (id_timer != -1)
        	killTimer(id_timer);*/

        timerStep->start(0);
    }
    else       {/*id_timer = startTimer(100);*/timerStep->stop();}
    if (getScene())  getScene()->getContext()->setAnimate(value);
}


void RealGUI::setGUI(void)
{
    textEdit1->setText(viewer->helpString());

#ifdef SOFA_GUI_QTOGREVIEWER
    //Hide unused options
    if (!strcmp(viewerName,"ogre"))
    {
        showVisual->hide();
        showBehavior->hide();
        showCollision->hide();
        showBoundingCollision->hide();
        showMapping->hide();
        showMechanicalMapping->hide();
        showForceField->hide();
        showInteractionForceField->hide();
        showWireFrame->hide();
        showNormals->hide();
    }
    else
    {
        showVisual->show();
        showBehavior->show();
        showCollision->show();
        showBoundingCollision->show();
        showMapping->show();
        showMechanicalMapping->show();
        showForceField->show();
        showInteractionForceField->show();
        showWireFrame->show();
        showNormals->show();
    }
#endif
}
//###################################################################################################################


//*****************************************************************************************
//called at each step of the rendering

void RealGUI::step()
{
    GNode* groot = viewer->getScene();
    if (groot == NULL) return;

    if (groot->getContext()->getMultiThreadSimulation())
    {
        static Node* n = NULL;

        if(ExecBus::getInstance() != NULL)
        {
            n = ExecBus::getInstance()->getNext("displayThread", n);

            if (n)
            {
                n->execute("displayThread");
            }
        }
    }
    else
    {
        if (viewer->ready()) return;
        //groot->setLogTime(true);

        Simulation::animate(groot);

        if( m_dumpState )
            Simulation::dumpState( groot, *m_dumpStateStream );
        if( m_exportGnuplot )
            Simulation::exportGnuplot( groot, groot->getTime() );

        viewer->wait();

        eventNewStep();
        eventNewTime();

        viewer->getQWidget()->update();
    }


    if (_animationOBJ)
    {
#ifdef CAPTURE_PERIOD
        static int counter = 0;
        if ((counter++ % CAPTURE_PERIOD)==0)
#endif
        {
            exportOBJ(false);
            ++_animationOBJcounter;
        }
    }
}



//*****************************************************************************************
// Update sofa Simulation with the time step

void RealGUI::eventNewStep()
{
    static ctime_t beginTime[10];
    static const ctime_t timeTicks = CTime::getRefTicksPerSec();
    static int frameCounter = 0;
    GNode* groot = getScene();
    if (frameCounter==0)
    {
        ctime_t t = CTime::getRefTime();
        for (int i=0; i<10; i++)
            beginTime[i] = t;
    }
    ++frameCounter;
    if ((frameCounter%10) == 0)
    {
        ctime_t curtime = CTime::getRefTime();
        int i = ((frameCounter/10)%10);
        double fps = ((double)timeTicks / (curtime - beginTime[i]))*(frameCounter<100?frameCounter:100);
        // 	    emit newFPS(fps);
        char buf[100];
        sprintf(buf, "%.1f FPS", fps);
        fpsLabel->setText(buf);
        // 	    emit newFPS(buf);
        beginTime[i] = curtime;
        //frameCounter = 0;
    }

    if (m_displayComputationTime && (frameCounter%100) == 0 && groot!=NULL)
    {

        std::cout << "========== ITERATION " << frameCounter << " ==========\n";
        const sofa::simulation::tree::GNode::NodeTimer& total = groot->getTotalTime();
        const std::map<std::string, sofa::simulation::tree::GNode::NodeTimer>& times = groot->getActionTime();
        const std::map<std::string, std::map<sofa::core::objectmodel::BaseObject*, sofa::simulation::tree::GNode::ObjectTimer> >& objtimes = groot->getObjectTime();
        const double fact = 1000000.0 / (100*groot->getTimeFreq());
        for (std::map<std::string, sofa::simulation::tree::GNode::NodeTimer>::const_iterator it = times.begin(); it != times.end(); ++it)
        {
            std::cout << "TIME "<<it->first<<": " << ((int)(fact*it->second.tTree+0.5))*0.001 << " ms (" << (1000*it->second.tTree/total.tTree)*0.1 << " %).\n";
            std::map<std::string, std::map<sofa::core::objectmodel::BaseObject*, sofa::simulation::tree::GNode::ObjectTimer> >::const_iterator it1 = objtimes.find(it->first);
            if (it1 != objtimes.end())
            {
                for (std::map<sofa::core::objectmodel::BaseObject*, sofa::simulation::tree::GNode::ObjectTimer>::const_iterator it2 = it1->second.begin(); it2 != it1->second.end(); ++it2)
                {
                    std::cout << "  "<< sofa::helper::gettypename(typeid(*(it2->first)))<<" "<< it2->first->getName() <<": "
                            << ((int)(fact*it2->second.tObject+0.5))*0.001 << " ms (" << (1000*it2->second.tObject/it->second.tTree)*0.1 << " %).\n";
                }
            }
        }
        for (std::map<std::string, std::map<sofa::core::objectmodel::BaseObject*, sofa::simulation::tree::GNode::ObjectTimer> >::const_iterator it = objtimes.begin(); it != objtimes.end(); ++it)
        {
            if (times.count(it->first)>0) continue;
            ctime_t ttotal = 0;
            for (std::map<sofa::core::objectmodel::BaseObject*, sofa::simulation::tree::GNode::ObjectTimer>::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2)
                ttotal += it2->second.tObject;
            std::cout << "TIME "<<it->first<<": " << ((int)(fact*ttotal+0.5))*0.001 << " ms (" << (1000*ttotal/total.tTree)*0.1 << " %).\n";
            if (ttotal > 0)
                for (std::map<sofa::core::objectmodel::BaseObject*, sofa::simulation::tree::GNode::ObjectTimer>::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2)
                {
                    std::cout << "  "<< sofa::helper::gettypename(typeid(*(it2->first)))<<" "<< it2->first->getName() <<": "
                            << ((int)(fact*it2->second.tObject+0.5))*0.001 << " ms (" << (1000*it2->second.tObject/ttotal)*0.1 << " %).\n";
                }
        }
        std::cout << "TOTAL TIME: " << ((int)(fact*total.tTree+0.5))*0.001 << " ms (" << ((int)(100/(fact*total.tTree*0.000001)+0.5))*0.01 << " FPS).\n";
        groot->resetTime();

    }
}



void RealGUI::eventNewTime()
{
    GNode* groot = getScene();
    if (groot)
    {

        double time = groot->getTime();
        // 	    emit newTime(time);
        char buf[100];
        sprintf(buf, "T: %.3f s", time);
        timeLabel->setText(buf);
        // 	    emit newTime(buf);

    }
}




//*****************************************************************************************
// Set the time between each iteration of the Sofa Simulation

void RealGUI::setDt(double value)
{
    GNode* groot = getScene();
    if (value > 0.0)
    {

        if (groot)
            groot->getContext()->setDt(value);
    }
}

void RealGUI::setDt(const QString& value)
{
    setDt(value.toDouble());
}


//*****************************************************************************************
// Reset the simulation to t=0
void RealGUI::resetScene()
{
    GNode* groot = getScene();
    if (groot)
    {
        Simulation::reset(groot);
        eventNewTime();
        viewer->getQWidget()->update();
    }
}


//*****************************************************************************************
// Set what to display
void RealGUI::slot_showVisual(bool value)
{
    GNode* groot = getScene();
    if (groot)
    {
        groot->getContext()->setShowVisualModels(value);
        Simulation::updateContext(groot);
    }
    viewer->getQWidget()->update();
}

void RealGUI::slot_showBehavior(bool value)
{
    GNode* groot = getScene();
    if (groot)
    {
        groot->getContext()->setShowBehaviorModels(value);
        Simulation::updateContext(groot);
    }
    viewer->getQWidget()->update();
}

void RealGUI::slot_showCollision(bool value)
{
    GNode* groot = getScene();
    if (groot)
    {
        groot->getContext()->setShowCollisionModels(value);
        Simulation::updateContext(groot);
    }
    viewer->getQWidget()->update();
}

void RealGUI::slot_showBoundingCollision(bool value)
{
    GNode* groot = getScene();
    if (groot)
    {
        groot->getContext()->setShowBoundingCollisionModels(value);
        Simulation::updateContext(groot);
    }
    viewer->getQWidget()->update();
}

void RealGUI::slot_showMapping(bool value)
{
    GNode* groot = getScene();
    if (groot)
    {
        groot->getContext()->setShowMappings(value);
        Simulation::updateContext(groot);
    }
    viewer->getQWidget()->update();
}

void RealGUI::slot_showMechanicalMapping(bool value)
{
    GNode* groot = getScene();
    if (groot)
    {
        groot->getContext()->setShowMechanicalMappings(value);
        Simulation::updateContext(groot);
    }
    viewer->getQWidget()->update();
}

void RealGUI::slot_showForceField(bool value)
{
    GNode* groot = getScene();
    if (groot)
    {
        groot->getContext()->setShowForceFields(value);
        Simulation::updateContext(groot);
    }
    viewer->getQWidget()->update();
}

void RealGUI::slot_showInteractionForceField(bool value)
{
    GNode* groot = getScene();
    if (groot)
    {
        groot->getContext()->setShowInteractionForceFields(value);
        Simulation::updateContext(groot);
    }
    viewer->getQWidget()->update();
}

void RealGUI::slot_showWireFrame(bool value)
{
    GNode* groot = getScene();
    if (groot)
    {
        groot->getContext()->setShowWireFrame(value);
        Simulation::updateContext(groot);
    }
    viewer->getQWidget()->update();
}

void RealGUI::slot_showNormals(bool value)
{
    GNode* groot = getScene();
    if (groot)
    {
        groot->getContext()->setShowNormals(value);
        Simulation::updateContext(groot);
    }
    viewer->getQWidget()->update();
}


//*****************************************************************************************
//
void RealGUI::exportGraph()
{
    exportGraph(getScene());
}


void RealGUI::exportGraph(sofa::simulation::tree::GNode* root)
{

    if (root == NULL) return;
    sofa::gui::guiviewer::GenGraphForm* form = new sofa::gui::guiviewer::GenGraphForm;
    form->setScene(root);
    form->show();
}



//*****************************************************************************************
//
void RealGUI::displayComputationTime(bool value)
{
    GNode* groot = getScene();
    m_displayComputationTime = value;
    if (groot)
    {
        groot->setLogTime(m_displayComputationTime);
    }
}



//*****************************************************************************************
//
void RealGUI::setExportGnuplot( bool exp )
{
    GNode* groot = getScene();
    m_exportGnuplot = exp;
    if( m_exportGnuplot && groot )
    {
        Simulation::initGnuplot( groot );
        Simulation::exportGnuplot( groot, groot->getTime() );
    }
}


//*****************************************************************************************
//
void RealGUI::dumpState(bool value)
{
    m_dumpState = value;
    if( m_dumpState )
    {
        m_dumpStateStream = new std::ofstream("dumpState.data");
    }
    else if( m_dumpStateStream!=NULL )
    {
        delete m_dumpStateStream;
        m_dumpStateStream = 0;
    }
}



//*****************************************************************************************
//
void RealGUI::exportOBJ(bool exportMTL)
{
    GNode* groot = getScene();
    if (!groot) return;
    std::string sceneFileName = viewer->getSceneFileName();
    std::ostringstream ofilename;
    if (!sceneFileName.empty())
    {
        const char* begin = sceneFileName.c_str();
        const char* end = strrchr(begin,'.');
        if (!end) end = begin + sceneFileName.length();
        ofilename << std::string(begin, end);
    }
    else
        ofilename << "scene";

    std::stringstream oss;
    oss.width(5);
    oss.fill('0');
    oss << _animationOBJcounter;

    ofilename << '_' << (oss.str().c_str());
    ofilename << ".obj";
    std::string filename = ofilename.str();
    std::cout << "Exporting OBJ Scene "<<filename<<std::endl;
    Simulation::exportOBJ(groot, filename.c_str(),exportMTL);
}


//*****************************************************************************************
// Called by the animate timer
void RealGUI::animate()
{
    viewer->getQWidget()->update();
}


void RealGUI::keyPressEvent ( QKeyEvent * e )
{
    switch(e->key())
    {

    case Qt::Key_O:
        // --- export to OBJ
    {
        exportOBJ();
        break;
    }
    case Qt::Key_P:
        // --- export to a succession of OBJ to make a video
    {
        _animationOBJ = !_animationOBJ;
        _animationOBJcounter = 0;
        break;
    }
    default:break;
    }
}



} // namespace qt

} // namespace gui

} // namespace sofa
