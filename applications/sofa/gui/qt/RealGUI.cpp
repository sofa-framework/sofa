/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
 *                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
 * Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
 * H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
 * M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
 *                                                                             *
 * Contact information: contact@sofa-framework.org                             *
 ******************************************************************************/
#include "RealGUI.h"
#include "ImageQt.h"
#ifndef SOFA_GUI_QT_NO_RECORDER
#include "sofa/gui/qt/QSofaRecorder.h"
#endif
#include "sofa/gui/qt/QSofaStatWidget.h"
#include "sofa/gui/qt/GenGraphForm.h"
#include "sofa/gui/qt/QSofaListView.h"
#include <algorithm>

#ifdef SOFA_HAVE_CHAI3D
#include <sofa/simulation/common/PropagateEventVisitor.h>
#include <sofa/core/objectmodel/GLInitializedEvent.h>
#endif // SOFA_HAVE_CHAI3D

#include <sofa/component/visualmodel/VisualModelImpl.h>
#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/common/xml/XML.h>
#include <sofa/simulation/common/InitVisitor.h>
#include <sofa/simulation/common/UpdateContextVisitor.h>
#include <sofa/simulation/common/DeleteVisitor.h>
#include <sofa/simulation/common/DeactivatedNodeVisitor.h>
#include <sofa/component/visualmodel/VisualStyle.h>
#include <sofa/helper/system/FileRepository.h>

#include <sofa/core/visual/VisualParams.h>

#ifdef SOFA_QT4
#include <QWidget>
#include <QDockWidget>
#include <QStackedWidget>
#include <QLayout>
#include <Q3ListViewItem>
#include <Q3ListView>
#include <QStatusBar>
#include <QRadioButton>
#include <QCheckBox>
#include <QSplitter>
#include <Q3TextEdit>
#include <QCursor>
#include <QAction>
#include <QMessageBox>
#include <QTabWidget>
#include <QToolTip>
#include <QButtonGroup>
#include <QRadioButton>
#include <QInputDialog>
#include <Q3DockWindow>
#include <Q3DockArea>
#else
#include <qwidget.h>
#include <qwidgetstack.h>
#include <qlayout.h>
#include <qlistview.h>
#include <qstatusbar.h>
#include <qheader.h>
#include <qimage.h>
#include <qsplitter.h>
#include <qtextedit.h>
#include <qcursor.h>
#include <qapplication.h>
#include <qaction.h>
#include <qmessagebox.h>
#include <qtabwidget.h>
#include <qtooltip.h>
#include <qbuttongroup.h>
#include <qradiobutton.h>
#include <qinputdialog.h>
#include <qmime.h>
#include <qdockwindow.h>
#include <qdockarea.h>
#endif



namespace sofa
{

namespace gui
{

namespace qt
{



SOFA_LINK_CLASS(ImageQt);


#ifdef SOFA_QT4
typedef Q3ListView QListView;
typedef Q3DockWindow QDockWindow;
typedef QStackedWidget QWidgetStack;
typedef Q3TextEdit QTextEdit;
#endif


using sofa::core::objectmodel::BaseObject;
using namespace sofa::helper::system::thread;
using namespace sofa::simulation;
using namespace sofa::core::visual;


///////////////////////////////////////////////////////////
//////////////////// SofaGUI Interface ////////////////////
///////////////////////////////////////////////////////////
extern QApplication* application; // = NULL;
extern RealGUI* gui;

#ifdef SOFA_QT4
/// Custom QApplication class handling FileOpen events for MacOS
class QSOFAApplication : public QApplication
{
public:
    QSOFAApplication(int &argc, char ** argv)
        : QApplication(argc,argv)
    {
    }

protected:
    bool event(QEvent *event)
    {
        switch (event->type())
        {
        case QEvent::FileOpen:
            static_cast<RealGUI*>(mainWidget())->fileOpen(static_cast<QFileOpenEvent *>(event)->file().ascii());

            return true;
        default:
            return QApplication::event(event);
        }
    }
};
#else
typedef QApplication QSOFAApplication;
#endif

RealGUI* gui = NULL;
QApplication* application = NULL;

const char* progname="";


int RealGUI::InitGUI ( const char* /*name*/, const std::vector<std::string>& /* options */ )
{
    if ( ImageQt::Init() )
    {
        return 0;
    }
    else
    {
        return 1;
    }
}


SofaGUI* RealGUI::CreateGUI ( const char* name, const std::vector<std::string>& options, sofa::simulation::Node* root, const char* filename )
{
    {
        int  *argc = new int;
        char **argv=new char*[2];
        *argc = 1;
        argv[0] = strdup ( SofaGUI::GetProgramName() );
        argv[1]=NULL;
        application = new QSOFAApplication ( *argc,argv );

    }
    // create interface
    gui = new RealGUI ( name, options );
    if ( root )
    {
        gui->setScene ( root, filename );
        gui->setWindowFilePath(QString(filename));
    }
    application->setMainWidget ( gui );

    QString pathIcon=(sofa::helper::system::DataRepository.getFirstPath() + std::string( "/icons/SOFA.png" )).c_str();
#ifdef SOFA_QT4
    application->setWindowIcon(QIcon(pathIcon));
#else
    gui->setIcon(QPixmap(pathIcon));
#endif

    // show the gui
    gui->show();

#ifdef SOFA_HAVE_CHAI3D
    // Tell nodes that openGl is initialized
    // especially the GL_MODELVIEW_MATRIX
    sofa::core::objectmodel::GLInitializedEvent ev;
    sofa::simulation::PropagateEventVisitor act(&ev);
    root->execute(act);
#endif // SOFA_HAVE_CHAI3D

    return gui;
}

int RealGUI::mainLoop()
{
    if (windowFilePath().isNull()) return application->exec();
    const std::string &filename=windowFilePath().ascii();
    const std::string &extension=sofa::helper::system::SetDirectory::GetExtension(filename.c_str());
    if (extension == "simu") fileOpenSimu(filename);
    return application->exec();
}

void RealGUI::redraw()
{
    emit newStep();
}

int RealGUI::closeGUI()
{

    delete this;
    return 0;
}

Node* RealGUI::currentSimulation()
{
    return viewer->getScene();
}

RealGUI::RealGUI ( const char* viewername, const std::vector<std::string>& /*options*/ )
    : viewerName ( viewername ),
      viewer ( NULL ),
      simulationGraph(NULL),
      currentTab ( NULL ),
      tabInstrument (NULL),
#ifndef SOFA_GUI_QT_NO_RECORDER
      recorder(NULL),
#else
      fpsLabel(NULL),
      timeLabel(NULL),
#endif
      statWidget(NULL),
      timerStep(NULL),
      backgroundImage(NULL),
      left_stack(NULL),
      recentlyOpenedFilesManager("config/Sofa.ini"),
      saveReloadFile(false),
      displayFlag(NULL)
{

#ifdef SOFA_GUI_INTERACTION
    interactionButton = new QPushButton(optionTabs);
    interactionButton->setObjectName(QString::fromUtf8("interactionButton"));
    interactionButton->setCheckable(true);
    interactionButton->setStyleSheet("background-color: cyan;");


    gridLayout->addWidget(interactionButton, 3, 0, 1, 1);
    gridLayout->removeWidget(screenshotButton);
    gridLayout->addWidget(screenshotButton, 3, 1, 1,1);

    interactionButton->setText(QApplication::translate("GUI", "&Interaction", 0, QApplication::UnicodeUTF8));
    interactionButton->setShortcut(QApplication::translate("GUI", "Alt+i", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
    interactionButton->setProperty("toolTip", QVariant(QApplication::translate("GUI", "Start interaction mode", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_TOOLTIP	
#endif

    connect(this, SIGNAL(quit()), this, SLOT(fileExit()));

    informationOnPickCallBack = InformationOnPickCallBack(this);

#ifdef SOFA_QT4
    fileMenu->removeAction(Action);
#endif
    //Configure Recently Opened Menu
    const int indexRecentlyOpened=fileMenu->count()-2;
    QMenu *recentMenu = recentlyOpenedFilesManager.createWidget(this);
    fileMenu->insertItem(QPixmap(),recentMenu,indexRecentlyOpened,indexRecentlyOpened);
    connect(recentMenu, SIGNAL(activated(int)), this, SLOT(fileRecentlyOpened(int)));

    left_stack = new QWidgetStack ( splitter2 );
    connect ( startButton, SIGNAL ( toggled ( bool ) ), this , SLOT ( playpauseGUI ( bool ) ) );
#ifdef SOFA_GUI_INTERACTION
    connect ( interactionButton, SIGNAL ( toggled ( bool ) ), this , SLOT ( interactionGUI ( bool ) ) );
#endif
    connect ( ResetSceneButton, SIGNAL ( clicked() ), this, SLOT ( resetScene() ) );
    connect ( dtEdit, SIGNAL ( textChanged ( const QString& ) ), this, SLOT ( setDt ( const QString& ) ) );
    connect ( stepButton, SIGNAL ( clicked() ), this, SLOT ( step() ) );
    connect ( dumpStateCheckBox, SIGNAL ( toggled ( bool ) ), this, SLOT ( dumpState ( bool ) ) );
    connect ( displayComputationTimeCheckBox, SIGNAL ( toggled ( bool ) ), this, SLOT ( displayComputationTime ( bool ) ) );
    connect ( exportGnuplotFilesCheckbox, SIGNAL ( toggled ( bool ) ), this, SLOT ( setExportGnuplot ( bool ) ) );
#ifdef SOFA_DUMP_VISITOR_INFO
    connect ( exportVisitorCheckbox, SIGNAL ( toggled ( bool ) ), this, SLOT ( setExportVisitor ( bool ) ) );
#endif


    pathDumpVisitor = sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str()) + std::string( "/dumpVisitor.xml" );

    connect ( tabs, SIGNAL ( currentChanged ( QWidget* ) ), this, SLOT ( currentTabChanged ( QWidget* ) ) );

    //Create a Dock Window to receive the Sofa Recorder
#ifndef SOFA_GUI_QT_NO_RECORDER
    QDockWindow *dockRecorder=new QDockWindow(this);
    dockRecorder->setResizeEnabled(true);
    this->moveDockWindow( dockRecorder, Qt::DockBottom);
    this->leftDock() ->setAcceptDockWindow(dockRecorder,false);
    this->rightDock()->setAcceptDockWindow(dockRecorder,false);

    recorder = new QSofaRecorder(dockRecorder);
    dockRecorder->setWidget(recorder);

    connect(startButton, SIGNAL(  toggled ( bool ) ), recorder, SLOT( TimerStart(bool) ) );
#else
    //Status Bar Configuration
    fpsLabel = new QLabel ( "9999.9 FPS", statusBar() );
    fpsLabel->setMinimumSize ( fpsLabel->sizeHint() );
    fpsLabel->clear();

    timeLabel = new QLabel ( "Time: 999.9999 s", statusBar() );
    timeLabel->setMinimumSize ( timeLabel->sizeHint() );
    timeLabel->clear();
    statusBar()->addWidget ( fpsLabel );
    statusBar()->addWidget ( timeLabel );
#endif

    statWidget = new QSofaStatWidget(TabStats);
    TabStats->layout()->add(statWidget);

    createViewers(viewername);


    currentTabChanged ( tabs->currentPage() );

    //ADD GUI for Background
    //------------------------------------------------------------------------
    //Informations
    QWidget *colour = new QWidget(TabPage);
    QHBoxLayout *colourLayout = new QHBoxLayout(colour);
    colourLayout->addWidget(new QLabel(QString("Colour "),colour));



    for (unsigned int i=0; i<3; ++i)
    {
        std::ostringstream s;
        s<<"background" <<i;
        background[i] = new WDoubleLineEdit(colour,s.str().c_str());
        background[i]->setMinValue( 0.0f);
        background[i]->setMaxValue( 1.0f);
        background[i]->setValue( 1.0f);

        colourLayout->addWidget(background[i]);
        connect( background[i], SIGNAL( returnPressed() ), this, SLOT( updateBackgroundColour() ) );
    }

    QWidget *image = new QWidget(TabPage);
    QHBoxLayout *imageLayout = new QHBoxLayout(image);
    imageLayout->addWidget(new QLabel(QString("Image "),image));

    backgroundImage = new QLineEdit(image,"backgroundImage");
    backgroundImage->setText( QString(viewer->getBackgroundImage().c_str()) );
    imageLayout->addWidget(backgroundImage);
    connect( backgroundImage, SIGNAL( returnPressed() ), this, SLOT( updateBackgroundImage() ) );

    ((QVBoxLayout*)(TabPage->layout()))->insertWidget(1,colour);
    ((QVBoxLayout*)(TabPage->layout()))->insertWidget(2,image);

    //---------------------------------------------------------------------------------------------------
#ifdef SOFA_PML
    pmlreader = NULL;
    lmlreader = NULL;
#endif
    simulationGraph = new QSofaListView(SIMULATION,TabGraph,"SimuGraph");
    ((QVBoxLayout*)TabGraph->layout())->addWidget(simulationGraph);
    connect ( ExportGraphButton, SIGNAL ( clicked() ), simulationGraph, SLOT ( Export() ) );
#ifdef SOFA_QT4
    tabs->removeTab(tabs->indexOf(TabVisualGraph));
#endif


#ifndef SOFA_DUMP_VISITOR_INFO
    //Remove option to see visitor trace
    this->exportVisitorCheckbox->hide();
#else
    //Main window containing a QListView only
    windowTraceVisitor = new WindowVisitor;
    windowTraceVisitor->graphView->setSorting(-1);
    windowTraceVisitor->hide();
    connect(windowTraceVisitor, SIGNAL(WindowVisitorClosed(bool)), this->exportVisitorCheckbox, SLOT(setChecked(bool)));
    handleTraceVisitor = new GraphVisitor(windowTraceVisitor);
#endif
    //--------
    descriptionScene = new QDialog(this);
    descriptionScene->resize(400,400);
    QVBoxLayout *descriptionLayout = new QVBoxLayout(descriptionScene);
    htmlPage = new QTextBrowser(descriptionScene);
    descriptionLayout->addWidget(htmlPage);
#ifdef SOFA_QT4
    connect(htmlPage, SIGNAL(sourceChanged(const QUrl&)), this, SLOT(changeHtmlPage(const QUrl&)));
#else
    // QMimeSourceFactory::defaultFactory()->setExtensionType("html", "text/utf8");
    htmlPage->mimeSourceFactory()->setExtensionType("html", "text/utf8");;
    connect(htmlPage, SIGNAL(sourceChanged(const QString&)), this, SLOT(changeHtmlPage(const QString&)));
#endif
    //--------
    pluginManager_dialog = new SofaPluginManager();
    pluginManager_dialog->hide();


    this->connect( pluginManager_dialog, SIGNAL( libraryAdded() ),  this, SLOT( updateViewerList() ));
    this->connect( pluginManager_dialog, SIGNAL( libraryRemoved() ),  this, SLOT( updateViewerList() ));


    SofaMouseManager::getInstance()->hide();
    SofaVideoRecorderManager::getInstance()->hide();


    //Center the application
    const QRect screen = QApplication::desktop()->availableGeometry(QApplication::desktop()->primaryScreen());
    this->move(  ( screen.width()- this->width()  ) / 2,  ( screen.height() - this->height()) / 2  );

    connect(simulationGraph, SIGNAL( RootNodeChanged(sofa::simulation::Node*, const char*) ), this, SLOT ( NewRootNode(sofa::simulation::Node* , const char*) ) );
    connect(simulationGraph, SIGNAL( NodeRemoved() ), this, SLOT( Update() ) );
    connect(simulationGraph, SIGNAL( Lock(bool) ), this, SLOT( LockAnimation(bool) ) );
    connect(simulationGraph, SIGNAL( RequestSaving(sofa::simulation::Node*) ), this, SLOT( fileSaveAs(sofa::simulation::Node*) ) );
    connect(simulationGraph, SIGNAL( RequestExportOBJ(sofa::simulation::Node*, bool) ), this, SLOT( exportOBJ(sofa::simulation::Node*, bool) ) );
    connect(simulationGraph, SIGNAL( RequestActivation(sofa::simulation::Node*, bool) ), this, SLOT( ActivateNode(sofa::simulation::Node*, bool) ) );
    connect(simulationGraph, SIGNAL( Updated() ), this, SLOT( redraw() ) );
    connect(simulationGraph, SIGNAL( NodeAdded() ), this, SLOT( Update() ) );
    connect(this, SIGNAL( newScene() ), simulationGraph, SLOT( CloseAllDialogs() ) );
    connect(this, SIGNAL( newStep() ), simulationGraph, SLOT( UpdateOpenedDialogs() ) );
    connect(simulationGraph, SIGNAL(focusChanged(sofa::core::objectmodel::BaseObject*)),
            viewer->getQWidget(), SLOT(fitObjectBBox(sofa::core::objectmodel::BaseObject*)) );
    connect(simulationGraph, SIGNAL( focusChanged(sofa::core::objectmodel::BaseNode*) ),
            viewer->getQWidget(), SLOT( fitNodeBBox(sofa::core::objectmodel::BaseNode*) ) );
#ifndef SOFA_GUI_QT_NO_RECORDER
    if (recorder)
        connect( recorder, SIGNAL( RecordSimulation(bool) ), startButton, SLOT( setOn(bool) ) );
    if (recorder)
        connect( recorder, SIGNAL( NewTime() ), viewer->getQWidget(), SLOT( update() ) );
#endif
    timerStep = new QTimer(this);
    connect ( timerStep, SIGNAL ( timeout() ), this, SLOT ( step() ) );

    animationState = false;

#ifdef SOFA_GUI_INTERACTION
    m_interactionActived = false;
    viewer->getQWidget()->installEventFilter(this);
#endif

}

void RealGUI::fileRecentlyOpened(int id)
{
    fileOpen(recentlyOpenedFilesManager.getFilename((unsigned int)id));
}


void RealGUI::setPixmap(std::string pixmap_filename, QPushButton* b)
{
    if ( sofa::helper::system::DataRepository.findFile ( pixmap_filename ) )
        pixmap_filename = sofa::helper::system::DataRepository.getFile ( pixmap_filename );

#ifdef SOFA_QT4
    b->setPixmap(QPixmap(QPixmap::fromImage(QImage(pixmap_filename.c_str()))));
#else
    b->setPixmap(QPixmap(QImage(pixmap_filename.c_str())));
#endif
}

RealGUI::~RealGUI()
{
#ifdef SOFA_PML
    if ( pmlreader )
    {
        delete pmlreader;
        pmlreader = NULL;
    }
    if ( lmlreader )
    {
        delete lmlreader;
        lmlreader = NULL;
    }
#endif
    if( displayFlag != NULL )
    {
        delete displayFlag;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    delete windowTraceVisitor;
    delete handleTraceVisitor;
#endif
    delete viewer;
}

void RealGUI::init()
{

    frameCounter = 0;
    _animationOBJ = false;
    _animationOBJcounter = 0;
    m_dumpState = false;
    m_dumpStateStream = 0;
    m_displayComputationTime = false;
    m_exportGnuplot = false;
    gnuplot_directory = "";
}

void RealGUI::createViewers(const char* viewerName)
{
    viewer::SofaViewerArgument arg;
    arg.name = "viewer";
    arg.parent = left_stack;
    if( viewer != NULL )
    {
        delete viewer;
        viewer = NULL;
    }
    viewerMap.clear();
    if ( viewerName[0] )
    {
        helper::vector< helper::SofaViewerFactory::Key > keys;
        helper::SofaViewerFactory::getInstance()->uniqueKeys(std::back_inserter(keys));
        helper::vector< helper::SofaViewerFactory::Key >::const_iterator iter;
        for ( iter = keys.begin(); iter != keys.end(); ++iter )
        {
            QAction* action = new QAction(this);
            action->setText( helper::SofaViewerFactory::getInstance()->getViewerName(*iter) );
            action->setMenuText(  helper::SofaViewerFactory::getInstance()->getAcceleratedViewerName(*iter) );
            action->setToggleAction(true);
            action->addTo(View);
            viewerMap[*iter] = action;
            action->setEnabled(true);
            connect(action, SIGNAL( activated() ), this, SLOT( changeViewer() ) );
            if( strcmp(iter->c_str(), viewerName )== 0 )
            {
                viewer = helper::SofaViewerFactory::CreateObject(*iter, arg);
                action->setOn(true);
            }
            else
            {
                action->setOn(false);
            }
        }
    }

    if( viewer == NULL )
    {
        std::cerr << "ERROR(QtGUI): unknown or disabled viewer name "<<viewerName<<std::endl;
        application->exit();
    }
    left_stack->addWidget ( viewer->getQWidget() );
    initViewer();
}

void RealGUI::initViewer()
{
    assert( viewer != NULL );
    frameCounter = 0;
    _animationOBJ = false;
    _animationOBJcounter = 0;
    m_dumpState = false;
    m_dumpStateStream = 0;
    m_displayComputationTime = false;
    m_exportGnuplot = false;
    gnuplot_directory = "";


#ifdef SOFA_QT4
    left_stack->setCurrentWidget ( viewer->getQWidget() );
    viewer->getQWidget()->setFocusPolicy ( Qt::StrongFocus );
#else
    int id_viewer = left_stack->addWidget ( viewer->getQWidget() );
    left_stack->raiseWidget ( id_viewer );
    viewer->getQWidget()->setFocusPolicy ( QWidget::StrongFocus );
    viewer->getQWidget()->setCursor ( QCursor ( 2 ) );
#endif
    viewer->getQWidget()->setSizePolicy ( QSizePolicy ( ( QSizePolicy::SizeType ) 7, ( QSizePolicy::SizeType ) 7, 100, 1, viewer->getQWidget()->sizePolicy().hasHeightForWidth() ) );
    viewer->getQWidget()->setMinimumSize ( QSize ( 0, 0 ) );
    viewer->getQWidget()->setMouseTracking ( TRUE );
    viewer->configureViewerTab(tabs);

    connect ( ResetViewButton, SIGNAL ( clicked() ), viewer->getQWidget(), SLOT ( resetView() ) );
    connect ( SaveViewButton, SIGNAL ( clicked() ), viewer->getQWidget(), SLOT ( saveView() ) );
    connect ( screenshotButton, SIGNAL ( clicked() ), this, SLOT ( screenshot() ) );
    connect ( sizeW, SIGNAL ( valueChanged ( int ) ), viewer->getQWidget(), SLOT ( setSizeW ( int ) ) );
    connect ( sizeH, SIGNAL ( valueChanged ( int ) ), viewer->getQWidget(), SLOT ( setSizeH ( int ) ) );
    connect ( viewer->getQWidget(), SIGNAL ( resizeW ( int ) ), sizeW, SLOT ( setValue ( int ) ) );
    connect ( viewer->getQWidget(), SIGNAL ( resizeH ( int ) ), sizeH, SLOT ( setValue ( int ) ) );
    connect ( viewer->getQWidget(), SIGNAL ( quit (  ) ), this, SLOT ( fileExit (  ) ) );

    QSplitter *splitter_ptr = dynamic_cast<QSplitter *> ( splitter2 );
    splitter_ptr->moveToLast ( left_stack );
    splitter_ptr->setOpaqueResize ( false );
#ifdef SOFA_QT4
    splitter_ptr->setStretchFactor( 0, 0);
    splitter_ptr->setStretchFactor( 1, 10);
    QList<int> list;
#else
    QValueList<int> list;
#endif
    list.push_back ( 250 );
    list.push_back ( 640 );
    splitter_ptr->setSizes ( list );
    setGUI();

    viewer->getQWidget()->setFocus();
    viewer->getQWidget()->show();
    viewer->getQWidget()->update();
    SofaMouseManager::getInstance()->setPickHandler(viewer->getPickHandler());
    viewer->getPickHandler()->addCallBack(&informationOnPickCallBack );

    guiName=viewerName;
}

void RealGUI::changeViewer()
{
    QObject* obj = const_cast<QObject*>( QObject::sender() );
    if( !obj) return;

    QAction* action = static_cast<QAction*>(obj);

    action->setOn(true);
    std::map< helper::SofaViewerFactory::Key, QAction*  >::const_iterator iter_map;
    for ( iter_map = viewerMap.begin(); iter_map != viewerMap.end() ; ++iter_map )
    {

        if ( (*iter_map).second == action )
        {

            /* cleanup previous viewer */
            if ( viewer->getScene() !=NULL )
            {
                viewer->getPickHandler()->unload();
                simulation::getSimulation()->unload ( viewer->getScene() );
                delete viewer->getScene() ;
                viewer->setScene(NULL);
            }
            viewer->removeViewerTab(tabs);
            left_stack->removeWidget(viewer->getQWidget() );
            delete viewer;
            viewer = NULL;

            viewer::SofaViewerArgument arg;
            arg.name = "viewer";
            arg.parent = left_stack;
            /* change viewer */
            viewer =  helper::SofaViewerFactory::CreateObject( (*iter_map).first, arg);
            left_stack->addWidget( viewer->getQWidget() );
            initViewer();
        }
        else
        {
            (*iter_map).second->setOn(false);
        }
    }
    /* reload the scene */
    std::string filename(this->windowFilePath().ascii());
    fileOpen ( filename.c_str() ); // keep the current display flags
}

void RealGUI::updateViewerList()
{
    helper::vector< helper::SofaViewerFactory::Key > currentKeys;
    std::map< helper::SofaViewerFactory::Key, QAction*>::const_iterator iter_map;
    for ( iter_map = viewerMap.begin(); iter_map != viewerMap.end() ; ++iter_map )
    {
        currentKeys.push_back((*iter_map).first);
    }
    std::sort(currentKeys.begin(),currentKeys.end());
    helper::vector< helper::SofaViewerFactory::Key > updatedKeys;
    helper::SofaViewerFactory::getInstance()->uniqueKeys(std::back_inserter(updatedKeys));
    std::sort(updatedKeys.begin(),updatedKeys.end());

    helper::vector< helper::SofaViewerFactory::Key > diffKeys;

    std::set_symmetric_difference(currentKeys.begin(),currentKeys.end(),updatedKeys.begin(),updatedKeys.end()
            ,std::back_inserter(diffKeys));

    helper::vector< helper::SofaViewerFactory::Key >::const_iterator it;
    for( it = diffKeys.begin(); it != diffKeys.end(); ++it)
    {

        std::map< helper::SofaViewerFactory::Key, QAction* >::iterator itViewerMap;

        if( (itViewerMap = viewerMap.find(*it)) != viewerMap.end() )
        {
            if( (*itViewerMap).second->isOn() )
            {
                if ( viewer->getScene() !=NULL )
                {
                    viewer->getPickHandler()->unload();
                    simulation::getSimulation()->unload ( viewer->getScene() );
                    delete viewer->getScene() ;
                    viewer->setScene(NULL);
                }
                viewer->removeViewerTab(tabs);
                left_stack->removeWidget(viewer->getQWidget() );
                delete viewer;
                viewer = NULL;
            }
            (*itViewerMap).second->removeFrom(View);
            viewerMap.erase(itViewerMap);
        }
        else
        {
            QAction* action = new QAction(this);
            action->setText( helper::SofaViewerFactory::getInstance()->getViewerName(*it) );
            action->setMenuText(  helper::SofaViewerFactory::getInstance()->getAcceleratedViewerName(*it) );
            action->setToggleAction(true);
            action->addTo(View);
            viewerMap[*it] = action;
            action->setEnabled(true);
            connect(action, SIGNAL( activated() ), this, SLOT( changeViewer() ) );
        }
    }

    if( viewer == NULL )
    {
        if(!viewerMap.empty())
        {
            viewer::SofaViewerArgument arg;
            arg.name = "viewer";
            arg.parent = left_stack;
            /* change viewer */
            viewer =  helper::SofaViewerFactory::CreateObject( viewerMap.begin()->first, arg);
            left_stack->addWidget( viewer->getQWidget() );
            initViewer();
        }
    }
}


void RealGUI::fileOpen ( std::string filename, bool temporaryFile )
{
    const std::string &extension=sofa::helper::system::SetDirectory::GetExtension(filename.c_str());
    if (extension == "simu")
    {
        return fileOpenSimu(filename);
    }

    startButton->setOn(false);
    descriptionScene->hide();
    htmlPage->clear();

    if ( sofa::helper::system::DataRepository.findFile (filename) )
        filename = sofa::helper::system::DataRepository.getFile ( filename );
    else
        return;
    startDumpVisitor();

    frameCounter = 0;
    sofa::simulation::xml::numDefault = 0;

    update();
    //Hide all the dialogs to modify the graph
    emit ( newScene() );

    if ( viewer->getScene() !=NULL )
    {
        viewer->getPickHandler()->reset();//activateRay(false);
        viewer->getPickHandler()->unload();

        simulation::getSimulation()->unload ( viewer->getScene() ); delete viewer->getScene() ;
    }
    //Clear the list of modified dialog opened

    simulation::Node* root = simulation::getSimulation()->load ( filename.c_str() );
    simulation::getSimulation()->init ( root );

    if ( root == NULL )
    {
        qFatal ( "Failed to load %s",filename.c_str() );
        stopDumpVisitor();
        return;
    }
    this->setWindowFilePath(filename.c_str());
    setScene ( root, filename.c_str(), temporaryFile );



    configureGUI(root);

    //need to create again the output streams !!
    simulation::getSimulation()->gnuplotDirectory.setValue(gnuplot_directory);
    setExportGnuplot(exportGnuplotFilesCheckbox->isChecked());
    displayComputationTime(m_displayComputationTime);
    stopDumpVisitor();
}

#ifdef SOFA_PML
void RealGUI::pmlOpen ( const char* filename, bool /*resetView*/ )
{
    std::string scene = "PML/default.scn";
    if ( !sofa::helper::system::DataRepository.findFile ( scene ) )
    {
        std::cerr << "File " << scene << " not found " << std::endl;
        return;
    }
    if ( viewer->getScene() !=NULL )
    {
        simulation::getSimulation()->unload ( viewer->getScene() ); delete viewer->getScene() ;

    }
    GNode *simuNode = dynamic_cast< GNode *> (simulation::getSimulation()->load ( scene.c_str() ));
    getSimulation()->init(simuNode);
    if ( simuNode )
    {
        if ( !pmlreader ) pmlreader = new PMLReader;
        pmlreader->BuildStructure ( filename, simuNode );
        setScene ( simuNode, filename );
        this->setWindowFilePath(filename); //.c_str());
    }
}

void RealGUI::lmlOpen ( const char* filename )
{
    if ( pmlreader )
    {
        Node* root;
        if ( lmlreader != NULL ) delete lmlreader;
        lmlreader = new LMLReader; std::cout <<"New lml reader\n";
        lmlreader->BuildStructure ( filename, pmlreader );

        root = viewer->getScene();
        simulation::getSimulation()->init ( root );

    }
    else
        std::cerr<<"You must load the pml file before the lml file"<<endl;
}
#endif



void RealGUI::setScene ( Node* root, const char* filename, bool temporaryFile )
{
    if (filename)
    {
        if (!temporaryFile) recentlyOpenedFilesManager.openFile(filename);
        setTitle ( filename );
        saveReloadFile=temporaryFile;
        std::string extension=sofa::helper::system::SetDirectory::GetExtension(filename);
        std::string htmlFile=filename; htmlFile.resize(htmlFile.size()-extension.size()-1);
        htmlFile+=".html";
        if (sofa::helper::system::DataRepository.findFile (htmlFile,"",NULL))
        {
#ifdef WIN32
            htmlFile = "file:///"+htmlFile;
#endif
            descriptionScene->show();
#ifdef SOFA_QT4
            htmlPage->setSource(QUrl(QString(htmlFile.c_str())));
#else
            htmlPage->mimeSourceFactory()->setFilePath(QString(htmlFile.c_str()));
            htmlPage->setSource(QString(htmlFile.c_str()));
#endif

        }

    }
    if (tabInstrument!= NULL)
    {
        tabs->removePage(tabInstrument);
        delete tabInstrument;
        tabInstrument = NULL;
    }
    viewer->setScene ( root, filename );

    if( displayFlag != NULL)
    {
        gridLayout1->removeWidget(displayFlag);
        delete displayFlag;
        displayFlag = NULL;
    }

    component::visualmodel::VisualStyle* visualStyle = NULL;

    if( root )
    {
        root->get(visualStyle);
        if(visualStyle)
        {
            displayFlag = new DisplayFlagsDataWidget(tabView,"displayFlagwidget",&visualStyle->displayFlags);
            displayFlag->createWidgets();
            displayFlag->updateWidgetValue();
            connect( displayFlag, SIGNAL( WidgetDirty(bool) ), this, SLOT(showhideElements() ));
            displayFlag->setMinimumSize(50,100);
            gridLayout1->addWidget(displayFlag,0,0);
            connect(tabs,SIGNAL(currentChanged(QWidget*)),displayFlag, SLOT( updateWidgetValue() ));
        }
    }


    this->setWindowFilePath(filename);
    viewer->resetView();

    eventNewTime();

    if (root)
    {


        //simulation::getSimulation()->updateVisualContext ( root );
        startButton->setOn ( root->getContext()->getAnimate() );
        dtEdit->setText ( QString::number ( root->getDt() ) );

        simulationGraph->Clear(root);

        statWidget->CreateStats(dynamic_cast<Node*>(simulation::getSimulation()->getContext()) );

#ifndef SOFA_GUI_QT_NO_RECORDER
        if (recorder)
            recorder->Clear();
#endif
    }

#ifdef SOFA_HAVE_CHAI3D
    // Tell nodes that openGl is initialized
    // especialy the GL_MODELVIEW_MATRIX
    sofa::core::objectmodel::GLInitializedEvent ev;
    sofa::simulation::PropagateEventVisitor act(&ev);
    root->execute(act);
#endif // SOFA_HAVE_CHAI3D

    viewer->getQWidget()->setFocus();
    viewer->getQWidget()->show();
    viewer->getQWidget()->update();
    resetScene();



}

void RealGUI::Clear()
{
    simulation::Simulation *s = simulation::getSimulation();

    //In case instruments are present in the scene, we create a new tab, and display the listr
    if (s->instruments.size() != 0)
    {
        tabInstrument = new QWidget();
        tabs->addTab(tabInstrument, QString("Instrument"));

        QVBoxLayout *layout = new QVBoxLayout( tabInstrument, 0, 1, "tabInstrument");

        QButtonGroup *list_instrument = new QButtonGroup(tabInstrument);
        list_instrument->setExclusive(true);

#ifdef SOFA_QT4
        connect ( list_instrument, SIGNAL ( buttonClicked(int) ), this, SLOT ( changeInstrument(int) ) );
#else
        connect ( list_instrument, SIGNAL ( clicked(int) ), this, SLOT ( changeInstrument(int) ) );
#endif

        QRadioButton *button = new QRadioButton(tabInstrument); button->setText("None");
#ifdef SOFA_QT4
        list_instrument->addButton(button, 0);
#else
        list_instrument->insert(button);
#endif
        layout->addWidget(button);

        for (unsigned int i=0; i<s->instruments.size(); i++)
        {
            QRadioButton *button = new QRadioButton(tabInstrument);  button->setText(QString( s->instruments[i]->getName().c_str() ) );
#ifdef SOFA_QT4
            list_instrument->addButton(button, i+1);
#else
            list_instrument->insert(button);
#endif
            layout->addWidget(button);
            if (i==0)
            {
                button->setChecked(true); changeInstrument(1);
            }
            else
                s->instruments[i]->setActive(false);

        }
#ifdef SOFA_QT4
        layout->addStretch(1);
#endif
#ifndef SOFA_QT4
        layout->addWidget(list_instrument);
#endif
    }

#ifndef SOFA_GUI_QT_NO_RECORDER
    if (recorder)
        recorder->Clear();
#endif

    simulationGraph->Clear(dynamic_cast<Node*>(simulation::getSimulation()->getContext()));

    statWidget->CreateStats(dynamic_cast<Node*>(simulation::getSimulation()->getContext()));


}

//----------------------------------
//Configuration
void RealGUI::setViewerResolution ( int w, int h )
{
    QSize winSize = size();
    QSize viewSize = viewer->getQWidget()->size();
    //viewer->getQWidget()->setMinimumSize ( QSize ( w, h ) );
    //viewer->getQWidget()->setMaximumSize ( QSize ( w, h ) );
    //viewer->getQWidget()->resize(w,h);
#ifdef SOFA_QT4
    QList<int> list;
#else
    QValueList<int> list;
#endif
    list.push_back ( 250 );
    list.push_back ( w );
    QSplitter *splitter_ptr = dynamic_cast<QSplitter *> ( splitter2 );
    splitter_ptr->setSizes ( list );
#ifdef SOFA_QT4
    layout()->update();
#endif
    resize(winSize.width() - viewSize.width() + w, winSize.height() - viewSize.height() + h);
    //std::cout << "Setting windows dimension to " << size().width() << " x " << size().height() << std::endl;
}
void RealGUI::setFullScreen ()
{

#ifdef SOFA_QT4
    QList<int> list;
#else
    QValueList<int> list;
#endif
    list.push_back ( 0 );
    list.push_back ( this->width() );
    QSplitter *splitter_ptr = dynamic_cast<QSplitter *> ( splitter2 );
    splitter_ptr->setSizes ( list );

    showFullScreen();

#ifndef SOFA_GUI_QT_NO_RECORDER
    if (recorder) recorder->parentWidget()->hide();
    statusBar()->addWidget( recorder->getFPSLabel());
    statusBar()->addWidget( recorder->getTimeLabel());
#endif

}

void RealGUI::setBackgroundColor(const defaulttype::Vector3& c)
{
    background[0]->setText(QString::number(c[0]));
    background[1]->setText(QString::number(c[1]));
    background[2]->setText(QString::number(c[2]));
    updateBackgroundColour();
}

void RealGUI::setBackgroundImage(const std::string& c)
{
    backgroundImage->setText(QString(c.c_str()));
    updateBackgroundImage();
}

void RealGUI::setDumpState(bool b)
{
    dumpStateCheckBox->setChecked(b);
}

void RealGUI::setLogTime(bool b)
{
    displayComputationTimeCheckBox->setChecked(b);
}

void RealGUI::setExportState(bool b)
{
    exportGnuplotFilesCheckbox->setChecked(b);
}

#ifdef SOFA_DUMP_VISITOR_INFO
void RealGUI::setTraceVisitors(bool b)
{
    exportVisitorCheckbox->setChecked(b);
}

#endif

void RealGUI::setRecordPath(const std::string &
#ifndef SOFA_GUI_QT_NO_RECORDER
        path
#endif
                           )
{
#ifndef SOFA_GUI_QT_NO_RECORDER
    if (recorder) recorder->SetRecordDirectory(path);
#endif
}

void RealGUI::setGnuplotPath(const std::string &path)
{
    simulation::getSimulation()->gnuplotDirectory.setValue(path);
}
void RealGUI::setViewerConfiguration(sofa::component::configurationsetting::ViewerSetting* viewerConf)
{
    const defaulttype::Vec<2,int> &res=viewerConf->resolution.getValue();
    if (viewerConf->fullscreen.getValue()) setFullScreen();
    else setViewerResolution(res[0], res[1]);
    viewer->configure(viewerConf);
}

void RealGUI::setMouseButtonConfiguration(sofa::component::configurationsetting::MouseButtonSetting *button)
{
    SofaMouseManager::getInstance()->updateOperation(button);
    //        SofaMouseManager::getInstance()->updateContent();
}

//--------------------------------------
void RealGUI::changeInstrument(int id)
{
    std::cout << "Activation instrument "<<id<<std::endl;
    simulation::Simulation *s = simulation::getSimulation();
    if (s->instrumentInUse.getValue() >= 0 && s->instrumentInUse.getValue() < (int)s->instruments.size())
        s->instruments[s->instrumentInUse.getValue()]->setActive(false);

    simulation::getSimulation()->instrumentInUse.setValue(id-1);
    if (s->instrumentInUse.getValue() >= 0 && s->instrumentInUse.getValue() < (int)s->instruments.size())
        s->instruments[s->instrumentInUse.getValue()]->setActive(true);
    viewer->getQWidget()->update();
}


void RealGUI::screenshot()
{

    QString filename;

#ifdef SOFA_HAVE_PNG
    const char* imageString = "Images (*.png)";
#else
    const char* imageString = "Images (*.bmp)";
#endif

    filename = getSaveFileName ( this,
            viewer->screenshotName().c_str(),
            imageString,
            "save file dialog"
            "Choose a filename to save under" );
    viewer->getQWidget()->repaint();
    if ( filename != "" )
    {
        std::ostringstream ofilename;
        const char* begin = filename;
        const char* end = strrchr ( begin,'_' );
        if ( !end ) end = begin + filename.length();
        ofilename << std::string ( begin, end );
        ofilename << "_";
        viewer->setPrefix ( ofilename.str() );
#ifdef SOFA_QT4
        viewer->screenshot ( filename.toStdString() );
#else
        viewer->screenshot ( filename );
#endif

    }
}

void RealGUI::fileOpenSimu ( std::string s )
{
    std::ifstream in(s.c_str());

    if (!in.fail())
    {
        std::string filename;
        std::string initT, endT, dT, writeName;
        in
                >> filename
                        >> initT >> initT
                        >> endT  >> endT >> endT
                        >> dT >> dT
                        >> writeName >> writeName;
        in.close();



        if ( sofa::helper::system::DataRepository.findFile (filename) )
        {
            filename = sofa::helper::system::DataRepository.getFile ( filename );
            simulation_name = s;
            std::string::size_type pointSimu = simulation_name.rfind(".simu");
            simulation_name.resize(pointSimu);
            fileOpen(filename.c_str());
            this->setWindowFilePath(QString(filename.c_str()));
            dtEdit->setText(QString(dT.c_str()));
#ifndef SOFA_GUI_QT_NO_RECORDER
            if (recorder)
                recorder->SetSimulation(initT,endT,writeName);
#endif

        }
    }
}
void RealGUI::fileNew()
{
    std::string newScene("config/newScene.scn");
    if (sofa::helper::system::DataRepository.findFile (newScene))
        fileOpen(sofa::helper::system::DataRepository.getFile ( newScene ).c_str());
}
void RealGUI::fileOpen()
{
    std::string filename(this->windowFilePath().ascii());

    QString s = getOpenFileName ( this, filename.empty() ?NULL:filename.c_str(),
#ifdef SOFA_PML
            "Scenes (*.scn *.xml);;Simulation (*.simu);;Php Scenes (*.pscn);;Pml Lml (*.pml *.lml);;All (*)",
#else
            "Scenes (*.scn *.xml);;Simulation (*.simu);;Php Scenes (*.pscn);;All (*)",
#endif
            "open file dialog",  "Choose a file to open" );

    if ( s.length() >0 )
    {
#ifdef SOFA_PML
        if ( s.endsWith ( ".pml" ) )
            pmlOpen ( s );
        else if ( s.endsWith ( ".lml" ) )
            lmlOpen ( s );
        else
#endif
            if (s.endsWith( ".simu") )
                fileOpenSimu(s.ascii());
            else
            {
                fileOpen (s.ascii());
            }
    }
}

void RealGUI::fileReload()
{

    std::string filename(this->windowFilePath().ascii());
    QString s = filename.c_str();

    if ( filename.empty() ) { std::cerr << "Reload failed: no file loaded.\n"; return;}

#ifdef SOFA_PML
    if ( s.length() >0 )
    {
        if ( s.endsWith ( ".pml" ) )
            pmlOpen ( s );
        else if ( s.endsWith ( ".lml" ) )
            lmlOpen ( s );
        else if (s.endsWith( ".simu") )
            fileOpenSimu(filename);
        else
            fileOpen ( filename, saveReloadFile);
    }
#else
    if (s.endsWith( ".simu") )
        fileOpenSimu(s.ascii());
    else
        fileOpen ( s.ascii(),saveReloadFile );
#endif

}

void RealGUI::fileSave()
{
    std::string filename(this->windowFilePath().ascii());
    std::string message="You are about to overwrite your current scene: "  + filename + "\nAre you sure you want to do that ?";

    if ( QMessageBox::warning ( this, "Saving the Scene",message.c_str(), QMessageBox::Yes | QMessageBox::Default, QMessageBox::No ) != QMessageBox::Yes )
        return;

    Node *node = viewer->getScene();
    fileSaveAs ( node,filename.c_str() );
}


void RealGUI::fileSaveAs(Node *node)
{
    if (node == NULL) node = viewer->getScene();
    QString s;
    std::string filename(this->windowFilePath().ascii());
#ifdef SOFA_PML
    s = getSaveFileName ( this, filename.empty() ?NULL:filename.c_str(), "Scenes (*.scn *.xml *.pml)", "save file dialog",  "Choose where the scene will be saved" );
    if ( s.length() >0 )
    {
        if ( pmlreader && s.endsWith ( ".pml" ) )
            pmlreader->saveAsPML ( s );
        else
            fileSaveAs ( node,s );
    }
#else
    s = getSaveFileName ( this, filename.empty() ?NULL:filename.c_str(), "Scenes (*.scn *.xml)", "save file dialog", "Choose where the scene will be saved" );
    if ( s.length() >0 )
        fileSaveAs ( node,s );
#endif

}

void RealGUI::fileSaveAs ( Node *node, const char* filename )
{
    simulation::getSimulation()->exportXML ( node, filename );
}

void RealGUI::fileExit()
{
    //Hide all opened ModifyObject windows
    emit ( newScene() );
    startButton->setOn ( false);
    this->close();
}

void RealGUI::saveXML()
{
    simulation::getSimulation()->exportXML ( viewer->getScene(), "scene.scn" );
}

void RealGUI::editRecordDirectory()
{
    std::string filename(this->windowFilePath().ascii());
    std::string record_directory;
    QString s = getExistingDirectory ( this, filename.empty() ?NULL:filename.c_str(), "open directory dialog",  "Choose a directory" );
    if (s.length() > 0)
    {
        record_directory = s.ascii();
        if (record_directory.at(record_directory.size()-1) != '/') record_directory+="/";
#ifndef SOFA_GUI_QT_NO_RECORDER
        if (recorder)
            recorder->SetRecordDirectory(record_directory);
#endif
    }

}

void RealGUI::showPluginManager()
{
    pluginManager_dialog->show();
}

void RealGUI::showMouseManager()
{
    SofaMouseManager::getInstance()->updateContent();
    SofaMouseManager::getInstance()->show();
}

void RealGUI::showVideoRecorderManager()
{
    SofaVideoRecorderManager::getInstance()->show();
}

void RealGUI::editGnuplotDirectory()
{
    std::string filename(this->windowFilePath().ascii());
    QString s = getExistingDirectory ( this, filename.empty() ?NULL:filename.c_str(), "open directory dialog",  "Choose a directory" );
    if (s.length() > 0)
    {
        gnuplot_directory = s.ascii();
        if (gnuplot_directory.at(gnuplot_directory.size()-1) != '/') gnuplot_directory+="/";

        simulation::getSimulation()->gnuplotDirectory.setValue(gnuplot_directory);
        setExportGnuplot(exportGnuplotFilesCheckbox->isChecked());
    }
}


void RealGUI::setTitle ( std::string windowTitle )
{
    std::string str = "Sofa";
    if ( !windowTitle.empty() )
    {
        str += " - ";
        str += windowTitle;
    }
#ifdef WIN32
    setWindowTitle ( str.c_str() );
#else
    setCaption ( str.c_str() );
#endif
}


void RealGUI::playpauseGUI ( bool value )
{
    startButton->setOn ( value );
    if ( getScene() )  getScene()->getContext()->setAnimate ( value );
    if(value)
    {
        timerStep->start(0);
    }
    else
    {
        timerStep->stop();
    }
}

#ifdef SOFA_GUI_INTERACTION
void RealGUI::interactionGUI ( bool value )
{
    interactionButton->setOn ( value );
    m_interactionActived = value;
    viewer->getQWidget()->setMouseTracking ( ! value);
    if (value==true) playpauseGUI(value);

    if(value)
    {
        interactionButton->setText(QApplication::translate("GUI", "ESC to qu&it", 0, QApplication::UnicodeUTF8));
        this->grabMouse();
        this->grabKeyboard();
        this->setMouseTracking(true);
        //this->setCursor(QCursor(Qt::BlankCursor));
        application->setOverrideCursor( QCursor( Qt::BlankCursor ) );
        QPoint p = mapToGlobal(QPoint((this->width()+2)/2,(this->height()+2)/2));
        QCursor::setPos(p);
    }
    else
    {
        interactionButton->setText(QApplication::translate("GUI", "&Interaction", 0, QApplication::UnicodeUTF8));
        this->releaseKeyboard();
        this->releaseMouse();
        this->setMouseTracking(false);
        //this->setCursor(QCursor(Qt::ArrowCursor));
        application->restoreOverrideCursor();
    }
    sofa::core::objectmodel::KeypressedEvent keyEvent(value?(char)0x81:(char)0x80);
    Node* groot = viewer->getScene();
    if (groot) groot->propagateEvent(core::ExecParams::defaultInstance(), &keyEvent);
}
#else
void RealGUI::interactionGUI ( bool )
{
}
#endif


void RealGUI::setGUI ( void )
{
    textEdit1->setText ( viewer->helpString() );
    connect ( this, SIGNAL( newStep()), viewer->getQWidget(), SLOT( update()));
}
//###################################################################################################################

void RealGUI::startDumpVisitor()
{
#ifdef SOFA_DUMP_VISITOR_INFO
    Node* root = viewer->getScene();
    if (root && this->exportVisitorCheckbox->isOn())
    {
        m_dumpVisitorStream.str("");
        Visitor::startDumpVisitor(&m_dumpVisitorStream, root->getTime());
    }
#endif
}
void RealGUI::stopDumpVisitor()
{
#ifdef SOFA_DUMP_VISITOR_INFO
    if (this->exportVisitorCheckbox->isOn())
    {
        Visitor::stopDumpVisitor();
        m_dumpVisitorStream.flush();
        //Creation of the graph
        std::string xmlDoc=m_dumpVisitorStream.str();
        handleTraceVisitor->load(xmlDoc);
        m_dumpVisitorStream.str("");
    }
#endif
}
//*****************************************************************************************
//called at each step of the rendering

void RealGUI::step()
{
    Node* root = viewer->getScene();
    if ( root == NULL ) return;

    startDumpVisitor();

    {
        if ( viewer->ready() ) return;



        //root->setLogTime(true);
        //T=T+DT
        SReal dt=root->getDt();
        simulation::getSimulation()->animate ( root, dt );
        simulation::getSimulation()->updateVisual( root );

        if ( m_dumpState )
            simulation::getSimulation()->dumpState ( root, *m_dumpStateStream );
        if ( m_exportGnuplot )
            simulation::getSimulation()->exportGnuplot ( root, root->getTime() );

        viewer->wait();

        eventNewStep();
        eventNewTime();
    }


    if ( _animationOBJ )
    {
#ifdef CAPTURE_PERIOD
        static int counter = 0;
        if ( ( counter++ % CAPTURE_PERIOD ) ==0 )
#endif
        {
            exportOBJ ( getScene(), false );

            ++_animationOBJcounter;
        }
    }

    stopDumpVisitor();
    emit newStep();
    if ( simulation::getSimulation()->getPaused() )
        startButton->setOn ( false );
}

//*****************************************************************************************
// Update sofa Simulation with the time step

void RealGUI::eventNewStep()
{
    static ctime_t beginTime[10];
    static const ctime_t timeTicks = CTime::getRefTicksPerSec();
    Node* root = getScene();
    if ( frameCounter==0 )
    {
        ctime_t t = CTime::getRefTime();
        for ( int i=0; i<10; i++ )
            beginTime[i] = t;
    }
    ++frameCounter;
    if ( ( frameCounter%10 ) == 0 )
    {
        ctime_t curtime = CTime::getRefTime();
        int i = ( ( frameCounter/10 ) %10 );
        double fps = ( ( double ) timeTicks / ( curtime - beginTime[i] ) ) * ( frameCounter<100?frameCounter:100 );

#ifndef SOFA_GUI_QT_NO_RECORDER
        if (recorder)
            recorder->setFPS(fps);
#else
        if (fpsLabel)
        {
            char buf[100];
            sprintf ( buf, "%.1f FPS", fps );
            fpsLabel->setText ( buf );
        }
#endif
        beginTime[i] = curtime;
    }

    if ( m_displayComputationTime && ( frameCounter%100 ) == 0 && root!=NULL )
    {

        std::cout << "========== ITERATION " << frameCounter << " ==========\n";
        const sofa::simulation::Node::NodeTimer& total = root->getTotalTime();
        const std::map<std::string, sofa::simulation::Node::NodeTimer>& times = root->getVisitorTime();
        const std::map<std::string, std::map<sofa::core::objectmodel::BaseObject*, sofa::simulation::Node::ObjectTimer> >& objtimes = root->getObjectTime();
        const double fact = 1000000.0 / ( 100*root->getTimeFreq() );
        for ( std::map<std::string, sofa::simulation::Node::NodeTimer>::const_iterator it = times.begin(); it != times.end(); ++it )
        {
            std::cout << "TIME "<<it->first<<": " << ( ( int ) ( fact*it->second.tTree+0.5 ) ) *0.001 << " ms (" << ( 1000*it->second.tTree/total.tTree ) *0.1 << " %).\n";
            std::map<std::string, std::map<sofa::core::objectmodel::BaseObject*, sofa::simulation::Node::ObjectTimer> >::const_iterator it1 = objtimes.find ( it->first );
            if ( it1 != objtimes.end() )
            {
                for ( std::map<sofa::core::objectmodel::BaseObject*, sofa::simulation::Node::ObjectTimer>::const_iterator it2 = it1->second.begin(); it2 != it1->second.end(); ++it2 )
                {
                    std::cout << "  "<< sofa::helper::gettypename ( typeid ( * ( it2->first ) ) ) <<" "<< it2->first->getName() <<": "
                            << ( ( int ) ( fact*it2->second.tObject+0.5 ) ) *0.001 << " ms (" << ( 1000*it2->second.tObject/it->second.tTree ) *0.1 << " %).\n";
                }
            }
        }
        for ( std::map<std::string, std::map<sofa::core::objectmodel::BaseObject*, sofa::simulation::Node::ObjectTimer> >::const_iterator it = objtimes.begin(); it != objtimes.end(); ++it )
        {
            if ( times.count ( it->first ) >0 ) continue;
            ctime_t ttotal = 0;
            for ( std::map<sofa::core::objectmodel::BaseObject*, sofa::simulation::Node::ObjectTimer>::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2 )
                ttotal += it2->second.tObject;
            std::cout << "TIME "<<it->first<<": " << ( ( int ) ( fact*ttotal+0.5 ) ) *0.001 << " ms (" << ( 1000*ttotal/total.tTree ) *0.1 << " %).\n";
            if ( ttotal > 0 )
                for ( std::map<sofa::core::objectmodel::BaseObject*, sofa::simulation::Node::ObjectTimer>::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2 )
                {
                    std::cout << "  "<< sofa::helper::gettypename ( typeid ( * ( it2->first ) ) ) <<" "<< it2->first->getName() <<": "
                            << ( ( int ) ( fact*it2->second.tObject+0.5 ) ) *0.001 << " ms (" << ( 1000*it2->second.tObject/ttotal ) *0.1 << " %).\n";
                }
        }
        std::cout << "TOTAL TIME: " << ( ( int ) ( fact*total.tTree+0.5 ) ) *0.001 << " ms (" << ( ( int ) ( 100/ ( fact*total.tTree*0.000001 ) +0.5 ) ) *0.01 << " FPS).\n";
        root->resetTime();

    }
}

void RealGUI::currentTabChanged ( QWidget* widget )
{
    if ( widget == currentTab ) return;
    if ( currentTab == NULL )
    {
        currentTab = widget;
    }
    if ( widget == TabGraph )
    {
        simulationGraph->Unfreeze( );
    }
    else if ( currentTab == TabGraph )
    {
        simulationGraph->Freeze();
    }

    else if (widget == TabStats)
        statWidget->CreateStats(dynamic_cast<Node*>(simulation::getSimulation()->getContext()));

    currentTab = widget;
}



void RealGUI::eventNewTime()
{
#ifndef SOFA_GUI_QT_NO_RECORDER
    if (recorder)
        recorder->UpdateTime();
#else
    Node* root = getScene();
    if (root && timeLabel)
    {
        double time = root->getTime();
        char buf[100];
        sprintf ( buf, "Time: %.3g s", time );
        timeLabel->setText ( buf );
    }
#endif
}




//*****************************************************************************************
// Set the time between each iteration of the Sofa Simulation

void RealGUI::setDt ( double value )
{
    Node* root = getScene();
    if ( value > 0.0 )
    {

        if ( root )
            root->getContext()->setDt ( value );
    }
}

void RealGUI::setDt ( const QString& value )
{
    setDt ( value.toDouble() );
}


//*****************************************************************************************
// Reset the simulation to t=0
void RealGUI::resetScene()
{
    Node* root = getScene();
    startDumpVisitor();
    emit ( newScene() );
    //Reset the scene
    if ( root )
    {
        root->setTime(0.);
        eventNewTime();
        simulation::getSimulation()->reset ( root );

        UpdateSimulationContextVisitor(sofa::core::ExecParams::defaultInstance()).execute(root);

        emit newStep();
    }

    viewer->getPickHandler()->reset();
    stopDumpVisitor();

}

//*****************************************************************************************
//
void RealGUI::displayComputationTime ( bool value )
{
    Node* root = getScene();
    m_displayComputationTime = value;
    if ( root )
    {
        root->setLogTime ( m_displayComputationTime );
    }
}



//*****************************************************************************************
//
void RealGUI::setExportGnuplot ( bool exp )
{
    Node* root = getScene();
    m_exportGnuplot = exp;
    if ( m_exportGnuplot && root )
    {
        simulation::getSimulation()->initGnuplot ( root );
        simulation::getSimulation()->exportGnuplot ( root, root->getTime() );
    }
}

//*****************************************************************************************
//
#ifdef SOFA_DUMP_VISITOR_INFO
void RealGUI::setExportVisitor ( bool exp )
#else
void RealGUI::setExportVisitor ( bool /*exp*/ )
#endif
{
#ifdef SOFA_DUMP_VISITOR_INFO
    if (exp)
    {
        windowTraceVisitor->show();
        handleTraceVisitor->clear();
    }
    else
    {
        windowTraceVisitor->hide();
    }
#endif
}

//*****************************************************************************************
//
void RealGUI::dumpState ( bool value )
{
    m_dumpState = value;
    if ( m_dumpState )
    {
        m_dumpStateStream = new std::ofstream ( "dumpState.data" );
    }
    else if ( m_dumpStateStream!=NULL )
    {
        delete m_dumpStateStream;
        m_dumpStateStream = 0;
    }
}



//*****************************************************************************************
//
void RealGUI::exportOBJ (simulation::Node* root,  bool exportMTL )
{
    if ( !root ) return;
    std::string sceneFileName(this->windowFilePath ().ascii());
    std::ostringstream ofilename;
    if ( !sceneFileName.empty() )
    {
        const char* begin = sceneFileName.c_str();
        const char* end = strrchr ( begin,'.' );
        if ( !end ) end = begin + sceneFileName.length();
        ofilename << std::string ( begin, end );
    }
    else
        ofilename << "scene";

    std::stringstream oss;
    oss.width ( 5 );
    oss.fill ( '0' );
    oss << _animationOBJcounter;

    ofilename << '_' << ( oss.str().c_str() );
    ofilename << ".obj";
    std::string filename = ofilename.str();
    std::cout << "Exporting OBJ Scene "<<filename<<std::endl;
    simulation::getSimulation()->exportOBJ ( root, filename.c_str(),exportMTL );
}


//*****************************************************************************************


void RealGUI::keyPressEvent ( QKeyEvent * e )
{
#ifdef SOFA_GUI_INTERACTION
    if(m_interactionActived)
    {
        if ((e->key()==Qt::Key_Escape) || (e->modifiers() && (e->key()=='I')))
        {
            this->interactionGUI (false);
        }
        else
        {
            sofa::core::objectmodel::KeypressedEvent keyEvent(e->key());
            Node* groot = viewer->getScene();
            if (groot) groot->propagateEvent(core::ExecParams::defaultInstance(), &keyEvent);
        }
        return;
    }
#endif


#ifdef SOFA_QT4
    if (e->modifiers()) return;
#else
    if (e->state() & (Qt::KeyButtonMask)) return;
#endif
    // ignore if there are modifiers (i.e. CTRL of SHIFT)
    switch ( e->key() )
    {

    case Qt::Key_O:
        // --- export to OBJ
    {
        exportOBJ ( getScene() );

        break;
    }
    case Qt::Key_P:
        // --- export to a succession of OBJ to make a video
    {
        _animationOBJ = !_animationOBJ;
        _animationOBJcounter = 0;
        break;
    }
    case Qt::Key_Escape:
    {
        emit(quit());
        break;
    }
    default:
    {
        e->ignore();
        break;
    }
    }


}

#ifdef SOFA_GUI_INTERACTION
bool RealGUI::eventFilter(QObject * /*obj*/, QEvent *e)
{
    if (m_interactionActived)
    {
        if (e->type() == QEvent::Wheel)
        {
            this->wheelEvent((QWheelEvent*)e);
            return true;
        }
    }
    return false; // pass other events
}

void RealGUI::mouseMoveEvent(QMouseEvent * /*e*/)
{
    if (m_interactionActived)
    {
        QPoint p = mapToGlobal(QPoint((this->width()+2)/2,(this->height()+2)/2));
        QPoint c = QCursor::pos();
        sofa::core::objectmodel::MouseEvent mouseEvent(sofa::core::objectmodel::MouseEvent::Move,c.x() - p.x(),c.y() - p.y());
        QCursor::setPos(p);
        Node* groot = viewer->getScene();
        if (groot)groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
        return;
    }
}

void RealGUI::wheelEvent(QWheelEvent* e)
{
    if(m_interactionActived)
    {
        sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::Wheel,e->delta());
        Node* groot = viewer->getScene();
        if (groot)groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
        e->accept();
        return;
    }
}

void RealGUI::mousePressEvent(QMouseEvent * e)
{
    if(m_interactionActived)
    {
        if (e->type() == QEvent::MouseButtonPress)
        {
            if (e->button() == Qt::LeftButton)
            {
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::LeftPressed);
                Node* groot = viewer->getScene();
                if (groot)groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            else if (e->button() == Qt::RightButton)
            {
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::RightPressed);
                Node* groot = viewer->getScene();
                if (groot)groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            else if (e->button() == Qt::MidButton)
            {
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::MiddlePressed);
                Node* groot = viewer->getScene();
                if (groot)groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            return;
        }
    }
}

void RealGUI::mouseReleaseEvent(QMouseEvent * e)
{
    if(m_interactionActived)
    {
        if (e->type() == QEvent::MouseButtonRelease)
        {
            if (e->button() == Qt::LeftButton)
            {
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::LeftReleased);
                Node* groot = viewer->getScene();
                if (groot)groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            else if (e->button() == Qt::RightButton)
            {
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::RightReleased);
                Node* groot = viewer->getScene();
                if (groot)groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            else if (e->button() == Qt::MidButton)
            {
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::MiddleReleased);
                Node* groot = viewer->getScene();
                if (groot)groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            return;
        }
    }
}

void RealGUI::keyReleaseEvent(QKeyEvent * e)
{
    if(m_interactionActived)
    {
        sofa::core::objectmodel::KeyreleasedEvent keyEvent(e->key());
        Node* groot = viewer->getScene();
        if (groot) groot->propagateEvent(core::ExecParams::defaultInstance(), &keyEvent);
        return;
    }
}
#endif

void RealGUI::dropEvent(QDropEvent* event)
{
    QString text;
    Q3TextDrag::decode(event, text);
    std::string filename(text.ascii());
#ifdef WIN32
    filename = filename.substr(8); //removing file:///
#else
    filename = filename.substr(7); //removing file://
#endif

    if (filename[filename.size()-1] == '\n')
    {
        filename.resize(filename.size()-1);
        filename[filename.size()-1]='\0';
    }

    if (filename.rfind(".simu") != std::string::npos) fileOpenSimu(filename);
    else fileOpen(filename);
}

#ifdef SOFA_QT4
void RealGUI::changeHtmlPage( const QUrl& u)
{
    std::string path=u.path().ascii();
#ifdef WIN32
    path = path.substr(1);
#endif
#else
void RealGUI::changeHtmlPage( const QString& u)
{
    std::string path=u.ascii();
#endif
    path  = sofa::helper::system::DataRepository.getFile(path);
    std::string extension=sofa::helper::system::SetDirectory::GetExtension(path.c_str());
    if (extension == "xml" || extension == "scn") fileOpen(path);
}

void RealGUI::updateViewerParameters()
{
    gui->viewer->getQWidget()->update();
}

void RealGUI::updateBackgroundColour()
{
    viewer->setBackgroundColour(atof(background[0]->text().ascii()),atof(background[1]->text().ascii()),atof(background[2]->text().ascii()));
    updateViewerParameters();
}

void RealGUI::updateBackgroundImage()
{
    viewer->setBackgroundImage( backgroundImage->text().ascii() );
    updateViewerParameters();
}

void RealGUI::showhideElements()
{
    displayFlag->updateDataValue();
    viewer->getQWidget()->update();
}

void RealGUI::Update()
{
    viewer->getQWidget()->update();
    statWidget->CreateStats(dynamic_cast<Node*>(simulation::getSimulation()->getContext()));
}

void RealGUI::NewRootNode(sofa::simulation::Node* root, const char* path)
{
    if(path != NULL)
    {
        viewer->setScene ( root, viewer->getSceneFileName().c_str() );
    }
    else
    {
        viewer->setScene(root , path);
    }
    viewer->resetView();
    viewer->getQWidget()->update();
    statWidget->CreateStats(root);
}

void RealGUI::LockAnimation(bool value)
{
    if(value)
    {
        animationState = startButton->isOn();
        playpauseGUI(false);
    }
    else
    {
        playpauseGUI(animationState);
    }
}

void RealGUI::ActivateNode(sofa::simulation::Node* node, bool activate)
{
    QSofaListView* sofalistview = (QSofaListView*)sender();

    if (activate) node->setActive(true);
    simulation::DeactivationVisitor v(sofa::core::ExecParams::defaultInstance(), activate);
    node->executeVisitor(&v);

    using core::objectmodel::BaseNode;
    std::list< BaseNode* > nodeToProcess;
    nodeToProcess.push_front((BaseNode*)node);


    std::list< BaseNode* > nodeToChange;
    //Breadth First approach to activate all the nodes
    while (!nodeToProcess.empty())
    {
        //We take the first element of the list
        Node* n= (Node*)nodeToProcess.front();
        nodeToProcess.pop_front();

        nodeToChange.push_front(n);

        //We add to the list of node to process all its children
        std::copy(n->child.begin(), n->child.end(), std::back_inserter(nodeToProcess));

    }
    {
        ActivationFunctor activator( activate, sofalistview->getListener() );
        std::for_each(nodeToChange.begin(),nodeToChange.end(),activator);
    }

    nodeToChange.clear();

    Update();

    if ( sofalistview == simulationGraph && activate )
    {
        if ( node == simulation::getSimulation()->getContext() )
        {
            simulation::getSimulation()->init(node);

        }
        else
        {
            simulation::getSimulation()->initNode(node);
        }
    }
}
} // namespace qt

} // namespace gui

} // namespace sofa
