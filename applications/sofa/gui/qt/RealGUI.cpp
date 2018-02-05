/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#include "RealGUI.h"

#ifdef SOFA_PML
#  include <sofa/filemanager/sofapml/PMLReader.h>
#  include <sofa/filemanager/sofapml/LMLReader.h>
#endif

#ifndef SOFA_GUI_QT_NO_RECORDER
#include "sofa/gui/qt/QSofaRecorder.h"
#endif

#ifdef SOFA_DUMP_VISITOR_INFO
#include "WindowVisitor.h"
#include "GraphVisitor.h"
#endif

#ifdef SOFA_PML
#include <sofa/simulation/Node.h>
#endif

#include "QSofaListView.h"
#include "QDisplayPropertyWidget.h"
#include "FileManagement.h"
#include "DisplayFlagsDataWidget.h"
#include "SofaPluginManager.h"
#include "SofaMouseManager.h"
#include "SofaVideoRecorderManager.h"
#include "WDoubleLineEdit.h"
#include "QSofaStatWidget.h"
#include "viewer/SofaViewer.h"

#include <sofa/gui/BaseViewer.h>
#include <SofaSimulationCommon/xml/XML.h>
#include <sofa/simulation/DeactivatedNodeVisitor.h>
#include <SofaBaseVisual/VisualStyle.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/system/SetDirectory.h>
using sofa::helper::system::SetDirectory ;

#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::FileSystem ;

#include <sofa/helper/system/FileRepository.h>
using sofa::helper::system::DataRepository ;

#include <sofa/simulation/SceneLoaderFactory.h>
using sofa::simulation::SceneLoaderFactory ;

#include <sofa/simulation/ExportGnuplotVisitor.h>

#include <QHBoxLayout>
#include <QApplication>
#include <QTimer>
#include <QTextBrowser>
#include <QWidget>
#include <QStackedWidget>
#include <QTreeWidget>
#include <QTextEdit>
#include <QAction>
#include <QMessageBox>
#include <QDockWidget>
#include <QDesktopWidget>
#include <QStatusBar>
#include <QDockWidget>
#include <QSettings>
#include <QMimeData>
#include <QCompleter>
#include <QDesktopServices>

#   ifdef SOFA_GUI_INTERACTION
#    include <QCursor>
#   endif

#   ifdef SOFA_GUI_INTERACTION
#    include <qcursor.h>
#   endif

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <ctime>

#include <sofa/core/objectmodel/IdleEvent.h>
using sofa::core::objectmodel::IdleEvent ;

#include <sofa/helper/system/FileMonitor.h>
using sofa::helper::system::FileMonitor ;

#include <SofaGraphComponent/SceneCheckerVisitor.h>
using sofa::simulation::SceneCheckerVisitor ;

#include <SofaGraphComponent/SceneCheckAPIChanges.h>
using sofa::simulation::scenecheckers::SceneCheckAPIChange ;

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory ;

#include "panels/QDocBrowser.h"
using sofa::gui::qt::DocBrowser ;

using sofa::core::ExecParams ;

namespace sofa
{

#ifdef SOFA_PML
using namespace filemanager::pml;
#endif

namespace gui
{

namespace qt
{


using sofa::core::objectmodel::BaseObject;
using namespace sofa::helper::system::thread;
using namespace sofa::simulation;
using namespace sofa::core::visual;

/// Custom QApplication class handling FileOpen events for MacOS
class QSOFAApplication : public QApplication
{
public:
    QSOFAApplication(int &argc, char ** argv)
        : QApplication(argc,argv)
    { }

#if QT_VERSION < 0x050000
    static inline QString translate(const char * context, const char * key, const char * disambiguation,
                                    QCoreApplication::Encoding encoding = QCoreApplication::UnicodeUTF8, int n = -1)
    { return QApplication::translate(context, key, disambiguation, encoding, n); }
#else
    static inline QString translate(const char * context, const char * key,
                                    const char * disambiguation = Q_NULLPTR, int n = -1)
    { return QApplication::translate(context, key, disambiguation, n); }
#endif

protected:
    bool event(QEvent *event)
    {
        switch (event->type())
        {
        case QEvent::FileOpen:
        {
            if(this->topLevelWidgets().count() < 1)
                return false;
            return true;
        }
        default:
            return QApplication::event(event);
        }
    }
};

RealGUI* gui = NULL;
QApplication* application = NULL;

const char* progname="";



class RealGUIFileListener : public sofa::helper::system::FileEventListener
{
public:
    RealGUIFileListener(RealGUI* realgui){
        m_realgui = realgui ;
    }
    virtual ~RealGUIFileListener(){}

    virtual void fileHasChanged(const std::string& filename) override
    {
        m_realgui->fileOpen(filename, false, true);
    }
    RealGUI* m_realgui ;
};


//======================= STATIC METHODS ========================= {

BaseGUI* RealGUI::CreateGUI ( const char* name, sofa::simulation::Node::SPtr root, const char* filename )
{
    CreateApplication();

    // create interface
    gui = new RealGUI ( name );
    if ( root )
    {
        gui->setScene ( root, filename );
        gui->setWindowFilePath(QString(filename));
    }

    InitApplication(gui);

    return gui;
}

//------------------------------------

void RealGUI::SetPixmap(std::string pixmap_filename, QPushButton* b)
{
    if ( DataRepository.findFile ( pixmap_filename ) )
        pixmap_filename = DataRepository.getFile ( pixmap_filename );

    b->setIcon(QIcon(QPixmap(QPixmap::fromImage(QImage(pixmap_filename.c_str())))));
}

//------------------------------------

void RealGUI::CreateApplication(int /*_argc*/, char** /*_argv*/)
{
    int  *argc = new int;
    char **argv=new char*[2];
    *argc = 1;
    argv[0] = strdup ( BaseGUI::GetProgramName() );
    argv[1]=NULL;
    application = new QSOFAApplication ( *argc,argv );

    //force locale to Standard C
    //(must be done immediatly after the QApplication has been created)
    QLocale locale(QLocale::C);
    QLocale::setDefault(locale);
}

//------------------------------------

void RealGUI::InitApplication( RealGUI* _gui)
{
    QString pathIcon=(DataRepository.getFirstPath() + std::string( "/icons/SOFA.png" )).c_str();
    application->setWindowIcon(QIcon(pathIcon));

#ifdef SOFA_GUI_NATIVE_MENU
    // Use the OS'native menu instead of the Qt one
    _gui->menubar->setNativeMenuBar(true);
#else
    // Use the qt menu instead of the native one in order to standardize the way the menu is showed on every OS
    _gui->menubar->setNativeMenuBar(false);
#endif

    // show the gui
    _gui->show(); // adding extra line in the console?
}
//======================= STATIC METHODS ========================= }




//======================= CONSTRUCTOR - DESTRUCTOR ========================= {
RealGUI::RealGUI ( const char* viewername)
    :
      #ifdef SOFA_GUI_INTERACTION
      interactionButton( NULL ),
      #endif

      #ifndef SOFA_GUI_QT_NO_RECORDER
      recorder(NULL),
      #else
      fpsLabel(NULL),
      timeLabel(NULL),
      #endif

      #ifdef SOFA_GUI_INTERACTION
      m_interactionActived(false),
      #endif

      #ifdef SOFA_PML
      pmlreader(NULL),
      lmlreader(NULL),
      #endif

      #ifdef SOFA_DUMP_VISITOR_INFO
      windowTraceVisitor(NULL),
      handleTraceVisitor(NULL),
      #endif

      simulationGraph(NULL),
      mCreateViewersOpt(true),
      mIsEmbeddedViewer(true),
      m_dumpState(false),
      m_dumpStateStream(NULL),
      m_exportGnuplot(false),
      _animationOBJ(false),
      _animationOBJcounter(0),
      m_displayComputationTime(false),
      m_fullScreen(false),
      mViewer(NULL),
      m_clockBeforeLastStep(0),
      propertyWidget(NULL),
      currentTab ( NULL ),
      statWidget(NULL),
      timerStep(NULL),
      backgroundImage(NULL),
      pluginManager_dialog(NULL),
      recentlyOpenedFilesManager(sofa::gui::BaseGUI::getConfigDirectoryPath() + "/runSofa.ini"),
      saveReloadFile(false),
      displayFlag(NULL),
      m_docbrowser(NULL),
      animationState(false),
      frameCounter(0),
      m_viewerMSAANbSampling(1)
{
    setupUi(this);

    parseOptions();

    createPluginManager();

    createRecentFilesMenu(); // configure Recently Opened Menu

    QDoubleValidator *dtValidator = new QDoubleValidator(dtEdit);
    dtValidator->setBottom(0.000000001);
    dtEdit->setValidator(dtValidator);

    timerStep = new QTimer(this);
    connect ( timerStep, SIGNAL ( timeout() ), this, SLOT ( step() ) );
    connect ( this, SIGNAL ( quit() ), this, SLOT ( fileExit() ) );
    connect ( startButton, SIGNAL ( toggled ( bool ) ), this , SLOT ( playpauseGUI ( bool ) ) );
    connect ( ResetSceneButton, SIGNAL ( clicked() ), this, SLOT ( resetScene() ) );
    connect ( dtEdit, SIGNAL ( textChanged ( const QString& ) ), this, SLOT ( setDt ( const QString& ) ) );
    connect ( realTimeCheckBox, SIGNAL ( stateChanged ( int ) ), this, SLOT ( updateDtEditState() ) );
    connect ( stepButton, SIGNAL ( clicked() ), this, SLOT ( step() ) );
    connect ( dumpStateCheckBox, SIGNAL ( toggled ( bool ) ), this, SLOT ( dumpState ( bool ) ) );
    connect ( displayComputationTimeCheckBox, SIGNAL ( toggled ( bool ) ), this, SLOT ( displayComputationTime ( bool ) ) );
    connect ( exportGnuplotFilesCheckbox, SIGNAL ( toggled ( bool ) ), this, SLOT ( setExportGnuplot ( bool ) ) );
    connect ( tabs, SIGNAL ( currentChanged ( int ) ), this, SLOT ( currentTabChanged ( int ) ) );

    /// We activate this timer only if the interactive mode is enabled (ie livecoding+mouse mouve event).
    if(m_enableInteraction){
        timerIdle = new QTimer(this);
        connect ( timerIdle, SIGNAL ( timeout() ), this, SLOT ( emitIdle() ) );
        timerIdle->start(50) ;
    }

    this->setDockOptions(QMainWindow::AnimatedDocks | QMainWindow::AllowTabbedDocks);
    dockWidget->setFeatures(QDockWidget::AllDockWidgetFeatures);
    dockWidget->setAllowedAreas(Qt::RightDockWidgetArea | Qt::LeftDockWidgetArea);

    connect(dockWidget, SIGNAL(dockLocationChanged(Qt::DockWidgetArea)), this, SLOT(toolsDockMoved()));

    // create a Dock Window to receive the Sofa Recorder
#ifndef SOFA_GUI_QT_NO_RECORDER
    QDockWidget *dockRecorder=new QDockWidget(this);
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
    TabStats->layout()->addWidget(statWidget);

    createSimulationGraph();

    //disable widget, can be bothersome with objects with a lot of data
    //createPropertyWidget();

    //viewer
    informationOnPickCallBack = InformationOnPickCallBack(this);

    viewerMap.clear();
    if (mCreateViewersOpt)
        createViewer(viewername, true);

    currentTabChanged ( tabs->currentIndex() );

    createBackgroundGUIInfos(); // add GUI for Background Informations

    createWindowVisitor();

    SofaMouseManager::getInstance()->hide();
    SofaVideoRecorderManager::getInstance()->hide();

    //Center the application
    const QRect screen = QApplication::desktop()->availableGeometry(QApplication::desktop()->primaryScreen());
    this->move(  ( screen.width()- this->width()  ) / 2 - 200,  ( screen.height() - this->height()) / 2 - 50  );

    tabs->removeTab(tabs->indexOf(TabVisualGraph));

#ifndef SOFA_GUI_QT_NO_RECORDER
    if (recorder)
        connect( recorder, SIGNAL( RecordSimulation(bool) ), startButton, SLOT( setChecked(bool) ) );
    if (recorder && getQtViewer())
        connect( recorder, SIGNAL( NewTime() ), getQtViewer()->getQWidget(), SLOT( update() ) );
#endif

#ifdef SOFA_GUI_INTERACTION
    interactionButton = new QPushButton(optionTabs);
    interactionButton->setObjectName(QString::fromUtf8("interactionButton"));
    interactionButton->setCheckable(true);
    interactionButton->setStyleSheet("background-color: cyan;");

    gridLayout->addWidget(interactionButton, 3, 0, 1, 1);
    gridLayout->removeWidget(screenshotButton);
    gridLayout->addWidget(screenshotButton, 3, 1, 1,1);

    interactionButton->setText(QSOFAApplication::translate("GUI", "&Interaction", 0));
    interactionButton->setShortcut(QSOFAApplication::translate("GUI", "Alt+i", 0));
#ifndef QT_NO_TOOLTIP
    interactionButton->setProperty("toolTip", QVariant(QSOFAApplication::translate("GUI", "Start interaction mode", 0)));
#endif

    connect ( interactionButton, SIGNAL ( toggled ( bool ) ), this , SLOT ( interactionGUI ( bool ) ) );

    m_interactionActived = false;

    if(mCreateViewersOpt)
        getQtViewer()->getQWidget()->installEventFilter(this);
#endif

    m_docbrowser = new DocBrowser(this) ;
    /// Signal to the realGUI that the visibility has changed (eg: to update the menu bar)
    connect(m_docbrowser, SIGNAL(visibilityChanged(bool)), this, SLOT(docBrowserVisibilityChanged(bool)));

    m_filelistener = new RealGUIFileListener(this) ;
}

//------------------------------------

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
        delete displayFlag;

#ifdef SOFA_DUMP_VISITOR_INFO
    delete windowTraceVisitor;
    delete handleTraceVisitor;
#endif

    removeViewer();

    FileMonitor::removeListener(m_filelistener) ;
    delete m_filelistener ;
}
//======================= CONSTRUCTOR - DESTRUCTOR ========================= }



//======================= OPTIONS DEFINITIONS ========================= {
#ifdef SOFA_DUMP_VISITOR_INFO
void RealGUI::setTraceVisitors(bool b)
{
    exportVisitorCheckbox->setChecked(b);
}
#endif


//------------------------------------

#ifdef SOFA_GUI_INTERACTION
void RealGUI::mouseMoveEvent(QMouseEvent * /*e*/)
{
    if (m_interactionActived)
    {
        QPoint p = mapToGlobal(QPoint((this->width()+2)/2,(this->height()+2)/2));
        QPoint c = QCursor::pos();
        sofa::core::objectmodel::MouseEvent mouseEvent(sofa::core::objectmodel::MouseEvent::Move,c.x() - p.x(),c.y() - p.y());
        QCursor::setPos(p);
        Node* groot = mViewer->getScene();
        if (groot)
            groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
        return;
    }
}

//------------------------------------

void RealGUI::wheelEvent(QWheelEvent* e)
{
    if(m_interactionActived)
    {
        sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::Wheel,e->delta());
        Node* groot = mViewer->getScene();
        if (groot)
            groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
        e->accept();
        return;
    }
}

//------------------------------------

void RealGUI::mousePressEvent(QMouseEvent * e)
{
    if(m_interactionActived)
    {
        if (e->type() == QEvent::MouseButtonPress)
        {
            if (e->button() == Qt::LeftButton)
            {
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::LeftPressed);
                Node* groot = mViewer->getScene();
                if (groot)
                    groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            else if (e->button() == Qt::RightButton)
            {
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::RightPressed);
                Node* groot = mViewer->getScene();
                if (groot)
                    groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            else if (e->button() == Qt::MidButton)
            {
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::MiddlePressed);
                Node* groot = mViewer->getScene();
                if (groot)
                    groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            return;
        }
    }
}

//------------------------------------

void RealGUI::mouseReleaseEvent(QMouseEvent * e)
{
    if(m_interactionActived)
    {
        if (e->type() == QEvent::MouseButtonRelease)
        {
            if (e->button() == Qt::LeftButton)
            {
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::LeftReleased);
                Node* groot = mViewer->getScene();
                if (groot)
                    groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            else if (e->button() == Qt::RightButton)
            {
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::RightReleased);
                Node* groot = mViewer->getScene();
                if (groot)
                    groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            else if (e->button() == Qt::MidButton)
            {
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::MiddleReleased);
                Node* groot = mViewer->getScene();
                if (groot)
                    groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            return;
        }
    }
}

//------------------------------------

void RealGUI::keyReleaseEvent(QKeyEvent * e)
{
    if(m_interactionActived)
    {
        sofa::core::objectmodel::KeyreleasedEvent keyEvent(e->key());
        Node* groot = mViewer->getScene();
        if (groot)
            groot->propagateEvent(core::ExecParams::defaultInstance(), &keyEvent);
        return;
    }
}

//------------------------------------

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
#endif

//------------------------------------

#ifdef SOFA_PML
void RealGUI::pmlOpen ( const char* filename, bool /*resetView*/ )
{
    std::string scene = "PML/default.scn";
    if ( !DataRepository.findFile ( scene ) )
    {
        msg_info("RealGUI") << "File '" << scene << "' not found " ;
        return;
    }
    this->unloadScene();
    mSimulation = dynamic_cast< Node *> (simulation::getSimulation()->load ( scene.c_str() ));
    getSimulation()->init(mSimulation);
    if ( mSimulation )
    {
        if ( !pmlreader ) pmlreader = new PMLReader;
        pmlreader->BuildStructure ( filename, mSimulation );
        setScene ( mSimulation, filename );
        this->setWindowFilePath(filename); //.c_str());
    }
}

//------------------------------------

//lmlOpen
void RealGUI::lmlOpen ( const char* filename )
{
    if ( pmlreader )
    {
        Node* root;
        if ( lmlreader != NULL ) delete lmlreader;
        lmlreader = new LMLReader; std::cout <<"New lml reader\n";
        lmlreader->BuildStructure ( filename, pmlreader );
        root = getScene();
        simulation::getSimulation()->init ( root );
    }
    else
        msg_info()<<"You must load the pml file before the lml file"<<endl;
}
#endif

//------------------------------------


//======================= OPTIONS DEFINITIONS ========================= }


//======================= METHODS ========================= {

void RealGUI::docBrowserVisibilityChanged(bool visibility)
{
    if(visibility)
        helpShowDocBrowser->setText("Hide doc browser");
    else
        helpShowDocBrowser->setText("Show doc browser");
}

void RealGUI::stepMainLoop () {
    application->processEvents();
}

int RealGUI::mainLoop()
{
    int retcode;
    if (windowFilePath().isNull())
    {
        retcode = application->exec();
    }
    else
    {
        const std::string &filename=windowFilePath().toStdString();
        const std::string &extension=SetDirectory::GetExtension(filename.c_str());
        if (extension == "simu") fileOpenSimu(filename);
        retcode = application->exec();
    }
    return exitApplication(retcode);
}

//------------------------------------

int RealGUI::closeGUI()
{
    QSettings settings;
    settings.beginGroup("viewer");
    settings.setValue("screenNumber", QApplication::desktop()->screenNumber(this));
    settings.endGroup();

    delete this;
    return 0;
}

//------------------------------------

sofa::simulation::Node* RealGUI::currentSimulation()
{
    return mSimulation.get();
}

//------------------------------------


void RealGUI::fileOpen ( std::string filename, bool temporaryFile, bool reload )
{
    const std::string &extension=SetDirectory::GetExtension(filename.c_str());
    if (extension == "simu")
    {
        return fileOpenSimu(filename);
    }

    startButton->setChecked(false);
    startDumpVisitor();
    update();

    //Hide all the dialogs to modify the graph
    emit ( newScene() );

    if ( DataRepository.findFile (filename) )
        filename = DataRepository.getFile ( filename );
    else
        return;

    sofa::simulation::xml::numDefault = 0;

    if( currentSimulation() ) this->unloadScene();
    mSimulation = simulation::getSimulation()->load ( filename.c_str() );
    simulation::getSimulation()->init ( mSimulation.get() );
    if ( mSimulation == NULL )
    {
        msg_warning("RealGUI")<<"Failed to load "<<filename.c_str();
        return;
    }
    if(reload)
        setSceneWithoutMonitor(mSimulation, filename.c_str(), temporaryFile);
    else
        setScene(mSimulation, filename.c_str(), temporaryFile);

    configureGUI(mSimulation.get());

    this->setWindowFilePath(filename.c_str());
    setExportGnuplot(exportGnuplotFilesCheckbox->isChecked());
    stopDumpVisitor();

    /// We want to warn user that there is component that are implemented in specific plugin
    /// and that there is no RequiredPlugin in their scene.
    /// But we don't want that to happen each reload in interactive mode.
    if(reload)
    {
        SceneCheckerVisitor checker(ExecParams::defaultInstance()) ;
        checker.addCheck(simulation::SceneCheckAPIChange::newSPtr());
        checker.addCheck(simulation::SceneCheckDuplicatedName::newSPtr());
        checker.addCheck(simulation::SceneCheckMissingRequiredPlugin::newSPtr());
        checker.validate(mSimulation.get()) ;
    }
}


//------------------------------------

void RealGUI::emitIdle()
{
    // Update all the registered monitor.
    FileMonitor::updates(0) ;

    IdleEvent hb;
    Node* groot = mViewer->getScene();
    if (groot)
    {
        groot->propagateEvent(core::ExecParams::defaultInstance(), &hb);
    }

    if(isEmbeddedViewer())
        getQtViewer()->getQWidget()->update();;
}

/// This open popup the file selection windows.
void RealGUI::popupOpenFileSelector()
{
    std::string filename(this->windowFilePath().toStdString());

    // build the filter with the SceneLoaderFactory
    std::string filter, allKnownFilters = "All known (";
    SceneLoaderFactory::SceneLoaderList* loaders = SceneLoaderFactory::getInstance()->getEntries();
    for (SceneLoaderFactory::SceneLoaderList::iterator it=loaders->begin(); it!=loaders->end(); ++it)
    {
        if (it!=loaders->begin()) filter +=";;";
        filter += (*it)->getFileTypeDesc();
        filter += " (";
        SceneLoader::ExtensionList extensions;
        (*it)->getExtensionList(&extensions);
        for (SceneLoader::ExtensionList::iterator itExt=extensions.begin(); itExt!=extensions.end(); ++itExt)
        {
            if (itExt!=extensions.begin()) filter +=" ";
            filter+="*.";
            filter+=(*itExt);

            allKnownFilters+="*."+(*itExt);
            if (*it!=loaders->back() || itExt!=extensions.end()-1) allKnownFilters += " ";
        }
        filter+=")";
    }
    allKnownFilters+=")";

#ifdef SOFA_PML
    //            "Scenes (*.scn *.xml);;Simulation (*.simu);;Php Scenes (*.pscn);;Pml Lml (*.pml *.lml);;All (*)",
    filter += ";;Simulation (*.simu);;Pml Lml (*.pml *.lml)";
#else
    //            "Scenes (*.scn *.xml);;Simulation (*.simu);;Php Scenes (*.pscn);;All (*)",
    filter += ";;Simulation (*.simu)";
#endif


    filter = allKnownFilters+";;"+filter+";;All (*)"; // the first filter is selected by default

    QString selectedFilter( tr(allKnownFilters.c_str()) ); // this does not select the desired filter

    QString s = getOpenFileName ( this, filename.empty() ?NULL:filename.c_str(),
                                  filter.c_str(),
                                  "open file dialog",  "Choose a file to open", &selectedFilter
                                  );
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
                fileOpenSimu(s.toStdString());
            else
                fileOpen (s.toStdString());
    }
}

//------------------------------------

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

        if ( DataRepository.findFile (filename) )
        {
            filename = DataRepository.getFile ( filename );
            simulation_name = s;
            std::string::size_type pointSimu = simulation_name.rfind(".simu");
            simulation_name.resize(pointSimu);
            fileOpen(filename.c_str());

            dtEdit->setText(QString(dT.c_str()));

#ifndef SOFA_GUI_QT_NO_RECORDER
            if (recorder)
                recorder->SetSimulation(currentSimulation(), initT, endT, writeName);
#endif
        }
    }
}

//------------------------------------

void RealGUI::setSceneWithoutMonitor (Node::SPtr root, const char* filename, bool temporaryFile)
{
    if (filename)
    {

        if (!temporaryFile)
            recentlyOpenedFilesManager.openFile(filename);
        saveReloadFile=temporaryFile;
        setTitle ( filename ) ;
        m_docbrowser->loadHtml( filename ) ;
    }

    if (root)
    {
        /// We want to warn user that there is component that are implemented in specific plugin
        /// and that there is no RequiredPlugin in their scene.
        SceneCheckerVisitor checker(ExecParams::defaultInstance()) ;
        checker.addCheck(simulation::SceneCheckAPIChange::newSPtr());
        checker.addCheck(simulation::SceneCheckDuplicatedName::newSPtr());
        checker.addCheck(simulation::SceneCheckMissingRequiredPlugin::newSPtr());
        checker.validate(root.get()) ;

        //Check the validity of the BBox
        const sofa::defaulttype::BoundingBox& nodeBBox = root->getContext()->f_bbox.getValue();
        if(nodeBBox.isNegligeable())
        {
            msg_error("RealGUI") << "Global Bounding Box seems invalid ; please implement updateBBox in your components "
                                    << "or force a value by adding the parameter bbox=\"minX minY minZ maxX maxY maxZ\" in your root node \n";
            msg_error("RealGUI") << "Your viewer settings (based on the bbox) are likely invalid.";
        }

        mSimulation = root;
        eventNewTime();
        startButton->setChecked(root->getContext()->getAnimate() );
        dtEdit->setText ( QString::number ( root->getDt() ) );
        simulationGraph->Clear(root.get());
        statWidget->CreateStats(root.get());

#ifndef SOFA_GUI_QT_NO_RECORDER
        if (recorder)
            recorder->Clear(root.get());
#endif

        getViewer()->setScene( root, filename );
        getViewer()->load();
        getViewer()->resetView();
        createDisplayFlags( root );

        if( isEmbeddedViewer() )
        {
            getQtViewer()->getQWidget()->setFocus();
            getQtViewer()->getQWidget()->show();
            getQtViewer()->getQWidget()->update();
        }

        resetScene();
    }
}

void RealGUI::setScene(Node::SPtr root, const char* filename, bool temporaryFile)
{
    if(m_enableInteraction &&  filename){
        FileMonitor::removeListener(m_filelistener) ;
        FileMonitor::addFile(filename, m_filelistener) ;
    }
    setSceneWithoutMonitor(root, filename, temporaryFile) ;
}

//------------------------------------

void RealGUI::unloadScene(bool _withViewer)
{
    if(_withViewer && getViewer())
        getViewer()->unload();

    simulation::getSimulation()->unload ( currentSimulation() );

    if(_withViewer && getViewer())
        getViewer()->setScene(NULL);
}

//------------------------------------

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
    this->setWindowTitle(QString(str.c_str()) );
#endif
    setWindowFilePath( windowTitle.c_str() );
}

//------------------------------------

void RealGUI::fileNew()
{
    std::string newScene("config/newScene.scn");
    if (DataRepository.findFile (newScene))
        fileOpen(DataRepository.getFile ( newScene ).c_str());
}

//------------------------------------

void RealGUI::fileSave()
{
    std::string filename(this->windowFilePath().toStdString());
    std::string message="You are about to overwrite your current scene: "  + filename + "\nAre you sure you want to do that ?";

    if ( QMessageBox::warning ( this, "Saving the Scene",message.c_str(), QMessageBox::Yes | QMessageBox::Default, QMessageBox::No ) != QMessageBox::Yes )
        return;

    fileSaveAs ( currentSimulation(), filename.c_str() );
}

//------------------------------------

void RealGUI::fileSaveAs ( Node *node, const char* filename )
{
    simulation::getSimulation()->exportGraph ( node, filename );
}

//------------------------------------

void RealGUI::fileReload()
{
    std::string filename(this->windowFilePath().toStdString());
    QString s = filename.c_str();

    if ( filename.empty() )
    {
        msg_error("RealGUI") << "Reload failed: no file loaded.";
        return;
    }

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
        fileOpenSimu(s.toStdString());
    else
        fileOpen ( s.toStdString(),saveReloadFile );
#endif
}

//------------------------------------

void RealGUI::fileExit()
{
    //Hide all opened ModifyObject windows
    emit ( newScene() );
    startButton->setChecked ( false);
    this->close();
}

void RealGUI::editRecordDirectory()
{
    std::string filename(this->windowFilePath().toStdString());
    std::string record_directory;
    QString s = getExistingDirectory ( this, filename.empty() ?NULL:filename.c_str(), "open directory dialog",  "Choose a directory" );
    if (s.length() > 0)
    {
        record_directory = s.toStdString();
        if (record_directory.at(record_directory.size()-1) != '/')
            record_directory+="/";
#ifndef SOFA_GUI_QT_NO_RECORDER
        if (recorder)
            recorder->SetRecordDirectory(record_directory);
#endif
    }
}

//------------------------------------

void RealGUI::editGnuplotDirectory()
{
    std::string filename(this->windowFilePath().toStdString());
    QString s = getExistingDirectory ( this, filename.empty() ?NULL:filename.c_str(), "open directory dialog",  "Choose a directory" );
    if (s.length() > 0)
    {
        gnuplot_directory = s.toStdString();
        if (gnuplot_directory.at(gnuplot_directory.size()-1) != '/')
            gnuplot_directory+="/";
        setExportGnuplot(exportGnuplotFilesCheckbox->isChecked());
    }
}

//------------------------------------

void RealGUI::showDocBrowser()
{
    m_docbrowser->flipVisibility() ;
}

void RealGUI::showPluginManager()
{
    pluginManager_dialog->updatePluginsListView();
    pluginManager_dialog->show();
}

//------------------------------------

void RealGUI::showMouseManager()
{
    SofaMouseManager::getInstance()->updateContent();
    SofaMouseManager::getInstance()->show();
}

//------------------------------------

void RealGUI::showVideoRecorderManager()
{
    SofaVideoRecorderManager::getInstance()->show();
}

//------------------------------------

void RealGUI::setViewerResolution ( int w, int h )
{
    if( isEmbeddedViewer() )
    {
        QSize winSize = size();
        QSize viewSize = ( getViewer() ) ? getQtViewer()->getQWidget()->size() : QSize(0,0);

        const QRect screen = QApplication::desktop()->availableGeometry(QApplication::desktop()->screenNumber(this));

        QSize newWinSize(winSize.width() - viewSize.width() + w, winSize.height() - viewSize.height() + h);
        if (newWinSize.width() > screen.width()) newWinSize.setWidth(screen.width()-20);
        if (newWinSize.height() > screen.height()) newWinSize.setHeight(screen.height()-20);

        this->resize(newWinSize);
    }
    else
    {
        getViewer()->setSizeW(w);
        getViewer()->setSizeH(h);
    }
}

//------------------------------------

void RealGUI::setFullScreen (bool enable)
{
    if (enable == m_fullScreen) return;

    if( isEmbeddedViewer() )
    {
        if (enable)
        {
            optionTabs->hide();
        }
        else if (m_fullScreen)
        {
            optionTabs->show();
        }

        if (enable)
        {
            std::cout << "Set Full Screen Mode" << std::endl;
            showFullScreen();
            m_fullScreen = true;

            dockWidget->setFloating(true);
            dockWidget->setVisible(false);
        }
        else
        {
            std::cout << "Set Windowed Mode" << std::endl;
            showNormal();
            m_fullScreen = false;
            dockWidget->setVisible(true);
            dockWidget->setFloating(false);
        }

        if (enable)
        {
            menuBar()->hide();
            statusBar()->hide();
#ifndef SOFA_GUI_QT_NO_RECORDER
            if (recorder) recorder->parentWidget()->hide();
            //statusBar()->addWidget( recorder->getFPSLabel());
            //statusBar()->addWidget( recorder->getTimeLabel());
#endif
        }
        else
        {
            menuBar()->show();
            statusBar()->show();
#ifndef SOFA_GUI_QT_NO_RECORDER
            recorder->parentWidget()->show();
#endif
        }
    }
    else
    {
        getViewer()->setFullScreen(enable);
    }
}

//------------------------------------

void RealGUI::setBackgroundColor(const defaulttype::RGBAColor& c)
{
    background[0]->setText(QString::number(c[0]));
    background[1]->setText(QString::number(c[1]));
    background[2]->setText(QString::number(c[2]));
    updateBackgroundColour();
}

//------------------------------------

void RealGUI::setBackgroundImage(const std::string& c)
{
    backgroundImage->setText(QString(c.c_str()));
    updateBackgroundImage();
}

//------------------------------------

void RealGUI::setViewerConfiguration(sofa::component::configurationsetting::ViewerSetting* viewerConf)
{
    const defaulttype::Vec<2,int> &res=viewerConf->resolution.getValue();

    if (viewerConf->fullscreen.getValue())
        setFullScreen();
    else
        setViewerResolution(res[0], res[1]);
    getViewer()->configure(viewerConf);
}

//------------------------------------

void RealGUI::setMouseButtonConfiguration(sofa::component::configurationsetting::MouseButtonSetting *button)
{
    SofaMouseManager::getInstance()->updateOperation(button);
}

//------------------------------------

void RealGUI::setDumpState(bool b)
{
    dumpStateCheckBox->setChecked(b);
}

//------------------------------------

void RealGUI::setLogTime(bool b)
{
    displayComputationTimeCheckBox->setChecked(b);
}

//------------------------------------

void RealGUI::setExportState(bool b)
{
    exportGnuplotFilesCheckbox->setChecked(b);
}

//------------------------------------

#ifndef SOFA_GUI_QT_NO_RECORDER
void RealGUI::setRecordPath(const std::string & path)
{
    if (recorder)
        recorder->SetRecordDirectory(path);
}
#else
void RealGUI::setRecordPath(const std::string&) {}
#endif

//------------------------------------

void RealGUI::setGnuplotPath(const std::string &path)
{
    gnuplot_directory = path;
}

//------------------------------------

void RealGUI::createViewer(const char* _viewerName, bool _updateViewerList/*=false*/)
{
    if(_updateViewerList)
    {
        this->updateViewerList();
        // the viewer with the key viewerName is already created
        if( mViewer != NULL && !viewerMap.begin()->first.compare( std::string(_viewerName) ) )
            return;
    }

    for (std::map< helper::SofaViewerFactory::Key, QAction*>::const_iterator iter_map = viewerMap.begin();
         iter_map != viewerMap.end() ; ++iter_map )
    {
        if( strcmp( iter_map->first.c_str(), _viewerName ) == 0 )
        {
            removeViewer();
            ViewerQtArgument viewerArg = ViewerQtArgument("viewer", this->widget, m_viewerMSAANbSampling);
            registerViewer( helper::SofaViewerFactory::CreateObject(iter_map->first, viewerArg) );
            //see to put on checkable
            iter_map->second->setChecked(true);
        }
        else
            iter_map->second->setChecked(false);
    }

    mGuiName = _viewerName;
    initViewer( getViewer() );
}

//------------------------------------

void RealGUI::registerViewer(BaseViewer* _viewer)
{
    // Change our viewer
    sofa::gui::BaseViewer* old = mViewer;
    mViewer = _viewer;
    if(mViewer != NULL)
        delete old;
    else
        msg_error("RealGUI")<<"when registerViewer, the viewer is NULL";
}

//------------------------------------

BaseViewer* RealGUI::getViewer()
{
    return mViewer!=NULL ? mViewer : NULL;
}

//------------------------------------

sofa::gui::qt::viewer::SofaViewer* RealGUI::getQtViewer()
{
    sofa::gui::qt::viewer::SofaViewer* qtViewer = dynamic_cast<sofa::gui::qt::viewer::SofaViewer*>(mViewer);
    return qtViewer ? qtViewer : NULL;
}

//------------------------------------

bool RealGUI::isEmbeddedViewer()
{
    return mIsEmbeddedViewer;
}

//------------------------------------

void RealGUI::removeViewer()
{
    if(mViewer != NULL)
    {
        if(isEmbeddedViewer())
        {
            getQtViewer()->removeViewerTab(tabs);
        }
        delete mViewer;
        mViewer = NULL;
    }
}

//------------------------------------

void RealGUI::dragEnterEvent( QDragEnterEvent* event)
{
    event->accept();
}

//------------------------------------

void RealGUI::dropEvent(QDropEvent* event)
{
    QString text;
    //Q3TextDrag::decode(event, text);
    if (event->mimeData()->hasText())
        text = event->mimeData()->text();
    std::string filename(text.toStdString());

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

    if (filename.rfind(".simu") != std::string::npos)
        fileOpenSimu(filename);
    else fileOpen(filename);
}

//------------------------------------

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
    m_fullScreen = false;
}

//------------------------------------

void RealGUI::createDisplayFlags(Node::SPtr root)
{
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
            displayFlag = new DisplayFlagsDataWidget(tabView,"displayFlagwidget",&visualStyle->displayFlags, true);
            displayFlag->createWidgets();
            displayFlag->updateWidgetValue();
            connect( displayFlag, SIGNAL( WidgetDirty(bool) ), this, SLOT(showhideElements() ));
            displayFlag->setMinimumSize(50,100);
            gridLayout1->addWidget(displayFlag,0,0);
            connect(tabs,SIGNAL(currentChanged(int)),displayFlag, SLOT( updateWidgetValue() ));
        }
    }
}

//------------------------------------

// Update sofa Simulation with the time step
void RealGUI::eventNewStep()
{
    static ctime_t beginTime[10];
    static const ctime_t timeTicks = CTime::getRefTicksPerSec();
    Node* root = currentSimulation();

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
        showFPS(fps);

        beginTime[i] = curtime;
    }

    if ( m_displayComputationTime && ( frameCounter%100 ) == 0 && root!=NULL )
    {
        /// @TODO: use AdvancedTimer in GUI to display time statistics
    }
}

void RealGUI::showFPS(double fps)
{
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
}

//------------------------------------

void RealGUI::eventNewTime()
{
#ifndef SOFA_GUI_QT_NO_RECORDER
    if (recorder)
        recorder->UpdateTime(currentSimulation());
#else
    Node* root = currentSimulation();
    if (root && timeLabel)
    {
        double time = root->getTime();
        char buf[100];
        sprintf ( buf, "Time: %.3g s", time );
        timeLabel->setText ( buf );
    }
#endif
}

//------------------------------------

void RealGUI::keyPressEvent ( QKeyEvent * e )
{
    sofa::gui::qt::viewer::SofaViewer* qtViewer = dynamic_cast<sofa::gui::qt::viewer::SofaViewer*>(getViewer());

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
            Node* groot = qtViewer->getScene();
            if (groot)
                groot->propagateEvent(core::ExecParams::defaultInstance(), &keyEvent);
        }
        return;
    }
#endif

    if (e->modifiers()) return;

    // ignore if there are modifiers (i.e. CTRL of SHIFT)
    switch ( e->key() )
    {
    case Qt::Key_O:
        // --- export to OBJ
    {
        exportOBJ ( currentSimulation() );
        break;
    }

    case Qt::Key_P:
        // --- export to a succession of OBJ to make a video
    {
        _animationOBJ = !_animationOBJ;
        _animationOBJcounter = 0;
        break;
    }
    case Qt::Key_Space:
    {
        playpauseGUI(!startButton->isChecked());
        break;
    }
    case Qt::Key_Backspace:
    {
        resetScene();
        break;
    }
    case Qt::Key_F11:
        // --- fullscreen mode
    {
        setFullScreen(!m_fullScreen);
        break;
    }
    case Qt::Key_Escape:
    {
        emit(quit());
        break;
    }
    default:
    {
        if (qtViewer)
            qtViewer->keyPressEvent(e);
        break;
    }
    }
}

//------------------------------------

void RealGUI::startDumpVisitor()
{
#ifdef SOFA_DUMP_VISITOR_INFO
    Node* root = currentSimulation();
    if (root && this->exportVisitorCheckbox->isChecked())
    {
        m_dumpVisitorStream.str("");
        Visitor::startDumpVisitor(&m_dumpVisitorStream, root->getTime());
    }
#endif
}

//------------------------------------

void RealGUI::stopDumpVisitor()
{
#ifdef SOFA_DUMP_VISITOR_INFO
    if (this->exportVisitorCheckbox->isChecked())
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

//------------------------------------

void RealGUI::initViewer(BaseViewer* _viewer)
{
    if(_viewer == NULL)
    {
        msg_error("RealGUI")<<"when initViewer, the viewer is NULL";
        return;
    }
    init(); //init data member from RealGUI for the viewer initialisation in the GUI

    // Is our viewer embedded or not ?
    sofa::gui::qt::viewer::SofaViewer* qtViewer = dynamic_cast<sofa::gui::qt::viewer::SofaViewer*>(_viewer);
    if( qtViewer == NULL )
    {
        isEmbeddedViewer(false);
        std::cout<<"initViewer: The viewer isn't embedded in the GUI"<<std::endl;
    }
    else
    {
        isEmbeddedViewer(true);
        this->mainWidgetLayout->addWidget(qtViewer->getQWidget());

        qtViewer->getQWidget()->setFocusPolicy ( Qt::StrongFocus );

        qtViewer->getQWidget()->setSizePolicy ( QSizePolicy ( ( QSizePolicy::Policy ) 7,
                                                              ( QSizePolicy::Policy ) 7
                                                              //, 100, 1,
                                                              //qtViewer->getQWidget()->sizePolicy().hasHeightForWidth() )
                                                              ));

        qtViewer->getQWidget()->setMinimumSize ( QSize ( 0, 0 ) );
        qtViewer->getQWidget()->setMouseTracking ( true );
        qtViewer->configureViewerTab(tabs);

        connect ( qtViewer->getQWidget(), SIGNAL ( resizeW ( int ) ), sizeW, SLOT ( setValue ( int ) ) );
        connect ( qtViewer->getQWidget(), SIGNAL ( resizeH ( int ) ), sizeH, SLOT ( setValue ( int ) ) );
        connect ( qtViewer->getQWidget(), SIGNAL ( quit (  ) ), this, SLOT ( fileExit (  ) ) );
        connect(simulationGraph, SIGNAL(focusChanged(sofa::core::objectmodel::BaseObject*)),
                qtViewer->getQWidget(), SLOT(fitObjectBBox(sofa::core::objectmodel::BaseObject*))
                );
        connect(simulationGraph, SIGNAL( focusChanged(sofa::core::objectmodel::BaseNode*) ),
                qtViewer->getQWidget(), SLOT( fitNodeBBox(sofa::core::objectmodel::BaseNode*) )
                );

        // setGUI
        textEdit1->setText ( qtViewer->helpString() );
        connect ( this, SIGNAL( newStep()), qtViewer->getQWidget(), SLOT( update()));

        qtViewer->getQWidget()->setFocus();
        qtViewer->getQWidget()->show();
        qtViewer->getQWidget()->update();

        qtViewer->getPickHandler()->addCallBack(&informationOnPickCallBack );
    }

    SofaMouseManager::getInstance()->setPickHandler(_viewer->getPickHandler());

    connect ( ResetViewButton, SIGNAL ( clicked() ), this, SLOT ( resetView() ) );
    connect ( SaveViewButton, SIGNAL ( clicked() ), this, SLOT ( saveView() ) );
    connect ( screenshotButton, SIGNAL ( clicked() ), this, SLOT ( screenshot() ) );
    connect ( sizeW, SIGNAL ( valueChanged ( int ) ), this, SLOT ( setSizeW ( int ) ) );
    connect ( sizeH, SIGNAL ( valueChanged ( int ) ), this, SLOT ( setSizeH ( int ) ) );

}

//------------------------------------

void RealGUI::parseOptions()
{
    po::variables_map vm = mArgumentParser->getVariableMap();
    if(vm.find("interactive") != vm.end())
        m_enableInteraction = vm["interactive"].as<bool>();
    if(vm.find("msaa") != vm.end())
        m_viewerMSAANbSampling = vm["msaa"].as<unsigned int>();

    if(m_enableInteraction)
        msg_warning("runSofa") << "you activated the interactive mode. This is currently an experimental feature "
                                  "that may change or be removed in the future. " ;
}

//------------------------------------

void RealGUI::createPluginManager()
{
    pluginManager_dialog = new SofaPluginManager();
    pluginManager_dialog->hide();
    this->connect( pluginManager_dialog, SIGNAL( libraryAdded() ),  this, SLOT( updateViewerList() ));
    this->connect( pluginManager_dialog, SIGNAL( libraryRemoved() ),  this, SLOT( updateViewerList() ));
}

//------------------------------------

void RealGUI::createRecentFilesMenu()
{
    fileMenu->removeAction(Action);

    //const int indexRecentlyOpened=fileMenu->count()-2;
    const int indexRecentlyOpened=fileMenu->actions().count();

    QMenu *recentMenu = recentlyOpenedFilesManager.createWidget(this);
    fileMenu->insertMenu(fileMenu->actions().at(indexRecentlyOpened-1),
                         recentMenu);
    connect(recentMenu, SIGNAL(triggered(QAction *)), this, SLOT(fileRecentlyOpened(QAction *)));
}

//------------------------------------

void RealGUI::createBackgroundGUIInfos()
{
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
        background[i]->setMaximumSize(50, 20);

        colourLayout->addWidget(background[i]);
        connect( background[i], SIGNAL( returnPressed() ), this, SLOT( updateBackgroundColour() ) );
    }

    QWidget *image = new QWidget(TabPage);
    QHBoxLayout *imageLayout = new QHBoxLayout(image);
    imageLayout->addWidget(new QLabel(QString("Image "),image));

    backgroundImage = new QLineEdit(image);
    backgroundImage->setText("backgroundImage");
    if ( getViewer() )
        backgroundImage->setText( QString(getViewer()->getBackgroundImage().c_str()) );
    else
        backgroundImage->setText( QString() );
    imageLayout->addWidget(backgroundImage);
    connect( backgroundImage, SIGNAL( returnPressed() ), this, SLOT( updateBackgroundImage() ) );

    ((QVBoxLayout*)(TabPage->layout()))->insertWidget(1,colour);
    ((QVBoxLayout*)(TabPage->layout()))->insertWidget(2,image);
}

//------------------------------------

void RealGUI::createSimulationGraph()
{
    simulationGraph = new QSofaListView(SIMULATION,TabGraph,"SimuGraph");
    TabGraph->layout()->addWidget(simulationGraph);

    connect ( ExportGraphButton, SIGNAL ( clicked() ), simulationGraph, SLOT ( Export() ) );
    connect(simulationGraph, SIGNAL( RootNodeChanged(sofa::simulation::Node*, const char*) ), this, SLOT ( NewRootNode(sofa::simulation::Node* , const char*) ) );
    connect(simulationGraph, SIGNAL( NodeRemoved() ), this, SLOT( Update() ) );
    connect(simulationGraph, SIGNAL( Lock(bool) ), this, SLOT( LockAnimation(bool) ) );
    connect(simulationGraph, SIGNAL( RequestSaving(sofa::simulation::Node*) ), this, SLOT( fileSaveAs(sofa::simulation::Node*) ) );
    connect(simulationGraph, SIGNAL( RequestExportOBJ(sofa::simulation::Node*, bool) ), this, SLOT( exportOBJ(sofa::simulation::Node*, bool) ) );
    connect(simulationGraph, SIGNAL( RequestActivation(sofa::simulation::Node*, bool) ), this, SLOT( ActivateNode(sofa::simulation::Node*, bool) ) );
    connect(simulationGraph, SIGNAL( RequestSleeping(sofa::simulation::Node*, bool) ), this, SLOT( setSleepingNode(sofa::simulation::Node*, bool) ) );
    connect(simulationGraph, SIGNAL( Updated() ), this, SLOT( redraw() ) );
    connect(simulationGraph, SIGNAL( NodeAdded() ), this, SLOT( Update() ) );
    connect(simulationGraph, SIGNAL( dataModified( QString ) ), this, SLOT( appendToDataLogFile(QString ) ) );
    connect(this, SIGNAL( newScene() ), simulationGraph, SLOT( CloseAllDialogs() ) );
    connect(this, SIGNAL( newStep() ), simulationGraph, SLOT( UpdateOpenedDialogs() ) );
}

void RealGUI::createPropertyWidget()
{
    ModifyObjectFlags modifyObjectFlags = ModifyObjectFlags();
    modifyObjectFlags.setFlagsForSofa();

    propertyWidget = new QDisplayPropertyWidget(modifyObjectFlags);

    QDockWidget *dockProperty=new QDockWidget(this);

    dockProperty->setAllowedAreas(Qt::RightDockWidgetArea | Qt::LeftDockWidgetArea);
    dockProperty->setMaximumSize(QSize(300,300));
    dockProperty->setWidget(propertyWidget);

    connect(dockProperty, SIGNAL(dockLocationChanged(QDockWidget::DockWidgetArea)),
            this, SLOT(propertyDockMoved(QDockWidget::DockWidgetArea)));
    simulationGraph->setPropertyWidget(propertyWidget);
}

//------------------------------------

void RealGUI::createWindowVisitor()
{
    pathDumpVisitor = SetDirectory::GetParentDir(DataRepository.getFirstPath().c_str()) + std::string( "/dumpVisitor.xml" );
#ifndef SOFA_DUMP_VISITOR_INFO
    //Remove option to see visitor trace
    this->exportVisitorCheckbox->hide();
#else
    //Main window containing a QListView only
    windowTraceVisitor = new WindowVisitor;
    windowTraceVisitor->graphView->setSortingEnabled(false);
    windowTraceVisitor->hide();
    connect ( exportVisitorCheckbox, SIGNAL ( toggled ( bool ) ), this, SLOT ( setExportVisitor ( bool ) ) );
    connect(windowTraceVisitor, SIGNAL(WindowVisitorClosed(bool)), this->exportVisitorCheckbox, SLOT(setChecked(bool)));
    handleTraceVisitor = new GraphVisitor(windowTraceVisitor);
#endif
}

void RealGUI::NewRootNode(sofa::simulation::Node* root, const char* path)
{
    std::string filename(this->windowFilePath().toStdString());
    std::string message="You are about to change the root node of the scene : "  + filename +
            "to the root node : " + std::string(path) +
            "\nThis implies that the simulation singleton has to change its root node.\nDo you want to proceed ?";
    if ( QMessageBox::warning ( this, "New root node: ",message.c_str(), QMessageBox::Yes | QMessageBox::Default, QMessageBox::No ) != QMessageBox::Yes )
        return;

    if(path != NULL && root != NULL)
    {
        getViewer()->setScene(root , path);
        getViewer()->load();
        getViewer()->resetView();
        if(isEmbeddedViewer())
            getQtViewer()->getQWidget()->update();;
        statWidget->CreateStats(root);
    }
}

//------------------------------------

void RealGUI::ActivateNode(sofa::simulation::Node* node, bool activate)
{
    QSofaListView* sofalistview = (QSofaListView*)sender();

    if (activate)
        node->setActive(true);
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
        for(Node::ChildIterator it = n->child.begin(), itend = n->child.end(); it != itend; ++it)
            nodeToProcess.push_back(it->get());
    }

    ActivationFunctor activator( activate, sofalistview->getListener() );
    std::for_each(nodeToChange.begin(),nodeToChange.end(),activator);
    nodeToChange.clear();
    Update();

    if ( sofalistview == simulationGraph && activate )
    {
        if ( node == currentSimulation() )
            simulation::getSimulation()->init(node);
        else
            simulation::getSimulation()->initNode(node);
    }
}

//------------------------------------

void RealGUI::setSleepingNode(sofa::simulation::Node* node, bool sleeping)
{
    node->setSleeping(sleeping);
}

//------------------------------------

void RealGUI::fileSaveAs(Node *node)
{
    if (node == NULL) node = currentSimulation();
    std::string filename(this->windowFilePath().toStdString());


    QString filter( "Scenes (");

    int nb=0;
    SceneLoaderFactory::SceneLoaderList* loaders = SceneLoaderFactory::getInstance()->getEntries();
    for (SceneLoaderFactory::SceneLoaderList::iterator it=loaders->begin(); it!=loaders->end(); it++)
    {
        SceneLoader::ExtensionList extensions;
        (*it)->getExtensionList(&extensions);
        for (SceneLoader::ExtensionList::iterator itExt=extensions.begin(); itExt!=extensions.end(); itExt++)
        {
            if( (*it)->canWriteFileExtension( itExt->c_str() ) )
            {
                if (nb!=0) filter +=" ";
                filter += "*.";
                filter += QString( itExt->c_str() );
                ++nb;
            }
        }
    }
#ifdef SOFA_PML
    filter += " *.pml";
#endif

    filter += ")";





    QString s = getSaveFileName ( this, filename.empty() ?NULL:filename.c_str(), filter, "save file dialog", "Choose where the scene will be saved" );
    if ( s.length() >0 )
#ifdef SOFA_PML
        if ( pmlreader && s.endsWith ( ".pml" ) )
            pmlreader->saveAsPML ( s );
        else
#endif
            fileSaveAs ( node,s.toStdString().c_str() );

}

//------------------------------------

void RealGUI::LockAnimation(bool value)
{
    if(value)
    {
        animationState = startButton->isChecked();
        playpauseGUI(false);
    }
    else
    {
        playpauseGUI(animationState);
    }
}

//------------------------------------

void RealGUI::fileRecentlyOpened(QAction *action)
{
    //fileOpen(recentlyOpenedFilesManager.getFilename((unsigned int)id));
    fileOpen(action->text().toStdString());
}

//------------------------------------

void RealGUI::playpauseGUI ( bool value )
{
    startButton->setChecked ( value );
    if ( currentSimulation() )
        currentSimulation()->getContext()->setAnimate ( value );
    if(value)
    {
        m_clockBeforeLastStep = 0;
        frameCounter=0;
        timerStep->start(0);
    }
    else
        timerStep->stop();
}

//------------------------------------

#ifdef SOFA_GUI_INTERACTION
void RealGUI::interactionGUI ( bool value )
{
    interactionButton->setChecked ( value );
    m_interactionActived = value;
    getQtViewer()->getQWidget()->setMouseTracking ( ! value);
    if (value==true)
        playpauseGUI(value);

    if(value)
    {
        interactionButton->setText(QSOFAApplication::translate("GUI", "ESC to qu&it", 0));
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
        interactionButton->setText(QSOFAApplication::translate("GUI", "&Interaction", 0));
        this->releaseKeyboard();
        this->releaseMouse();
        this->setMouseTracking(false);
        //this->setCursor(QCursor(Qt::ArrowCursor));
        application->restoreOverrideCursor();
    }

    sofa::core::objectmodel::KeypressedEvent keyEvent(value?(char)0x81:(char)0x80);
    Node* groot = mViewer->getScene();
    if (groot)
        groot->propagateEvent(core::ExecParams::defaultInstance(), &keyEvent);
}
#else
void RealGUI::interactionGUI ( bool )
{
}
#endif

//------------------------------------

//called at each step of the rendering
void RealGUI::step()
{
    sofa::helper::AdvancedTimer::begin("Animate");

    Node* root = currentSimulation();
    if ( root == NULL ) return;

    startDumpVisitor();

    if ( !getViewer()->ready() ) return;

    //root->setLogTime(true);

    // If dt is zero, the actual value of dt will be taken from the graph.
    double dt = 0.0;
    if (realTimeCheckBox->isChecked() && startButton->isChecked())
    {
        const clock_t currentClock = clock();
        // If animation has already started
        if (m_clockBeforeLastStep != 0)
        {
            // then dt <- "time since last step"
            dt = (double)(currentClock - m_clockBeforeLastStep) / CLOCKS_PER_SEC;
            // dt = std::min(dt, dtEdit->text().toDouble());
        }
        m_clockBeforeLastStep = currentClock;
    }

    simulation::getSimulation()->animate ( root, dt );
    simulation::getSimulation()->updateVisual( root );

    if ( m_dumpState )
        simulation::getSimulation()->dumpState ( root, *m_dumpStateStream );
    if ( m_exportGnuplot )
        exportGnuplot(root,gnuplot_directory);

    getViewer()->wait();

    eventNewStep();
    eventNewTime();

    if ( _animationOBJ )
    {
#ifdef CAPTURE_PERIOD
        static int counter = 0;
        if ( ( counter++ % CAPTURE_PERIOD ) ==0 )
#endif
        {
            exportOBJ ( currentSimulation(), false );
            ++_animationOBJcounter;
        }
    }

    stopDumpVisitor();
    emit newStep();
    if ( !currentSimulation()->getContext()->getAnimate() )
        startButton->setChecked ( false );


    sofa::helper::AdvancedTimer::end("Animate");
}

//------------------------------------

void RealGUI::updateDtEditState()
{
    dtEdit->setEnabled(!realTimeCheckBox->isChecked());
}

void RealGUI::setDt(const QString& value)
{
    const double dt = value.toDouble();
    // Input is validated, but value may be 0 anywway, while it is being entered.
    if (dt > 0.0)
        currentSimulation()->getContext()->setDt(dt);
}

//------------------------------------

// Reset the simulation to t=0
void RealGUI::resetScene()
{
    Node* root = currentSimulation();
    startDumpVisitor();
    emit ( newScene() );
    if (root)
    {
        frameCounter=0;

        simulation::getSimulation()->reset ( root );
        eventNewTime();
        emit newStep();
    }
    getViewer()->getPickHandler()->reset();
    stopDumpVisitor();
}

//------------------------------------

void RealGUI::screenshot()
{
    QString filename;

    bool pngSupport = helper::io::Image::FactoryImage::getInstance()->hasKey("png")
            || helper::io::Image::FactoryImage::getInstance()->hasKey("PNG");
    bool bmpSupport = helper::io::Image::FactoryImage::getInstance()->hasKey("bmp")
            || helper::io::Image::FactoryImage::getInstance()->hasKey("BMP");

    if(!pngSupport && !bmpSupport)
    {
        QMessageBox::warning(this, tr("runSofa"),
                             tr("Screenshot is not available (PNG or BMP support not found).\n"),
                             QMessageBox::Cancel);
        return;
    }
    std::string imageString = "Images (*.bmp)";
    if(pngSupport)
        imageString = "Images (*.png)";

    filename = getSaveFileName ( this,
                                 getViewer()->screenshotName().c_str(),
                                 imageString.c_str(),
                                 "save file dialog"
                                 "Choose a filename to save under"
                                 );

    viewer::SofaViewer* qtViewer = getQtViewer();
    if( qtViewer )
        qtViewer->getQWidget()->repaint();

    if ( filename != "" )
    {
        QString prefix;
        int end = filename.lastIndexOf('_');
        if (end > -1) {
            prefix = filename.mid(
                        0,
                        end+1
                        );
        } else {
            prefix = QString::fromStdString(
                        SetDirectory::GetFileNameWithoutExtension(filename.toStdString().c_str()) + "_");
        }

        if (!prefix.isEmpty())
            getViewer()->setPrefix ( prefix.toStdString(), false );

        getViewer()->screenshot ( filename.toStdString() );
    }
}

//------------------------------------

void RealGUI::showhideElements()
{
    displayFlag->updateDataValue();
    if(isEmbeddedViewer())
        getQtViewer()->getQWidget()->update();;
}

//------------------------------------

void RealGUI::Update()
{
    if(isEmbeddedViewer())
        getQtViewer()->getQWidget()->update();;
    statWidget->CreateStats(currentSimulation());
}

//------------------------------------

void RealGUI::updateBackgroundColour()
{
    if(getViewer())
        getViewer()->setBackgroundColour(background[0]->text().toFloat(),background[1]->text().toFloat(),background[2]->text().toFloat());
    if(isEmbeddedViewer())
        getQtViewer()->getQWidget()->update();;
}

//------------------------------------

void RealGUI::updateBackgroundImage()
{
    if(getViewer())
        getViewer()->setBackgroundImage( backgroundImage->text().toStdString() );
    if(isEmbeddedViewer())
        getQtViewer()->getQWidget()->update();;
}

//------------------------------------

void RealGUI::clear()
{
#ifndef SOFA_GUI_QT_NO_RECORDER
    if (recorder)
        recorder->Clear(currentSimulation());
#endif
    simulationGraph->Clear(currentSimulation());
    statWidget->CreateStats(currentSimulation());
}

//----------------------------------

void RealGUI::redraw()
{
    emit newStep();
}

//------------------------------------

void RealGUI::exportOBJ (simulation::Node* root,  bool exportMTL )
{
    if ( !root ) return;

    std::string sceneFileName(this->windowFilePath ().toStdString());
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

//------------------------------------

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

//------------------------------------

void RealGUI::displayComputationTime ( bool value )
{
    Node* root = currentSimulation();
    m_displayComputationTime = value;
    if ( root )
    {
        if (value)
            std::cout << "Activating Timer" << std::endl;
        else
            std::cout << "Deactivating Timer" << std::endl;
        sofa::helper::AdvancedTimer::setEnabled("Animate", value);
    }
}

//------------------------------------

void RealGUI::setExportGnuplot ( bool exp )
{
    Node* root = currentSimulation();
    m_exportGnuplot = exp;
    if ( exp && root )
    {
        sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
        InitGnuplotVisitor v(params , gnuplot_directory);
        root->execute( v );
        exportGnuplot(root,gnuplot_directory);
    }
}

//------------------------------------

#ifdef SOFA_DUMP_VISITOR_INFO
void RealGUI::setExportVisitor ( bool exp )
{
    if (exp)
    {
        windowTraceVisitor->show();
        handleTraceVisitor->clear();
    }
    else
    {
        windowTraceVisitor->hide();
    }
}
#else
void RealGUI::setExportVisitor ( bool )
{
}
#endif

//------------------------------------

void RealGUI::currentTabChanged ( int index )
{
    QWidget* widget = tabs->widget(index);

    if ( widget == currentTab ) return;

    if ( currentTab == NULL )
        currentTab = widget;

    if ( widget == TabGraph )
        simulationGraph->Unfreeze( );
    else if ( currentTab == TabGraph )
        simulationGraph->Freeze();
    else if (widget == TabStats)
        statWidget->CreateStats(currentSimulation());

    currentTab = widget;
}

//------------------------------------

void RealGUI::changeViewer()
{
    QObject* obj = const_cast<QObject*>( QObject::sender() );
    if( !obj) return;

    QAction* action = static_cast<QAction*>(obj);
    action->setChecked(true);

    std::map< helper::SofaViewerFactory::Key, QAction*  >::const_iterator iter_map;
    for ( iter_map = viewerMap.begin(); iter_map != viewerMap.end() ; ++iter_map )
    {
        if ( iter_map->second == action )
        {
            this->unloadScene();
            removeViewer();
            createViewer(iter_map->first.c_str());
        }
        else
        {
            (*iter_map).second->setChecked(false);
        }
    }

    // Reload the scene
    std::string filename(this->windowFilePath().toStdString());
    fileOpen ( filename.c_str() ); // keep the current display flags
}

//------------------------------------

void RealGUI::updateViewerList()
{
    // the current list of viewer key with associate QAction
    helper::vector< helper::SofaViewerFactory::Key > currentKeys;
    std::map< helper::SofaViewerFactory::Key, QAction*>::const_iterator iter_map;
    for ( iter_map = viewerMap.begin(); iter_map != viewerMap.end() ; ++iter_map )
        currentKeys.push_back((*iter_map).first);
    std::sort(currentKeys.begin(),currentKeys.end());

    // the new list (most recent since we load/unload viewer plugin)
    helper::vector< helper::SofaViewerFactory::Key > updatedKeys;
    helper::SofaViewerFactory::getInstance()->uniqueKeys(std::back_inserter(updatedKeys));
    std::sort(updatedKeys.begin(),updatedKeys.end());

    helper::vector< helper::SofaViewerFactory::Key > diffKeys;
    std::set_symmetric_difference(currentKeys.begin(),
                                  currentKeys.end(),
                                  updatedKeys.begin(),
                                  updatedKeys.end(),
                                  std::back_inserter(diffKeys)
                                  );

    bool viewerRemoved=false;
    helper::vector< helper::SofaViewerFactory::Key >::const_iterator it;
    for( it = diffKeys.begin(); it != diffKeys.end(); ++it)
    {
        // delete old
        std::map< helper::SofaViewerFactory::Key, QAction* >::iterator itViewerMap;
        if( (itViewerMap = viewerMap.find(*it)) != viewerMap.end() )
        {
            if( (*itViewerMap).second->isChecked() )
            {
                this->unloadScene();
                removeViewer();
                viewerRemoved = true;
            }
            //(*itViewerMap).second->disconnect(View);
            View->removeAction( (*itViewerMap).second);
            viewerMap.erase(itViewerMap);
        }
        else // add new
        {
            QAction* action = new QAction(this);
            action->setText( helper::SofaViewerFactory::getInstance()->getViewerName(*it) );
            //action->setMenuText(  helper::SofaViewerFactory::getInstance()->getAcceleratedViewerName(*it) );
            action->setToolTip(  helper::SofaViewerFactory::getInstance()->getAcceleratedViewerName(*it) );

            //action->setToggleAction(true);
            action->setCheckable(true);


            //action->addTo(View);
            View->addAction(action);

            viewerMap[*it] = action;
            action->setEnabled(true);
            connect(action, SIGNAL( triggered() ), this, SLOT( changeViewer() ) );
        }
    }

    // if we unloaded a viewer plugin actually in use
    if( viewerRemoved && !viewerMap.empty() )
    {
        createViewer(viewerMap.begin()->first.c_str());
        viewerMap.begin()->second->setChecked(true);
    }
}

void RealGUI::toolsDockMoved()
{
    QDockWidget* dockWindow = qobject_cast<QDockWidget*>(sender());
    if(!dockWindow)
        return;

    if(dockWindow->isFloating())
        dockWindow->resize(500, 700);
}

void RealGUI::propertyDockMoved(Qt::DockWidgetArea /*a*/)
{
    QDockWidget* dockWindow = qobject_cast<QDockWidget*>(sender());
    if(!dockWindow)
        return;

    if(dockWindow->isFloating())
        dockWindow->resize(500, 700);
}

namespace
{

std::string getFormattedLocalTimeFromTimestamp(time_t timestamp)
{
    const tm *timeinfo = localtime(&timestamp);
    std::ostringstream oss;
    oss << std::setfill('0')
        << std::setw(2) << timeinfo->tm_mday << "/" // Day
        << std::setw(2) << (timeinfo->tm_mon + 1) << "/"  // Month
        << std::setw(4) << (1900 + timeinfo->tm_year) << " " // Year
        << std::setw(2) << timeinfo->tm_hour << ":" // Hours
        << std::setw(2) << timeinfo->tm_min << ":"  // Minutes
        << std::setw(2) << timeinfo->tm_sec;        // Seconds
    return oss.str();
}

std::string getFormattedLocalTime()
{
    return getFormattedLocalTimeFromTimestamp( time( NULL ) );
}

} // namespace

//------------------------------------
void RealGUI::appendToDataLogFile(QString dataModifiedString)
{
    const std::string filename = this->windowFilePath().toStdString() + std::string(".log");

    std::ofstream ofs( filename.c_str(), std::ofstream::out | std::ofstream::app );

    if (ofs.good())
    {
        if (m_modifiedLogFiles.find(filename) == m_modifiedLogFiles.end())
        {
            ofs << std::endl << "--- NEW SESSION: " << getFormattedLocalTime() << " ---" << std::endl;
            m_modifiedLogFiles.insert(filename);
        }

        ofs << dataModifiedString.toStdString();
    }

    ofs.close();
}

//======================= SIGNALS-SLOTS ========================= }

} // namespace qt

} // namespace gui

} // namespace sofa
