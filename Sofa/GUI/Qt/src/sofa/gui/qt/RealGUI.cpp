/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/version.h>

#ifdef SOFA_DUMP_VISITOR_INFO
#include "WindowVisitor.h"
#include "GraphVisitor.h"
#endif

#if SOFA_GUI_QT_HAVE_QT_CHARTS
#include "SofaWindowProfiler.h"
#endif

#if SOFA_GUI_QT_HAVE_NODEEDITOR
#include "SofaWindowDataGraph.h"
#endif


#include <mutex>
#include <QScreen>
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

#include <sofa/gui/common/BaseViewer.h>
#include <sofa/simulation/common/xml/XML.h>
#include <sofa/simulation/DeactivatedNodeVisitor.h>
#include <sofa/component/visual/VisualStyle.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/simulation/SimulationLoop.h>

#include <sofa/helper/system/SetDirectory.h>
using sofa::helper::system::SetDirectory;

#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::FileSystem;

#include <sofa/helper/Utils.h>
using sofa::helper::Utils;

#include <sofa/helper/system/FileRepository.h>
using sofa::helper::system::DataRepository;

#include <sofa/gui/common/GuiDataRepository.h>
using sofa::gui::common::GuiDataRepository;

#include <sofa/simulation/SceneLoaderFactory.h>
using sofa::simulation::SceneLoaderFactory;

#include <sofa/simulation/Simulation.h>
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
#include <QStatusBar>
#include <QSettings>
#include <QMimeData>
#include <QCompleter>
#include <QDesktopServices>

#if (QT_VERSION < QT_VERSION_CHECK(5, 11, 0))
#include <QDesktopWidget>
#endif

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <ctime>

#include <sofa/core/objectmodel/IdleEvent.h>
using sofa::core::objectmodel::IdleEvent;

#include <sofa/simulation/events/SimulationStartEvent.h>
using sofa::simulation::SimulationStartEvent;

#include <sofa/simulation/events/SimulationStopEvent.h>
using sofa::simulation::SimulationStopEvent;

#include <sofa/helper/system/FileMonitor.h>
using sofa::helper::system::FileMonitor;

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory;

#if(SOFA_GUI_QT_HAVE_QT5_WEBENGINE)
#include "panels/QDocBrowser.h"
using sofa::gui::qt::DocBrowser;
#endif

using sofa::core::ExecParams;

#include <sofa/gui/common/ArgumentParser.h>


using namespace sofa::gui::common;

namespace sofa::gui::qt
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
    {
        QCoreApplication::setOrganizationName("Sofa Consortium");
        QCoreApplication::setOrganizationDomain("sofa");
        QCoreApplication::setApplicationName("runSofa");
    }

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
    bool event(QEvent *event) override
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

RealGUI* gui = nullptr;
QApplication* application = nullptr;

const char* progname="";



class RealGUIFileListener : public sofa::helper::system::FileEventListener
{
public:
    RealGUIFileListener(RealGUI* realgui){
        m_realgui = realgui;
    }
    ~RealGUIFileListener() override{}

    void fileHasChanged(const std::string& filename) override
    {
        m_realgui->fileOpen(filename, false, true);
    }
    RealGUI* m_realgui;
};


//======================= STATIC METHODS ========================= {

void RealGUI::setupSurfaceFormat()
{
    static std::once_flag flag;
    std::call_once(flag, []
    {
        QSurfaceFormat format;
        if(!SOFA_GUI_QT_ENABLE_VSYNC)
        {
            format.setSwapInterval(0); //Setting an interval value of 0 will turn the vertical refresh syncing off
        }

        static constexpr int vmajor = 3, vminor = 2;
        format.setVersion(vmajor,vminor); //Sets the desired major and minor OpenGL versions.
        format.setProfile(QSurfaceFormat::CompatibilityProfile); //Sets the desired OpenGL context profile. CompatibilityProfile = Functionality from earlier OpenGL versions is available.
        format.setOption(QSurfaceFormat::DeprecatedFunctions, true); //Used to request that deprecated functions be included in the OpenGL context profile. If not specified, you should get a forward compatible context without support functionality marked as deprecated. This requires OpenGL version 3.0 or higher.
        format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);

        if (mArgumentParser)
        {
            unsigned int viewerMSAANbSampling = 0;
            mArgumentParser->getValueFromKey("msaa", viewerMSAANbSampling);
            if (viewerMSAANbSampling > 1)
            {
                msg_info("RealGUI") << "Set multisampling anti-aliasing (MSAA) with " << viewerMSAANbSampling << " samples." ;
                format.setSamples(static_cast<int>(viewerMSAANbSampling));
            }
        }

        QSurfaceFormat::setDefaultFormat(format);
    });
}

BaseGUI* RealGUI::CreateGUI ( const char* name, sofa::simulation::Node::SPtr root, const char* filename )
{
    setupSurfaceFormat();

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
    argv[1]=nullptr;
    application = new QSOFAApplication ( *argc,argv );

    //force locale to Standard C
    //(must be done immediatly after the QApplication has been created)
    const QLocale locale(QLocale::C);
    QLocale::setDefault(locale);
}

//------------------------------------

void RealGUI::InitApplication( RealGUI* _gui)
{
    const QString pathIcon=(DataRepository.getFirstPath() + std::string( "/icons/SOFA.png" )).c_str();
    application->setWindowIcon(QIcon(pathIcon));

    if(SOFA_GUI_QT_ENABLE_NATIVE_MENU)
    {
        // Use the OS'native menu instead of the Qt one
        _gui->menubar->setNativeMenuBar(true);
    }
    else
    {
        // Use the qt menu instead of the native one in order to standardize the way the menu is showed on every OS
        _gui->menubar->setNativeMenuBar(false);
    }
    // show the gui
    _gui->show(); // adding extra line in the console?
}
//======================= STATIC METHODS ========================= }




//======================= CONSTRUCTOR - DESTRUCTOR ========================= {
RealGUI::RealGUI ( const char* viewername)
    :
      fpsLabel(nullptr),
      timeLabel(nullptr),

      #ifdef SOFA_DUMP_VISITOR_INFO
      windowTraceVisitor(nullptr),
      handleTraceVisitor(nullptr),
      #endif

      m_sofaMouseManager(nullptr),
      #if SOFAGUIQT_HAVE_QT5_CHARTS
      m_windowTimerProfiler(nullptr),
      #endif
      #if SOFAGUIQT_HAVE_NODEEDITOR
      m_sofaWindowDataGraph(nullptr),
      #endif
      simulationGraph(nullptr),
      m_createViewersOpt(true),
      m_isEmbeddedViewer(true),
      m_dumpState(false),
      m_dumpStateStream(nullptr),
      m_exportGnuplot(false),
      m_animateOBJ(false),
      m_animationOBJcounter(0),
      m_displayComputationTime(false),
      m_fullScreen(false),
      m_viewer(nullptr),
      m_clockBeforeLastStep(0),
      propertyWidget(nullptr),
      currentTab ( nullptr ),
      statWidget(nullptr),
      timerStep(nullptr),
      backgroundImage(nullptr),
      pluginManagerDialog(nullptr),
      recentlyOpenedFilesManager(BaseGUI::getConfigDirectoryPath() + "/runSofa.ini"),
      m_saveReloadFile(false),
      displayFlag(nullptr),
#if(SOFA_GUI_QT_HAVE_QT5_WEBENGINE)
      m_docbrowser(nullptr),
#endif
      m_animationState(false),
      m_frameCounter(0),
      m_viewerMSAANbSampling(1)
{
    setupUi(this);
    
    ExpandAllButton->setIcon(QIcon(":/RealGUI/expandAll"));
    CollapseAllButton->setIcon(QIcon(":/RealGUI/collapseAll"));
    sceneGraphRefreshToggleButton->setIcon(QIcon(":/RealGUI/sceneGraphRefresh"));
    for (auto* button : {ExpandAllButton, CollapseAllButton, sceneGraphRefreshToggleButton})
    {
        button->setFixedWidth(button->height());
    }
    
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
    connect ( ReloadSceneButton, SIGNAL ( clicked() ), this, SLOT ( fileReload() ) );
    connect ( dtEdit, SIGNAL ( textChanged ( const QString& ) ), this, SLOT ( setDt ( const QString& ) ) );
    connect ( realTimeCheckBox, SIGNAL ( stateChanged ( int ) ), this, SLOT ( updateDtEditState() ) );
    connect ( stepButton, SIGNAL ( clicked() ), this, SLOT ( step() ) );
    connect ( dumpStateCheckBox, SIGNAL ( toggled ( bool ) ), this, SLOT ( dumpState ( bool ) ) );
    connect ( displayComputationTimeCheckBox, SIGNAL ( toggled ( bool ) ), this, SLOT ( displayComputationTime ( bool ) ) );
    connect ( exportGnuplotFilesCheckbox, SIGNAL ( toggled ( bool ) ), this, SLOT ( setExportGnuplot ( bool ) ) );
    connect ( tabs, SIGNAL ( currentChanged ( int ) ), this, SLOT ( currentTabChanged ( int ) ) );

    connect ( ResetViewButton, SIGNAL ( clicked() ), this, SLOT ( resetView() ) );
    connect ( SaveViewButton, SIGNAL ( clicked() ), this, SLOT ( saveView() ) );
    connect ( screenshotButton, SIGNAL ( clicked() ), this, SLOT ( screenshot() ) );
    connect ( sizeW, SIGNAL ( valueChanged ( int ) ), this, SLOT ( setSizeW ( int ) ) );
    connect ( sizeH, SIGNAL ( valueChanged ( int ) ), this, SLOT ( setSizeH ( int ) ) );

    /// We activate this timer only if the interactive mode is enabled (ie livecoding+mouse mouve event).
    if(m_enableInteraction){
        timerIdle = new QTimer(this);
        connect ( timerIdle, SIGNAL ( timeout() ), this, SLOT ( emitIdle() ) );
        timerIdle->start(50);
    }

    this->setDockOptions(QMainWindow::AnimatedDocks | QMainWindow::AllowTabbedDocks);
#if (QT_VERSION < QT_VERSION_CHECK(5, 15, 0))
    dockWidget->setFeatures(QDockWidget::AllDockWidgetFeatures);
#else
    dockWidget->setFeatures(QDockWidget::DockWidgetClosable|QDockWidget::DockWidgetMovable|QDockWidget::DockWidgetFloatable);
#endif
    dockWidget->setAllowedAreas(Qt::RightDockWidgetArea | Qt::LeftDockWidgetArea);

    connect(dockWidget, SIGNAL(dockLocationChanged(Qt::DockWidgetArea)), this, SLOT(toolsDockMoved()));

    //Status Bar Configuration
    fpsLabel = new QLabel ( "9999.9 FPS", statusBar() );
    fpsLabel->setMinimumSize ( fpsLabel->sizeHint() );
    fpsLabel->clear();

    timeLabel = new QLabel ( "Time: 999.9999", statusBar() );
    timeLabel->setMinimumSize ( timeLabel->sizeHint() );
    timeLabel->clear();
    statusBar()->addWidget ( fpsLabel );
    statusBar()->addWidget ( timeLabel );

    statWidget = new QSofaStatWidget(TabStats);
    TabStats->layout()->addWidget(statWidget);

    // create al widgets first
    m_sofaMouseManager = new SofaMouseManager(this);

    createSimulationGraph();

    //disable widget, can be bothersome with objects with a lot of data
    //createPropertyWidget();

    //viewer
    informationOnPickCallBack = InformationOnPickCallBack(this);

    viewerMap.clear();
    if (m_createViewersOpt)
        createViewer(viewername, true);

    currentTabChanged ( tabs->currentIndex() );

    createBackgroundGUIInfos(); // add GUI for Background Informations

    createWindowVisitor();

    createAdvancedTimerProfilerWindow();

    m_sofaMouseManager->hide();
    SofaVideoRecorderManager::getInstance()->hide();

    centerWindow();

    tabs->removeTab(tabs->indexOf(TabVisualGraph));

#if(SOFA_GUI_QT_HAVE_QT5_WEBENGINE)
    m_docbrowser = new DocBrowser(this);
    /// Signal to the realGUI that the visibility has changed (eg: to update the menu bar)
    connect(m_docbrowser, SIGNAL(visibilityChanged(bool)), this, SLOT(docBrowserVisibilityChanged(bool)));
#endif

    // Trigger QDialog for "About" section
    connect(helpAboutAction, SIGNAL(triggered()), this, SLOT(showAbout()));

    m_filelistener = new RealGUIFileListener(this);
}

//------------------------------------

RealGUI::~RealGUI()
{
    if( displayFlag != nullptr )
        delete displayFlag;

#ifdef SOFA_DUMP_VISITOR_INFO
    delete windowTraceVisitor;
    delete handleTraceVisitor;
#endif

    removeViewer();

    FileMonitor::removeListener(m_filelistener);
    delete m_filelistener;
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
    QScreen* screen = widget->window()->windowHandle()->screen();
    settings.beginGroup("viewer");
    settings.setValue("screenNumber", QGuiApplication::screens().indexOf(screen));
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
    std::vector<std::string> expandedNodes;

    if(reload)
    {
        saveView();

        if(simulationGraph)
            simulationGraph->getExpandedNodes(expandedNodes);
    }

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

    const std::vector<std::string> sceneArgs = ArgumentParser::extra_args();
    mSimulation = sofa::simulation::node::load ( filename, reload, sceneArgs );

    sofa::simulation::node::initRoot(mSimulation.get());
    if ( mSimulation == nullptr )
    {
        msg_warning("RealGUI")<<"Failed to load "<<filename.c_str();
        return;
    }

    if(reload)
        setSceneWithoutMonitor(mSimulation, filename.c_str(), temporaryFile);
    else{
        setScene(mSimulation, filename.c_str(), temporaryFile);
#if(SOFA_GUI_QT_HAVE_QT5_WEBENGINE)
        m_docbrowser->loadHtml( filename ) ;
#endif
    }

    configureGUI(mSimulation.get());

    this->setWindowFilePath(filename.c_str());
    setExportGnuplot(exportGnuplotFilesCheckbox->isChecked());
    stopDumpVisitor();

    if(!expandedNodes.empty())
    {
        simulationGraph->expandPathFrom(expandedNodes);
    }

#if SOFA_GUI_QT_HAVE_QT_CHARTS
    if (m_windowTimerProfiler)
        m_windowTimerProfiler->resetGraph();
#endif

#if SOFA_GUI_QT_HAVE_NODEEDITOR
    if (m_sofaWindowDataGraph)
        m_sofaWindowDataGraph->resetNodeGraph(currentSimulation());
#endif

}


//------------------------------------

void RealGUI::emitIdle()
{
    // Update all the registered monitor.
    FileMonitor::updates(0);

    IdleEvent hb;
    Node* groot = m_viewer->getScene();
    if (groot)
    {
        groot->propagateEvent(core::execparams::defaultInstance(), &hb);
    }

    if(isEmbeddedViewer())
        getSofaViewer()->getQWidget()->update();
}

/// This open popup the file selection windows.
void RealGUI::popupOpenFileSelector()
{
    const std::string filename(this->windowFilePath().toStdString());

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

    filter += ";;Simulation (*.simu)";

    filter = allKnownFilters+";;"+filter+";;All (*)"; // the first filter is selected by default

    QString selectedFilter( tr(allKnownFilters.c_str()) ); // this does not select the desired filter

    QString s = getOpenFileName ( this, filename.empty() ?nullptr:filename.c_str(),
                                  filter.c_str(),
                                  "open file dialog",  "Choose a file to open", &selectedFilter
                                  );
    if ( s.length() >0 )
    {
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
            simulationName = s;
            const std::string::size_type pointSimu = simulationName.rfind(".simu");
            simulationName.resize(pointSimu);
            fileOpen(filename.c_str());

            dtEdit->setText(QString(dT.c_str()));
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
        m_saveReloadFile=temporaryFile;
        setTitle ( filename );
#if(SOFA_GUI_QT_HAVE_QT5_WEBENGINE)
        if (m_docbrowser && filename)
        {
            m_docbrowser->loadHtml( filename );
        }
#endif
    }

    if (root)
    {
        //Check the validity of the BBox
        const sofa::type::BoundingBox& nodeBBox = root->getContext()->f_bbox.getValue();
        if(nodeBBox.isNegligeable())
        {
            msg_warning("RealGUI") << "Global Bounding Box seems very small; Your viewer settings (based on the bbox) are likely invalid, switching to default value of [-1,-1,-1,1,1,1]."
                                   << "This is caused by using component which does not implement properly the computeBBox function."
                                   << "You can remove this warning by manually forcing a value in the parameter bbox=\"minX minY minZ maxX maxY maxZ\" in your root node \n";
            const sofa::type::BoundingBox b(-1.0,-1.0,-1.0,1.0,1.0,1.0);
            root->f_bbox.setValue(b);
        }


        mSimulation = root;
        eventNewTime();
        startButton->setChecked(root->getContext()->getAnimate() );

        dtEdit->setText ( QString::number ( root->getDt() ) );
        simulationGraph->setRoot(root.get());
        simulationGraph->collapseAll();
        simulationGraph->expandToDepth(0);
        simulationGraph->resizeColumnToContents(0);
        statWidget->CreateStats(root.get());

        getViewer()->setScene( root, filename );
        getViewer()->load();
        getViewer()->resetView();
        createDisplayFlags( root );

        if( isEmbeddedViewer() )
        {
            getSofaViewer()->getQWidget()->setFocus();
            getSofaViewer()->getQWidget()->show();
            getSofaViewer()->getQWidget()->update();
        }

        resetScene();
    }
}

void RealGUI::setScene(Node::SPtr root, const char* filename, bool temporaryFile)
{
    if(m_enableInteraction &&  filename){
        FileMonitor::removeListener(m_filelistener);
        FileMonitor::addFile(filename, m_filelistener);
    }
    setSceneWithoutMonitor(root, filename, temporaryFile) ;
#if(SOFA_GUI_QT_HAVE_QT5_WEBENGINE)
    if (m_docbrowser && filename)
    {
        m_docbrowser->loadHtml( filename ) ;
    }
#endif
}

//------------------------------------

void RealGUI::unloadScene(bool _withViewer)
{
    if(_withViewer && getViewer())
        getViewer()->unload();

    sofa::simulation::node::unload ( currentSimulation() );

    if(_withViewer && getViewer())
        getViewer()->setScene(nullptr);
}

//------------------------------------

void RealGUI::setTitle ( std::string windowTitle )
{
    std::string str = "SOFA v" + std::string(SOFA_VERSION_STR);
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
    const std::string filename(this->windowFilePath().toStdString());
    const std::string message="You are about to overwrite your current scene: "  + filename + "\nAre you sure you want to do that ?";

    if ( QMessageBox::warning ( this, "Saving the Scene",message.c_str(), QMessageBox::Yes | QMessageBox::Default, QMessageBox::No ) != QMessageBox::Yes )
        return;

    fileSaveAs ( currentSimulation(), filename.c_str() );
}

//------------------------------------

void RealGUI::fileSaveAs ( Node *node, const char* filename )
{
    sofa::simulation::node::exportGraph ( node, filename );
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

    if (s.endsWith( ".simu") )
        fileOpenSimu(s.toStdString());
    else
        fileOpen ( s.toStdString(),m_saveReloadFile );
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
    const std::string filename(this->windowFilePath().toStdString());
    std::string record_directory;
    const QString s = getExistingDirectory ( this, filename.empty() ?nullptr:filename.c_str(), "open directory dialog",  "Choose a directory" );
    if (s.length() > 0)
    {
        record_directory = s.toStdString();
        if (record_directory.at(record_directory.size()-1) != '/')
            record_directory+="/";
    }
}

//------------------------------------

void RealGUI::editGnuplotDirectory()
{
    const std::string filename(this->windowFilePath().toStdString());
    const QString s = getExistingDirectory ( this, filename.empty() ?nullptr:filename.c_str(), "open directory dialog",  "Choose a directory" );
    if (s.length() > 0)
    {
        gnuplotDirectory = s.toStdString();
        if (gnuplotDirectory.at(gnuplotDirectory.size()-1) != '/')
            gnuplotDirectory+="/";
        setExportGnuplot(exportGnuplotFilesCheckbox->isChecked());
    }
}

//------------------------------------

void RealGUI::showDocBrowser()
{
#if(SOFA_GUI_QT_HAVE_QT5_WEBENGINE)
    m_docbrowser->flipVisibility();
#else
    msg_warning("RealGUI") << "Doc browser has been disabled because Qt5WebEngine is not available";
#endif
}

//------------------------------------

void RealGUI::showAbout()
{
    //create the QDialog for About
    AboutSOFADialog* aboutSOFA_dialog = new sofa::gui::qt::AboutSOFADialog(this);
    aboutSOFA_dialog->show();
}

//------------------------------------

void RealGUI::showPluginManager()
{
    pluginManagerDialog->updatePluginsListView();
    pluginManagerDialog->show();
}

//------------------------------------

void RealGUI::showMouseManager()
{
    m_sofaMouseManager->updateContent();
    m_sofaMouseManager->show();
}

//------------------------------------

void RealGUI::showVideoRecorderManager()
{
    SofaVideoRecorderManager::getInstance()->show();
}

//------------------------------------

void RealGUI::showWindowDataGraph()
{
#if SOFA_GUI_QT_HAVE_NODEEDITOR
    std::cout << "RealGUI::showWindowDataGraph()" << std::endl;
    //m_sofaMouseManager->createGraph();
    if (m_sofaWindowDataGraph == nullptr)
    {
        createSofaWindowDataGraph();
    }
    m_sofaWindowDataGraph->show();

#endif
}

//------------------------------------

void RealGUI::setViewerResolution ( int w, int h )
{
    if( isEmbeddedViewer() )
    {
        const QSize winSize = size();
        const QSize viewSize = ( getViewer() ) ? getSofaViewer()->getQWidget()->size() : QSize(0,0);

#if (QT_VERSION < QT_VERSION_CHECK(5, 11, 0))
        const QRect screen = QApplication::desktop()->availableGeometry(QApplication::desktop()->screenNumber(this));
#else
        const QRect screen = QGuiApplication::primaryScreen()->availableGeometry();// QGuiApplication::screens().at(QApplication::desktop()->screenNumber(this))->availableGeometry();
#endif
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
        }
        else
        {
            menuBar()->show();
            statusBar()->show();
        }
    }
    else
    {
        getViewer()->setFullScreen(enable);
    }
}

void RealGUI::centerWindow()
{
    //Center the application
#if (QT_VERSION < QT_VERSION_CHECK(5, 11, 0))
    const QRect screen = QApplication::desktop()->availableGeometry(QApplication::desktop()->primaryScreen());
#else
    const QRect screen = QGuiApplication::primaryScreen()->availableGeometry();
#endif
    this->move(  ( screen.width() - this->width()  ) / 2,  ( screen.height() - this->height()) / 2 );
}

//------------------------------------

void RealGUI::setBackgroundColor(const sofa::type::RGBAColor& c)
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

void RealGUI::setViewerConfiguration(sofa::component::setting::ViewerSetting* viewerConf)
{
    const type::Vec<2,int> &res=viewerConf->resolution.getValue();

    if (viewerConf->fullscreen.getValue())
        setFullScreen();
    else
        setViewerResolution(res[0], res[1]);
    getViewer()->configure(viewerConf);
}

//------------------------------------

void RealGUI::setMouseButtonConfiguration(sofa::component::setting::MouseButtonSetting *button)
{
    m_sofaMouseManager->updateOperation(button);
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

void RealGUI::setGnuplotPath(const std::string &path)
{
    gnuplotDirectory = path;
}

//------------------------------------

void RealGUI::createViewer(const char* _viewerName, bool _updateViewerList/*=false*/)
{
    if(_updateViewerList)
    {
        this->updateViewerList();
        // the viewer with the key viewerName is already created
        if( m_viewer != nullptr && !viewerMap.begin()->first.compare( std::string(_viewerName) ) )
            return;
    }

    for (std::map< helper::SofaViewerFactory::Key, QAction*>::const_iterator iter_map = viewerMap.begin();
         iter_map != viewerMap.end(); ++iter_map )
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
    const BaseViewer* old = m_viewer;
    m_viewer = _viewer;
    if(m_viewer != nullptr)
        delete old;
    else
        msg_error("RealGUI")<<"when registerViewer, the viewer is nullptr";
}

//------------------------------------

BaseViewer* RealGUI::getViewer()
{
    return m_viewer;
}

//------------------------------------

sofa::gui::qt::viewer::SofaViewer* RealGUI::getSofaViewer()
{
    return dynamic_cast<sofa::gui::qt::viewer::SofaViewer*>(m_viewer);
}

//------------------------------------

bool RealGUI::isEmbeddedViewer()
{
    return m_isEmbeddedViewer;
}

//------------------------------------

void RealGUI::removeViewer()
{
    if(m_viewer != nullptr)
    {
        if(isEmbeddedViewer())
        {
            getSofaViewer()->removeViewerTab(tabs);
        }
        delete m_viewer;
        m_viewer = nullptr;
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
    m_frameCounter = 0;
    m_animateOBJ = false;
    m_animationOBJcounter = 0;
    m_dumpState = false;
    m_dumpStateStream = 0;
    m_displayComputationTime = false;
    m_exportGnuplot = false;
    gnuplotDirectory = "";
    m_fullScreen = false;
}

//------------------------------------

void RealGUI::createDisplayFlags(Node::SPtr root)
{
    if( displayFlag != nullptr)
    {
        gridLayout1->removeWidget(displayFlag);
        delete displayFlag;
        displayFlag = nullptr;
    }

    sofa::component::visual::VisualStyle* visualStyle = nullptr;

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
    const Node* root = currentSimulation();

    if ( m_frameCounter==0 )
    {
        const ctime_t t = CTime::getRefTime();
        for ( int i=0; i<10; i++ )
            beginTime[i] = t;
    }

    ++m_frameCounter;
    if ( ( m_frameCounter%10 ) == 0 )
    {
        const ctime_t curtime = CTime::getRefTime();
        const int i = ( ( m_frameCounter/10 ) %10 );
        const double fps = ( ( double ) timeTicks / ( curtime - beginTime[i] ) ) * ( m_frameCounter<100?m_frameCounter:100 );
        showFPS(fps);

        beginTime[i] = curtime;
    }

    if ( m_displayComputationTime && ( m_frameCounter%100 ) == 0 && root!=nullptr )
    {
        /// @TODO: use AdvancedTimer in GUI to display time statistics
    }
}

void RealGUI::showFPS(double fps)
{
    if (fpsLabel)
    {
        char buf[100];
        sprintf ( buf, "%.1f FPS", fps );
        fpsLabel->setText ( buf );
    }
}

//------------------------------------

void RealGUI::eventNewTime()
{
    const Node* root = currentSimulation();
    if (root && timeLabel)
    {
        const double time = root->getTime();
        char buf[100];
        sprintf ( buf, "Time: %.3g,   Steps:  %i", time, m_frameCounter );
        timeLabel->setText ( buf );
    }
}

//------------------------------------

void RealGUI::keyPressEvent ( QKeyEvent * e )
{
    sofa::gui::qt::viewer::SofaViewer* sofaViewer = dynamic_cast<sofa::gui::qt::viewer::SofaViewer*>(getViewer());

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
        m_animateOBJ = !m_animateOBJ;
        m_animationOBJcounter = 0;
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
    case Qt::Key_S:
    {
        screenshot();
        break;
    }
    default:
    {
        if (sofaViewer)
            sofaViewer->keyPressEvent(e);
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
    if(_viewer == nullptr)
    {
        msg_error("RealGUI")<<"when initViewer, the viewer is nullptr";
        return;
    }
    init(); //init data member from RealGUI for the viewer initialisation in the GUI

    // Is our viewer embedded or not ?
    sofa::gui::qt::viewer::SofaViewer* sofaViewer = dynamic_cast<sofa::gui::qt::viewer::SofaViewer*>(_viewer);
    if( sofaViewer == nullptr )
    {
        isEmbeddedViewer(false);
        std::cout<<"initViewer: The viewer isn't embedded in the GUI"<<std::endl;
    }
    else
    {
        isEmbeddedViewer(true);
        this->mainWidgetLayout->addWidget(sofaViewer->getQWidget());

        sofaViewer->getQWidget()->setFocusPolicy ( Qt::StrongFocus );

        sofaViewer->getQWidget()->setSizePolicy ( QSizePolicy ( ( QSizePolicy::Policy ) 7,
                                                                ( QSizePolicy::Policy ) 7
                                                                //, 100, 1,
                                                                //sofaViewer->getQWidget()->sizePolicy().hasHeightForWidth() )
                                                                ));

        sofaViewer->getQWidget()->setMinimumSize ( QSize ( 0, 0 ) );
        sofaViewer->getQWidget()->setMouseTracking ( true );
        sofaViewer->configureViewerTab(tabs);

        connect ( sofaViewer->getQWidget(), SIGNAL ( resizeW ( int ) ), sizeW, SLOT ( setValue ( int ) ) );
        connect ( sofaViewer->getQWidget(), SIGNAL ( resizeH ( int ) ), sizeH, SLOT ( setValue ( int ) ) );
        connect ( sofaViewer->getQWidget(), SIGNAL ( quit (  ) ), this, SLOT ( fileExit (  ) ) );
        connect(simulationGraph, SIGNAL(focusChanged(sofa::core::objectmodel::BaseObject*)),
                sofaViewer->getQWidget(), SLOT(fitObjectBBox(sofa::core::objectmodel::BaseObject*))
                );
        connect(simulationGraph, SIGNAL( focusChanged(sofa::core::objectmodel::BaseNode*) ),
                sofaViewer->getQWidget(), SLOT( fitNodeBBox(sofa::core::objectmodel::BaseNode*) )
                );

        // setGUI
        textEdit1->setText ( sofaViewer->helpString() );
        connect ( this, SIGNAL( newStep()), sofaViewer->getQWidget(), SLOT( update()));

        sofaViewer->getQWidget()->setFocus();
        sofaViewer->getQWidget()->show();
        sofaViewer->getQWidget()->update();

        sofaViewer->getPickHandler()->addCallBack(&informationOnPickCallBack );
    }

    m_sofaMouseManager->setPickHandler(_viewer->getPickHandler());
}

//------------------------------------

void RealGUI::parseOptions()
{

    if (mArgumentParser) {
        mArgumentParser->getValueFromKey("interactive", m_enableInteraction);
        mArgumentParser->getValueFromKey("msaa", m_viewerMSAANbSampling);

        if (m_enableInteraction)
            msg_warning("runSofa") << "you activated the interactive mode. This is currently an experimental feature "
                                      "that may change or be removed in the future. ";
    }
}

//------------------------------------

void RealGUI::createPluginManager()
{
    pluginManagerDialog = new SofaPluginManager(this);
    pluginManagerDialog->hide();
    this->connect( pluginManagerDialog, SIGNAL( libraryAdded() ),  this, SLOT( updateViewerList() ));
    this->connect( pluginManagerDialog, SIGNAL( libraryRemoved() ),  this, SLOT( updateViewerList() ));
}

void RealGUI::createSofaWindowDataGraph()
{
#if SOFA_GUI_QT_HAVE_NODEEDITOR
    m_sofaWindowDataGraph = new SofaWindowDataGraph(this, currentSimulation());
    m_sofaWindowDataGraph->hide();
#endif
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
    connect ( ExpandAllButton, SIGNAL ( clicked() ), simulationGraph, SLOT ( expandAll() ) );
    connect ( CollapseAllButton, SIGNAL ( clicked() ), simulationGraph, SLOT ( ExpandRootNodeOnly() ) );
    connect ( sceneGraphRefreshToggleButton, &QPushButton::clicked , this, &RealGUI::onSceneGraphRefreshButtonClicked );
    connect(simulationGraph, &QSofaListView::dirtynessChanged, this, &RealGUI::sceneGraphViewDirtynessChanged);
    connect(simulationGraph, &QSofaListView::lockingChanged, this, &RealGUI::sceneGraphViewLockingChanged);

    connect(simulationGraph, SIGNAL( RootNodeChanged(sofa::simulation::Node*, const char*) ), this, SLOT ( newRootNode(sofa::simulation::Node* , const char*) ) );
    connect(simulationGraph, SIGNAL( NodeRemoved() ), this, SLOT( update() ) );
    connect(simulationGraph, SIGNAL( Lock(bool) ), this, SLOT( lockAnimation(bool) ) );
    connect(simulationGraph, SIGNAL( RequestSaving(sofa::simulation::Node*) ), this, SLOT( fileSaveAs(sofa::simulation::Node*) ) );
    connect(simulationGraph, SIGNAL( RequestExportOBJ(sofa::simulation::Node*, bool) ), this, SLOT( exportOBJ(sofa::simulation::Node*, bool) ) );
    connect(simulationGraph, SIGNAL( RequestActivation(sofa::simulation::Node*, bool) ), this, SLOT( activateNode(sofa::simulation::Node*, bool) ) );
    connect(simulationGraph, SIGNAL( RequestSleeping(sofa::simulation::Node*, bool) ), this, SLOT( setSleepingNode(sofa::simulation::Node*, bool) ) );
    connect(simulationGraph, SIGNAL( Updated() ), this, SLOT( redraw() ) );
    connect(simulationGraph, SIGNAL( NodeAdded() ), this, SLOT( update() ) );
    connect(simulationGraph, SIGNAL( dataModified( QString ) ), this, SLOT( appendToDataLogFile(QString ) ) );
    connect(this, SIGNAL( newScene() ), simulationGraph, SLOT( CloseAllDialogs() ) );
    connect(this, SIGNAL( newStep() ), simulationGraph, SLOT( UpdateOpenedDialogs() ) );

    std::ifstream file( BaseGUI::getConfigDirectoryPath() + "/sceneGraphLock" );
    if(file.is_open())
    {
        bool isLocked;
        file >> isLocked;
        if (isLocked)
        {
            simulationGraph->lock();
        }
        else
        {
            simulationGraph->unLock();
        }
        file.close();
    }
    else
    {
        simulationGraph->unLock();
    }
}

// This slot is called when the sceneGraph view is set to dirty
void RealGUI::sceneGraphViewDirtynessChanged(bool isDirty)
{
    if(isDirty)
    {
        sceneGraphRefreshToggleButton->setIcon(QIcon(":/RealGUI/sceneGraphRefresh-dirty"));
    }
    else if(simulationGraph->isLocked())
    {
        sceneGraphRefreshToggleButton->setIcon(QIcon(":/RealGUI/sceneGraphRefresh-locked"));
    }
    else
    {
        sceneGraphRefreshToggleButton->setIcon(QIcon(":/RealGUI/sceneGraphRefresh-unlocked"));
    }
}

// This slot is called when the sceneGraph view has been locked/unlocked.
// The locking state indicates how the view is taking into account the scene graph data changes.
// when locked, the vue is not updated. When unlocked all changes are taken into account.
void RealGUI::sceneGraphViewLockingChanged(bool isLocked)
{
    if(isLocked)
    {
        sceneGraphRefreshToggleButton->setIcon(QIcon(":/RealGUI/sceneGraphRefresh-locked"));
    }
    else
    {
        sceneGraphRefreshToggleButton->setIcon(QIcon(":/RealGUI/sceneGraphRefresh-unlocked"));
    }
}

// The scene graph update button has three states
// State 0: unlocked (all the changes on the graph are immediately taken into account)
// State 1: locked (the changes on the graph are not done but the simulation graph is set to dirty if
//          there is some changes on the graph. A click on the button unlocks the graph (go to state 0).
// State 2: dirty, in that state the button reflect the fact that the scene graph view has changed but not displayed.
//          A click on the button refreshes the graph view but does not change the Lock/Unlock state
void RealGUI::onSceneGraphRefreshButtonClicked()
{
    if(simulationGraph->isLocked())
    {
        simulationGraph->unLock();
    }
    else
    {
        simulationGraph->lock();
    }

    std::ofstream file( BaseGUI::getConfigDirectoryPath() + "/sceneGraphLock" );
    if(file)
    {
        file << simulationGraph->isLocked();
        file.close();
    }
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
    windowTraceVisitor = new WindowVisitor(this);
    windowTraceVisitor->graphView->setSortingEnabled(false);
    windowTraceVisitor->hide();
    connect ( exportVisitorCheckbox, SIGNAL ( toggled ( bool ) ), this, SLOT ( setExportVisitor ( bool ) ) );
    connect(windowTraceVisitor, SIGNAL(WindowVisitorClosed(bool)), this->exportVisitorCheckbox, SLOT(setChecked(bool)));
    handleTraceVisitor = new GraphVisitor(windowTraceVisitor);
#endif
}

void RealGUI::createAdvancedTimerProfilerWindow()
{
#if SOFA_GUI_QT_HAVE_QT_CHARTS
    m_windowTimerProfiler = new SofaWindowProfiler(this);
    m_windowTimerProfiler->hide();
    connect( displayTimeProfiler, SIGNAL ( toggled ( bool ) ), this, SLOT ( displayProflierWindow ( bool ) ) );
    connect( m_windowTimerProfiler, SIGNAL(closeWindow(bool)), this->displayTimeProfiler, SLOT(setChecked(bool)));
#else
    displayTimeProfiler->setEnabled(false);
#endif
}

void RealGUI::newRootNode(sofa::simulation::Node* root, const char* path)
{
    const std::string filename(this->windowFilePath().toStdString());
    const std::string message="You are about to change the root node of the scene : "  + filename +
            "to the root node : " + std::string(path) +
            "\nThis implies that the simulation singleton has to change its root node.\nDo you want to proceed ?";
    if ( QMessageBox::warning ( this, "New root node: ",message.c_str(), QMessageBox::Yes | QMessageBox::Default, QMessageBox::No ) != QMessageBox::Yes )
        return;

    if(path != nullptr && root != nullptr)
    {
        getViewer()->setScene(root , path);
        getViewer()->load();
        getViewer()->resetView();
        if(isEmbeddedViewer())
            getSofaViewer()->getQWidget()->update();
        statWidget->CreateStats(root);
    }
}

//------------------------------------

void RealGUI::activateNode(sofa::simulation::Node* node, bool activate)
{
    const QSofaListView* sofalistview = (QSofaListView*)sender();

    if (activate)
        node->setActive(true);
    simulation::DeactivationVisitor v(sofa::core::execparams::defaultInstance(), activate);
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

    const ActivationFunctor activator( activate, sofalistview->getListener() );
    std::for_each(nodeToChange.begin(),nodeToChange.end(),activator);
    nodeToChange.clear();
    update();

    if ( sofalistview == simulationGraph && activate )
    {
        if (node == currentSimulation())
        {
            sofa::simulation::node::initRoot(node);
        }
        else
        {
            sofa::simulation::node::init(node);
        }
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
    if (node == nullptr) node = currentSimulation();
    const std::string filename(this->windowFilePath().toStdString());


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

    filter += ")";


    QString s = getSaveFileName ( this, filename.empty() ?nullptr:filename.c_str(), filter, "save file dialog", "Choose where the scene will be saved" );
    if (s.length() > 0) {
        fileSaveAs(node, s.toStdString().c_str());
    }

}

//------------------------------------

void RealGUI::lockAnimation(bool value)
{
    if(value)
    {
        m_animationState = startButton->isChecked();
        playpauseGUI(false);
    }
    else
    {
        playpauseGUI(m_animationState);
    }
}

//------------------------------------

void RealGUI::fileRecentlyOpened(QAction *action)
{
    //fileOpen(recentlyOpenedFilesManager.getFilename((unsigned int)id));
    fileOpen(action->text().toStdString());
}

//------------------------------------

void RealGUI::playpauseGUI ( bool startSimulation )
{
    startButton->setChecked ( startSimulation );

    /// If there is no root node we do nothing.
    Node* root = currentSimulation();
    if (root==nullptr)
        return;

    /// Set the animation 'on' in the getContext()
    currentSimulation()->getContext()->setAnimate ( startSimulation );

    if(startSimulation)
    {
        SimulationStopEvent startEvt;
        root->propagateEvent(core::execparams::defaultInstance(), &startEvt);
        m_clockBeforeLastStep = 0;
        timerStep->start(0);
        return;
    }

    SimulationStartEvent stopEvt;
    root->propagateEvent(core::execparams::defaultInstance(), &stopEvt);

    timerStep->stop();
    return;
}

//------------------------------------

void RealGUI::interactionGUI ( bool )
{
}


//------------------------------------

//called at each step of the rendering
void RealGUI::step()
{
    SIMULATION_LOOP_SCOPE
    sofa::helper::AdvancedTimer::begin("Animate");

    Node* root = currentSimulation();
    if ( root == nullptr ) return;

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

    sofa::simulation::node::animate(root, dt);
    sofa::simulation::node::updateVisual(root);

    if ( m_dumpState )
        sofa::simulation::node::dumpState ( root, *m_dumpStateStream );
    if ( m_exportGnuplot )
        exportGnuplot(root,gnuplotDirectory);

    getViewer()->wait();

    eventNewStep();
    eventNewTime();

    if ( m_animateOBJ )
    {
#ifdef CAPTURE_PERIOD
        static int counter = 0;
        if ( ( counter++ % CAPTURE_PERIOD ) ==0 )
#endif
        {
            exportOBJ ( currentSimulation(), false );
            ++m_animationOBJcounter;
        }
    }

    stopDumpVisitor();
    emit newStep();
    if ( !currentSimulation()->getContext()->getAnimate() )
        startButton->setChecked ( false );

#if SOFA_GUI_QT_HAVE_QT_CHARTS
    if (displayTimeProfiler->isChecked())
    {
        m_windowTimerProfiler->pushStepData();
    }
#endif

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
        m_frameCounter=0;

        sofa::simulation::node::reset(root);
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

    const bool pngSupport = helper::io::Image::FactoryImage::getInstance()->hasKey("png")
            || helper::io::Image::FactoryImage::getInstance()->hasKey("PNG");
    const bool bmpSupport = helper::io::Image::FactoryImage::getInstance()->hasKey("bmp")
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

    viewer::SofaViewer* sofaViewer = getSofaViewer();
    if( sofaViewer )
        sofaViewer->getQWidget()->repaint();

    if ( filename != "" )
    {
        QString prefix;
        const int end = filename.lastIndexOf('_');
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
        getSofaViewer()->getQWidget()->update();
}

//------------------------------------

void RealGUI::update()
{
    if(isEmbeddedViewer())
        getSofaViewer()->getQWidget()->update();
    statWidget->CreateStats(currentSimulation());
}

//------------------------------------

void RealGUI::updateBackgroundColour()
{
    if(getViewer())
        getViewer()->setBackgroundColour(background[0]->text().toFloat(),background[1]->text().toFloat(),background[2]->text().toFloat());
    if(isEmbeddedViewer())
        getSofaViewer()->getQWidget()->update();
}

//------------------------------------

void RealGUI::updateBackgroundImage()
{
    if(getViewer())
        getViewer()->setBackgroundImage( backgroundImage->text().toStdString() );
    if(isEmbeddedViewer())
        getSofaViewer()->getQWidget()->update();
}

//------------------------------------

void RealGUI::clear()
{
    simulationGraph->setRoot(currentSimulation());
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

    SCOPED_TIMER_VARNAME(exportOBJTimer, "exportOBJ");

    const std::string sceneFileName(this->windowFilePath ().toStdString());
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
    oss << m_animationOBJcounter;

    ofilename << '_' << ( oss.str().c_str() );
    ofilename << ".obj";
    const std::string filename = ofilename.str();
    std::cout << "Exporting OBJ Scene "<<filename<<std::endl;
    sofa::simulation::node::exportOBJ ( root, filename.c_str(),exportMTL );
}

//------------------------------------

void RealGUI::dumpState ( bool value )
{
    m_dumpState = value;
    if ( m_dumpState )
    {
        m_dumpStateStream = new std::ofstream ( "dumpState.data" );
    }
    else if ( m_dumpStateStream!=nullptr )
    {
        delete m_dumpStateStream;
        m_dumpStateStream = 0;
    }
}

//------------------------------------

void RealGUI::displayComputationTime ( bool value )
{
    const Node* root = currentSimulation();
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
        const sofa::core::ExecParams* params = sofa::core::execparams::defaultInstance();
        InitGnuplotVisitor v(params , gnuplotDirectory);
        root->execute( v );
        exportGnuplot(root,gnuplotDirectory);
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

void RealGUI::displayProflierWindow (bool value)
{
#if SOFA_GUI_QT_HAVE_QT_CHARTS
    if (m_windowTimerProfiler == nullptr)
        return;

    m_windowTimerProfiler->activateATimer(value);
    if (value)
        m_windowTimerProfiler->show();
    else
        m_windowTimerProfiler->hide();
#else
    SOFA_UNUSED(value);
#endif
}


//------------------------------------

void RealGUI::currentTabChanged ( int index )
{
    QWidget* widget = tabs->widget(index);

    if ( widget == currentTab ) return;

    if ( currentTab == nullptr )
        currentTab = widget;

    if ( widget == TabGraph )
        simulationGraph->update();
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
    for ( iter_map = viewerMap.begin(); iter_map != viewerMap.end(); ++iter_map )
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
    const std::string filename(this->windowFilePath().toStdString());
    fileOpen ( filename.c_str() ); // keep the current display flags
}

//------------------------------------

void RealGUI::updateViewerList()
{
    // the current list of viewer key with associate QAction
    type::vector< helper::SofaViewerFactory::Key > currentKeys;
    std::map< helper::SofaViewerFactory::Key, QAction*>::const_iterator iter_map;
    for ( iter_map = viewerMap.begin(); iter_map != viewerMap.end(); ++iter_map )
        currentKeys.push_back((*iter_map).first);
    std::sort(currentKeys.begin(),currentKeys.end());

    // the new list (most recent since we load/unload viewer plugin)
    type::vector< helper::SofaViewerFactory::Key > updatedKeys;
    helper::SofaViewerFactory::getInstance()->uniqueKeys(std::back_inserter(updatedKeys));
    std::sort(updatedKeys.begin(),updatedKeys.end());

    type::vector< helper::SofaViewerFactory::Key > diffKeys;
    std::set_symmetric_difference(currentKeys.begin(),
                                  currentKeys.end(),
                                  updatedKeys.begin(),
                                  updatedKeys.end(),
                                  std::back_inserter(diffKeys)
                                  );

    bool viewerRemoved=false;
    type::vector< helper::SofaViewerFactory::Key >::const_iterator it;
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
    return getFormattedLocalTimeFromTimestamp( time( nullptr ) );
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

} // namespace sofa::gui::qt
