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
#include "RealGUI.h"
#include "ImageQt.h"
#ifndef SOFA_GUI_QT_NO_RECORDER
#include "sofa/gui/qt/QSofaRecorder.h"
#endif
#include "sofa/gui/qt/QSofaStatWidget.h"
#include "sofa/gui/qt/GenGraphForm.h"
#include "sofa/gui/qt/QSofaListView.h"
#include <algorithm>

#include <sofa/simulation/common/PropagateEventVisitor.h>
#include <sofa/component/visualmodel/VisualModelImpl.h>
#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/common/xml/XML.h>
#include <sofa/simulation/common/InitVisitor.h>
#include <sofa/simulation/common/UpdateContextVisitor.h>
#include <sofa/simulation/common/DeleteVisitor.h>
#include <sofa/simulation/common/DeactivatedNodeVisitor.h>
#include <sofa/component/visualmodel/VisualStyle.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/AdvancedTimer.h>

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

#ifdef SOFA_QT4
/// Custom QApplication class handling FileOpen events for MacOS
class QSOFAApplication : public QApplication
{
public:
    QSOFAApplication(int &argc, char ** argv)
        : QApplication(argc,argv)
    { }

protected:
    bool event(QEvent *event)
    {
        switch (event->type())
        {
        case QEvent::FileOpen:
        {
            std::string filename = static_cast<QFileOpenEvent *>(event)->file().ascii();
            if (filename != std::string(static_cast<RealGUI*>(mainWidget())->windowFilePath().ascii()))
                static_cast<RealGUI*>(mainWidget())->fileOpen(static_cast<QFileOpenEvent *>(event)->file().ascii());
            return true;
        }
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


//======================= STATIC METHODS ========================= {
int RealGUI::InitGUI ( const char* /*name*/, const std::vector<std::string>& /* options */ )
{
    return ImageQt::Init() ? 0 : 1;
}

//------------------------------------

BaseGUI* RealGUI::CreateGUI ( const char* name, const std::vector<std::string>& options, sofa::simulation::Node::SPtr root, const char* filename )
{
    CreateApplication();

    // create interface
    gui = new RealGUI ( name, options );
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
    if ( sofa::helper::system::DataRepository.findFile ( pixmap_filename ) )
        pixmap_filename = sofa::helper::system::DataRepository.getFile ( pixmap_filename );

#ifdef SOFA_QT4
    b->setPixmap(QPixmap(QPixmap::fromImage(QImage(pixmap_filename.c_str()))));
#else
    b->setPixmap(QPixmap(QImage(pixmap_filename.c_str())));
#endif
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
}

//------------------------------------

void RealGUI::InitApplication( RealGUI* _gui)
{
    application->setMainWidget ( _gui );

    QString pathIcon=(sofa::helper::system::DataRepository.getFirstPath() + std::string( "/icons/SOFA.png" )).c_str();
#ifdef SOFA_QT4
    application->setWindowIcon(QIcon(pathIcon));
#else
    gui->setIcon(QPixmap(pathIcon));
#endif

    // show the gui
    _gui->show();
}
//======================= STATIC METHODS ========================= }




//======================= CONSTRUCTOR - DESTRUCTOR ========================= {
RealGUI::RealGUI ( const char* viewername, const std::vector<std::string>& options )
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
    m_dumpState(false),
    m_dumpStateStream(NULL),
    m_exportGnuplot(false),
    _animationOBJ(false),
    _animationOBJcounter(0),
    m_displayComputationTime(false),
    m_fullScreen(false),
    currentTab ( NULL ),
    statWidget(NULL),
    timerStep(NULL),
    backgroundImage(NULL),
    left_stack(NULL),
    pluginManager_dialog(NULL),
    recentlyOpenedFilesManager("config/Sofa.ini"),
    saveReloadFile(false),
    displayFlag(NULL),
    descriptionScene(NULL),
    htmlPage(NULL),
    animationState(false)
{
    parseOptions(options);

    createPluginManager();

    // TODO: deal with this...
    informationOnPickCallBack = InformationOnPickCallBack(this);

    createRecentFilesMenu(); // configure Recently Opened Menu

    timerStep = new QTimer(this);
    connect ( timerStep, SIGNAL ( timeout() ), this, SLOT ( step() ) );
    connect(this, SIGNAL(quit()), this, SLOT(fileExit()));
    connect ( startButton, SIGNAL ( toggled ( bool ) ), this , SLOT ( playpauseGUI ( bool ) ) );
    connect ( ResetSceneButton, SIGNAL ( clicked() ), this, SLOT ( resetScene() ) );
    connect ( dtEdit, SIGNAL ( textChanged ( const QString& ) ), this, SLOT ( setDt ( const QString& ) ) );
    connect ( stepButton, SIGNAL ( clicked() ), this, SLOT ( step() ) );
    connect ( dumpStateCheckBox, SIGNAL ( toggled ( bool ) ), this, SLOT ( dumpState ( bool ) ) );
    connect ( displayComputationTimeCheckBox, SIGNAL ( toggled ( bool ) ), this, SLOT ( displayComputationTime ( bool ) ) );
    connect ( exportGnuplotFilesCheckbox, SIGNAL ( toggled ( bool ) ), this, SLOT ( setExportGnuplot ( bool ) ) );
    connect ( tabs, SIGNAL ( currentChanged ( QWidget* ) ), this, SLOT ( currentTabChanged ( QWidget* ) ) );

    // create a Dock Window to receive the Sofa Recorder
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

    createSimulationGraph();

    //TODO : viewer
    mViewerName = viewername;
    left_stack = new QWidgetStack ( splitter2 );
    if (mCreateViewersOpt)
        createViewers(viewername);

    currentTabChanged ( tabs->currentPage() );

    createBackgroundGUIInfos(); // add GUI for Background Informations

    createWindowVisitor();

    createSceneDescription();

    SofaMouseManager::getInstance()->hide();
    SofaVideoRecorderManager::getInstance()->hide();

    //Center the application
    const QRect screen = QApplication::desktop()->availableGeometry(QApplication::desktop()->primaryScreen());
    this->move(  ( screen.width()- this->width()  ) / 2,  ( screen.height() - this->height()) / 2  );

#ifdef SOFA_QT4
    tabs->removeTab(tabs->indexOf(TabVisualGraph));
#endif

#ifndef SOFA_GUI_QT_NO_RECORDER
    if (recorder)
        connect( recorder, SIGNAL( RecordSimulation(bool) ), startButton, SLOT( setOn(bool) ) );
    if (recorder && mCreateViewersOpt)
        connect( recorder, SIGNAL( NewTime() ), getViewerWidget(), SLOT( update() ) );
#endif

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
#endif

    connect ( interactionButton, SIGNAL ( toggled ( bool ) ), this , SLOT ( interactionGUI ( bool ) ) );

    m_interactionActived = false;

    if(mCreateViewersOpt)
        getViewerWidget()->installEventFilter(this);
#endif
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
    if ( !sofa::helper::system::DataRepository.findFile ( scene ) )
    {
        std::cerr << "File " << scene << " not found " << std::endl;
        return;
    }
    this->unloadScene();
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
        std::cerr<<"You must load the pml file before the lml file"<<endl;
}
#endif

//------------------------------------


//======================= OPTIONS DEFINITIONS ========================= }



//======================= METHODS ========================= {
int RealGUI::mainLoop()
{
    int retcode;
    if (windowFilePath().isNull())
    {
        retcode = application->exec();
    }
    else
    {
        const std::string &filename=windowFilePath().ascii();
        const std::string &extension=sofa::helper::system::SetDirectory::GetExtension(filename.c_str());
        if (extension == "simu") fileOpenSimu(filename);
        retcode = application->exec();
    }
    return exitApplication(retcode);
}

//------------------------------------

int RealGUI::closeGUI()
{
    delete this;
    return 0;
}

//------------------------------------

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
    startDumpVisitor();
    update();
    //Hide all the dialogs to modify the graph
    emit ( newScene() );

    BaseGUIUtil::fileOpen(filename, temporaryFile);

    this->setWindowFilePath(filename.c_str());
    setExportGnuplot(exportGnuplotFilesCheckbox->isChecked());
    //  displayComputationTime(m_displayComputationTime);  // (FF) This can be set outside of the GUI and should not be changed implicitly by the GUI
    stopDumpVisitor();
}

//------------------------------------

void RealGUI::fileOpen()
{
    std::string filename(this->windowFilePath().ascii());

    QString s = getOpenFileName ( this, filename.empty() ?NULL:filename.c_str(),
#ifdef SOFA_PML
            "Scenes (*.scn *.xml);;Simulation (*.simu);;Php Scenes (*.pscn);;Pml Lml (*.pml *.lml);;All (*)",
#else
            "Scenes (*.scn *.xml);;Simulation (*.simu);;Php Scenes (*.pscn);;All (*)",
#endif
            "open file dialog",  "Choose a file to open"
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
                fileOpenSimu(s.ascii());
            else
                fileOpen (s.ascii());
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
                recorder->SetSimulation(currentSimulation(),initT,endT,writeName);
#endif
        }
    }
}

//------------------------------------

void RealGUI::setScene ( Node::SPtr root, const char* filename, bool temporaryFile )
{
    if (filename)
    {
        if (!temporaryFile)
            recentlyOpenedFilesManager.openFile(filename);
        saveReloadFile=temporaryFile;
        setTitle ( filename );
        loadHtmlDescription( filename );
    }

    mViewer->setScene( root, filename );
    createDisplayFlags( root );
    mViewer->resetView();

    eventNewTime();

    if (root)
    {
        //simulation::getSimulation()->updateVisualContext ( root );
        startButton->setOn ( root->getContext()->getAnimate() );
        dtEdit->setText ( QString::number ( root->getDt() ) );
        simulationGraph->Clear(root.get());
        statWidget->CreateStats(root.get());

#ifndef SOFA_GUI_QT_NO_RECORDER
        if (recorder)
            recorder->Clear(root.get());
#endif
    }

    if( isEmbeddedViewer() )
    {
        getViewerWidget()->setFocus();
        getViewerWidget()->show();
        updateViewerParameters();
    }

    resetScene();
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
    setCaption ( str.c_str() );
#endif
    setWindowFilePath( windowTitle.c_str() );
}

//------------------------------------

void RealGUI::fileNew()
{
    std::string newScene("config/newScene.scn");
    if (sofa::helper::system::DataRepository.findFile (newScene))
        fileOpen(sofa::helper::system::DataRepository.getFile ( newScene ).c_str());
}

//------------------------------------

void RealGUI::fileSave()
{
    std::string filename(this->windowFilePath().ascii());
    std::string message="You are about to overwrite your current scene: "  + filename + "\nAre you sure you want to do that ?";

    if ( QMessageBox::warning ( this, "Saving the Scene",message.c_str(), QMessageBox::Yes | QMessageBox::Default, QMessageBox::No ) != QMessageBox::Yes )
        return;

    Node *node = currentSimulation();
    fileSaveAs ( node,filename.c_str() );
}

//------------------------------------

void RealGUI::fileSaveAs ( Node *node, const char* filename )
{
    simulation::getSimulation()->exportXML ( node, filename );
}

//------------------------------------

void RealGUI::fileReload()
{
    std::string filename(this->windowFilePath().ascii());
    QString s = filename.c_str();

    if ( filename.empty() )
    {
        std::cerr << "Reload failed: no file loaded.\n";
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
        fileOpenSimu(s.ascii());
    else
        fileOpen ( s.ascii(),saveReloadFile );
#endif
}

//------------------------------------

void RealGUI::fileExit()
{
    //Hide all opened ModifyObject windows
    emit ( newScene() );
    startButton->setOn ( false);
    this->close();
}

//------------------------------------

void RealGUI::saveXML()
{
    simulation::getSimulation()->exportXML ( currentSimulation(), "scene.scn" );
}

//------------------------------------

void RealGUI::editRecordDirectory()
{
    std::string filename(this->windowFilePath().ascii());
    std::string record_directory;
    QString s = getExistingDirectory ( this, filename.empty() ?NULL:filename.c_str(), "open directory dialog",  "Choose a directory" );
    if (s.length() > 0)
    {
        record_directory = s.ascii();
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
    std::string filename(this->windowFilePath().ascii());
    QString s = getExistingDirectory ( this, filename.empty() ?NULL:filename.c_str(), "open directory dialog",  "Choose a directory" );
    if (s.length() > 0)
    {
        gnuplot_directory = s.ascii();
        if (gnuplot_directory.at(gnuplot_directory.size()-1) != '/')
            gnuplot_directory+="/";
        setExportGnuplot(exportGnuplotFilesCheckbox->isChecked());
    }
}

//------------------------------------

void RealGUI::showPluginManager()
{
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

void RealGUI::createViewers(const char* viewerName)
{
    ViewerArgument arg;
    arg.name = "viewer";
    arg.parent = left_stack;

    if( mViewer != NULL )
        removeViewer();

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
                mViewer = helper::SofaViewerFactory::CreateObject(*iter, arg);
                action->setOn(true);
            }
            else
            {
                action->setOn(false);
            }
        }
    }

    if( mViewer == NULL )
    {
        std::cerr << "ERROR(QtGUI): unknown or disabled viewer name "<<viewerName<<std::endl;
        application->exit();
    }

    if( isEmbeddedViewer() )
        left_stack->addWidget ( getViewerWidget() );

    initViewer();
}

//------------------------------------

void RealGUI::initViewer()
{
    if(mViewer == NULL)
    {
        std::cerr<<"ERROR when initViewer, the viewer is NULL"<<std::endl;
        return;
    }

    frameCounter = 0;
    _animationOBJ = false;
    _animationOBJcounter = 0;
    m_dumpState = false;
    m_dumpStateStream = 0;
    m_displayComputationTime = false;
    m_exportGnuplot = false;
    gnuplot_directory = "";

    if( isEmbeddedViewer() )
    {
#ifdef SOFA_QT4
        left_stack->setCurrentWidget ( getViewerWidget() );
        getViewerWidget()->setFocusPolicy ( Qt::StrongFocus );
#else
        int id_viewer = left_stack->addWidget ( getViewerWidget() );
        left_stack->raiseWidget ( id_viewer );
        getViewerWidget()->setFocusPolicy ( QWidget::StrongFocus );
        getViewerWidget()->setCursor ( QCursor ( 2 ) );
#endif

        getViewerWidget()->setSizePolicy ( QSizePolicy ( ( QSizePolicy::SizeType ) 7, ( QSizePolicy::SizeType ) 7, 100, 1, getViewerWidget()->sizePolicy().hasHeightForWidth() ) );
        getViewerWidget()->setMinimumSize ( QSize ( 0, 0 ) );
        getViewerWidget()->setMouseTracking ( TRUE );
    }
    dynamic_cast<sofa::gui::qt::viewer::SofaViewer*>(mViewer)->configureViewerTab(tabs);

    connect ( ResetViewButton, SIGNAL ( clicked() ), this, SLOT ( resetView() ) );
    connect ( SaveViewButton, SIGNAL ( clicked() ), this, SLOT ( saveView() ) );
    connect ( screenshotButton, SIGNAL ( clicked() ), this, SLOT ( screenshot() ) );
    connect ( sizeW, SIGNAL ( valueChanged ( int ) ), this, SLOT ( setSizeW ( int ) ) );
    connect ( sizeH, SIGNAL ( valueChanged ( int ) ), this, SLOT ( setSizeH ( int ) ) );
    if(isEmbeddedViewer())
    {
        connect ( getViewerWidget(), SIGNAL ( resizeW ( int ) ), sizeW, SLOT ( setValue ( int ) ) );
        connect ( getViewerWidget(), SIGNAL ( resizeH ( int ) ), sizeH, SLOT ( setValue ( int ) ) );
        connect ( getViewerWidget(), SIGNAL ( quit (  ) ), this, SLOT ( fileExit (  ) ) );
        connect(simulationGraph, SIGNAL(focusChanged(sofa::core::objectmodel::BaseObject*)),
                getViewerWidget(), SLOT(fitObjectBBox(sofa::core::objectmodel::BaseObject*))
               );
        connect(simulationGraph, SIGNAL( focusChanged(sofa::core::objectmodel::BaseNode*) ),
                getViewerWidget(), SLOT( fitNodeBBox(sofa::core::objectmodel::BaseNode*) )
               );
    }

    // splitter2 separates horizontally the OptionTab widget and the viewer widget
    QSplitter *splitter_ptr = dynamic_cast<QSplitter *> ( splitter2 );
    if(isEmbeddedViewer())
    {
        splitter_ptr->moveToLast ( left_stack ); // Use addWidget(widget) instead.
        splitter_ptr->setOpaqueResize ( false );

#ifdef SOFA_QT4
        // rescale factor for the space occuped by the widget index
        splitter_ptr->setStretchFactor( 0, 0); // OptionTab
        splitter_ptr->setStretchFactor( 1, 10); // Viewer
        QList<int> list;
#else
        QValueList<int> list;
#endif

        list.push_back ( 250 ); // OptionTab
        list.push_back ( 640 ); // Viewer
        splitter_ptr->setSizes ( list );
    }
    else
    {
        splitter_ptr->setOpaqueResize ( false );

#ifdef SOFA_QT4
        // rescale factor for the space occuped by the widget index
        splitter_ptr->setStretchFactor( 0, 0); // OptionTab
        splitter_ptr->setStretchFactor( 1, 0); // Viewer
        QList<int> list;
#else
        QValueList<int> list;
#endif

        list.push_back ( 250 ); // OptionTab
        list.push_back ( 0 );  // Viewer
        splitter_ptr->setSizes ( list );
    }

    // setGUI
    textEdit1->setText ( dynamic_cast<sofa::gui::qt::viewer::SofaViewer*>(mViewer)->helpString() );
    if(isEmbeddedViewer())
        connect ( this, SIGNAL( newStep()), getViewerWidget(), SLOT( update()));

    if(isEmbeddedViewer())
    {
        getViewerWidget()->setFocus();
        getViewerWidget()->show();
        updateViewerParameters();
    }

    SofaMouseManager::getInstance()->setPickHandler(mViewer->getPickHandler());

    if(isEmbeddedViewer())
        mViewer->getPickHandler()->addCallBack(&informationOnPickCallBack );

    mGuiName=mViewerName;
}

//------------------------------------

void RealGUI::setViewerResolution ( int w, int h )
{
    if(mViewer == NULL)
        return;

    if(isEmbeddedViewer())
    {
        QSize winSize = size();
        QSize viewSize = ( mViewer != NULL ) ? getViewerWidget()->size() : QSize(0,0);

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
        //std::cout<<"winSize.width() - viewSize.width() + w = "<< winSize.width()<<"-"<< viewSize.width()<<"+"<<w<<std::endl;
        //std::cout<<"winSize.height() - viewSize.height() + h = "<< winSize.height()<<"-"<< viewSize.height()<<"+"<<h<<std::endl;
        //std::cout << "Setting windows dimension to " << size().width() << " x " << size().height() << std::endl;
    }
}

//------------------------------------

void RealGUI::setFullScreen (bool enable)
{
    if (enable == m_fullScreen) return;

    QSplitter *splitter_ptr = dynamic_cast<QSplitter *> ( splitter2 );

#ifdef SOFA_QT4
    QList<int> list;
    static QList<int> savedsizes;
#else
    QValueList<int> list;
    static QValueList<int> savedsizes;
#endif

    if (enable)
    {
        savedsizes = splitter_ptr->sizes();
        optionTabs->hide();
        optionTabs->setParent(static_cast<QWidget*>(splitter_ptr->parent()));
    }
    else if (m_fullScreen)
    {
        splitter_ptr->insertWidget(0,optionTabs);
        optionTabs->show();
        splitter_ptr->setSizes ( savedsizes );
    }

    if (enable)
    {
        std::cout << "Set Full Screen Mode" << std::endl;
        showFullScreen();
        m_fullScreen = true;
    }
    else
    {
        std::cout << "Set Windowed Mode" << std::endl;
        showNormal();
        m_fullScreen = false;
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

//------------------------------------

void RealGUI::setBackgroundColor(const defaulttype::Vector3& c)
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
    mViewer->configure(viewerConf);
}

//------------------------------------

void RealGUI::setMouseButtonConfiguration(sofa::component::configurationsetting::MouseButtonSetting *button)
{
    SofaMouseManager::getInstance()->updateOperation(button);
    // SofaMouseManager::getInstance()->updateContent();
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

void RealGUI::setRecordPath(const std::string & path)
{
#ifndef SOFA_GUI_QT_NO_RECORDER
    if (recorder)
        recorder->SetRecordDirectory(path);
#endif
}

//------------------------------------

void RealGUI::setGnuplotPath(const std::string &path)
{
    gnuplot_directory = path;
}

//------------------------------------

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
            connect(tabs,SIGNAL(currentChanged(QWidget*)),displayFlag, SLOT( updateWidgetValue() ));
        }
    }
}

//------------------------------------

void RealGUI::loadHtmlDescription(const char* filename)
{
    std::string extension=sofa::helper::system::SetDirectory::GetExtension(filename);
    std::string htmlFile=filename;
    htmlFile.resize(htmlFile.size()-extension.size()-1);
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
        /// @TODO: use AdvancedTimer in GUI to display time statistics
    }
}

//------------------------------------

void RealGUI::eventNewTime()
{
#ifndef SOFA_GUI_QT_NO_RECORDER
    if (recorder)
        recorder->UpdateTime(currentSimulation());
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

//------------------------------------

void RealGUI::keyPressEvent ( QKeyEvent * e )
{
    sofa::gui::qt::viewer::SofaViewer* qtViewer = dynamic_cast<sofa::gui::qt::viewer::SofaViewer*>(mViewer);

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
        playpauseGUI(!startButton->isOn());
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
    if (root && this->exportVisitorCheckbox->isOn())
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

//------------------------------------

void RealGUI::parseOptions(const std::vector<std::string>& options)
{
    for (unsigned int i=0; i<options.size(); ++i)
    {
        if (options[i] == "noViewers")
            mCreateViewersOpt = false;
    }
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
#ifdef SOFA_QT4
    fileMenu->removeAction(Action);
#endif
    const int indexRecentlyOpened=fileMenu->count()-2;
    QMenu *recentMenu = recentlyOpenedFilesManager.createWidget(this);
    fileMenu->insertItem(QPixmap(),recentMenu,indexRecentlyOpened,indexRecentlyOpened);
    connect(recentMenu, SIGNAL(activated(int)), this, SLOT(fileRecentlyOpened(int)));
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

        colourLayout->addWidget(background[i]);
        connect( background[i], SIGNAL( returnPressed() ), this, SLOT( updateBackgroundColour() ) );
    }

    QWidget *image = new QWidget(TabPage);
    QHBoxLayout *imageLayout = new QHBoxLayout(image);
    imageLayout->addWidget(new QLabel(QString("Image "),image));

    backgroundImage = new QLineEdit(image,"backgroundImage");
    if ( mViewer != NULL )
        backgroundImage->setText( QString(mViewer->getBackgroundImage().c_str()) );
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
    ((QVBoxLayout*)TabGraph->layout())->addWidget(simulationGraph);
    connect ( ExportGraphButton, SIGNAL ( clicked() ), simulationGraph, SLOT ( Export() ) );
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
}

//------------------------------------

void RealGUI::createWindowVisitor()
{
    pathDumpVisitor = sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str()) + std::string( "/dumpVisitor.xml" );
#ifndef SOFA_DUMP_VISITOR_INFO
    //Remove option to see visitor trace
    this->exportVisitorCheckbox->hide();
#else
    //Main window containing a QListView only
    windowTraceVisitor = new WindowVisitor;
    windowTraceVisitor->graphView->setSorting(-1);
    windowTraceVisitor->hide();
    connect ( exportVisitorCheckbox, SIGNAL ( toggled ( bool ) ), this, SLOT ( setExportVisitor ( bool ) ) );
    connect(windowTraceVisitor, SIGNAL(WindowVisitorClosed(bool)), this->exportVisitorCheckbox, SLOT(setChecked(bool)));
    handleTraceVisitor = new GraphVisitor(windowTraceVisitor);
#endif
}

//------------------------------------

void RealGUI::createSceneDescription()
{
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
}
//======================= METHODS ========================= }



//======================= SIGNALS-SLOTS ========================= {
void RealGUI::NewRootNode(sofa::simulation::Node* root, const char* path)
{
    if(path != NULL)
        mViewer->setScene ( root, mViewer->getSceneFileName().c_str() );
    else
        mViewer->setScene(root , path);

    mViewer->resetView();
    updateViewerParameters();
    statWidget->CreateStats(root);
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
        if ( node == mViewer->getScene() )
            simulation::getSimulation()->init(node);
        else
            simulation::getSimulation()->initNode(node);
    }
}

//------------------------------------

void RealGUI::fileSaveAs(Node *node)
{
    if (node == NULL) node = currentSimulation();
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

//------------------------------------

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

//------------------------------------

void RealGUI::fileRecentlyOpened(int id)
{
    fileOpen(recentlyOpenedFilesManager.getFilename((unsigned int)id));
}

//------------------------------------

void RealGUI::playpauseGUI ( bool value )
{
    startButton->setOn ( value );
    if ( currentSimulation() )
        currentSimulation()->getContext()->setAnimate ( value );
    if(value)
        timerStep->start(0);
    else
        timerStep->stop();
}

//------------------------------------

#ifdef SOFA_GUI_INTERACTION
void RealGUI::interactionGUI ( bool value )
{
    interactionButton->setOn ( value );
    m_interactionActived = value;
    getViewerWidget()->setMouseTracking ( ! value);
    if (value==true)
        playpauseGUI(value);

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
    Node* groot = getScene();
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
    Node* root = currentSimulation();
    if ( root == NULL ) return;

    startDumpVisitor();

    if ( mViewer->ready() ) return;

    //root->setLogTime(true);
    //T=T+DT
    SReal dt=root->getDt();
    simulation::getSimulation()->animate ( root, dt );
    simulation::getSimulation()->updateVisual( root );

    if ( m_dumpState )
        simulation::getSimulation()->dumpState ( root, *m_dumpStateStream );
    if ( m_exportGnuplot )
        exportGnuplot(root,gnuplot_directory);

    mViewer->wait();

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
        startButton->setOn ( false );
}

//------------------------------------

// Set the time between each iteration of the Sofa Simulation
void RealGUI::setDt ( double value )
{
    Node* root = currentSimulation();
    if ( value > 0.0 && root)
        root->getContext()->setDt ( value );
}

//------------------------------------

void RealGUI::setDt ( const QString& value )
{
    setDt ( value.toDouble() );
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
        simulation::getSimulation()->reset ( root );
        eventNewTime();
        emit newStep();
    }
    mViewer->getPickHandler()->reset();
    stopDumpVisitor();
}

//------------------------------------

void RealGUI::screenshot()
{
    sofa::gui::qt::viewer::SofaViewer* qtViewer = dynamic_cast<sofa::gui::qt::viewer::SofaViewer*>(mViewer);
    if( !qtViewer ) return;

    QString filename;

#ifdef SOFA_HAVE_PNG
    const char* imageString = "Images (*.png)";
#else
    const char* imageString = "Images (*.bmp)";
#endif

    filename = getSaveFileName ( this,
            qtViewer->screenshotName().c_str(),
            imageString,
            "save file dialog"
            "Choose a filename to save under"
                               );
    qtViewer->getQWidget()->repaint();
    if ( filename != "" )
    {
        std::ostringstream ofilename;
        const char* begin = filename;
        const char* end = strrchr ( begin,'_' );
        if ( !end )
            end = begin + filename.length();
        ofilename << std::string ( begin, end );
        ofilename << "_";
        qtViewer->setPrefix ( ofilename.str() );
#ifdef SOFA_QT4
        qtViewer->screenshot ( filename.toStdString() );
#else
        qtViewer->screenshot ( filename );
#endif
    }
}

//------------------------------------

void RealGUI::showhideElements()
{
    displayFlag->updateDataValue();
    updateViewerParameters();
}

//------------------------------------

void RealGUI::Update()
{
    updateViewerParameters();
    statWidget->CreateStats(mViewer->getScene());
}

//------------------------------------

void RealGUI::updateViewerParameters()
{
    if(isEmbeddedViewer())
        getViewerWidget()->update();
}

//------------------------------------

void RealGUI::updateBackgroundColour()
{
    mViewer->setBackgroundColour(atof(background[0]->text().ascii()),atof(background[1]->text().ascii()),atof(background[2]->text().ascii()));
    updateViewerParameters();
}

//------------------------------------

void RealGUI::updateBackgroundImage()
{
    mViewer->setBackgroundImage( backgroundImage->text().ascii() );
    updateViewerParameters();
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
        exportGnuplot(root,gnuplot_directory);
}

//------------------------------------

void RealGUI::setExportVisitor ( bool exp )
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

//------------------------------------

void RealGUI::currentTabChanged ( QWidget* widget )
{
    if ( widget == currentTab ) return;

    if ( currentTab == NULL )
        currentTab = widget;

    if ( widget == TabGraph )
        simulationGraph->Unfreeze( );
    else if ( currentTab == TabGraph )
        simulationGraph->Freeze();
    else if (widget == TabStats)
        statWidget->CreateStats(mViewer->getScene());

    currentTab = widget;
}

//------------------------------------

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
            // Cleanup previous viewer
            this->unloadScene();
            dynamic_cast<sofa::gui::qt::viewer::SofaViewer*>(mViewer)->removeViewerTab(tabs);
            if(isEmbeddedViewer())
                left_stack->removeWidget( getViewerWidget() );
            removeViewer();

            // Need for the next creation
            ViewerArgument arg;
            arg.name = "viewer";
            arg.parent = left_stack; // Set it by default but in case of standalone viewer, we don't use it in tis constructor

            // Create new viewer
            mViewer =  helper::SofaViewerFactory::CreateObject( (*iter_map).first, arg);

            if(isEmbeddedViewer())
                left_stack->addWidget( getViewerWidget() );

            initViewer();
        }
        else
        {
            (*iter_map).second->setOn(false);
        }
    }

    // Reload the scene
    std::string filename(this->windowFilePath().ascii());
    fileOpen ( filename.c_str() ); // keep the current display flags
}

//------------------------------------

void RealGUI::updateViewerList()
{
    helper::vector< helper::SofaViewerFactory::Key > currentKeys;
    std::map< helper::SofaViewerFactory::Key, QAction*>::const_iterator iter_map;
    for ( iter_map = viewerMap.begin(); iter_map != viewerMap.end() ; ++iter_map )
        currentKeys.push_back((*iter_map).first);

    std::sort(currentKeys.begin(),currentKeys.end());
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

    helper::vector< helper::SofaViewerFactory::Key >::const_iterator it;
    for( it = diffKeys.begin(); it != diffKeys.end(); ++it)
    {
        std::map< helper::SofaViewerFactory::Key, QAction* >::iterator itViewerMap;
        if( (itViewerMap = viewerMap.find(*it)) != viewerMap.end() )
        {
            if( (*itViewerMap).second->isOn() )
            {
                this->unloadScene();
                dynamic_cast<sofa::gui::qt::viewer::SofaViewer*>(mViewer)->removeViewerTab(tabs);
                if(isEmbeddedViewer())
                    left_stack->removeWidget(getViewerWidget() );
                removeViewer();
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

    if( mViewer == NULL )
    {
        if(!viewerMap.empty())
        {
            ViewerArgument arg;
            arg.name = "viewer";
            arg.parent = left_stack;
            /* change viewer */
            mViewer =  helper::SofaViewerFactory::CreateObject( viewerMap.begin()->first, arg);

            if(isEmbeddedViewer())
                left_stack->addWidget( getViewerWidget() );

            initViewer();
        }
    }
}
//======================= SIGNALS-SLOTS ========================= }

} // namespace qt

} // namespace gui

} // namespace sofa
