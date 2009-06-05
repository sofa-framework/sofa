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
#include <sofa/gui/qt/RealGUI.h>
#include <sofa/gui/qt/ImageQt.h>


#ifdef SOFA_GUI_QTOGREVIEWER
#include <sofa/gui/qt/viewer/qtogre/QtOgreViewer.h>
#endif

#ifdef SOFA_GUI_QTVIEWER
#include <sofa/gui/qt/viewer/qt/QtViewer.h>
#endif

#ifdef SOFA_GUI_QGLVIEWER
#include <sofa/gui/qt/viewer/qgl/QtGLViewer.h>
#endif

#include <sofa/simulation/common/InitVisitor.h>
#include <sofa/simulation/common/DesactivatedNodeVisitor.h>

#ifdef SOFA_DEV

#include <sofa/simulation/automatescheduler/ThreadSimulation.h>
#include <sofa/simulation/automatescheduler/ExecBus.h>
#include <sofa/simulation/automatescheduler/Node.h>
#include <sofa/simulation/automatescheduler/AutomateUtils.h>

#endif // SOFA_DEV

#ifdef SOFA_HAVE_CHAI3D
#include <sofa/simulation/common/PropagateEventVisitor.h>
#include <sofa/core/objectmodel/GLInitializedEvent.h>
#endif // SOFA_HAVE_CHAI3D


#include <sofa/component/visualmodel/VisualModelImpl.h>

#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/tree/xml/XML.h>
#include <sofa/simulation/common/TransformationVisitor.h>
#include <sofa/helper/system/FileRepository.h>

#define MAX_RECENTLY_OPENED 10

#ifdef SOFA_QT4
#include <QWidget>
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
#endif

#include <GenGraphForm.h>

#include <stdio.h>


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
//       using namespace sofa::simulation::tree;

#ifdef SOFA_DEV
using namespace sofa::simulation::automatescheduler;
typedef sofa::simulation::automatescheduler::Node AutomateNode;
#endif // SOFA_DEV


///////////////////////////////////////////////////////////
//////////////////// SofaGUI Interface ////////////////////
///////////////////////////////////////////////////////////

#ifdef SOFA_GUI_QGLVIEWER
SOFA_DECL_CLASS ( QGLViewerGUI )
int QGLViewerGUIClass = SofaGUI::RegisterGUI ( "qglviewer", &RealGUI::CreateGUI, &RealGUI::InitGUI, 3 );
#endif
#ifdef SOFA_GUI_QTVIEWER
SOFA_DECL_CLASS ( QTGUI )
int QtGUIClass = SofaGUI::RegisterGUI ( "qt", &RealGUI::CreateGUI, &RealGUI::InitGUI, 2 );
#endif
#ifdef SOFA_GUI_QTOGREVIEWER
SOFA_DECL_CLASS ( OgreGUI )
int QtOGREGUIClass = SofaGUI::RegisterGUI ( "ogre", &RealGUI::CreateGUI, &RealGUI::InitGUI, 1 );
#endif

int RealGUI::InitGUI ( const char* name, const std::vector<std::string>& /* options */ )
{
    ImageQt::Init();
#ifdef SOFA_GUI_QGLVIEWER
    if ( !name[0] || !strcmp ( name,"qglviewer" ) )
    {
        return sofa::gui::qt::viewer::qgl::QtGLViewer::EnableViewer();
    }
    else
#endif
#ifdef SOFA_GUI_QTVIEWER
        if ( !name[0] || !strcmp ( name,"qt" ) )
        {
            return sofa::gui::qt::viewer::qt::QtViewer::EnableViewer();
        }
        else
#endif
#ifdef SOFA_GUI_QTOGREVIEWER
            if ( !name[0] || !strcmp ( name,"ogre" ) )
            {
                return sofa::gui::qt::viewer::qtogre::QtOgreViewer::EnableViewer();
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

SofaGUI* RealGUI::CreateGUI ( const char* name, const std::vector<std::string>& options, sofa::simulation::Node* node, const char* filename )
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
    Node *root = dynamic_cast< Node* >(node);
    if ( !root )
        root = simulation::getSimulation()->newNode("Root");

    gui->setScene ( root, filename );

    //gui->viewer->resetView();

    application->setMainWidget ( gui );

#ifdef SOFA_DEV

    // Threads Management
    if ( sofa::simulation::automatescheduler::ThreadSimulation::initialized() )
    {
        sofa::simulation::automatescheduler::ThreadSimulation::getInstance()->computeVModelsList ( root );
        root->setMultiThreadSimulation ( true );
        sofa::simulation::automatescheduler::groot = root;

        sofa::simulation::automatescheduler::Automate::setDrawCB ( gui->viewer );

        gui->viewer->getQWidget()->update();
        sofa::simulation::automatescheduler::ThreadSimulation::getInstance()->start();
    }

#endif // SOFA_DEV

    // show the gui
    gui->show();


#ifdef SOFA_HAVE_CHAI3D
    // Tell nodes that openGl is initialized
    // especialy the GL_MODELVIEW_MATRIX
    sofa::core::objectmodel::GLInitializedEvent ev;
    sofa::simulation::PropagateEventVisitor act(&ev);
    root->execute(act);
#endif // SOFA_HAVE_CHAI3D

    return gui;
}

int RealGUI::mainLoop()
{

#ifdef SOFA_QT4
    QString title = windowTitle();
#else
    QString title = caption();
#endif

    title.remove(QString("Sofa - "), true);
    std::string title_str(title.ascii());

    if ( (title_str.rfind(".simu") != std::string::npos) && sofa::helper::system::DataRepository.findFile (title_str) )
    {
        title_str = sofa::helper::system::DataRepository.getFile ( title_str );

        fileOpenSimu(title_str.c_str() );
    }



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
    : viewerName ( viewername ), viewer ( NULL ), currentTab ( NULL ), tabInstrument (NULL),  graphListener ( NULL ), dialog ( NULL )
{
    connect(this, SIGNAL(quit()), this, SLOT(fileExit()));

#ifdef SOFA_QT4
    fileMenu->removeAction(Action);
#endif

    displayFlag = new DisplayFlagWidget(tabView);
    connect( displayFlag, SIGNAL( change(int,bool)), this, SLOT(showhideElements(int,bool) ));
    gridLayout1->addWidget(displayFlag,0,0);


    left_stack = new QWidgetStack ( splitter2 );
    connect ( startButton, SIGNAL ( toggled ( bool ) ), this , SLOT ( playpauseGUI ( bool ) ) );

    //Status Bar Configuration
    fpsLabel = new QLabel ( "9999.9 FPS", statusBar() );
    fpsLabel->setMinimumSize ( fpsLabel->sizeHint() );
    fpsLabel->clear();

    timeLabel = new QLabel ( "Time: 999.9999 s", statusBar() );
    timeLabel->setMinimumSize ( timeLabel->sizeHint() );
    timeLabel->clear();

    initialTime = new QLabel( "Init:", statusBar() );
    initialTime->setMinimumSize ( initialTime->sizeHint() );

    timeSlider = new QSlider( Qt::Horizontal, statusBar(), "Time Slider");
    timeSlider->setTickmarks(QSlider::Both);
    timeSlider->setMinValue(0);
    timeSlider->setMaxValue(0);

    finalTime = new QLabel( "End:", statusBar() );
    finalTime->setMinimumSize ( finalTime->sizeHint() );

    std::string pixmap_filename;

    record                 = new QPushButton( statusBar(), "Record");  	record->setToggleButton(true);
    backward_record        = new QPushButton( statusBar(), "Backward");
    stepbackward_record    = new QPushButton( statusBar(), "Step Backward");
    playforward_record     = new QPushButton( statusBar(), "Play Forward"); playforward_record->setToggleButton(true);
    stepforward_record     = new QPushButton( statusBar(), "Step Forward");
    forward_record         = new QPushButton( statusBar(), "Forward");

    QToolTip::add(record               , tr( "Record" ) );
    QToolTip::add(backward_record      , tr( "Load Initial Time" ) );
    QToolTip::add(stepbackward_record  , tr( "Make one step backward" ) );
    QToolTip::add(playforward_record   , tr( "Continuous play forward" ) );
    QToolTip::add(stepforward_record   , tr( "Make one step forward" ) );
    QToolTip::add(forward_record       , tr( "Load Final Time" ) );


    setPixmap("textures/media-record.png", record);
    setPixmap("textures/media-seek-backward.png", backward_record);
    setPixmap("textures/media-skip-backward.png", stepbackward_record);
    setPixmap("textures/media-playback-start.png", playforward_record);
    setPixmap("textures/media-skip-forward.png", stepforward_record);
    setPixmap("textures/media-seek-forward.png", forward_record);


    QLabel *timeRecord = new QLabel("T=",statusBar());
    loadRecordTime = new QLineEdit(statusBar());
    loadRecordTime->setMaximumSize(QSize(75, 100));


    statusBar()->addWidget ( fpsLabel );
    statusBar()->addWidget ( timeLabel );


    statusBar()->addWidget( record);
    statusBar()->addWidget( backward_record);
    statusBar()->addWidget( stepbackward_record);
    statusBar()->addWidget( playforward_record);
    statusBar()->addWidget( stepforward_record);
    statusBar()->addWidget( forward_record);
    statusBar()->addWidget( timeRecord);

    statusBar()->addWidget( loadRecordTime);

    statusBar()->addWidget ( initialTime );
    statusBar()->addWidget( timeSlider);
    statusBar()->addWidget ( finalTime );

    timerStep       = new QTimer ( this );
    timerRecordStep = new QTimer ( this );


    connect ( timerStep, SIGNAL ( timeout() ), this, SLOT ( step() ) );
    connect ( timerRecordStep, SIGNAL ( timeout() ), this, SLOT ( slot_stepforward() ) );
    connect ( ResetSceneButton, SIGNAL ( clicked() ), this, SLOT ( resetScene() ) );
    connect ( dtEdit, SIGNAL ( textChanged ( const QString& ) ), this, SLOT ( setDt ( const QString& ) ) );
    connect ( stepButton, SIGNAL ( clicked() ), this, SLOT ( step() ) );
    connect ( ExportGraphButton, SIGNAL ( clicked() ), this, SLOT ( exportGraph() ) );
    connect ( dumpStateCheckBox, SIGNAL ( toggled ( bool ) ), this, SLOT ( dumpState ( bool ) ) );
    connect ( exportGnuplotFilesCheckbox, SIGNAL ( toggled ( bool ) ), this, SLOT ( setExportGnuplot ( bool ) ) );
    connect ( exportVisitorCheckbox, SIGNAL ( toggled ( bool ) ), this, SLOT ( setExportVisitor ( bool ) ) );
    connect ( displayComputationTimeCheckBox, SIGNAL ( toggled ( bool ) ), this, SLOT ( displayComputationTime ( bool ) ) );

    connect ( record, SIGNAL (toggled (bool) ),              this, SLOT( slot_recordSimulation( bool) ) );
    connect ( backward_record, SIGNAL (clicked () ),         this, SLOT( slot_backward( ) ) );
    connect ( stepbackward_record, SIGNAL (clicked () ),     this, SLOT( slot_stepbackward( ) ) );
    connect ( playforward_record,  SIGNAL (clicked () ),     this, SLOT( slot_playforward( ) ) );
    connect ( stepforward_record,  SIGNAL (clicked () ),     this, SLOT( slot_stepforward( ) ) );
    connect ( forward_record, SIGNAL (clicked () ),          this, SLOT( slot_forward( ) ) );
    connect ( loadRecordTime, SIGNAL(returnPressed ()),       this, SLOT( slot_loadrecord_timevalue()));
    connect ( timeSlider, SIGNAL (sliderMoved (int) ),   this, SLOT( slot_sliderValue( int) ) );


    connect( recentlyOpened, SIGNAL(activated(int)), this, SLOT(fileRecentlyOpened(int)));
    //Recently Opened Files
    std::string scenes ( "config/Sofa.ini" );
    if ( !sofa::helper::system::DataRepository.findFile ( scenes ) )
    {
        std::string fileToBeCreated = sofa::helper::system::DataRepository.getFirstPath() + "/" + scenes;

        std::ofstream ofile(fileToBeCreated.c_str());
        ofile << "";
        ofile.close();
    }
    pathDumpVisitor = sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str()) + std::string( "/dumpVisitor.xml" );
    scenes = sofa::helper::system::DataRepository.getFile ( scenes );

    updateRecentlyOpened("");


    //Dialog Add Object
    connect ( tabs, SIGNAL ( currentChanged ( QWidget* ) ), this, SLOT ( currentTabChanged ( QWidget* ) ) );

    addViewer();
    currentTabChanged ( tabs->currentPage() );

    //ADD GUI for Background
    //------------------------------------------------------------------------
    //Informations
    Q3GroupBox *groupInfo = new Q3GroupBox(QString("Background"), tabs->page(3));
    groupInfo->setColumns(4);
    QWidget     *global    = new QWidget(groupInfo);
    QGridLayout *globalLayout = new QGridLayout(global);


    globalLayout->addWidget(new QLabel(QString("Colour "),global),1,0);
    for (unsigned int i=0; i<3; ++i)
    {
        std::ostringstream s;
        s<<"background" <<i;
        background[i] = new WFloatLineEdit(global,s.str().c_str());
        background[i]->setMinFloatValue( 0.0f);
        background[i]->setMaxFloatValue( 1.0f);
        background[i]->setFloatValue( 1.0f);
        globalLayout->addWidget(background[i],1,i+1);
        connect( background[i], SIGNAL( returnPressed() ), this, SLOT( updateBackgroundColour() ) );
    }

    QWidget     *global2    = new QWidget(groupInfo);
    groupInfo->setColumns(1);
    QGridLayout *globalLayout2 = new QGridLayout(global2);
    globalLayout2->addWidget(new QLabel(QString("Image "),global2),2,0);
    backgroundImage = new QLineEdit(global2,"backgroundImage");
    backgroundImage->setMinimumWidth( 200 );
    backgroundImage->setText( QString(viewer->getBackgroundImage().c_str()) );
    globalLayout2->addWidget(backgroundImage,2,1);
    connect( backgroundImage, SIGNAL( returnPressed() ), this, SLOT( updateBackgroundImage() ) );


#ifdef SOFA_QT4
    vboxLayout4->insertWidget(1,groupInfo);
#else
    TabPageLayout->insertWidget(1,groupInfo);
#endif

    //---------------------------------------------------------------------------------------------------
#ifdef SOFA_PML
    pmlreader = NULL;
    lmlreader = NULL;
#endif

    //Center the application
    const QRect screen = QApplication::desktop()->availableGeometry(QApplication::desktop()->primaryScreen());
    this->move(  ( screen.width()- this->width()  ) / 2,  ( screen.height() - this->height()) / 2  );

}

void RealGUI::fileRecentlyOpened(int id)
{
    fileOpen(recentlyOpened->text(id).ascii());
}

void RealGUI::updateRecentlyOpened(std::string fileLoaded)
{

#ifdef WIN32
    for (unsigned int i=0; i<fileLoaded.size(); ++i)
    {
        if (fileLoaded[i] == '\\') fileLoaded[i] = '/';
    }
#endif
    std::string scenes ( "config/Sofa.ini" );

    scenes = sofa::helper::system::DataRepository.getFile ( scenes );

    std::vector< std::string > list_files;
    std::ifstream end(scenes.c_str());
    std::string s;
    while( std::getline(end,s) )
    {
        if (s != fileLoaded)
            list_files.push_back(sofa::helper::system::DataRepository.getFile(s));
    }
    end.close();


    recentlyOpened->clear();
    std::ofstream out;
    out.open(scenes.c_str(),std::ios::out);
    if (sofa::helper::system::DataRepository.findFile(fileLoaded))
    {
        fileLoaded = sofa::helper::system::DataRepository.getFile(fileLoaded);
        out << fileLoaded << "\n";

        recentlyOpened->insertItem(QString(fileLoaded.c_str()));
    }
    for (unsigned int i=0; i<list_files.size() && i<MAX_RECENTLY_OPENED ; ++i)
    {
        if (fileLoaded != list_files[i])
        {
            recentlyOpened->insertItem(QString(list_files[i].c_str()));
            out << list_files[i] << "\n";
        }
    }

    out.close();


}

void RealGUI::setPixmap(std::string pixmap_filename, QPushButton* b)
{
    if ( sofa::helper::system::DataRepository.findFile ( pixmap_filename ) )
        pixmap_filename = sofa::helper::system::DataRepository.getFile ( pixmap_filename );
    b->setPixmap(QPixmap(QImage(pixmap_filename.c_str())));
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
    delete displayFlag;
    delete windowTraceVisitor;
    delete handleTraceVisitor;
    if (dialog) delete dialog;
}

void RealGUI::init()
{

    frameCounter = 0;
    node_clicked = NULL;
    item_clicked = NULL;
    _animationOBJ = false;
    _animationOBJcounter = 0;
    m_dumpState = false;
    m_dumpStateStream = 0;
    m_displayComputationTime = false;
    m_exportGnuplot = false;
    record_directory = sofa::helper::system::SetDirectory::GetRelativeFromDir("../../examples/Simulation/",sofa::helper::system::SetDirectory::GetProcessFullPath("").c_str());
    gnuplot_directory = "";
    writeSceneName = "";

    current_Id_modifyDialog = 0;
    map_modifyDialogOpened.clear();



#ifndef SOFA_DUMP_VISITOR_INFO
    //Remove option to see visitor trace
    this->exportVisitorCheckbox->hide();
#endif
    //Main window containing a QListView only
    windowTraceVisitor = new WindowVisitor;
    windowTraceVisitor->graphView->setSorting(-1);
    windowTraceVisitor->hide();
    connect(windowTraceVisitor, SIGNAL(WindowVisitorClosed(bool)), this->exportVisitorCheckbox, SLOT(setChecked(bool)));
    handleTraceVisitor = new GraphVisitor(windowTraceVisitor);

    //--------
    descriptionScene = new QDialog(this);
    descriptionScene->resize(400,400);
    QVBoxLayout *descriptionLayout = new QVBoxLayout(descriptionScene);
    htmlPage = new QTextBrowser(descriptionScene);
    descriptionLayout->addWidget(htmlPage);
    connect(htmlPage, SIGNAL(sourceChanged(const QUrl&)), this, SLOT(changeHtmlPage(const QUrl&)));
    //--------
    pluginManager = new SofaPluginManager;
    pluginManager->hide();
    //*********************************************************************************************************************************
    //List of objects
    //Read the object.txt that contains the information about the objects which can be added to the scenes whithin a given BoundingBox and scale range
    std::string object ( "config/object.txt" );
    if ( !sofa::helper::system::DataRepository.findFile ( object ) )
        return;

    object = sofa::helper::system::DataRepository.getFile ( object );
    list_object.clear();
    std::ifstream end(object.c_str());
    std::string s;
    while( end >> s )
    {
        list_object.push_back(s);
    }
    end.close();
}

void RealGUI::addViewer()
{
    init();
    const char* name = viewerName;

    // set menu state

#ifdef SOFA_GUI_QTVIEWER
    viewerOpenGLAction->setEnabled ( true );
#else
    viewerOpenGLAction->setEnabled ( false );
    viewerOpenGLAction->setToolTip ( "enable SOFA_GUI_QTVIEWER in sofa-local.cfg to activate" );
#endif
#ifdef SOFA_GUI_QGLVIEWER
    viewerQGLViewerAction->setEnabled ( true );
#else
    viewerQGLViewerAction->setEnabled ( false );
    viewerQGLViewerAction->setToolTip ( "enable SOFA_GUI_QGLVIEWER in sofa-local.cfg to activate" );
#endif
#ifdef SOFA_GUI_QTOGREVIEWER
    viewerOGREAction->setEnabled ( true );
#else
    viewerOGREAction->setEnabled ( false );
    viewerOGREAction->setToolTip ( "enable SOFA_GUI_QTOGREVIEWER in sofa-local.cfg to activate" );
#endif

#ifdef SOFA_GUI_QGLVIEWER
    if ( !name[0] || !strcmp ( name,"qglviewer" ) )
    {
        viewer = new sofa::gui::qt::viewer::qgl::QtGLViewer ( left_stack, "viewer" );
        viewerQGLViewerAction->setOn ( true );
    }
    else
#endif
#ifdef SOFA_GUI_QTVIEWER
        if ( !name[0] || !strcmp ( name,"qt" ) )
        {
            viewer = new sofa::gui::qt::viewer::qt::QtViewer ( left_stack, "viewer" );
            viewerOpenGLAction->setOn ( true );
        }
        else
#endif
#ifdef SOFA_GUI_QTOGREVIEWER
            if ( !name[0] || !strcmp ( name,"ogre" ) )
            {
                viewer = new sofa::gui::qt::viewer::qtogre::QtOgreViewer ( left_stack , "viewer" );
                viewerOGREAction->setOn ( true );
            }
            else
#endif
            {
                std::cerr << "ERROR(QtGUI): unknown or disabled viewer name "<<name<<std::endl;
                application->exit();
            }

#ifdef SOFA_QT4
    left_stack->addWidget ( viewer->getQWidget() );
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

    viewer->setup();
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
    QList<int> list;
#else
    QValueList<int> list;
#endif
    list.push_back ( 259 );
    list.push_back ( 525 );
    splitter_ptr->setSizes ( list );

    viewer->getQWidget()->setFocus();
    viewer->getQWidget()->show();
    viewer->getQWidget()->update();
    setGUI();


}

void RealGUI::viewerOpenGL()
{
    setViewer ( "qt" );
    viewerOpenGLAction->setOn(true);
    viewerQGLViewerAction->setOn(false);
    viewerOGREAction->setOn(false);
}

void RealGUI::viewerQGLViewer()
{
    setViewer ( "qglviewer" );
    viewerOpenGLAction->setOn(false);
    viewerQGLViewerAction->setOn(true);
    viewerOGREAction->setOn(false);
}

void RealGUI::viewerOGRE()
{
    setViewer ( "ogre" );
    viewerOpenGLAction->setOn(false);
    viewerQGLViewerAction->setOn(false);
    viewerOGREAction->setOn(true);

}

bool RealGUI::setViewer ( const char* name )
{

    if ( !strcmp ( name,viewerName ) )
        return true; // nothing to do
    if ( !strcmp ( name,"qt" ) )
    {
#ifndef SOFA_GUI_QTVIEWER
        std::cerr << "OpenGL viewer not activated. Enable SOFA_GUI_QGLVIEWER in sofa-local.cfg to activate.\n";
        return false;
#endif
    }
    else if ( !strcmp ( name,"qglviewer" ) )
    {
#ifndef SOFA_GUI_QGLVIEWER
        std::cerr << "QGLViewer viewer not activated. Enable SOFA_GUI_QGLVIEWER in sofa-local.cfg to activate.\n";
        return false;
#endif
    }
    else if ( !strcmp ( name,"ogre" ) )
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

    if ( viewer->getScene() !=NULL )
    {
        simulation::getSimulation()->unload ( viewer->getScene() );
        if ( graphListener!=NULL )
        {
            delete graphListener;
            graphListener = NULL;
        }
        graphView->clear();
    }

// 	fileOpen(filename);
// 	GNode* groot = new GNode; // empty scene to do the transition
// 	setScene ( groot,filename.c_str() ); // keep the current display flags


    viewer->removeViewerTab(tabs);

    left_stack->removeWidget ( viewer->getQWidget() );
    delete viewer;
    viewer = NULL;
    // Disable Viewer-specific classes
#ifdef SOFA_GUI_QTVIEWER
    if ( !strcmp ( viewerName,"qt" ) )
    {
        sofa::gui::qt::viewer::qt::QtViewer::DisableViewer();
    }
    else
#endif
#ifdef SOFA_GUI_QGLVIEWER
        if ( !strcmp ( viewerName,"qglviewer" ) )
        {
            sofa::gui::qt::viewer::qgl::QtGLViewer::DisableViewer();
        }
        else
#endif
#ifdef SOFA_GUI_QTOGREVIEWER
            if ( !strcmp ( viewerName,"ogre" ) )
            {
                sofa::gui::qt::viewer::qtogre::QtOgreViewer::DisableViewer();
            }
            else
#endif
            {}
    // Enable Viewer-specific classes
#ifdef SOFA_GUI_QTVIEWER
    if ( !strcmp ( name,"qt" ) )
    {
        sofa::gui::qt::viewer::qt::QtViewer::EnableViewer();
    }
    else
#endif
#ifdef SOFA_GUI_QGLVIEWER
        if ( !strcmp ( name,"qglviewer" ) )
        {
            sofa::gui::qt::viewer::qgl::QtGLViewer::EnableViewer();
        }
        else
#endif
#ifdef SOFA_GUI_QTOGREVIEWER
            if ( !strcmp ( name,"ogre" ) )
            {
                sofa::gui::qt::viewer::qtogre::QtOgreViewer::EnableViewer();
            }
#endif


    viewerName = name;

    addViewer();

    viewer->configureViewerTab(tabs);



    if (filename.rfind(".simu") != std::string::npos)
        fileOpenSimu(filename.c_str() );
    //else if (filename.rfind(".pscn") != std::string::npos)
    //  fileOpenScript(filename.c_str(), PHP );
    //else
    fileOpen ( filename.c_str() ); // keep the current display flags
    return true;
}

void RealGUI::fileOpen ( std::string filename )
{
    if ( sofa::helper::system::DataRepository.findFile (filename) )
        filename = sofa::helper::system::DataRepository.getFile ( filename );
    else
        return;

    startDumpVisitor();

    frameCounter = 0;
    sofa::simulation::tree::xml::numDefault = 0;
    list_object_added.clear();
    list_object_removed.clear();
    list_object_initial.clear();
    writeSceneName="";

    update();

    //Hide the dialog to add a new object in the graph
    if ( dialog != NULL ) dialog->hide();
    //Hide all the dialogs to modify the graph
    emit ( newScene() );

    if ( viewer->getScene() !=NULL )
    {
        simulation::getSimulation()->unload ( viewer->getScene() );
        if ( graphListener!=NULL )
        {
            delete graphListener;
            graphListener = NULL;
        }
        graphView->clear();
    }

    //Clear the list of modified dialog opened
    current_Id_modifyDialog=0;
    map_modifyDialogOpened.clear();

    simulation::Node* root = simulation::getSimulation()->load ( filename.c_str() );
    simulation::getSimulation()->init ( root );
    if ( root == NULL )
    {
        qFatal ( "Failed to load %s",filename.c_str() );
        stopDumpVisitor();
        return;
    }

    setScene ( root, filename.c_str() );
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
        simulation::getSimulation()->unload ( viewer->getScene() );
        if ( graphListener!=NULL )
        {
            delete graphListener;
            graphListener = NULL;
        }
        graphView->clear();
    }
    GNode *simuNode = dynamic_cast< GNode *> (simulation::getSimulation()->load ( scene.c_str() ));
    getSimulation()->init(simuNode);
    if ( simuNode )
    {
        if ( !pmlreader ) pmlreader = new PMLReader;
        pmlreader->BuildStructure ( filename, simuNode );
        setScene ( simuNode, filename );
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

void RealGUI::initDesactivatedNode()
{

    std::map<core::objectmodel::Base*, QListViewItem* >::iterator graph_iterator;

    for (graph_iterator = graphListener->items.begin(); graph_iterator != graphListener->items.end(); graph_iterator++)
    {
        node_clicked = dynamic_cast< Node* >(graph_iterator->first);
        if (node_clicked!=NULL )
        {
            if (!node_clicked->isActive() )
            {
                item_clicked =  graphListener->items[node_clicked];
                graphDesactivateNode();
            }
        }
    }
}


void RealGUI::setScene ( Node* root, const char* filename )
{
    if (filename)
    {
        updateRecentlyOpened(filename);
        setTitle ( filename );
        std::string extension=sofa::helper::system::SetDirectory::GetExtension(filename);
        std::string htmlFile=filename; htmlFile.resize(htmlFile.size()-extension.size()-1);
        htmlFile+=".html";
        if (sofa::helper::system::DataRepository.findFile (htmlFile))
        {
            htmlFile = sofa::helper::system::DataRepository.getFile (htmlFile);
#ifdef WIN32
            htmlFile = "file:///"+htmlFile;
#endif
            descriptionScene->show();
            htmlPage->setSource(QUrl(QString(htmlFile.c_str())));
        }
        else
        {
            htmlPage->clear();
            descriptionScene->hide();
        }
    }
    else
    {
        htmlPage->clear();
        descriptionScene->hide();
    }

    if (tabInstrument!= NULL)
    {
        tabs->removePage(tabInstrument);
        delete tabInstrument;
        tabInstrument = NULL;
    }


    graphView->clear();
    viewer->setScene ( root, filename );
    viewer->resetView();
    initial_time = (root != NULL)?root->getTime():0;

    record_simulation = false;
    clearRecord();
    clearGraph();

    initDesactivatedNode();

    eventNewTime();

    if (root)
    {
        // set state of display flags
        displayFlag->setFlag(DisplayFlagWidget::VISUAL,root->getContext()->getShowVisualModels());
        displayFlag->setFlag(DisplayFlagWidget::BEHAVIOR,root->getContext()->getShowBehaviorModels());
        displayFlag->setFlag(DisplayFlagWidget::COLLISION,root->getContext()->getShowCollisionModels());
        displayFlag->setFlag(DisplayFlagWidget::BOUNDING,root->getContext()->getShowBoundingCollisionModels());
        displayFlag->setFlag(DisplayFlagWidget::MAPPING,root->getContext()->getShowMappings());
        displayFlag->setFlag(DisplayFlagWidget::MECHANICALMAPPING,root->getContext()->getShowMechanicalMappings());
        displayFlag->setFlag(DisplayFlagWidget::FORCEFIELD,root->getContext()->getShowForceFields());
        displayFlag->setFlag(DisplayFlagWidget::INTERACTION,root->getContext()->getShowInteractionForceFields());
        displayFlag->setFlag(DisplayFlagWidget::WIREFRAME,root->getContext()->getShowWireFrame());
        displayFlag->setFlag(DisplayFlagWidget::NORMALS,root->getContext()->getShowNormals());

        //simulation::getSimulation()->updateVisualContext ( root );
        startButton->setOn ( root->getContext()->getAnimate() );
        dtEdit->setText ( QString::number ( root->getDt() ) );
    }
    record->setOn(false);

#ifdef SOFA_HAVE_CHAI3D
    // Tell nodes that openGl is initialized
    // especialy the GL_MODELVIEW_MATRIX
    sofa::core::objectmodel::GLInitializedEvent ev;
    sofa::simulation::PropagateEventVisitor act(&ev);
    root->execute(act);
#endif // SOFA_HAVE_CHAI3D

#ifdef SOFA_GUI_QTOGREVIEWER
    if (std::string(sofa::gui::SofaGUI::GetGUIName()) == "ogre")
        resetScene();
#endif

}

void RealGUI::setDimension ( int w, int h )
{
    resize(w,h);
}
void RealGUI::setFullScreen ()
{
    showFullScreen();
}
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

            writeSceneName = writeName;
            addReadState(true);


            char buf[100];

            sprintf ( buf, "Init: %s s",initT.c_str()  );
            initialTime->setText ( buf );

            sprintf ( buf, "End: %s s",endT.c_str()  );
            finalTime->setText ( buf );

            loadRecordTime->setText( QString(initT.c_str()) );

            dtEdit->setText(QString(dT.c_str()));
            timeSlider->setMaxValue( (int)((atof(endT.c_str())-atof(initT.c_str()))/(atof(dT.c_str()))+0.5));

            record_directory = sofa::helper::system::SetDirectory::GetParentDir(filename.c_str()) + "/";
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
    std::string filename = viewer->getSceneFileName();

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

    std::string filename = viewer->getSceneFileName();
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
            fileOpen ( filename );
    }
#else
    if (s.endsWith( ".simu") )
        fileOpenSimu(s.ascii());
    else
        fileOpen ( s.ascii() );
#endif

}

void RealGUI::fileSave()
{
    Node *node = viewer->getScene();
    std::string filename = viewer->getSceneFileName();
    fileSaveAs ( node,filename.c_str() );
}


void RealGUI::fileSaveAs(Node *node)
{
    if (node == NULL) node = viewer->getScene();
    QString s;
    std::string filename = viewer->getSceneFileName();
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
    simulation::getSimulation()->printXML ( node, filename );
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
    simulation::getSimulation()->printXML ( viewer->getScene(), "scene.scn" );
}

void RealGUI::editRecordDirectory()
{
    std::string filename = viewer->getSceneFileName();
    QString s = getExistingDirectory ( this, filename.empty() ?NULL:filename.c_str(), "open directory dialog",  "Choose a directory" );
    if (s.length() > 0)
    {
        record_directory = s.ascii();
        if (record_directory.at(record_directory.size()-1) != '/') record_directory+="/";
    }

}

void RealGUI::showPluginManager()
{
    pluginManager->show();
}

void RealGUI::editGnuplotDirectory()
{
    std::string filename = viewer->getSceneFileName();
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
    if ( value ) {timerStep->start ( 0 );}
    else {timerStep->stop();}
    if ( getScene() )  getScene()->getContext()->setAnimate ( value );
}


void RealGUI::setGUI ( void )
{
    textEdit1->setText ( viewer->helpString() );
    connect ( this, SIGNAL( newStep()), viewer->getQWidget(), SLOT( update()));
    /*
    #ifdef SOFA_GUI_QTOGREVIEWER
    	//Hide unused options
    	if ( !strcmp ( viewerName,"ogre" ) )
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
    #endif*/
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

#ifdef SOFA_DEV

    if ( root->getContext()->getMultiThreadSimulation() )
    {
        static AutomateNode* n = NULL;

        if ( ExecBus::getInstance() != NULL )
        {
            n = ExecBus::getInstance()->getNext ( "displayThread", n );

            if ( n )
            {
                n->execute ( "displayThread" );
            }
        }
    }
    else

#endif // SOFA_DEV

    {
        if ( viewer->ready() ) return;



        //root->setLogTime(true);
        simulation::getSimulation()->animate ( root, root->getDt() );

        if ( m_dumpState )
            simulation::getSimulation()->dumpState ( root, *m_dumpStateStream );
        if ( m_exportGnuplot )
            simulation::getSimulation()->exportGnuplot ( root, root->getTime() );

        viewer->wait();

        eventNewStep();
        eventNewTime();

//    	    viewer->getQWidget()->update();

        if (currentTab == TabStats) graphCreateStats(viewer->getScene());
    }


    if ( _animationOBJ )
    {
#ifdef CAPTURE_PERIOD
        static int counter = 0;
        if ( ( counter++ % CAPTURE_PERIOD ) ==0 )
#endif
        {
            exportOBJ ( false );
            ++_animationOBJcounter;
        }
    }

    stopDumpVisitor();
    emit newStep();
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
        // 	    emit newFPS(fps);
        char buf[100];
        sprintf ( buf, "%.1f FPS", fps );
        fpsLabel->setText ( buf );
        // 	    emit newFPS(buf);
        beginTime[i] = curtime;
        //frameCounter = 0;
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



void RealGUI::eventNewTime()
{
    Node* root = getScene();
    if ( root )
    {

        double time = root->getTime();
        char buf[100];
        sprintf ( buf, "Time: %.3g s", time );
        timeLabel->setText ( buf );

        if (record_simulation)
        {
            setRecordTime(time);
            double final_time = getRecordFinalTime();

            if ((int)(1000*final_time) < (int)(1000*time))
            {
                setRecordFinalTime(time);
                timeSlider->setMaxValue(timeSlider->maxValue()+1);
                timeSlider->setValue(timeSlider->maxValue());
            }
            else
            {
                timeSlider->setValue(timeSlider->value()+1);
            }
            timeSlider->update();
        }
    }
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

    //Hide the dialog to add a new object in the graph
    if ( dialog != NULL ) dialog->hide();
    //Hide all the dialogs to modify the graph
    emit ( newScene() );

    //Clear the list of modified dialog opened
    current_Id_modifyDialog=0;
    map_modifyDialogOpened.clear();


    std::list< Node *>::iterator it;
    //**************************************************************
    //GRAPH MANAGER
    bool isFrozen = graphListener->frozen;
    if (root) graphListener->unfreeze ( root );
    std::map<core::objectmodel::Base*, Q3ListViewItem* >::iterator graph_iterator;


    //Remove all the objects added
    bool node_removed ;
    for ( it=list_object_added.begin(); it != list_object_added.end(); it++ )
    {
        node_removed = false;

        //Verify if they have not been removed before
        std::list< Node *>::iterator it_removed;
        for ( it_removed=list_object_removed.begin(); it_removed != list_object_removed.end(); it_removed++ )
        {
            if ( ( *it_removed ) == ( *it ) ) { node_removed=true; continue;} //node already removed
        }
        if ( node_removed ) continue;
        ( *it )->detachFromGraph();
        graphListener->removeChild ( NULL, ( *it ) );
        delete ( *it );
    }

    list_object_added.clear();


    //Add all the objects present at initial time
    //Begin from the last removed item: the last one can be the parent of one removed lately in the simulation
    it=list_object_removed.end();

    while ( true )
    {
        if ( it == list_object_removed.begin() ) break;
        --it;
        std::list< std::pair<Node *,Node *> >::iterator it_initial;
        for ( it_initial=list_object_initial.begin(); it_initial != list_object_initial.end(); it_initial++ )
        {
            if ( ( it_initial->second ) == ( *it ) )
            {
                ( it_initial->first )->addChild ( (Node*)( *it ) );
                graphListener->addObject (  ( it_initial->second ), ( core::objectmodel::BaseObject* ) ( *it ) );
                continue;
            }
        }
    }

    list_object_removed.clear();




    if ( root && isFrozen ) graphListener->freeze ( root );

    //Reset the scene
    if ( root )
    {
        simulation::getSimulation()->reset ( root );
        root->setTime(initial_time);
        eventNewTime();

        //viewer->resetView();
        emit newStep();
// 	    viewer->getQWidget()->update();
    }
}


//*****************************************************************************************
//
void RealGUI::exportGraph()
{
    exportGraph ( getScene() );
}


void RealGUI::exportGraph ( sofa::simulation::Node* root )
{

    if ( root == NULL ) return;
    sofa::gui::qt::GenGraphForm* form = new sofa::gui::qt::GenGraphForm;
    form->setScene ( root );
    std::string gname = viewer->getSceneFileName();
    std::size_t gpath = gname.find_last_of("/\\");
    std::size_t gext = gname.rfind('.');
    if (gext != std::string::npos && (gpath == std::string::npos || gext > gpath))
        gname = gname.substr(0,gext);
    form->filename->setText(gname.c_str());
    form->show();
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
void RealGUI::setExportVisitor ( bool exp )
{
    if (exp)
    {
        std::string pFilename = sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str()) + std::string( "/dumpVisitor.xml" );
        windowTraceVisitor->show();
        handleTraceVisitor->clear();
    }
    else
    {
        windowTraceVisitor->hide();
    }
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
void RealGUI::exportOBJ ( bool exportMTL )
{
    Node* root = getScene();
    if ( !root ) return;
    std::string sceneFileName = viewer->getSceneFileName();
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
    // ignore if there are modifiers (i.e. CTRL of SHIFT)
#ifdef SOFA_QT4
    if (e->modifiers()) return;
#else
    if (e->state() & (Qt::KeyButtonMask)) return;
#endif
    switch ( e->key() )
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


/*****************************************************************************************************************/
//Translate an object
void RealGUI::transformObject ( Node *node, double dx, double dy, double dz,  double rx, double ry, double rz, double scale )
{
    if ( node == NULL ) return;
    const SReal conversionDegRad = 3.141592653/180.0;
    Vector3 rotationVector = Vector3(rx,ry,rz)*conversionDegRad;

    TransformationVisitor transform;
    transform.setTranslation(dx,dy,dz);
    transform.setRotation(rx,ry,rz);
    transform.setScale(scale);
    transform.execute(node);

}

/*****************************************************************************************************************/
void RealGUI::loadObject ( std::string path, double dx, double dy, double dz,  double rx, double ry, double rz,double scale )
{
    //Verify if the file exists
    if ( !sofa::helper::system::DataRepository.findFile ( path ) )
    {
        return;
    }
    path = sofa::helper::system::DataRepository.getFile ( path );

    //Desactivate the animate-> no more graph modification

    bool isAnimated = startButton->isOn();

    playpauseGUI ( false );
    //If we add the object without clicking on the graph (direct use of the method),
    //the object will be added to the root node
    if ( node_clicked == NULL )
    {
        for ( std::map<core::objectmodel::Base*, Q3ListViewItem* >::iterator it = graphListener->items.begin() ; it != graphListener->items.end() ; ++ it )
        {
            if ( ( *it ).second->itemPos() == 0 ) //Root node position
            {
                node_clicked = dynamic_cast< sofa::simulation::Node *> ( ( *it ).first );
                break;
            }
        }
        if ( node_clicked == NULL ) return;
    }

    //We allow unlock the graph to make all the changes now
    if ( currentTab != TabGraph )
        graphListener->unfreeze ( node_clicked );

    //Loading of the xml file
    simulation::tree::xml::BaseElement* xml = simulation::tree::xml::loadFromFile ( path.c_str() );
    if ( xml == NULL ) return;


    helper::system::SetDirectory chdir ( path.c_str() );

    //std::cout << "Initializing objects"<<std::endl;
    if ( !xml->init() )
    {
        std::cerr << "Objects initialization failed."<<std::endl;
    }

    Node* new_node = dynamic_cast<Node*> ( xml->getObject() );

    if ( new_node == NULL )
    {
        std::cerr << "Objects initialization failed."<<std::endl;
        delete xml;
        return ;
    }

    //std::cout << "Initializing simulation "<<new_node->getName() <<std::endl;
    new_node->execute<InitVisitor>();
    if (node_clicked && new_node)
    {
        if ( node_clicked->child.empty() &&  node_clicked->object.empty() )
        {
            //Temporary Root : the current graph is empty, and has only a single node "Root"
            viewer->setScene ( new_node, path.c_str() );
            graphListener->removeChild ( NULL, node_clicked );
            graphListener->addChild ( NULL, new_node );
        }
        else
        {
            node_clicked->addChild (new_node );
            graphListener->addObject ( node_clicked, (sofa::core::objectmodel::BaseObject*) new_node );

            list_object_added.push_back ( new_node );

        }
    }
    //update the stats graph
    graphCreateStats(viewer->getScene());
    //Apply the Transformation
    transformObject ( new_node, dx, dy, dz, rx,ry,rz,scale );

    //Update the view
    viewer->resetView();
    viewer->getQWidget()->update();

    //freeze the graph if needed and animate
    if ( currentTab != TabGraph )
        graphListener->freeze ( node_clicked );

    node_clicked = NULL;
    item_clicked = NULL;
    playpauseGUI ( isAnimated );
}


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
    else  	                                    fileOpen(filename);
}


void RealGUI::changeHtmlPage( const QUrl& u)
{
    std::string path=u.path().ascii();
#ifdef WIN32
    path = path.substr(1);
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

void RealGUI::showhideElements(int FILTER, bool value)
{
    Node* root = getScene();
    if ( root )
    {
        switch(FILTER)
        {
        case DisplayFlagWidget::ALL:
            root->getContext()->setShowVisualModels ( value );
            root->getContext()->setShowBehaviorModels ( value );
            root->getContext()->setShowCollisionModels ( value );
            root->getContext()->setShowBoundingCollisionModels ( value );
            root->getContext()->setShowMappings ( value );
            root->getContext()->setShowMechanicalMappings ( value );
            root->getContext()->setShowForceFields ( value );
            root->getContext()->setShowInteractionForceFields ( value );
            break;
        case  DisplayFlagWidget::VISUAL:            root->getContext()->setShowVisualModels ( value ); break;
        case  DisplayFlagWidget::BEHAVIOR:          root->getContext()->setShowBehaviorModels ( value ); break;
        case  DisplayFlagWidget::COLLISION:         root->getContext()->setShowCollisionModels ( value ); break;
        case  DisplayFlagWidget::BOUNDING:          root->getContext()->setShowBoundingCollisionModels ( value );  break;
        case  DisplayFlagWidget::MAPPING:           root->getContext()->setShowMappings ( value ); break;
        case  DisplayFlagWidget::MECHANICALMAPPING: root->getContext()->setShowMechanicalMappings ( value ); break;
        case  DisplayFlagWidget::FORCEFIELD:        root->getContext()->setShowForceFields ( value ); break;
        case  DisplayFlagWidget::INTERACTION:       root->getContext()->setShowInteractionForceFields ( value ); break;
        case  DisplayFlagWidget::WIREFRAME:         root->getContext()->setShowWireFrame ( value ); break;
        case  DisplayFlagWidget::NORMALS:           root->getContext()->setShowNormals ( value ); break;
        }
        sofa::simulation::getSimulation()->updateVisualContext ( root, FILTER );
    }
    viewer->getQWidget()->update();
}


} // namespace qt

} // namespace gui

} // namespace sofa
