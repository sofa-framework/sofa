/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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

#ifdef SOFA_GUI_QTOGREVIEWER
#include <sofa/gui/qt/viewer/qtogre/QtOgreViewer.h>
#endif

#ifdef SOFA_GUI_QTVIEWER
#include <sofa/gui/qt/viewer/qt/QtViewer.h>
#endif

#ifdef SOFA_GUI_QGLVIEWER
#include <sofa/gui/qt/viewer/qgl/QtGLViewer.h>
#endif

#include <sofa/simulation/tree/Simulation.h>
#include <sofa/simulation/common/InitVisitor.h>
#include <sofa/simulation/common/DesactivatedNodeVisitor.h>

#ifdef SOFA_DEV

#include <sofa/simulation/automatescheduler/ThreadSimulation.h>
#include <sofa/simulation/automatescheduler/ExecBus.h>
#include <sofa/simulation/automatescheduler/Node.h>

#endif // SOFA_DEV

#ifdef SOFA_HAVE_CHAI3D
#include <sofa/simulation/common/PropagateEventVisitor.h>
#include <sofa/core/objectmodel/GLInitializedEvent.h>
#endif // SOFA_HAVE_CHAI3D


#include <sofa/component/visualmodel/VisualModelImpl.h>

#include <sofa/simulation/tree/xml/XML.h>
#include <sofa/simulation/common/TransformationVisitor.h>
#include <sofa/helper/system/FileRepository.h>

#ifdef SOFA_DEV

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

#endif // SOFA_DEV


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



namespace sofa
{

namespace gui
{

namespace qt
{

#ifdef SOFA_QT4
typedef Q3ListView QListView;
typedef Q3DockWindow QDockWindow;
typedef QStackedWidget QWidgetStack;
typedef Q3TextEdit QTextEdit;
#endif


using sofa::core::objectmodel::BaseObject;
using sofa::simulation::tree::GNode;
using namespace sofa::helper::system::thread;
using namespace sofa::simulation;
using namespace sofa::simulation::tree;

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
    QSOFAApplication(int argc, char ** argv)
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
        int argc=1;
        char* argv[1];
        argv[0] = strdup ( SofaGUI::GetProgramName() );
        application = new QSOFAApplication ( argc,argv );
        free ( argv[0] );
    }
    // create interface
    gui = new RealGUI ( name, options );
    GNode *groot = dynamic_cast< GNode* >(node);
    if ( groot )
        gui->setScene ( groot, filename );


    else
        return NULL;

    //gui->viewer->resetView();

    application->setMainWidget ( gui );

#ifdef SOFA_DEV

    // Threads Management
    if ( sofa::simulation::automatescheduler::ThreadSimulation::initialized() )
    {
        sofa::simulation::automatescheduler::ThreadSimulation::getInstance()->computeVModelsList ( groot );
        groot->setMultiThreadSimulation ( true );
        sofa::simulation::automatescheduler::groot = groot;

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
    groot->execute(act);
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
    viewer->getQWidget()->update();
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
    //Add Filemenu Recently Opened files
    recentlyOpened = new QPopupMenu(this);
    this->fileMenu->insertItem( QIconSet( ), tr( "Recently Opened Files..."), recentlyOpened, -1, 7);


    listDisplayFlags->header()->hide();
#ifdef SOFA_QT4
    listDisplayFlags->setBackgroundRole(QPalette::NoRole);
    listDisplayFlags->viewport()->setBackgroundRole(QPalette::NoRole);
#endif
    // remove everything...
    listDisplayFlags->clear();
    listDisplayFlags->setSortColumn(-1);
    itemShowAll = new DisplayFlagItem(this, listDisplayFlags, ALL, "All", Q3CheckListItem::CheckBoxController);
    itemShowAll->setOpen(true);
    Q3CheckListItem* itemShowVisual = new Q3CheckListItem(itemShowAll, "Visual", Q3CheckListItem::CheckBoxController);
    itemShowVisualModels = new DisplayFlagItem(this, itemShowVisual, VISUALMODELS, "Visual Models");
    Q3CheckListItem* itemShowBehavior = new Q3CheckListItem(itemShowAll, itemShowVisual, "Behavior", Q3CheckListItem::CheckBoxController);
    itemShowBehaviorModels = new DisplayFlagItem(this, itemShowBehavior, BEHAVIORMODELS, "Behavior Models");
    itemShowForceFields = new DisplayFlagItem(this, itemShowBehavior, itemShowBehaviorModels, FORCEFIELDS, "Force Fields");
    itemShowInteractions = new DisplayFlagItem(this, itemShowBehavior, itemShowForceFields, INTERACTIONS, "Interactions");
    Q3CheckListItem* itemShowCollision = new Q3CheckListItem(itemShowAll, itemShowBehavior, "Collision", Q3CheckListItem::CheckBoxController);
    itemShowCollisionModels = new DisplayFlagItem(this, itemShowCollision, COLLISIONMODELS, "Collision Models");
    itemShowBoundingTrees = new DisplayFlagItem(this, itemShowCollision, itemShowCollisionModels, BOUNDINGTREES, "Bounding Trees");
    Q3CheckListItem* itemShowMapping = new Q3CheckListItem(itemShowAll, itemShowCollision, "Mapping", Q3CheckListItem::CheckBoxController);
    itemShowMappings = new DisplayFlagItem(this, itemShowMapping, itemShowInteractions, MAPPINGS, "Visual Mappings");
    itemShowMechanicalMappings = new DisplayFlagItem(this, itemShowMapping, itemShowMappings, MECHANICALMAPPINGS, "Mechanical Mappings");
    Q3ListViewItem* itemShowOptions = new Q3ListViewItem(listDisplayFlags, itemShowAll, "Options");
    itemShowWireFrame = new DisplayFlagItem(this, itemShowOptions, WIREFRAME, "Wire Frame");
    itemShowNormals = new DisplayFlagItem(this, itemShowOptions, itemShowWireFrame, NORMALS, "Normals");
#ifdef SOFA_QT4
    connect( listDisplayFlags, SIGNAL( pressed(Q3ListViewItem *)), this, SLOT(flagChanged(Q3ListViewItem *)));
    connect( listDisplayFlags, SIGNAL( doubleClicked(Q3ListViewItem *)), this, SLOT(flagDoubleClicked(Q3ListViewItem *)));
#else
    connect( listDisplayFlags, SIGNAL( pressed(QListViewItem *)), this, SLOT(flagChanged(QListViewItem *)));
    connect( listDisplayFlags, SIGNAL( doubleClicked(QListViewItem *)), this, SLOT(flagDoubleClicked(QListViewItem *)));
#endif
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
    loadRecordTime->setMaximumSize(QSize(50, 100));


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

    scenes = sofa::helper::system::DataRepository.getFile ( scenes );

    updateRecentlyOpened("");


    //Dialog Add Object
    connect ( tabs, SIGNAL ( currentChanged ( QWidget* ) ), this, SLOT ( currentTabChanged ( QWidget* ) ) );

    addViewer();
    currentTabChanged ( tabs->currentPage() );


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
    for (unsigned int i=0; i<list_files.size() && i<5 ; ++i)
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
#else
    int id_viewer = left_stack->addWidget ( viewer->getQWidget() );
    left_stack->raiseWidget ( id_viewer );
#endif
    viewer->getQWidget()->setSizePolicy ( QSizePolicy ( ( QSizePolicy::SizeType ) 7, ( QSizePolicy::SizeType ) 7, 100, 1,
            viewer->getQWidget()->sizePolicy().hasHeightForWidth() ) );
    viewer->getQWidget()->setMinimumSize ( QSize ( 0, 0 ) );
#ifndef SOFA_QT4
    viewer->getQWidget()->setCursor ( QCursor ( 2 ) );
#endif
    viewer->getQWidget()->setMouseTracking ( TRUE );

#ifdef SOFA_QT4
    viewer->getQWidget()->setFocusPolicy ( Qt::StrongFocus );
#else
    viewer->getQWidget()->setFocusPolicy ( QWidget::StrongFocus );
#endif

    viewer->setup();

    connect ( ResetViewButton, SIGNAL ( clicked() ), viewer->getQWidget(), SLOT ( resetView() ) );
    connect ( SaveViewButton, SIGNAL ( clicked() ), viewer->getQWidget(), SLOT ( saveView() ) );
    connect ( screenshotButton, SIGNAL ( clicked() ), this, SLOT ( screenshot() ) );
    connect ( sizeW, SIGNAL ( valueChanged ( int ) ), viewer->getQWidget(), SLOT ( setSizeW ( int ) ) );
    connect ( sizeH, SIGNAL ( valueChanged ( int ) ), viewer->getQWidget(), SLOT ( setSizeH ( int ) ) );
    connect ( viewer->getQWidget(), SIGNAL ( resizeW ( int ) ), sizeW, SLOT ( setValue ( int ) ) );
    connect ( viewer->getQWidget(), SIGNAL ( resizeH ( int ) ), sizeH, SLOT ( setValue ( int ) ) );

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
        getSimulation()->unload ( viewer->getScene() );
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
        getSimulation()->unload ( viewer->getScene() );
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

    simulation::Node* groot = getSimulation()->load ( filename.c_str() );

    if ( groot == NULL )
    {
        qFatal ( "Failed to load %s",filename.c_str() );
        return;
    }

    setScene ( groot, filename.c_str() );
    //need to create again the output streams !!

    getSimulation()->gnuplotDirectory.setValue(gnuplot_directory);
    setExportGnuplot(exportGnuplotFilesCheckbox->isChecked());

    displayComputationTime(m_displayComputationTime);
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
        getSimulation()->unload ( viewer->getScene() );
        if ( graphListener!=NULL )
        {
            delete graphListener;
            graphListener = NULL;
        }
        graphView->clear();
    }
    GNode *simuNode = dynamic_cast< GNode *> (getSimulation()->load ( scene.c_str() ));
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
        Node* groot;
        if ( lmlreader != NULL ) delete lmlreader;
        lmlreader = new LMLReader; std::cout <<"New lml reader\n";
        lmlreader->BuildStructure ( filename, pmlreader );

        groot = viewer->getScene();
        getSimulation()->init ( groot );

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
        node_clicked = dynamic_cast< GNode* >(graph_iterator->first);
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


void RealGUI::setScene ( Node* groot, const char* filename )
{
    if (filename)
    {
        updateRecentlyOpened(filename);
        setTitle ( filename );
    }

    if (tabInstrument!= NULL)
    {
        tabs->removePage(tabInstrument);
        delete tabInstrument;
        tabInstrument = NULL;
    }



    viewer->setScene ( dynamic_cast< GNode *>(groot), filename );
    viewer->resetView();
    initial_time = groot->getTime();

    record_simulation = false;
    clearRecord();
    clearGraph();

    initDesactivatedNode();

    eventNewTime();

    // set state of display flags
    itemShowVisualModels->init(groot->getContext()->getShowVisualModels());
    itemShowBehaviorModels->init(groot->getContext()->getShowBehaviorModels());
    itemShowCollisionModels->init(groot->getContext()->getShowCollisionModels());
    itemShowBoundingTrees->init(groot->getContext()->getShowBoundingCollisionModels());
    itemShowMappings->init(groot->getContext()->getShowMappings());
    itemShowMechanicalMappings->init(groot->getContext()->getShowMechanicalMappings());
    itemShowForceFields->init(groot->getContext()->getShowForceFields());
    itemShowInteractions->init(groot->getContext()->getShowInteractionForceFields());
    itemShowWireFrame->init(groot->getContext()->getShowWireFrame());
    itemShowNormals->init(groot->getContext()->getShowNormals());

    //getSimulation()->updateVisualContext ( groot );
    startButton->setOn ( groot->getContext()->getAnimate() );
    dtEdit->setText ( QString::number ( groot->getDt() ) );
    record->setOn(false);

#ifdef SOFA_HAVE_CHAI3D
    // Tell nodes that openGl is initialized
    // especialy the GL_MODELVIEW_MATRIX
    sofa::core::objectmodel::GLInitializedEvent ev;
    sofa::simulation::PropagateEventVisitor act(&ev);
    groot->execute(act);
#endif // SOFA_HAVE_CHAI3D


}

void RealGUI::changeInstrument(int id)
{
    std::cout << "Activation instrument "<<id<<std::endl;
    Simulation *s = getSimulation();
    if (s->instrumentInUse.getValue() >= 0 && s->instrumentInUse.getValue() < (int)s->instruments.size())
        s->instruments[s->instrumentInUse.getValue()]->setActive(false);

    getSimulation()->instrumentInUse.setValue(id-1);
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
            "Scenes (*.scn *.xml *.simu *.pml *.lml)",
#else
            "Scenes (*.scn *.xml *.simu *.pscn)",
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
    GNode *node = viewer->getScene();
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
    getSimulation()->printXML ( node, filename );
}

void RealGUI::fileExit()
{
    startButton->setOn ( false);
    close();
}

void RealGUI::saveXML()
{
    getSimulation()->printXML ( viewer->getScene(), "scene.scn" );
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

void RealGUI::editGnuplotDirectory()
{
    std::string filename = viewer->getSceneFileName();
    QString s = getExistingDirectory ( this, filename.empty() ?NULL:filename.c_str(), "open directory dialog",  "Choose a directory" );
    if (s.length() > 0)
    {
        gnuplot_directory = s.ascii();
        if (gnuplot_directory.at(gnuplot_directory.size()-1) != '/') gnuplot_directory+="/";

        getSimulation()->gnuplotDirectory.setValue(gnuplot_directory);
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


//*****************************************************************************************
//called at each step of the rendering

void RealGUI::step()
{
    GNode* groot = viewer->getScene();
    if ( groot == NULL ) return;

#ifdef SOFA_DEV

    if ( groot->getContext()->getMultiThreadSimulation() )
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
        //groot->setLogTime(true);

        getSimulation()->animate ( groot );

        if ( m_dumpState )
            getSimulation()->dumpState ( groot, *m_dumpStateStream );
        if ( m_exportGnuplot )
            getSimulation()->exportGnuplot ( groot, groot->getTime() );

        viewer->wait();

        eventNewStep();
        eventNewTime();

#ifdef SOFA_QT4
        viewer->getQWidget()->setUpdatesEnabled ( true );
#endif
        viewer->getQWidget()->update();
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
    emit newStep();
}

//*****************************************************************************************
// Update sofa Simulation with the time step

void RealGUI::eventNewStep()
{
    static ctime_t beginTime[10];
    static const ctime_t timeTicks = CTime::getRefTicksPerSec();
    Node* groot = getScene();
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

    if ( m_displayComputationTime && ( frameCounter%100 ) == 0 && groot!=NULL )
    {

        std::cout << "========== ITERATION " << frameCounter << " ==========\n";
        const sofa::simulation::Node::NodeTimer& total = groot->getTotalTime();
        const std::map<std::string, sofa::simulation::Node::NodeTimer>& times = groot->getVisitorTime();
        const std::map<std::string, std::map<sofa::core::objectmodel::BaseObject*, sofa::simulation::Node::ObjectTimer> >& objtimes = groot->getObjectTime();
        const double fact = 1000000.0 / ( 100*groot->getTimeFreq() );
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
        groot->resetTime();

    }
}



void RealGUI::eventNewTime()
{
    Node* groot = getScene();
    if ( groot )
    {

        double time = groot->getTime();
        char buf[100];
        sprintf ( buf, "Time: %.3f s", time );
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
    Node* groot = getScene();
    if ( value > 0.0 )
    {

        if ( groot )
            groot->getContext()->setDt ( value );
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
    GNode* groot = getScene();

    //Hide the dialog to add a new object in the graph
    if ( dialog != NULL ) dialog->hide();
    //Hide all the dialogs to modify the graph
    emit ( newScene() );

    //Clear the list of modified dialog opened
    current_Id_modifyDialog=0;
    map_modifyDialogOpened.clear();


    std::list< GNode *>::iterator it;
    //**************************************************************
    //GRAPH MANAGER
    bool isFrozen = graphListener->frozen;
    graphListener->unfreeze ( groot );
    std::map<core::objectmodel::Base*, Q3ListViewItem* >::iterator graph_iterator;


    //Remove all the objects added
    bool node_removed ;
    for ( it=list_object_added.begin(); it != list_object_added.end(); it++ )
    {
        node_removed = false;

        //Verify if they have not been removed before
        std::list< GNode *>::iterator it_removed;
        for ( it_removed=list_object_removed.begin(); it_removed != list_object_removed.end(); it_removed++ )
        {
            if ( ( *it_removed ) == ( *it ) ) { node_removed=true; continue;} //node already removed
        }
        if ( node_removed ) continue;
        ( *it )->getParent()->removeChild ( ( *it ) );
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
        std::list< GNode *>::iterator it_initial;
        for ( it_initial=list_object_initial.begin(); it_initial != list_object_initial.end(); it_initial++ )
        {
            if ( ( *it_initial ) == ( *it ) )
            {
                it_initial++; //points to the parent of the node
                ( *it_initial )->addChild ( ( *it ) );
                graphListener->addObject ( dynamic_cast<GNode *> ( *it_initial ), ( core::objectmodel::BaseObject* ) ( *it ) );
                continue;
            }
            //We have to increment 2 times the iterator: le list_object_initial contains first the node, then its father
            it_initial++;
        }
    }

    list_object_removed.clear();




    if ( isFrozen ) graphListener->freeze ( groot );

    //Reset the scene
    if ( groot )
    {
        getSimulation()->reset ( groot );
        groot->setTime(initial_time);
        eventNewTime();

        //viewer->resetView();
        viewer->getQWidget()->update();
    }
}


//*****************************************************************************************
//
void RealGUI::exportGraph()
{
    exportGraph ( getScene() );
}


void RealGUI::exportGraph ( sofa::simulation::tree::GNode* root )
{

    if ( root == NULL ) return;
    sofa::gui::qt::GenGraphForm* form = new sofa::gui::qt::GenGraphForm;
    form->setScene ( root );
    form->show();
}



//*****************************************************************************************
//
void RealGUI::displayComputationTime ( bool value )
{
    Node* groot = getScene();
    m_displayComputationTime = value;
    if ( groot )
    {
        groot->setLogTime ( m_displayComputationTime );
    }
}



//*****************************************************************************************
//
void RealGUI::setExportGnuplot ( bool exp )
{
    GNode* groot = getScene();
    m_exportGnuplot = exp;
    if ( m_exportGnuplot && groot )
    {
        getSimulation()->initGnuplot ( groot );
        getSimulation()->exportGnuplot ( groot, groot->getTime() );
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
    GNode* groot = getScene();
    if ( !groot ) return;
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
    getSimulation()->exportOBJ ( groot, filename.c_str(),exportMTL );
}


//*****************************************************************************************
// Called by the animate timer
void RealGUI::animate()
{
    viewer->getQWidget()->update();
}


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
                node_clicked = dynamic_cast< sofa::simulation::tree::GNode *> ( ( *it ).first );
                break;
            }
        }
        if ( node_clicked == NULL ) return;
    }

    //We allow unlock the graph to make all the changes now
    if ( currentTab != TabGraph )
        graphListener->unfreeze ( node_clicked );

    //Loading of the xml file
    xml::BaseElement* xml = xml::loadFromFile ( path.c_str() );
    if ( xml == NULL ) return;


    helper::system::SetDirectory chdir ( path.c_str() );

    //std::cout << "Initializing objects"<<std::endl;
    if ( !xml->init() )
    {
        std::cerr << "Objects initialization failed."<<std::endl;
    }

    GNode* new_node = dynamic_cast<GNode*> ( xml->getObject() );
    if ( new_node == NULL )
    {
        std::cerr << "Objects initialization failed."<<std::endl;
        delete xml;
        return ;
    }

    //std::cout << "Initializing simulation "<<new_node->getName() <<std::endl;
    new_node->execute<InitVisitor>();

    if ( node_clicked->child.begin() ==  node_clicked->child.end() &&  node_clicked->object.begin() == node_clicked->object.end() )
    {
        //Temporary Root : the current graph is empty, and has only a single node "Root"
        viewer->setScene ( new_node, path.c_str() );
        graphListener->removeChild ( NULL, node_clicked );
        graphListener->addChild ( NULL, new_node );
    }
    else
    {
        node_clicked->addChild ( new_node );
        graphListener->addObject ( node_clicked, ( core::objectmodel::BaseObject* ) new_node );

        list_object_added.push_back ( new_node );
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



void RealGUI::showhideElements(int FILTER, bool value)
{
    Node* groot = getScene();
    if ( groot )
    {
        switch(FILTER)
        {
        case ALL:
            groot->getContext()->setShowVisualModels ( value );
            groot->getContext()->setShowBehaviorModels ( value );
            groot->getContext()->setShowCollisionModels ( value );
            groot->getContext()->setShowBoundingCollisionModels ( value );
            groot->getContext()->setShowMappings ( value );
            groot->getContext()->setShowMechanicalMappings ( value );
            groot->getContext()->setShowForceFields ( value );
            groot->getContext()->setShowInteractionForceFields ( value );
            break;
        case VISUALMODELS:       groot->getContext()->setShowVisualModels ( value ); break;
        case BEHAVIORMODELS:     groot->getContext()->setShowBehaviorModels ( value ); break;
        case COLLISIONMODELS:    groot->getContext()->setShowCollisionModels ( value ); break;
        case BOUNDINGTREES:      groot->getContext()->setShowBoundingCollisionModels ( value );  break;
        case MAPPINGS:           groot->getContext()->setShowMappings ( value ); break;
        case MECHANICALMAPPINGS: groot->getContext()->setShowMechanicalMappings ( value ); break;
        case FORCEFIELDS:        groot->getContext()->setShowForceFields ( value ); break;
        case INTERACTIONS:       groot->getContext()->setShowInteractionForceFields ( value ); break;
        case WIREFRAME:          groot->getContext()->setShowWireFrame ( value ); break;
        case NORMALS:            groot->getContext()->setShowNormals ( value ); break;
        }
        sofa::simulation::tree::getSimulation()->updateVisualContext ( groot, FILTER );
    }
    viewer->getQWidget()->update();
}


} // namespace qt

} // namespace gui

} // namespace sofa
