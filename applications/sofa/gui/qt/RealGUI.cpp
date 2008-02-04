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
#include "RealGUI.h"

#ifdef SOFA_GUI_QTOGREVIEWER
#include "viewer/qtogre/QtOgreViewer.h"
#endif

#ifdef SOFA_GUI_QTVIEWER
#include "viewer/qt/QtViewer.h"
#endif

#ifdef SOFA_GUI_QGLVIEWER
#include "viewer/qgl/QtGLViewer.h"
#endif



#include <sofa/simulation/tree/Simulation.h>
#include <sofa/simulation/tree/InitVisitor.h>


#include <sofa/simulation/tree/MutationListener.h>
#include <sofa/simulation/tree/Colors.h>

#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileRepository.h>

#include <sofa/simulation/automatescheduler/ThreadSimulation.h>
#include <sofa/simulation/automatescheduler/ExecBus.h>
#include <sofa/simulation/automatescheduler/Node.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/visualmodel/VisualModelImpl.h>

#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/LineModel.h>
#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/SphereModel.h>

#include <limits.h>

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
#include <QFileDialog>
#include <Q3FileDialog>
#include <QTabWidget>
#include <Q3PopupMenu>
#include <QToolTip>
#include <QButtonGroup>
#include <QRadioButton>
#else
#include <qwidget.h>
#include <qwidgetstack.h>
#include <qlayout.h>
#include <qlistview.h>
#include <qstatusbar.h>
#include <qfiledialog.h>
#include <qheader.h>
#include <qimage.h>
#include <qsplitter.h>
#include <qtextedit.h>
#include <qcursor.h>
#include <qapplication.h>
#include <qaction.h>
#include <qmessagebox.h>
#include <qfiledialog.h>
#include <qtabwidget.h>
#include <qpopupmenu.h>
#include <qtooltip.h>
#include <qbuttongroup.h>
#include <qradiobutton.h>
#endif

#include <GenGraphForm.h>



namespace sofa
{

namespace gui
{

namespace qt
{

#ifdef QT_MODULE_QT3SUPPORT
typedef Q3ListView QListView;
//      typedef Q3FileDialog QFileDialog;
typedef Q3DockWindow QDockWindow;
typedef QStackedWidget QWidgetStack;
typedef Q3TextEdit QTextEdit;
typedef Q3PopupMenu QPopupMenu;
#else
typedef QListViewItem Q3ListViewItem;
typedef QListView Q3ListView;
typedef QFileDialog Q3FileDialog;
//typedef QWidgetStack QStackedWidget;
typedef QTextEdit Q3TextEdit;
typedef QPopupMenu Q3PopupMenu;
#endif


using sofa::core::objectmodel::BaseObject;
using sofa::simulation::tree::GNode;
using namespace sofa::helper::system::thread;
using namespace sofa::simulation::tree;
using namespace sofa::simulation::automatescheduler;


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
            static_cast<RealGUI*>(mainWidget())->fileOpen(static_cast<QFileOpenEvent *>(event)->file());
            return true;
        default:
            return QApplication::event(event);
        }
    }
};
#else
typedef QApplication QSOFAApplication;
#endif

SofaGUI* RealGUI::CreateGUI ( const char* name, const std::vector<std::string>& options, sofa::simulation::tree::GNode* groot, const char* filename )
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
    if ( groot )
        gui->setScene ( groot, filename );

    //gui->viewer->resetView();

    application->setMainWidget ( gui );

    // Threads Management
    if ( ThreadSimulation::initialized() )
    {
        ThreadSimulation::getInstance()->computeVModelsList ( groot );
        groot->setMultiThreadSimulation ( true );
        sofa::simulation::automatescheduler::groot = groot;

        Automate::setDrawCB ( gui->viewer );

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




RealGUI::RealGUI ( const char* viewername, const std::vector<std::string>& /*options*/ )
    : viewerName ( viewername ), viewer ( NULL ), currentTab ( NULL ), tabInstrument (NULL),  graphListener ( NULL ), dialog ( NULL )
{

    left_stack = new QWidgetStack ( splitter2 );
    connect ( startButton, SIGNAL ( toggled ( bool ) ), this , SLOT ( playpauseGUI ( bool ) ) );

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
    playbackward_record    = new QPushButton( statusBar(), "Play Backward"); playbackward_record->setToggleButton(true);
    playforward_record     = new QPushButton( statusBar(), "Play Forward"); playforward_record->setToggleButton(true);
    stepforward_record     = new QPushButton( statusBar(), "Step Forward");
    forward_record         = new QPushButton( statusBar(), "Forward");

    QToolTip::add(record               , tr( "Record" ) );
    QToolTip::add(backward_record      , tr( "Load Initial Time" ) );
    QToolTip::add(stepbackward_record  , tr( "Make one step backward" ) );
    QToolTip::add(playbackward_record  , tr( "Continuous play backward" ) );
    QToolTip::add(playforward_record   , tr( "Continuous play forward" ) );
    QToolTip::add(stepforward_record   , tr( "Make one step forward" ) );
    QToolTip::add(forward_record       , tr( "Load Final Time" ) );


    // Image for the record button
    pixmap_filename = "textures/media-record.png";
    if ( sofa::helper::system::DataRepository.findFile ( pixmap_filename ) )
        pixmap_filename = sofa::helper::system::DataRepository.getFile ( pixmap_filename );
    record->setPixmap(QPixmap(QImage(pixmap_filename.c_str())));

    // Image for the backward button
    pixmap_filename = "textures/media-seek-backward.png";
    if ( sofa::helper::system::DataRepository.findFile ( pixmap_filename ) )
        pixmap_filename = sofa::helper::system::DataRepository.getFile ( pixmap_filename );
    backward_record->setPixmap(QPixmap(QImage(pixmap_filename.c_str())));

    // Image for the backward button
    pixmap_filename = "textures/media-skip-backward.png";
    if ( sofa::helper::system::DataRepository.findFile ( pixmap_filename ) )
        pixmap_filename = sofa::helper::system::DataRepository.getFile ( pixmap_filename );
    stepbackward_record->setPixmap(QPixmap(QImage(pixmap_filename.c_str())));

    // Image for the play backward button
    pixmap_filename = "textures/media-back-start.png";
    if ( sofa::helper::system::DataRepository.findFile ( pixmap_filename ) )
        pixmap_filename = sofa::helper::system::DataRepository.getFile ( pixmap_filename );
    playbackward_record->setPixmap(QPixmap(QImage(pixmap_filename.c_str())));

    // Image for the play forward button
    pixmap_filename = "textures/media-playback-start.png";
    if ( sofa::helper::system::DataRepository.findFile ( pixmap_filename ) )
        pixmap_filename = sofa::helper::system::DataRepository.getFile ( pixmap_filename );
    playforward_record->setPixmap(QPixmap(QImage(pixmap_filename.c_str())));

    // Image for the play forward button
    pixmap_filename = "textures/media-skip-forward.png";
    if ( sofa::helper::system::DataRepository.findFile ( pixmap_filename ) )
        pixmap_filename = sofa::helper::system::DataRepository.getFile ( pixmap_filename );
    stepforward_record->setPixmap(QPixmap(QImage(pixmap_filename.c_str())));

    // Image for the forward button
    pixmap_filename = "textures/media-seek-forward.png";
    if ( sofa::helper::system::DataRepository.findFile ( pixmap_filename ) )
        pixmap_filename = sofa::helper::system::DataRepository.getFile ( pixmap_filename );
    forward_record->setPixmap(QPixmap(QImage(pixmap_filename.c_str())));

    QLabel *timeRecord = new QLabel("T=",statusBar());
    loadRecordTime = new QLineEdit(statusBar());
    loadRecordTime->setMaximumSize(QSize(50, 100));


    statusBar()->addWidget ( fpsLabel );
    statusBar()->addWidget ( timeLabel );

    statusBar()->addWidget( record);
    statusBar()->addWidget( backward_record);
    statusBar()->addWidget( stepbackward_record);
    statusBar()->addWidget( playbackward_record);
    statusBar()->addWidget( playforward_record);
    statusBar()->addWidget( stepforward_record);
    statusBar()->addWidget( forward_record);
    statusBar()->addWidget( timeRecord);

    statusBar()->addWidget( loadRecordTime);

    statusBar()->addWidget ( initialTime );
    statusBar()->addWidget( timeSlider);
    statusBar()->addWidget ( finalTime );

    timerStep = new QTimer ( this );


    connect ( timerStep, SIGNAL ( timeout() ), this, SLOT ( step() ) );
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
    connect ( playbackward_record, SIGNAL (clicked () ),     this, SLOT( slot_playbackward( ) ) );
    connect ( playforward_record,  SIGNAL (clicked () ),     this, SLOT( slot_playforward( ) ) );
    connect ( stepforward_record,  SIGNAL (clicked () ),     this, SLOT( slot_stepforward( ) ) );
    connect ( forward_record, SIGNAL (clicked () ),          this, SLOT( slot_forward( ) ) );

    connect (loadRecordTime, SIGNAL(returnPressed ()),       this, SLOT( slot_loadrecord_timevalue()));

    connect ( timeSlider, SIGNAL (sliderMoved (int) ),  this, SLOT( slot_sliderValue( int) ) );
    //Dialog Add Object

    connect ( tabs, SIGNAL ( currentChanged ( QWidget* ) ), this, SLOT ( currentTabChanged ( QWidget* ) ) );

    addViewer();

    currentTabChanged ( tabs->currentPage() );


#ifdef SOFA_PML
    pmlreader = NULL;
    lmlreader = NULL;
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
}

void RealGUI::init()
{
    node_clicked = NULL;
    item_clicked = NULL;
    _animationOBJ = false;
    _animationOBJcounter = 0;
    m_dumpState = false;
    m_dumpStateStream = 0;
    m_displayComputationTime = false;
    m_exportGnuplot = false;
    record_directory = sofa::helper::system::SetDirectory::GetRelativeFromDir("../../scenes/simulation/",sofa::helper::system::SetDirectory::GetProcessFullPath("").c_str());
    gnuplot_directory = "";

    current_Id_modifyDialog = 0;
    map_modifyDialogOpened.clear();


    //*********************************************************************************************************************************
    //List of objects
    //Read the object.txt that contains the information about the objects which can be added to the scenes whithin a given BoundingBox and scale range
    std::string object ( "object.txt" );
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
    viewerOpenGLAction->setOn ( false );
#ifdef SOFA_GUI_QGLVIEWER
    viewerQGLViewerAction->setEnabled ( true );
#else
    viewerQGLViewerAction->setEnabled ( false );
    viewerQGLViewerAction->setToolTip ( "enable SOFA_GUI_QGLVIEWER in sofa-local.cfg to activate" );
#endif
    viewerQGLViewerAction->setOn ( false );
#ifdef SOFA_GUI_QTOGREVIEWER
    viewerOGREAction->setEnabled ( true );
#else
    viewerOGREAction->setEnabled ( false );
    viewerOGREAction->setToolTip ( "enable SOFA_GUI_QTOGREVIEWER in sofa-local.cfg to activate" );
#endif
    viewerOGREAction->setOn ( false );

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
#ifdef QT_MODULE_QT3SUPPORT
    left_stack->addWidget ( viewer->getQWidget() );
    left_stack->setCurrentWidget ( viewer->getQWidget() );
#else
    int id_viewer = left_stack->addWidget ( viewer->getQWidget() );
    left_stack->raiseWidget ( id_viewer );
#endif
    viewer->getQWidget()->setSizePolicy ( QSizePolicy ( ( QSizePolicy::SizeType ) 7, ( QSizePolicy::SizeType ) 7, 100, 1,
            viewer->getQWidget()->sizePolicy().hasHeightForWidth() ) );
    viewer->getQWidget()->setMinimumSize ( QSize ( 0, 0 ) );
#ifndef QT_MODULE_QT3SUPPORT
    viewer->getQWidget()->setCursor ( QCursor ( 2 ) );
#endif
    viewer->getQWidget()->setMouseTracking ( TRUE );

#ifdef QT_MODULE_QT3SUPPORT
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
#ifdef QT_MODULE_QT3SUPPORT
    QList<int> list;
#else
    QValueList<int> list;
#endif
    list.push_back ( 75 );
    list.push_back ( 500 );
    splitter_ptr->setSizes ( list );

    viewer->getQWidget()->setFocus();
    viewer->getQWidget()->show();
    viewer->getQWidget()->update();
    setGUI();
}

void RealGUI::viewerOpenGL()
{
    viewerOpenGLAction->setOn ( setViewer ( "qt" ) );
}

void RealGUI::viewerQGLViewer()
{
    viewerOpenGLAction->setOn ( setViewer ( "qglviewer" ) );
}

void RealGUI::viewerOGRE()
{
    viewerOpenGLAction->setOn ( setViewer ( "ogre" ) );
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
    GNode* groot = new GNode; // empty scene to do the transition
    setScene ( groot,filename.c_str() ); // keep the current display flags
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


    if ( graphListener )
        graphListener->removeChild ( NULL, groot );

    addViewer();

    if (filename.rfind(".simu") != std::string::npos)
        fileOpenSimu(filename.c_str() );
    else
        fileOpen ( filename.c_str() ); // keep the current display flags
    return true;
}

void RealGUI::fileOpen ( const char* filename )
{
    list_object_added.clear();
    list_object_removed.clear();
    list_object_initial.clear();


    //Hide the dialog to add a new object in the graph
    if ( dialog != NULL ) dialog->hide();
    //Hide all the dialogs to modify the graph
    emit ( newScene() );

    //Clear the list of modified dialog opened
    current_Id_modifyDialog=0;
    map_modifyDialogOpened.clear();


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

    GNode* groot = getSimulation()->load ( filename );
    if ( groot == NULL )
    {
        qFatal ( "Failed to load %s",filename );
        return;
    }

    setScene ( groot, filename );
    //need to create again the output streams !!

    getSimulation()->gnuplotDirectory.setValue(gnuplot_directory);
    setExportGnuplot(exportGnuplotFilesCheckbox->isChecked());
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
    groot = getSimulation()->load ( scene.c_str() );
    if ( groot )
    {
        if ( !pmlreader ) pmlreader = new PMLReader;
        pmlreader->BuildStructure ( filename, groot );
        setScene ( groot, filename );
    }
}

void RealGUI::lmlOpen ( const char* filename )
{
    if ( pmlreader )
    {
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

void RealGUI::setScene ( GNode* groot, const char* filename )
{
    if (tabInstrument!= NULL)
    {
        tabs->removePage(tabInstrument);
        delete tabInstrument;
        tabInstrument = NULL;
    }

    setTitle ( filename );
    record_simulation = false;
    viewer->setScene ( groot, filename );
    initial_time = groot->getTime();

    if (timeSlider->maxValue() == 0)
    {
        setRecordInitialTime(initial_time);
        setRecordFinalTime(initial_time);
        setRecordTime(initial_time);
    }
    eventNewTime();


    //getSimulation()->updateVisualContext ( groot );


    startButton->setOn ( groot->getContext()->getAnimate() );
    dtEdit->setText ( QString::number ( groot->getDt() ) );


    graphView->setSorting ( -1 );
    //graphView->setTreeStepSize(10);
    graphView->header()->hide();
    //dumpGraph(groot, new Q3ListViewItem(graphView));
    graphListener = new GraphListenerQListView ( graphView );
    graphListener->addChild ( NULL, groot );

    //Create Stats about the simulation
    graphCreateStats(groot,NULL);

    //Create the list of the object present at the beginning of the scene
    for ( std::map<core::objectmodel::Base*, Q3ListViewItem* >::iterator it = graphListener->items.begin() ; it != graphListener->items.end() ; ++ it )
    {
        if ( GNode *current_node = dynamic_cast< GNode *> ( ( *it ).first ) )
        {
            list_object_initial.push_back ( current_node );
            list_object_initial.push_back ( dynamic_cast< GNode *> ( current_node->getParent() ) );
        }
    }


    if ( currentTab != TabGraph )
    {
        graphListener->freeze ( groot );
    }

    simulation::tree::Simulation *s = simulation::tree::getSimulation();

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

    filename = getSaveFileName ( this,
            viewer->screenshotName().c_str(),
            "Images (*.png *.bmp *.jpg)",
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
#ifdef QT_MODULE_QT3SUPPORT
        viewer->screenshot ( filename.toStdString() );
#else
        viewer->screenshot ( filename );
#endif

    }
}

void RealGUI::fileOpenSimu ( const char* s )
{
    std::ifstream in(s);
    if (!in.fail())
    {
        std::string filename;
        std::string initT, endT, dT;
        in
                >> filename
                        >> initT >> initT
                        >> endT  >> endT >> endT
                        >> dT >> dT;
        in.close();

        if ( sofa::helper::system::DataRepository.findFile (filename) )
        {
            filename = sofa::helper::system::DataRepository.getFile ( filename );
            simulation_name = s;
            std::string::size_type pointSimu = simulation_name.rfind(".simu");
            simulation_name.resize(pointSimu);
            fileOpen(filename.c_str());
            char buf[100];

            sprintf ( buf, "Init: %s s",initT.c_str()  );
            initialTime->setText ( buf );

            sprintf ( buf, "End: %s s",endT.c_str()  );
            finalTime->setText ( buf );

            loadRecordTime->setText( QString(initT.c_str()) );

            dtEdit->setText(QString(dT.c_str()));
            timeSlider->setMinValue(0);
            timeSlider->setValue(0);
            timeSlider->setMaxValue( (int)((atof(endT.c_str())-atof(initT.c_str()))/(atof(dT.c_str()))+0.5));

            record_directory = sofa::helper::system::SetDirectory::GetParentDir(filename.c_str()) + "/";
        }
    }
}
void RealGUI::fileNew()
{
    std::string newScene("newScene.scn");
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
            "Scenes (*.scn *.xml *.simu)",
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
                fileOpenSimu(s);
            else
            {
                fileOpen (s);
                timeSlider->setValue(0);
                timeSlider->setMaxValue(0);
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
            fileOpenSimu(s);
        else
            fileOpen ( s );
    }
#else
    if (s.endsWith( ".simu") )
        fileOpenSimu(s);
    else
        fileOpen ( s );
#endif

}

void RealGUI::fileSave()
{
    GNode *node = viewer->getScene();
    std::string filename = viewer->getSceneFileName();
    fileSaveAs ( node,filename.c_str() );
}


void RealGUI::fileSaveAs(GNode *node)
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

void RealGUI::fileSaveAs ( GNode *node, const char* filename )
{
    getSimulation()->printXML ( node, filename );
}

void RealGUI::fileExit()
{
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

QString RealGUI::getExistingDirectory ( QWidget* parent, const QString & dir , const char * name , const QString & caption )
{
#ifdef SOFA_QT4
    QFileDialog::Options options = QFileDialog::ShowDirsOnly;
    //	options |= QFileDialog::DontUseNativeDialog;
    options |= QFileDialog::DontUseSheet;
    return QFileDialog::getExistingDirectory ( parent, name?QString(name):caption, dir, options );
#else
    return Q3FileDialog::getExistingDirectory( dir, parent, name, caption );
#endif
}

QString RealGUI::getOpenFileName ( QWidget* parent, const QString & startWith, const QString & filter, const char * name, const QString & caption, QString * selectedFilter )
{
#ifdef SOFA_QT4
    QFileDialog::Options options = 0;
    //	options |= QFileDialog::DontUseNativeDialog;
    options |= QFileDialog::DontUseSheet;
    return QFileDialog::getOpenFileName ( parent, name?QString(name):caption, startWith, filter, selectedFilter, options );
#else
    return Q3FileDialog::getOpenFileName ( startWith, filter, parent, name, caption, selectedFilter );
#endif
}

QString RealGUI::getSaveFileName ( QWidget* parent, const QString & startWith, const QString & filter, const char * name, const QString & caption, QString * selectedFilter )
{
#ifdef SOFA_QT4
    QFileDialog::Options options = 0;
    //	options |= QFileDialog::DontUseNativeDialog;
    options |= QFileDialog::DontUseSheet;
    return QFileDialog::getSaveFileName ( parent, name?QString(name):caption, startWith, filter, selectedFilter, options );
#else
    return Q3FileDialog::getSaveFileName ( startWith, filter, parent, name, caption, selectedFilter );
#endif
}

void RealGUI::setTitle ( const char* windowTitle )
{
    std::string str = "Sofa";
    if ( windowTitle && *windowTitle )
    {
        str += " - ";
        str += windowTitle;
    }
#ifdef _WIN32
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

    if ( groot->getContext()->getMultiThreadSimulation() )
    {
        static Node* n = NULL;

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

#ifdef QT_MODULE_QT3SUPPORT
        viewer->getQWidget()->setUpdatesEnabled ( true );
#endif
        viewer->getQWidget()->update();
        if (currentTab == TabStats) graphCreateStats(viewer->getScene(),NULL);
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
    static int frameCounter = 0;
    GNode* groot = getScene();
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
        const sofa::simulation::tree::GNode::NodeTimer& total = groot->getTotalTime();
        const std::map<std::string, sofa::simulation::tree::GNode::NodeTimer>& times = groot->getVisitorTime();
        const std::map<std::string, std::map<sofa::core::objectmodel::BaseObject*, sofa::simulation::tree::GNode::ObjectTimer> >& objtimes = groot->getObjectTime();
        const double fact = 1000000.0 / ( 100*groot->getTimeFreq() );
        for ( std::map<std::string, sofa::simulation::tree::GNode::NodeTimer>::const_iterator it = times.begin(); it != times.end(); ++it )
        {
            std::cout << "TIME "<<it->first<<": " << ( ( int ) ( fact*it->second.tTree+0.5 ) ) *0.001 << " ms (" << ( 1000*it->second.tTree/total.tTree ) *0.1 << " %).\n";
            std::map<std::string, std::map<sofa::core::objectmodel::BaseObject*, sofa::simulation::tree::GNode::ObjectTimer> >::const_iterator it1 = objtimes.find ( it->first );
            if ( it1 != objtimes.end() )
            {
                for ( std::map<sofa::core::objectmodel::BaseObject*, sofa::simulation::tree::GNode::ObjectTimer>::const_iterator it2 = it1->second.begin(); it2 != it1->second.end(); ++it2 )
                {
                    std::cout << "  "<< sofa::helper::gettypename ( typeid ( * ( it2->first ) ) ) <<" "<< it2->first->getName() <<": "
                            << ( ( int ) ( fact*it2->second.tObject+0.5 ) ) *0.001 << " ms (" << ( 1000*it2->second.tObject/it->second.tTree ) *0.1 << " %).\n";
                }
            }
        }
        for ( std::map<std::string, std::map<sofa::core::objectmodel::BaseObject*, sofa::simulation::tree::GNode::ObjectTimer> >::const_iterator it = objtimes.begin(); it != objtimes.end(); ++it )
        {
            if ( times.count ( it->first ) >0 ) continue;
            ctime_t ttotal = 0;
            for ( std::map<sofa::core::objectmodel::BaseObject*, sofa::simulation::tree::GNode::ObjectTimer>::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2 )
                ttotal += it2->second.tObject;
            std::cout << "TIME "<<it->first<<": " << ( ( int ) ( fact*ttotal+0.5 ) ) *0.001 << " ms (" << ( 1000*ttotal/total.tTree ) *0.1 << " %).\n";
            if ( ttotal > 0 )
                for ( std::map<sofa::core::objectmodel::BaseObject*, sofa::simulation::tree::GNode::ObjectTimer>::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2 )
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
    GNode* groot = getScene();
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
            std::string filename = simulation_name;
            filename = sofa::helper::system::SetDirectory::GetFileName(filename.c_str());

            std::string::size_type point = filename.rfind('.');
            if (point != std::string::npos) filename.resize(point);

            sprintf ( buf, "%s_%.3f.scn",filename.c_str(), time );
            std::string output(record_directory + buf);

            fileSaveAs(viewer->getScene(),output.c_str());
        }
    }
}




//*****************************************************************************************
// Set the time between each iteration of the Sofa Simulation

void RealGUI::setDt ( double value )
{
    GNode* groot = getScene();
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



//-----------------------------------------------------------------------------------
//Recording a simulation
void RealGUI::slot_recordSimulation( bool value)
{
    if (value)
    {
        GNode* groot = getScene();
        if (groot)
        {
            if(timeSlider->maxValue() == 0)
            {
                double time = groot->getTime();
                setRecordInitialTime(time);
            }
            else
            {
                if (atof(loadRecordTime->text()) != groot->getTime() )
                {
                    playSimulation(true); //set the correct sample of the simulation
                }
            }

            //save first sample
            std::string filename = viewer->getSceneFileName();

            //if no simulation has been recorded
            if (timeSlider->maxValue() == 0)
            {
                simulation_name = filename;

                filename = sofa::helper::system::SetDirectory::GetFileName(filename.c_str());

                std::string::size_type point = filename.rfind('.');
                if (point != std::string::npos) filename.resize(point);

                std::ostringstream ofilename;
                ofilename << record_directory << filename << "_" << loadRecordTime->text().ascii() << ".scn";


                fileSaveAs(viewer->getScene(),ofilename.str().c_str());
                fileOpen(ofilename.str().c_str()); //change the local directory
            }
            record_simulation=true;
            playpauseGUI(true);
        }
    }
    else
    {
        record_simulation=false;
        playpauseGUI(false);
        //Save simulation file

        std::string filename = viewer->getSceneFileName();
        filename = sofa::helper::system::SetDirectory::GetFileName(filename.c_str());
        std::string simulationFileName = simulation_name + ".simu";

        std::string output(record_directory + filename);

        std::ofstream out(simulationFileName.c_str());
        if (!out.fail())
        {
            out << output << " " << initialTime->text().ascii() << " " << finalTime->text().ascii() << " " << dtEdit->text().ascii();
            out.close();
        }
        std::cout << "Simulation parameters saved in "<<simulationFileName<<std::endl;

    }
}
void RealGUI::slot_backward(  )
{
    if (timeSlider->value() != timeSlider->minValue())
    {
        setRecordTime(getRecordInitialTime());
        slot_sliderValue(timeSlider->minValue());
    }
}

void RealGUI::slot_stepbackward()
{
    playforward_record->setOn(false);
    playbackward_record->setOn(false);

    playSimulation(false);
}

void RealGUI::slot_playbackward()
{
    if (playbackward_record->isOn())
    {
        playforward_record->setOn(false);
        playSimulation(false);
    }
}

void RealGUI::slot_playforward( )
{
    if (playforward_record->isOn())
    {
        playbackward_record->setOn(false);
        playSimulation(true);
    }
}

void RealGUI::slot_stepforward()
{
    playforward_record->setOn(false);
    playbackward_record->setOn(false);

    playSimulation(true);
}

void RealGUI::slot_forward(  )
{
    if (timeSlider->value() != timeSlider->maxValue())
    {
        setRecordTime(getRecordFinalTime());
        slot_sliderValue(timeSlider->maxValue());
    }
}

void RealGUI::slot_sliderValue(int value)
{
    double init_time  = getRecordInitialTime();
    double delta_time = atof(dtEdit->text().ascii());
    double time = init_time + delta_time*(value);
    setRecordTime(time);
    timeSlider->setValue(value);
    timeSlider->update();
    playSimulation(true);
}

void RealGUI::slot_loadrecord_timevalue()
{
    double init_time = getRecordInitialTime();
    double current_time = getRecordTime();
    int value = (int)((current_time - init_time)/( atof(dtEdit->text()))+0.5);

    if (value >= timeSlider->minValue() && value <= timeSlider->maxValue())
    {
        timeSlider->setValue(value);
        setRecordTime(getRecordTime());
    }
    else if (value < timeSlider->minValue())
    {
        timeSlider->setValue(timeSlider->minValue());
        setRecordTime(getRecordInitialTime());
    }
    else if (value > timeSlider->maxValue())
    {
        timeSlider->setValue(timeSlider->maxValue());
        setRecordTime(getRecordFinalTime());
    }
    playSimulation(true);

}


//Given a direction  (forward, or backward), load the samples of the simulation between the initial and final time.
void RealGUI::playSimulation(bool forward)
{
    if (timeSlider->maxValue() == 0)
    {
        playbackward_record->setOn(false);
        playforward_record->setOn(false);
        return;
    }

    const std::string originalName = viewer->getSceneFileName();
    std::string filename = viewer->getSceneFileName();
    std::string::size_type point = filename.rfind('_');
    if (point != std::string::npos) filename.resize(point);


    QString init_time  = initialTime->text();
    QString final_time = finalTime->text();

    double init_time_value = getRecordInitialTime();
    double final_time_value = getRecordFinalTime();

    int max = timeSlider->maxValue();


    if (getScene()!=NULL && (getRecordTime() == getScene()->getTime()) )
    {
        if (forward)
        {
            double time=getScene()->getTime() + atof(dtEdit->text());
            if ((int)(1000*time) <= (int)(1000*final_time_value) )
            {
                setRecordTime(time);
                timeSlider->setValue(timeSlider->value()+1);
            }
        }
        else
        {
            double time=getScene()->getTime() - atof(dtEdit->text());
            if ( (int)(1000*time) >= (int)(1000*init_time_value))
            {
                setRecordTime(time);
                timeSlider->setValue(timeSlider->value()-1);
            }
        }
    }

    do
    {
        std::string stepFilename;
        std::ostringstream ofilename;
        ofilename << filename << "_" << loadRecordTime->text().ascii() << ".scn";
        stepFilename = ofilename.str();

        if ( sofa::helper::system::DataRepository.findFile (stepFilename) )
        {
            stepFilename = sofa::helper::system::DataRepository.getFile ( stepFilename );
            fileOpen(stepFilename.c_str());

            if ( m_exportGnuplot )
                getSimulation()->exportGnuplot ( getScene(), getScene()->getTime() );

            playpauseGUI(false);

            viewer->setSceneFileName(originalName);
            initialTime->setText(init_time);
            finalTime->setText(final_time);
            timeSlider->setMaxValue(max);

            viewer->getQWidget()->repaint();
            statusBar()->repaint();
            repaint();
        }
        else
            std::cout << "Not Found " << stepFilename << "\n";

        float time=getRecordTime();
        if (forward && playforward_record->isOn())
        {
            time+=atof(dtEdit->text());
            if ((int)(1000*time+0.5) <= (int)(1000*final_time_value+0.5))
            {
                setRecordTime(time);
                timeSlider->setValue(timeSlider->value()+1);
            }
            else
                playforward_record->setOn(false);
        }
        else if (!forward && playbackward_record->isOn())
        {
            time-=atof(dtEdit->text());

            if ((int)(1000*time+0.5) >= (int)(1000*init_time_value+0.5))
            {
                setRecordTime(time);
                timeSlider->setValue(timeSlider->value()-1);
            }
            else
                playbackward_record->setOn(false);
        }
    }
    while ((forward && playforward_record->isOn()) || (!forward && playbackward_record->isOn()));
}


double RealGUI::getRecordInitialTime() const
{
    std::string init_time = initialTime->text().ascii();
    init_time.resize(init_time.size()-2);
    return fabs(atof((init_time.substr(6)).c_str()));
}
void RealGUI::setRecordInitialTime(double time)
{
    char buf[100];
    sprintf ( buf, "Init: %.3f s", fabs(time) );
    initialTime->setText ( buf );
}
double RealGUI::getRecordFinalTime() const
{
    std::string final_time = finalTime->text().ascii();
    final_time.resize(final_time.size()-2);
    return fabs(atof((final_time.substr(5)).c_str()));
}
void RealGUI::setRecordFinalTime(double time)
{
    char buf[100];
    sprintf ( buf, "End: %.3f s", fabs(time) );
    finalTime->setText( buf );
}
double RealGUI::getRecordTime() const
{
    return fabs(atof(loadRecordTime->text().ascii()));
}
void RealGUI::setRecordTime(double time)
{
    char buf[100];
    sprintf ( buf, "%.3f", fabs(time) );
    loadRecordTime->setText( buf );
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
    GNode* groot = getScene();
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
    default:
    {
        break;
    }
    }
}


/*****************************************************************************************************************/
// INTERACTION WITH THE GRAPH
/*****************************************************************************************************************/
void RealGUI::currentTabChanged ( QWidget* widget )
{
    if ( widget == currentTab ) return;
    GNode* groot = viewer==NULL ? NULL : viewer->getScene();
    if ( widget == TabGraph )
    {
        if ( groot && graphListener )
        {
            //graphListener->addChild(NULL, groot);
            graphListener->unfreeze ( groot );
        }
    }
    else if ( currentTab == TabGraph )
    {
        if ( groot && graphListener )
        {
            //graphListener->removeChild(NULL, groot);
            graphListener->freeze ( groot );
        }
    }
    else if (widget == TabStats)
        graphCreateStats(viewer->getScene(),NULL);

    currentTab = widget;
}

/*****************************************************************************************************************/
void RealGUI::DoubleClickeItemInSceneView ( QListViewItem *item )
{
    // This happens because the clicked() signal also calls the select callback with
    // NULL as a parameter.
    if ( item == NULL )
        return;

    item_clicked = item;

    // cancel the visibility action caused by the double click
    item_clicked->setOpen ( !item_clicked->isOpen() );
    graphModify();
}


/*****************************************************************************************************************/
void RealGUI::RightClickedItemInSceneView ( QListViewItem *item, const QPoint& point, int index )
{
    if ( dialog == NULL )
    {
        //Creation of the file dialog
        dialog = new AddObject ( &list_object, this );
        dialog->setPath ( viewer->getSceneFileName() );
        dialog->hide();
    }


    //Creation of a popup menu at the mouse position
    item_clicked=item;

    //Search in the graph if the element clicked is a node
    node_clicked = NULL;
    if ( item_clicked == NULL ) return;



    std::map<core::objectmodel::Base*, Q3ListViewItem* >::iterator graph_iterator;

    for (graph_iterator = graphListener->items.begin(); graph_iterator != graphListener->items.end(); graph_iterator++)
    {
        if ( (*graph_iterator).second == item) {node_clicked = dynamic_cast< GNode* >( (*graph_iterator).first); break;}
    }



    QPopupMenu *contextMenu = new QPopupMenu ( graphView, "ContextMenu" );


    //Creation of the context Menu
    if ( node_clicked != NULL )
    {
        contextMenu->insertItem ( "Collapse", this, SLOT ( graphCollapse() ) );
        contextMenu->insertItem ( "Expand", this, SLOT ( graphExpand() ) );
        contextMenu->insertSeparator ();
        /*****************************************************************************************************************/
        if (node_clicked->isActive())
            contextMenu->insertItem ( "Desactivate", this, SLOT ( graphDesactivateNode() ) );
        else
            contextMenu->insertItem ( "Activate", this, SLOT ( graphActivateNode() ) );
        contextMenu->insertSeparator ();
        /*****************************************************************************************************************/

        contextMenu->insertItem ( "Save Node", this, SLOT ( graphSaveObject() ) );
        contextMenu->insertItem ( "Add Node", this, SLOT ( graphAddObject() ) );
        int index_menu = contextMenu->insertItem ( "Remove Node", this, SLOT ( graphRemoveObject() ) );

        //If one of the elements or child of the current node is beeing modified, you cannot allow the user to erase the node
        if ( !isErasable ( node_clicked ) )
            contextMenu->setItemEnabled ( index_menu,false );

    }
    contextMenu->insertItem ( "Modify", this, SLOT ( graphModify() ) );
    contextMenu->popup ( point, index );

}

/*****************************************************************************************************************/
void RealGUI::graphSaveObject()
{
    bool isAnimated = startButton->isOn();

    playpauseGUI ( false );
    //Just pop up the dialog window
    if ( node_clicked != NULL )
    {
        fileSaveAs(node_clicked);
        item_clicked = NULL;
    }
    playpauseGUI ( isAnimated );
}
/*****************************************************************************************************************/
void RealGUI::graphAddObject()
{

    bool isAnimated = startButton->isOn();

    playpauseGUI ( false );
    //Just pop up the dialog window
    if ( node_clicked != NULL )
    {
        dialog->show();
        dialog->raise();

        item_clicked = NULL;
    }
    playpauseGUI ( isAnimated );
}

/*****************************************************************************************************************/
void RealGUI::graphRemoveObject()
{

    bool isAnimated = startButton->isOn();

    playpauseGUI ( false );
    if ( node_clicked != NULL )
    {
        if ( node_clicked->getParent() == NULL )
        {
            //Attempt to destroy the Root node : create an empty node to handle new graph interaction
            GNode *groot = new GNode ( "Root" );

            groot->setShowVisualModels ( 1 );
            groot->setShowCollisionModels ( 0 );
            groot->setShowBoundingCollisionModels ( 0 );
            groot->setShowBehaviorModels ( 0 );
            groot->setShowMappings ( 0 );
            groot->setShowMechanicalMappings ( 0 );
            groot->setShowForceFields ( 0 );
            groot->setShowInteractionForceFields ( 0 );
            groot->setShowWireFrame ( 0 );
            groot->setShowNormals ( 0 );

// 		showVisual->setChecked ( groot->getShowVisualModels() );
// 		showBehavior->setChecked ( groot->getShowBehaviorModels() );
// 		showCollision->setChecked ( groot->getShowCollisionModels() );
// 		showBoundingCollision->setChecked ( groot->getShowBoundingCollisionModels() );
// 		showForceField->setChecked ( groot->getShowForceFields() );
// 		showInteractionForceField->setChecked ( groot->getShowInteractionForceFields() );
// 		showMapping->setChecked ( groot->getShowMappings() );
// 		showMechanicalMapping->setChecked ( groot->getShowMechanicalMappings() );
// 		showWireFrame->setChecked ( groot->getShowWireFrame() );
// 		showNormals->setChecked ( groot->getShowNormals() );

            viewer->setScene ( groot, viewer->getSceneFileName().c_str() );
            graphListener->removeChild ( NULL, node_clicked );
            graphListener->addChild ( NULL, groot );
        }
        else
        {
            node_clicked->getParent()->removeChild ( node_clicked );
            graphListener->removeChild ( NULL, node_clicked );
            list_object_removed.push_back ( node_clicked );
        }

        viewer->resetView();
        viewer->getQWidget()->update();
        node_clicked = NULL;
        item_clicked = NULL;
    }
    playpauseGUI ( isAnimated );

    graphCreateStats(viewer->getScene(),NULL);
}

/*****************************************************************************************************************/
void RealGUI::graphModify()
{

    bool isAnimated = startButton->isOn();

    playpauseGUI ( false );
    if ( item_clicked != NULL )
    {
        core::objectmodel::Base* node=NULL;
        for ( std::map<core::objectmodel::Base*, Q3ListViewItem* >::iterator it = graphListener->items.begin() ; it != graphListener->items.end() ; ++ it )
        {
            if ( ( *it ).second == item_clicked )
            {
                node = ( *it ).first;
                break;
            }
        }

        //Opening of a dialog window automatically created
        current_Id_modifyDialog = node;
        std::map< void*, QDialog* >::iterator testWindow =  map_modifyObjectWindow.find( current_Id_modifyDialog);
        if ( testWindow != map_modifyObjectWindow.end())
        {
            //Object already being modified: no need to open a new window
            (*testWindow).second->raise();
            playpauseGUI ( isAnimated );
            return;
        }

        ModifyObject *dialogModify = new ModifyObject ( current_Id_modifyDialog, node, item_clicked,this,node->getName().data() );
        map_modifyObjectWindow.insert( std::make_pair(current_Id_modifyDialog, dialogModify));

        //If the item clicked is a node, we add it to the list of the element modified
        if ( dynamic_cast<GNode *> ( node ) )
            map_modifyDialogOpened.insert ( std::make_pair ( current_Id_modifyDialog, node ) );
        else
        {
            //If the item clicked is just an element of the node, we add the node containing it
            for ( std::map<core::objectmodel::Base*, Q3ListViewItem* >::iterator it = graphListener->items.begin() ; it != graphListener->items.end() ; ++ it )
            {
                if ( ( *it ).second == item_clicked->parent() )
                {
                    map_modifyDialogOpened.insert ( std::make_pair ( current_Id_modifyDialog, ( *it ).first ) );
                    break;
                }
            }
        }

        dialogModify->show();
        dialogModify->raise();

        connect ( this, SIGNAL ( newScene() ), dialogModify, SLOT ( closeNow() ) );
        connect ( this, SIGNAL ( newStep() ),  dialogModify, SLOT ( updateTables() ) );
        item_clicked = NULL;
    }
    playpauseGUI ( isAnimated );
}

/*****************************************************************************************************************/
void RealGUI::graphDesactivateNode()
{
    node_clicked->setActive(false);
    viewer->getQWidget()->update();
    node_clicked->reinit();

    graphCreateStats(viewer->getScene(),NULL);
}

void RealGUI::graphActivateNode()
{
    node_clicked->setActive(true);
    viewer->getQWidget()->update();
    node_clicked->reinit();

    graphCreateStats(viewer->getScene(),NULL);
}


/*****************************************************************************************************************/
// Test if a node can be erased in the graph : the condition is that none of its children has a menu modify opened
bool RealGUI::isErasable ( core::objectmodel::Base* element )
{

    if ( GNode *node = dynamic_cast<GNode *> ( element ) )
    {
        //we look into the list of element currently modified if the element is present.
        std::map< void *, core::objectmodel::Base* >::iterator dialog_it;
        for ( dialog_it = map_modifyDialogOpened.begin(); dialog_it !=map_modifyDialogOpened.end(); dialog_it++ )
        {
            if ( element == ( *dialog_it ).second ) return false;
        }

        //The element has not been found, we explorate its childs
        GNode::ChildIterator it;
        for ( it = node->child.begin(); it != node->child.end(); it++ )
        {
            if ( !isErasable ( ( *it ) ) ) return false; //If a child of the current node cannot be erased, the parent (the currend node) cannot be too.
        }

    }
    return true;
}

/*****************************************************************************************************************/
//Translate an object
void RealGUI::transformObject ( GNode *node, double dx, double dy, double dz,  double rx, double ry, double rz, double scale )
{
    if ( node == NULL ) return;
    GNode::ObjectIterator obj_it = node->object.begin();
    const double conversionDegRad = 3.141592653/180.0;
    Vec<3, double> rotationVector = Vec<3,double>(rx,ry,rz)*conversionDegRad;
    //We translate the elements

    while ( obj_it != node->object.end() )
    {
        if ( dynamic_cast< sofa::component::visualmodel::VisualModelImpl* > ( *obj_it ) )
        {
            sofa::component::visualmodel::VisualModelImpl *visual = dynamic_cast< sofa::component::visualmodel::VisualModelImpl* > ( *obj_it );
            visual->applyTranslation ( dx, dy, dz );
            visual->applyRotation(defaulttype::Quat::createFromRotationVector( rotationVector));
            visual->applyScale ( scale );
        }

        if ( dynamic_cast< core::componentmodel::behavior::BaseMechanicalState *> ( *obj_it ) )
        {
            core::componentmodel::behavior::BaseMechanicalState *mechanical = dynamic_cast< core::componentmodel::behavior::BaseMechanicalState *> ( *obj_it );
            mechanical->applyTranslation ( dx, dy, dz );
            mechanical->applyRotation(defaulttype::Quat::createFromRotationVector( rotationVector));
            mechanical->applyScale ( scale );
            // 		mechanical_object = true;
        }

        obj_it++;
    }
    //We don't need to go any further:
    // 	if (mechanical_object && !mesh_topology)
    // 	    return;


    //We search recursively with the childs of the currend node
    GNode::ChildIterator it  = node->child.begin();
    GNode::ChildIterator end = node->child.end();
    while ( it != end )
    {
        transformObject ( *it, dx, dy, dz, rx,ry,rz,scale );
        it++;
    }

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
    xml::BaseElement* xml = xml::load ( path.c_str() );
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
    graphCreateStats(viewer->getScene(),NULL);
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


/*****************************************************************************************************************/
//Visibility Option in grah : expand or collapse a node : easier to get access to a node, and see its properties properly
void RealGUI::graphCollapse()
{
    bool isAnimated = startButton->isOn ();

    playpauseGUI ( false );
    if ( item_clicked != NULL )
    {
        QListViewItem* child;
        child = item_clicked->firstChild();
        while ( child != NULL )
        {
            child->setOpen ( false );
            child = child->nextSibling();
        }
        item_clicked->setOpen ( true );
    }

    playpauseGUI ( isAnimated );
}

void RealGUI::graphExpand()
{
    bool isAnimated = startButton->isOn();
    playpauseGUI ( false );
    item_clicked->setOpen ( true );
    Q3ListViewItem *item_clicked_back = item_clicked;
    if ( item_clicked != NULL )
    {
        QListViewItem* child;
        child = item_clicked->firstChild();
        while ( child != NULL )
        {
            item_clicked = child;

            child->setOpen ( true );
            graphExpand();
            child = child->nextSibling();
        }
    }
    item_clicked = item_clicked_back;
    playpauseGUI ( isAnimated );
}
/*****************************************************************************************************************/
void RealGUI::modifyUnlock ( void *Id )
{
    graphCreateStats(viewer->getScene(),NULL);

    map_modifyDialogOpened.erase( Id );
    map_modifyObjectWindow.erase( Id );
}

/*****************************************************************************************************************/
// Fill the listview in the stats tab with information about the number of points/line/triangle/sphere of the collision models present in the scene

//Add the current node and its child to the stats graph.
bool RealGUI::graphCreateStats( GNode *node, QListViewItem *parent)
{
    bool initialization = false;
    sofa::helper::vector< sofa::core::CollisionModel* > list_collisionModels;
    node->get< sofa::core::CollisionModel >( list_collisionModels);
    //Creation of the item in the graph
    Q3ListViewItem *item;
    if (parent == NULL)
    {
        if (items_stats.size() != 0)
        {
            delete items_stats[0].second;
            items_stats.clear();
            GUI::StatsCounter->clear();
        }

        item = new Q3ListViewItem(GUI::StatsCounter);
        initialization = true;
    }
    else
        item = new Q3ListViewItem(parent);

    int index = items_stats.size();

    items_stats.push_back(std::make_pair(node, item));

    item->setText(0,node->getName().c_str());  item->setOpen(true);

    QPixmap* pix = sofa::gui::qt::getPixmap(node);
    if (pix)
        item->setPixmap(0, *pix);

    bool usedNode = false;
    //Add the current Collision Models
    usedNode = graphAddCollisionModelsStat(list_collisionModels,item);

    //Recursive call
    for (GNode::ChildIterator it= node->child.begin(); it != node->child.end(); ++it)
    {
        usedNode |= graphCreateStats((*it), item);
    }
    if (!usedNode)
    {
        items_stats.erase(items_stats.begin()+index);
        delete item; return false;
    }

    if (initialization)
    {
        graphSummary();
    }
    return true;
}

//Add a list of Collision model to the graph
bool RealGUI::graphAddCollisionModelsStat(sofa::helper::vector< sofa::core::CollisionModel* > &v,QListViewItem *parent)
{
    bool oneAdded = false;
    for (unsigned int i=0; i<v.size(); i++)
    {
        if (!v[i]->isActive()) continue;
        Q3ListViewItem *item = new Q3ListViewItem(parent);
        item->setText(0,v[i]->getName().c_str());

        if      (dynamic_cast< sofa::component::collision::TriangleModel* >(v[i])) item->setText(1, "Triangle");
        else if (dynamic_cast< sofa::component::collision::LineModel* >(v[i]))     item->setText(1, "Line");
        else if (dynamic_cast< sofa::component::collision::PointModel* >(v[i]))    item->setText(1, "Point");
        else if (dynamic_cast< sofa::component::collision::TSphereModel<Vec3Types>* >(v[i]))  item->setText(1, "Sphere");
        else continue;
        item->setText(2,QString::number(v[i]->getSize()));
        items_stats.push_back(std::make_pair(v[i], item));
        oneAdded = true;
    }
    return oneAdded;
}


//create global stats
void RealGUI::graphSummary()
{
    unsigned int counter[4]= {0,0,0,0};
    for (unsigned int i=0; i < items_stats.size(); i++)
    {
        if (!dynamic_cast< GNode *>(items_stats[i].first))
        {
            if ( std::string(items_stats[i].second->text(1).ascii()) == "Triangle")
            {
                counter[0]+=atoi(items_stats[i].second->text(2));
            }
            else if ( std::string(items_stats[i].second->text(1).ascii()) == "Line")
            {
                counter[1]+=atoi(items_stats[i].second->text(2));
            }
            else if ( std::string(items_stats[i].second->text(1).ascii()) == "Point")
            {
                counter[2]+=atoi(items_stats[i].second->text(2));
            }
            else if ( std::string(items_stats[i].second->text(1).ascii()) == "Sphere")
            {
                counter[3]+=atoi(items_stats[i].second->text(2));
            }
        }
    }
    std::string textStats("Collision Elements present: <ul>");
    if (counter[0] != 0)
    {
        char buf[100];
        sprintf ( buf, "<li>Triangles: %d</li>", counter[0] );
        textStats += buf;
    }

    if (counter[1] != 0)
    {
        char buf[100];
        sprintf ( buf, "<li>Lines: %d</li>", counter[1] );
        textStats += buf;
    }

    if (counter[2] != 0)
    {
        char buf[100];
        sprintf ( buf, "<li>Points: %d</li>", counter[2] );
        textStats += buf;
    }

    if (counter[3] != 0)
    {
        char buf[100];
        sprintf ( buf, "<li>Spheres: %d</li>", counter[3] );
        textStats += buf;
    }

    textStats += "</ul>";
    statsLabel->setText( textStats.c_str());
    statsLabel->update();
}

void RealGUI::viewExecutionGraph()
{

}

} // namespace qt

} // namespace gui

} // namespace sofa
