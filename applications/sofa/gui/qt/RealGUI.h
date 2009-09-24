/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     **
 under the terms of the GNU General Public License as published by the Free  *
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
#ifndef SOFA_GUI_VIEWER_REALGUI_H
#define SOFA_GUI_VIEWER_REALGUI_H

#ifdef SOFA_PML
#  include <sofa/filemanager/sofapml/PMLReader.h>
#  include <sofa/filemanager/sofapml/LMLReader.h>
#endif

#include <time.h>


#include <sofa/gui/SofaGUI.h>

#include "GUI.h"
#include <sofa/gui/qt/GraphListenerQListView.h>
#include <sofa/gui/qt/FileManagement.h>
#include <sofa/gui/qt/viewer/SofaViewer.h>
#include <sofa/gui/qt/AddObject.h>
#include <sofa/gui/qt/ModifyObject.h>
#include <sofa/gui/qt/DisplayFlagWidget.h>
#include <sofa/gui/qt/SofaPluginManager.h>
#include <sofa/gui/qt/SofaMouseManager.h>

#include <sofa/simulation/common/xml/XML.h>
#include <sofa/helper/system/SetDirectory.h>

#ifdef SOFA_DUMP_VISITOR_INFO
#include <sofa/gui/qt/WindowVisitor.h>
#include <sofa/gui/qt/GraphVisitor.h>
#endif

#ifdef SOFA_QT4
#include <QApplication>
#include <QDesktopWidget>
#include <Q3ListViewItem>
#include <QStackedWidget>
#include <QSlider>
#include <QTimer>
#include <Q3TextDrag>
#include <Q3PopupMenu>
#include <QLibrary>
#include <QTextBrowser>
#include <QUrl>
typedef Q3ListViewItem QListViewItem;
typedef QStackedWidget QWidgetStack;
typedef Q3PopupMenu QPopupMenu;
#else
typedef QTextDrag Q3TextDrag;
#include <qapplication.h>
#include <qdesktopwidget.h>
#include <qdragobject.h>
#include <qwidgetstack.h>
#include <qlistview.h>
#include <qslider.h>
#include <qpopupmenu.h>
#include <qlibrary.h>
#include <qtextbrowser.h>
#include <qurl.h>
#endif

#ifdef SOFA_PML
#include <sofa/simulation/tree/GNode.h>
#endif

namespace sofa
{

namespace gui
{

namespace qt
{

//enum TYPE{ NORMAL, PML, LML};
enum SCRIPT_TYPE { PHP, PERL };

using sofa::simulation::Node;
#ifdef SOFA_PML
using namespace sofa::filemanager::pml;
#endif


class RealGUI : public ::GUI, public SofaGUI
{
    Q_OBJECT

    /// @name SofaGUI Interface
    /// @{


public:

    static int InitGUI(const char* name, const std::vector<std::string>& options);
    static SofaGUI* CreateGUI(const char* name, const std::vector<std::string>& options, sofa::simulation::Node* groot = NULL, const char* filename = NULL);

    int mainLoop();

    int closeGUI();

    Node* currentSimulation();

    /// @}

    const char* viewerName;

    sofa::gui::qt::viewer::SofaViewer* viewer;

    RealGUI( const char* viewername, const std::vector<std::string>& options = std::vector<std::string>() );
    ~RealGUI();


    virtual void fileOpen(std::string filename); //, int TYPE=NORMAL);
    virtual void fileOpenSimu(std::string filename); //, int TYPE=NORMAL);
    virtual void setScene(Node* groot, const char* filename=NULL, bool temporaryFile=false);
    virtual void setDimension(int w, int h);
    virtual void setFullScreen();
    virtual void setTitle( std::string windowTitle );

    //public slots:
    virtual void fileNew();
    virtual void fileOpen();
    virtual void fileSave();
    virtual void fileSaveAs() {fileSaveAs((Node *)NULL);};
    virtual void fileSaveAs(Node *node);
    virtual void fileSaveAs(Node* node,const char* filename);

    virtual void fileReload();
    virtual void fileExit();
    virtual void saveXML();
    virtual void viewerOpenGL();
    virtual void viewerQGLViewer();
    virtual void viewerOGRE();

    virtual void editRecordDirectory();
    virtual void editGnuplotDirectory();
    virtual void showPluginManager();
    virtual void showMouseManager();

    void dragEnterEvent( QDragEnterEvent* event) {event->accept();}
    void dropEvent(QDropEvent* event);

    void initRecentlyOpened();

public slots:
    void fileRecentlyOpened(int id);

    void updateRecentlyOpened(std::string fileLoaded);
    void DoubleClickeItemInSceneView(QListViewItem * item);
    void RightClickedItemInSceneView(QListViewItem *item, const QPoint& point, int index);
    void playpauseGUI(bool value);
    void step();
    void setDt(double);
    void setDt(const QString&);
    void resetScene();
    void screenshot();

    void showhideElements(int FILTER, bool value);

    void clearRecord();
    void slot_recordSimulation( bool);
    void slot_backward( );
    void slot_stepbackward( );
// 	  void slot_playbackward(  );
    void slot_playforward(  ) ;
    void slot_stepforward( ) ;
    void slot_forward( );
    void slot_sliderValue(int value,bool updateTime=true);
    void slot_loadrecord_timevalue(bool updateTime=true);


    void updateViewerParameters();
    void updateBackgroundColour();
    void updateBackgroundImage();

    void changeHtmlPage( const QUrl&);
    void changeInstrument(int);

    void clearGraph();
    //Used in Context Menu
    void graphSaveObject();
    void graphAddObject();
    void graphRemoveObject();
    void graphModify();
    void graphCollapse();
    void graphExpand();
    void graphDesactivateNode();
    void graphActivateNode();
    //When adding an object in the graph
    void loadObject(std::string path, double dx, double dy, double dz,double rx, double ry, double rz, double scale=1.0);
    //refresh the visualization window
    void redraw();
    //when a dialog modify object is closed
    void modifyUnlock(void *Id);
    void transformObject( Node *node, double dx, double dy, double dz, double rx, double ry, double rz, double scale=1.0);


    void exportGraph();
    void exportGraph(sofa::simulation::Node*);
    void exportOBJ(bool exportMTL=true);
    void dumpState(bool);
    void displayComputationTime(bool);
    void setExportGnuplot(bool);
    void setExportVisitor(bool);
    void currentTabChanged(QWidget*);

signals:
    void reload();
    void newScene();
    void newStep();
    void quit();

protected:

    void eventNewStep();
    void eventNewTime();
    void init();
    void keyPressEvent ( QKeyEvent * e );

    void loadSimulation(bool one_step=false);

    void initDesactivatedNode();
    //Graph Stats
    bool graphCreateStats(Node *groot);
    void graphAddCollisionModelsStat(sofa::helper::vector< sofa::core::CollisionModel* > &v);
    void graphSummary();

    void addInitialNodes( Node* node);
    bool isErasable(core::objectmodel::Base* element);

    void startDumpVisitor();
    void stopDumpVisitor();

    bool m_dumpState;
    std::ofstream* m_dumpStateStream;
    std::ostringstream m_dumpVisitorStream;
    bool m_exportGnuplot;
    bool _animationOBJ; int _animationOBJcounter;// save a succession of .obj indexed by _animationOBJcounter
    bool m_displayComputationTime;


    QWidget* currentTab;
    QWidget *tabInstrument;

    GraphListenerQListView* graphListener;
    QListViewItem *item_clicked;
    Node *node_clicked;
    QTimer* timerStep;
    QTimer* timerRecordStep;
    QLabel* fpsLabel;
    QLabel* timeLabel;
    WFloatLineEdit *background[3];
    QLineEdit *backgroundImage;


    void setPixmap(std::string pixmap_filename, QPushButton* b);

    double getRecordInitialTime() const;
    void   setRecordInitialTime(const double time);
    double getRecordFinalTime  () const;
    void   setRecordFinalTime  (const double time);
    double getRecordTime       () const;
    void   setRecordTime       (const double time);
    void   setTimeSimulation   (const double time);

    QPushButton* record;
    QPushButton* backward_record;
    QPushButton* stepbackward_record;
// 	  QPushButton* playbackward_record;
    QPushButton* playforward_record;
    QPushButton* stepforward_record;
    QPushButton* forward_record;

    QLineEdit* loadRecordTime;
    QLabel* initialTime;
    QLabel* finalTime;
    QSlider* timeSlider;

    std::string simulation_name;
    std::string record_directory;
    std::string gnuplot_directory;
    std::string writeSceneName;
    std::string pathDumpVisitor;

    QWidgetStack* left_stack;
    AddObject *dialog;



    sofa::simulation::Node* getScene() { if (viewer) return viewer->getScene(); else return NULL; }

    void sleep(float seconds, float init_time)
    {
        unsigned int t = 0;
        clock_t goal = (clock_t) (seconds + init_time);
        while (goal > clock()/(float)CLOCKS_PER_SEC) t++;
    }

private:
    //Map: Id -> Node currently modified. Used to avoid dependancies during removing actions
    std::map< void*, core::objectmodel::Base* >            map_modifyDialogOpened;

    std::map< void*, QDialog* >                       map_modifyObjectWindow;
    std::vector<std::pair<core::objectmodel::Base*, Q3ListViewItem*> > items_stats;
    //unique ID to pass to a modify object dialog
    void *current_Id_modifyDialog;


    //currently unused: scale is experimental
    float object_Scale[2];

    float initial_time;
    int frameCounter;
    //At initialization: list of the path to the basic objects you can add to the scene
    std::vector< std::string > list_object;
    std::list< Node *> list_object_added;
    std::list< Node *> list_object_removed;
    //Pair: parent->child
    std::list< std::pair< Node *, Node* > > list_object_initial;
    bool record_simulation;

    bool setViewer(const char* name);
    void addViewer();
    void setGUI(void);

    bool setWriteSceneName();
    void addReadState(bool init);
    void addWriteState();

#ifdef SOFA_PML
    virtual void pmlOpen(const char* filename, bool resetView=true);
    virtual void lmlOpen(const char* filename);
    PMLReader *pmlreader;
    LMLReader *lmlreader;
#endif

    DisplayFlagWidget *displayFlag;

#ifdef SOFA_DUMP_VISITOR_INFO
    WindowVisitor* windowTraceVisitor;
    GraphVisitor* handleTraceVisitor;
#endif
    QDialog* descriptionScene;
    QTextBrowser* htmlPage;

};

} // namespace qt

} // namespace gui

} // namespace sofa

#endif // SOFA_GUI_VIEWER_REALGUI_H
