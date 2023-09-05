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
#pragma once
#include <sofa/gui/qt/config.h>

#include <string>
#include <vector>

#include <ui_GUI.h>
#include "GraphListenerQListView.h"
#include "QMenuFilesRecentlyOpened.h"
#include "AboutSOFADialog.h"
#include "PickHandlerCallBacks.h"

#include <sofa/gui/common/BaseGUI.h>
#include <sofa/gui/common/ViewerFactory.h>

#include <set>
#include <string>

#include <QListView>
#include <QUrl>
#include <QTimer>
#include <QTextBrowser>
#include <QDockWidget>
#include <QWindow>
#include <ctime>

#include <sofa/helper/system/FileMonitor.h>

class WDoubleLineEdit;
class QDragEnterEvent;

namespace sofa::gui::qt
{
#if(SOFA_GUI_QT_HAVE_QT5_WEBENGINE)
class DocBrowser ;
#endif

//enum TYPE{ NORMAL, PML, LML};
enum SCRIPT_TYPE { PHP, PERL };

class QSofaListView;
class QDisplayPropertyWidget;
class QSofaStatWidget;
class GraphListenerQListView;
class DisplayFlagsDataWidget;
class SofaPluginManager;
#ifdef SOFA_DUMP_VISITOR_INFO
class WindowVisitor;
class GraphVisitor;
#endif

class SofaMouseManager;

#if SOFA_GUI_QT_HAVE_QT_CHARTS
class SofaWindowProfiler;
#endif

#if SOFA_GUI_QT_HAVE_NODEEDITOR
class SofaWindowDataGraph;
#endif

namespace viewer
{
class SofaViewer;
}


class SOFA_GUI_QT_API RealGUI : public QMainWindow, public Ui::GUI, public sofa::gui::common::BaseGUI
{
    Q_OBJECT    

//-----------------STATIC METHODS------------------------{
public:
    static void setupSurfaceFormat();
    static common::BaseGUI* CreateGUI(const char* name, sofa::simulation::Node::SPtr groot = nullptr, const char* filename = nullptr);

    static void SetPixmap(std::string pixmap_filename, QPushButton* b);

protected:
    static void CreateApplication(int _argc=0, char** _argv=nullptr);
    static void InitApplication( RealGUI* _gui);
//-----------------STATIC METHODS------------------------}



//-----------------CONSTRUCTOR - DESTRUCTOR ------------------------{
public:
    RealGUI( const char* viewername);

    ~RealGUI() override;
//-----------------CONSTRUCTOR - DESTRUCTOR ------------------------}



//-----------------OPTIONS DEFINITIONS------------------------{
//public:

#ifdef SOFA_DUMP_VISITOR_INFO
    void setTraceVisitors(bool) override;
#endif

    void showFPS(double fps) override;

protected:
    QLabel* fpsLabel;
    QLabel* timeLabel;


private:

#ifdef SOFA_DUMP_VISITOR_INFO
    WindowVisitor* windowTraceVisitor;    
    GraphVisitor* handleTraceVisitor;
#endif
    SofaMouseManager* m_sofaMouseManager;
#if SOFA_GUI_QT_HAVE_QT_CHARTS
    SofaWindowProfiler* m_windowTimerProfiler;
#endif

#if SOFA_GUI_QT_HAVE_NODEEDITOR
    SofaWindowDataGraph* m_sofaWindowDataGraph;
#endif
//-----------------OPTIONS DEFINITIONS------------------------}



//-----------------DATAS MEMBER------------------------{
public:
    //TODO: make a protected data with an accessor
    QSofaListView* simulationGraph;

protected:
    /// create a viewer by default, otherwise you have to manage your own viewer
    bool m_createViewersOpt;
    bool m_isEmbeddedViewer;
    bool m_dumpState;
    std::ofstream* m_dumpStateStream;
    std::ostringstream m_dumpVisitorStream;
    bool m_exportGnuplot;
    bool m_animateOBJ;
    int m_animationOBJcounter;// save a succession of .obj indexed by _animationOBJcounter
    bool m_displayComputationTime;
    bool m_fullScreen;
    common::BaseViewer* m_viewer;
    // Clock before the last simulation step (or zero if the
    // simulation hasn't run yet).
    clock_t m_clockBeforeLastStep;

    // Component Properties
    QDisplayPropertyWidget* propertyWidget;

    /// list of all viewer key name (for creation) mapped to its QAction in the GUI
    std::map< helper::SofaViewerFactory::Key, QAction* > viewerMap;
    InformationOnPickCallBack informationOnPickCallBack;

    QWidget* currentTab;
    QSofaStatWidget* statWidget;
    QTimer* timerStep;
    QTimer* timerIdle;
    WDoubleLineEdit *background[3];
    QLineEdit *backgroundImage;
    SofaPluginManager* pluginManagerDialog;
    QMenuFilesRecentlyOpened recentlyOpenedFilesManager;

    std::string simulationName;
    std::string gnuplotDirectory;
    std::string pathDumpVisitor;

    /// Keep track of log files that have been modified since the GUI started
    std::set<std::string>   m_modifiedLogFiles;

    bool m_enableInteraction {false};
private:
    //currently unused: scale is experimental
    float m_objectScale[2];
    bool m_saveReloadFile;
    DisplayFlagsDataWidget*  displayFlag  {nullptr};
#if(SOFA_GUI_QT_HAVE_QT5_WEBENGINE)
    DocBrowser*              m_docbrowser {nullptr};
#endif
    bool m_animationState;
    int m_frameCounter;
    unsigned int m_viewerMSAANbSampling;
//-----------------DATAS MEMBER------------------------}



//-----------------METHODS------------------------{
public:
    void stepMainLoop () override;

    int mainLoop() override;
    int closeGUI() override;
    sofa::simulation::Node* currentSimulation() override;
    virtual void fileOpen(std::string filename, bool temporaryFile=false, bool reload=false);

    // virtual void fileOpen();
    virtual void fileOpenSimu(std::string filename);
    virtual void setScene(Node::SPtr groot, const char* filename=nullptr, bool temporaryFile=false) override;
    virtual void setSceneWithoutMonitor(Node::SPtr groot, const char* filename=nullptr, bool temporaryFile=false);

    virtual void unloadScene(bool _withViewer = true);

    virtual void setTitle( std::string windowTitle );
    virtual void fileSaveAs(Node* node,const char* filename);
//    virtual void saveXML();

    void setViewerResolution(int w, int h) override;
    void setFullScreen() override { setFullScreen(true); }
    virtual void setFullScreen(bool enable);
    void centerWindow() override;
    void setBackgroundColor(const sofa::type::RGBAColor& c) override;
    virtual void setBackgroundImage(const std::string& i) override;
    void setViewerConfiguration(sofa::component::setting::ViewerSetting* viewerConf) override;
    void setMouseButtonConfiguration(sofa::component::setting::MouseButtonSetting *button) override;

    //Configuration methods
    void setDumpState(bool) override;
    void setLogTime(bool) override;
    void setExportState(bool) override;
    virtual void setGnuplotPath(const std::string & path) override;

    /// create a viewer according to the argument key
    /// \note the viewerMap have to be initialize at least once before
    /// \arg _updateViewerList is used only if you want to reactualise the viewerMap in the GUI
    /// TODO: find a better way to propagate the argument when we construct the viewer
    virtual void createViewer(const char* _viewerName, bool _updateViewerList=false);

    /// Used to directly replace the current viewer
    void registerViewer(common::BaseViewer* _viewer) override;

    common::BaseViewer* getViewer() override;

    /// A way to know if our viewer is embedded or not... (see initViewer)
    /// TODO: Find a better way to do this
    sofa::gui::qt::viewer::SofaViewer* getSofaViewer();

    /// Our viewer is a QObject SofaViewer
    bool isEmbeddedViewer();

    virtual void removeViewer();

    void dragEnterEvent( QDragEnterEvent* event) override;

    void dropEvent(QDropEvent* event) override;

protected:
    /// init data member from RealGUI for the viewer initialisation in the GUI
    void init();
    void createDisplayFlags(Node::SPtr root);
    void loadSimulation(bool one_step=false); //? where is the implementation ?
    void eventNewStep();
    void eventNewTime();
    void keyPressEvent ( QKeyEvent * e ) override;
    void startDumpVisitor();
    void stopDumpVisitor();

    /// init the viewer for the GUI (embeded or not we have to connect some info about viewer in the GUI)
    void initViewer(common::BaseViewer* _viewer) override;

    /// Our viewer is a QObject SofaViewer
    void isEmbeddedViewer(bool _onOff)
    {
        m_isEmbeddedViewer = _onOff;
    }

    virtual int exitApplication(unsigned int _retcode = 0)
    {
        return _retcode;
    }

    void sleep(float seconds, float init_time)
    {
        [[maybe_unused]] unsigned int t = 0;
        const clock_t goal = (clock_t) (seconds + init_time);
        while (goal > clock()/(float)CLOCKS_PER_SEC) t++;
    }

    sofa::simulation::Node::SPtr mSimulation;

    sofa::helper::system::FileEventListener* m_filelistener {nullptr};
private:
    void addViewer();//? where is the implementation ?

    /// Parse options from the RealGUI constructor
    void parseOptions();

    void createPluginManager();
    void createSofaWindowDataGraph();

    /// configure Recently Opened Menu
    void createRecentFilesMenu();

    void createBackgroundGUIInfos();
    void createSimulationGraph();
    void createPropertyWidget();
    void createWindowVisitor();
    void createAdvancedTimerProfilerWindow();

public slots:
    virtual void newRootNode(sofa::simulation::Node* root, const char* path);
    virtual void activateNode(sofa::simulation::Node* , bool );
    virtual void setSleepingNode(sofa::simulation::Node*, bool);
    virtual void fileSaveAs(sofa::simulation::Node *node);
    virtual void lockAnimation(bool);
    virtual void fileRecentlyOpened(QAction * action);
    virtual void playpauseGUI(bool value);
    virtual void interactionGUI(bool value);
    virtual void step();
    virtual void emitIdle();
    // virtual void setDt(double);
    virtual void setDt(const QString&);
    // Disable dtEdit when realTimeCheckBox is checked
    virtual void updateDtEditState();
    virtual void resetScene();
    virtual void screenshot();
    virtual void showhideElements();
    virtual void update();
    virtual void updateBackgroundColour();
    virtual void updateBackgroundImage();

    // Propagate signal to call viewer method in case of it is not a widget
    virtual void resetView()            {if(getViewer())getViewer()->resetView();       }
    virtual void saveView()             {if(getViewer())getViewer()->saveView();        }
    virtual void setSizeW ( int _valW ) {if(getViewer())getViewer()->setSizeW(_valW);   }
    virtual void setSizeH ( int _valH ) {if(getViewer())getViewer()->setSizeH(_valH);   }

    virtual void clear();
    /// refresh the visualization window
    void redraw() override;
    virtual void exportOBJ(sofa::simulation::Node* node, bool exportMTL=true);
    virtual void dumpState(bool);
    virtual void displayComputationTime(bool);
    virtual void setExportGnuplot(bool);
    virtual void setExportVisitor(bool);
    virtual void displayProflierWindow(bool);
    virtual void currentTabChanged(int index);

    virtual void fileNew();
    virtual void popupOpenFileSelector();
    virtual void fileReload();
    virtual void fileSave();
    virtual void fileExit();
    virtual void fileSaveAs() {
        fileSaveAs((Node *)nullptr);
    }
    virtual void helpAbout() { /* TODO */ }
    virtual void editRecordDirectory();
    virtual void editGnuplotDirectory();
    virtual void showDocBrowser() ;
    virtual void showAbout() ;
    virtual void showPluginManager();
    virtual void showMouseManager();
    virtual void showVideoRecorderManager();
    virtual void showWindowDataGraph();
    virtual void toolsDockMoved();

protected slots:
    /// Allow to dynamicly change viewer. Called when click on another viewer in GUI Qt viewer list (see viewerMap).
    /// TODO: find a better way to propagate the argument when we construct the viewer
    virtual void changeViewer();

    /// Update the viewerMap and create viewer if we haven't yet one (the first of the list)
    /// TODO: find a better way to propagate the argument when we construct the viewer
    virtual void updateViewerList();

    /// Update the scenegraph and activate the automatic refresh.
    virtual void onSceneGraphRefreshButtonClicked();

    /// Update the SceneGraph update button to reflect the dirtyness status.
    virtual void sceneGraphViewDirtynessChanged(bool isDirty);

    /// Update the SceneGraph update button to reflect the locking status.
    virtual void sceneGraphViewLockingChanged(bool isLocked);

    void propertyDockMoved(Qt::DockWidgetArea a);

    void appendToDataLogFile(QString);

    void docBrowserVisibilityChanged(bool) ;

signals:
    void reload();
    void newScene();
    void newStep();
    void quit();
//-----------------SIGNALS-SLOTS------------------------}

};





struct ActivationFunctor
{
    ActivationFunctor(bool act, GraphListenerQListView* l):active(act), listener(l)
    {
        pixmap_filename= std::string("textures/media-record.png");
        if ( sofa::helper::system::DataRepository.findFile ( pixmap_filename ) )
            pixmap_filename = sofa::helper::system::DataRepository.getFile ( pixmap_filename );
    }
    void operator()(core::objectmodel::BaseNode* n)
    {
        if (active)
        {
            //Find the corresponding node in the Qt Graph
            QTreeWidgetItem *item=listener->items[n];
            //Remove the text
            QString desact_text = item->text(0);
            desact_text.remove(QString("Deactivated "), Qt::CaseInsensitive);
            item->setText(0,desact_text);
            //Remove the icon
            const QPixmap *p = getPixmap(n, false,false, false);
            item->setIcon(0, QIcon(*p));
//            item->setOpen(true);
            item->setExpanded(true);
        }
        else
        {
            //Find the corresponding node in the Qt Graph
            QTreeWidgetItem *item=listener->items[n];
            //Remove the text
            item->setText(0, QString("Deactivated ") + item->text(0));
            item->setIcon(0, QIcon(QPixmap::fromImage(QImage(pixmap_filename.c_str()))));

            item->setExpanded(false);
        }
    }
protected:
    std::string pixmap_filename;
    bool active;
    GraphListenerQListView* listener;
};

} //namespace sofa::gui::qt
