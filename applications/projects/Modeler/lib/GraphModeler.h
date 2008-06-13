
#ifndef SOFA_GRAPHMODELER_H
#define SOFA_GRAPHMODELER_H

#include <deque>

#include <sofa/simulation/tree/GNode.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseObject.h>

#ifdef SOFA_QT4
#include <Q3ListView>
#include <Q3ListViewItem>
#include <Q3TextDrag>
#else
#include <qlistview.h>
#include <qdragobject.h>
#endif
#include <sofa/gui/qt/GraphListenerQListView.h>
#include <sofa/gui/qt/ModifyObject.h>
namespace sofa
{

namespace gui
{

namespace qt
{

#ifdef SOFA_QT4
typedef Q3ListView QListView;
typedef Q3ListViewItem QListViewItem;
typedef Q3TextDrag QTextDrag;
#endif

typedef sofa::core::ObjectFactory::ClassEntry ClassInfo;
typedef sofa::core::ObjectFactory::Creator    ClassCreator;
using sofa::simulation::tree::GNode;
using namespace sofa::core::objectmodel;
class GraphModeler : public QListView
{
    typedef std::map< const QObject* , std::pair< ClassInfo*, QObject*> > ComponentMap;
    Q_OBJECT
public:
    GraphModeler( QWidget* parent=0, const char* name=0, Qt::WFlags f = 0 ):QListView(parent, name, f), graphListener(NULL)
    {
        graphListener = new GraphListenerQListView(this);
        addColumn("Graph");
        header()->hide();
        setSorting ( -1 );
        connect(this, SIGNAL(doubleClicked ( QListViewItem *, const QPoint &, int )), this, SLOT( doubleClick(QListViewItem *)));
        connect(this, SIGNAL(rightButtonClicked ( QListViewItem *, const QPoint &, int )),  this, SLOT( rightClick(QListViewItem *, const QPoint &, int )));
    };

    void dragEnterEvent(QDragEnterEvent* event)
    {
        event->accept(QTextDrag::canDecode(event));
    }

    void dropEvent(QDropEvent* event);

    void setLibrary(ComponentMap &s) {library=s;}


    GNode *getGNode(const QPoint &pos);
    GNode *getGNode(QListViewItem *item);
    BaseObject *getObject(QListViewItem *item);

    void keyPressEvent ( QKeyEvent * e );

signals:
    void changeNameWindow(std::string);

public slots:
    void collapseNode();
    void collapseNode(QListViewItem* item);
    void expandNode();
    void expandNode(QListViewItem* item);
    void saveNode();
    void saveNode(QListViewItem* item);
    void openModifyObject();
    void openModifyObject(QListViewItem *);
    void doubleClick(QListViewItem *);
    void rightClick(QListViewItem *, const QPoint &, int );
    void addGNode(GNode *parent, bool saveHistory=true);
    void addComponent(GNode *parent, ClassInfo *entry, std::string templateName, bool saveHistory=true );
    void deleteComponent();
    void deleteComponent(QListViewItem *item, bool saveHistory=true);
    void modifyUnlock ( void *Id );

    //File Menu
    void changeName(std::string filename);
    void fileNew(GNode* root=NULL);
    void fileReload();
    void fileOpen();
    void fileOpen(std::string filename);
    void fileSave();
    void fileSave(std::string filename);
    void fileSaveAs();
    void editUndo();
    void editRedo();
protected:
    class Operation
    {
    public:
        Operation() {};
        enum op {DELETE_OBJECT, ADD_OBJECT};
        Operation(QListViewItem* item_,Base* sofaComponent_,  op ID_):
            item(item_),sofaComponent(sofaComponent_), ID(ID_)
        {}
        QListViewItem* item;
        Base* sofaComponent;
        op ID;
    };


    GraphListenerQListView *graphListener;
    ComponentMap library;

    //Modify windows management: avoid duplicity, and dependencies
    void *current_Id_modifyDialog;
    std::map< void*, Base* >       map_modifyDialogOpened;
    std::map< void*, QDialog* >                       map_modifyObjectWindow;

    std::string filenameXML;
    std::deque< Operation > historyOperation;
    std::deque< Operation >::iterator currentStateHistory;

};













//Overloading ModifyObject to display all the elements
class ModifyObjectModeler: public ModifyObject
{
public:
    ModifyObjectModeler( void *Id_, core::objectmodel::Base* node_clicked, Q3ListViewItem* item_clicked, QWidget* parent_, const char* name= 0 )
    {
        parent = parent_;
        node = NULL;
        Id = Id_;
        visualContentModified=false;
        setCaption(name);
        HIDE_FLAG = false;
        EMPTY_FLAG = true;
        RESIZABLE_FLAG = true;

        energy_curve[0]=NULL;	        energy_curve[1]=NULL;	        energy_curve[2]=NULL;
        //Initialization of the Widget
        setNode(node_clicked, item_clicked);
        connect ( this, SIGNAL( dialogClosed(void *) ) , parent_, SLOT( modifyUnlock(void *)));
    }

};

}
}
}

#endif
