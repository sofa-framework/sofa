
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
#include <iostream>

namespace sofa
{

namespace gui
{

namespace qt
{

#ifndef SOFA_QT4
typedef QListView Q3ListView;
typedef QListViewItem Q3ListViewItem;
typedef QTextDrag Q3TextDrag;
#endif

typedef sofa::core::ObjectFactory::ClassEntry ClassInfo;
typedef sofa::core::ObjectFactory::Creator    ClassCreator;
using sofa::simulation::tree::GNode;
using namespace sofa::core::objectmodel;
class GraphModeler : public Q3ListView
{


    typedef std::map< const QObject* , std::pair< ClassInfo*, QObject*> > ComponentMap;
    Q_OBJECT
public:
    enum DataType {VEC,RIGID,LAPAROSCOPIC, UNKNOWN}; //type possible for mechanical state
    GraphModeler( QWidget* parent=0, const char* name=0, Qt::WFlags f = 0 ):Q3ListView(parent, name, f), graphListener(NULL)
    {
        graphListener = new GraphListenerQListView(this);
        addColumn("Graph");
        header()->hide();
        setSorting ( -1 );

#ifdef SOFA_QT4
        connect(this, SIGNAL(doubleClicked ( Q3ListViewItem *, const QPoint &, int )), this, SLOT( doubleClick(Q3ListViewItem *)));
        connect(this, SIGNAL(rightButtonClicked ( Q3ListViewItem *, const QPoint &, int )),  this, SLOT( rightClick(Q3ListViewItem *, const QPoint &, int )));
#else
        connect(this, SIGNAL(doubleClicked ( QListViewItem *, const QPoint &, int )), this, SLOT( doubleClick(QListViewItem *)));
        connect(this, SIGNAL(rightButtonClicked ( QListViewItem *, const QPoint &, int )),  this, SLOT( rightClick(QListViewItem *, const QPoint &, int )));
#endif
    };

    void dragEnterEvent( QDragEnterEvent* event)
    {
        QString text;
        Q3TextDrag::decode(event, text);
        std::string filename(text.ascii());
        std::string test = filename; test.resize(4);
        if (test == "file") {event->accept();}
        else
        {
            if ( getGNode(event->pos()))
                event->accept(event->answerRect());
            else
                event->ignore(event->answerRect());
        }
    }
    void dragMoveEvent( QDragMoveEvent* event)
    {
        QString text;
        Q3TextDrag::decode(event, text);
        std::string filename(text.ascii());
        std::string test = filename; test.resize(4);
        if (test == "file") {event->accept();}
        else
        {
            if ( getGNode(event->pos()))
                event->accept(event->answerRect());
            else
                event->ignore(event->answerRect());
        }
    }

    bool verifyInsertion(GNode *parent, BaseObject *object);
    void dropEvent(QDropEvent* event);

    void setLibrary(ComponentMap &s) {library=s;}


    GNode *getGNode(const QPoint &pos);
    GNode *getGNode(Q3ListViewItem *item);
    BaseObject *getObject(Q3ListViewItem *item);

    void keyPressEvent ( QKeyEvent * e );

signals:
    void changeNameWindow(std::string);
    void updateRecentlyOpened(std::string);

public slots:
    void collapseNode();
    void collapseNode(Q3ListViewItem* item);
    void expandNode();
    void expandNode(Q3ListViewItem* item);
    void saveNode();
    void saveNode(Q3ListViewItem* item);
    void openModifyObject();
    void openModifyObject(Q3ListViewItem *);
#ifdef SOFA_QT4
    void doubleClick(Q3ListViewItem *);
    void rightClick(Q3ListViewItem *, const QPoint &, int );
#else
    void doubleClick(QListViewItem *);
    void rightClick(QListViewItem *, const QPoint &, int );
#endif
    void addGNode(GNode *parent, bool saveHistory=true);
    void addComponent(GNode *parent, ClassInfo *entry, std::string templateName, bool saveHistory=true );
    void deleteComponent();
    void deleteComponent(Q3ListViewItem *item, bool saveHistory=true);
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
        Operation(Q3ListViewItem* item_,Base* sofaComponent_,  op ID_):
            item(item_),sofaComponent(sofaComponent_), ID(ID_)
        {}
        Q3ListViewItem* item;
        Base* sofaComponent;
        op ID;
    };

    struct TemplateInfo
    {
        unsigned int dim;
        DataType type;
        bool isFloat;

        bool operator==(const TemplateInfo& i)
        {
            return (i.dim == dim) && (i.type == type) && (i.isFloat == isFloat);
        }

        friend std::ostream& operator<< (std::ostream& out, const TemplateInfo& t)
        {
            out << "Dimenstion : " << t.dim << " Type: ";
            if (t.type == VEC) out <<"Vec";
            else if (t.type == RIGID) out << "Rigid";
            else if (t.type == LAPAROSCOPIC) out << "Laparoscopic";
            else if (t.type == UNKNOWN) out << "Unknown";

            if (t.isFloat) out << " FLOAT";
            else           out << " DOUBLE";
            return out;
        }

    };

    void getInfoTemplate(std::string templateName, TemplateInfo &info);

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
