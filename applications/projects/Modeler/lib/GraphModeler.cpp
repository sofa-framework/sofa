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
#include "GraphModeler.h"
#include "AddPreset.h"


#include <sofa/core/ComponentLibrary.h>
#include <sofa/core/objectmodel/ConfigurationSetting.h>

#include <sofa/simulation/Simulation.h>
#include <sofa/gui/qt/FileManagement.h> //static functions to manage opening/ saving of files

#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>

#include <SofaSimulationCommon/xml/ObjectElement.h>
#include <SofaSimulationCommon/xml/AttributeElement.h>
#include <SofaSimulationCommon/xml/DataElement.h>
#include <SofaSimulationCommon/xml/XML.h>
#include <sofa/simulation/XMLPrintVisitor.h>

#include <QMenu>
#include <QMessageBox>
#include <QHeaderView>
#include <QDrag>
#include <QMimeData>

using sofa::core::ComponentLibrary;

namespace sofa
{

namespace gui
{

namespace qt
{
namespace
{
int numNode= 0;
int numComponent= 0;
}

GraphModeler::GraphModeler(QWidget* parent, const char* name, Qt::WindowFlags f): QTreeWidget(parent), graphListener(NULL), propertyWidget(NULL)
{
    this->setObjectName(name);
    setWindowFlags(f);

    graphListener = new GraphListenerQListView(this);
    //addColumn("Graph");
    header()->hide();
    setSortingEnabled(false);
    setSelectionMode(QAbstractItemView::ExtendedSelection);
    this->setItemsExpandable(true);

    historyManager=new GraphHistoryManager(this);
    //Make the connections
    connect(this, SIGNAL(operationPerformed(GraphHistoryManager::Operation&)), historyManager, SLOT(operationPerformed(GraphHistoryManager::Operation&)));
    connect(this, SIGNAL(graphClean()), historyManager, SLOT(graphClean()));

    connect(historyManager, SIGNAL(undoEnabled(bool)),   this, SIGNAL(undoEnabled(bool)));
    connect(historyManager, SIGNAL(redoEnabled(bool)),   this, SIGNAL(redoEnabled(bool)));
    connect(historyManager, SIGNAL(graphModified(bool)), this, SIGNAL(graphModified(bool)));
    connect(historyManager, SIGNAL(displayMessage(const std::string&)), this, SIGNAL(displayMessage(const std::string&)));

    connect(this, SIGNAL(itemDoubleClicked(QTreeWidgetItem*,int)), this, SLOT( doubleClick(QTreeWidgetItem *, int)));
    connect(this, SIGNAL(itemSelectionChanged()),  this, SLOT( addInPropertyWidget()));
    //connect(this, SIGNAL(itemClicked(QTreeWidgetItem*,int)),  this, SLOT( rightClick(QTreeWidgetItem *, int )));
    this->setContextMenuPolicy(Qt::CustomContextMenu);

    connect(this, SIGNAL(customContextMenuRequested(const QPoint&) ),  this, SLOT( rightClick(const QPoint& )));
    DialogAdd=NULL;
}

GraphModeler::~GraphModeler()
{
    delete historyManager;
    simulation::getSimulation()->unload(getRoot());
    //delete getRoot();
    graphRoot.reset();
    delete graphListener;
    if (DialogAdd) delete DialogAdd;
}

/*
void GraphModeler::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::RightButton)
    {
        QTreeWidgetItem* item;
        item = dynamic_cast<QTreeWidgetItem *>(this->childAt(event->pos()));

        if(item)
            rightClick(item, 0 );
    }
}

*/

Node::SPtr GraphModeler::addNode(Node::SPtr parent, Node::SPtr child, bool saveHistory)
{
    Node::SPtr lastRoot = getRoot();
    if (!child)
    {
        std::ostringstream oss;
        oss << Node::shortName(child.get()) << numNode++;
        child = Node::create(oss.str() );
        if (!parent)
            child->setName("Root");
    }

    if (parent != NULL)
    {
        parent->addChild(child);

        if (saveHistory)
        {
            GraphHistoryManager::Operation adding(child, GraphHistoryManager::Operation::ADD_Node);
            adding.info=std::string("Adding Node ") + child->getClassName();
            emit operationPerformed(adding);
        }
    }
    else
    {
        graphListener->addChild(NULL, child.get());
        //Set up the root
        this->topLevelItem(0)->setExpanded(true);

        if (saveHistory) historyManager->graphClean();
        graphRoot = child;
    }

    if (!parent && this->children().size() > 1)
    {
        deleteComponent(graphListener->items[lastRoot.get()], saveHistory);
    }
    return child;
}

BaseObject::SPtr GraphModeler::addComponent(Node::SPtr parent, const ClassEntry::SPtr entry, const std::string &templateName, bool saveHistory, bool displayWarning)
{
    BaseObject::SPtr object=NULL;
    if (!parent || !entry) return object;

    std::string templateUsed = templateName;

    xml::ObjectElement description("Default", entry->className.c_str() );

    if (!templateName.empty()) description.setAttribute("template", templateName.c_str());

    Creator::SPtr c;
    if (entry->creatorMap.size() <= 1)
        c=entry->creatorMap.begin()->second;
    else
    {
        if (templateName.empty())
        {
            if (entry->creatorMap.find(entry->defaultTemplate) == entry->creatorMap.end()) { std::cerr << "Error: No template specified" << std::endl; return object;}

            c=entry->creatorMap.find(entry->defaultTemplate)->second;
            templateUsed=entry->defaultTemplate;
        }
        else
        {
            c=entry->creatorMap.find(templateName)->second;
        }
    }
    if (c->canCreate(parent->getContext(), &description))
    {
        core::objectmodel::BaseObjectDescription arg;
        std::ostringstream oss;
        oss << c->shortName(&description) << numComponent++;
        arg.setName(oss.str());

        object = c->createInstance(parent->getContext(), &arg);

        if (saveHistory)
        {
            GraphHistoryManager::Operation adding(object, GraphHistoryManager::Operation::ADD_OBJECT);

            adding.info=std::string("Adding Object ") + object->getClassName();
            emit operationPerformed(adding);
        }
    }
    else
    {
        BaseObject* reference = parent->getContext()->getMechanicalState();

        if (displayWarning)
        {
            if (entry->className.find("Mapping") == std::string::npos)
            {
                if (!reference)
                {
                    //we accept the mappings as no initialization of the object has been done
                    const QString caption("Creation Impossible");
                    const QString warning=QString("No MechanicalState found in your Node ") + QString(parent->getName().c_str());
                    if ( QMessageBox::warning ( this, caption,warning, QMessageBox::Cancel | QMessageBox::Default | QMessageBox::Escape, QMessageBox::Ignore ) == QMessageBox::Cancel )
                        return object;
                }
                else
                {
                    //TODO: try to give the possibility to change the template by the correct detected one
                    const QString caption("Creation Impossible");
                    const QString warning=
                        QString("Your component won't be created: \n \t * <")
                        + QString(reference->getTemplateName().c_str()) + QString("> DOFs are used in the Node ") + QString(parent->getName().c_str()) + QString("\n\t * <")
                        + QString(templateUsed.c_str()) + QString("> is the type of your ") + QString(entry->className.c_str());
                    if ( QMessageBox::warning ( this, caption,warning, QMessageBox::Cancel | QMessageBox::Default | QMessageBox::Escape, QMessageBox::Ignore ) == QMessageBox::Cancel )
                        return object;
                }
            }
        }
        object = c->createInstance(parent->getContext(), NULL);
        GraphHistoryManager::Operation adding(object, GraphHistoryManager::Operation::ADD_OBJECT);
        adding.info=std::string("Adding Object ") + object->getClassName();
        emit operationPerformed(adding);
        // 	    parent->addObject(object);
    }
    return object;
}




void GraphModeler::dropEvent(QDropEvent* event)
{
    QString text;

    if (event->mimeData()->hasText())
        text = event->mimeData()->text();

    if (!text.isEmpty())
    {
        std::string filename(text.toStdString());
        std::string test = filename; test.resize(4);

        if (test == "file")
        {
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

            QString f(filename.c_str());
            emit(fileOpen(f));
            return;
        }
    }
    if (text == QString("ComponentCreation"))
    {
        BaseObject::SPtr newComponent = addComponent(getNode(event->pos()), lastSelectedComponent.second, lastSelectedComponent.first );
        if (newComponent)
        {
            QTreeWidgetItem *after = graphListener->items[newComponent.get()];
            std::ostringstream oss;
            oss << newComponent->getClassName() << " " << newComponent->getName();
            after->setText(0, QString(oss.str().c_str()));
            QTreeWidgetItem *item = itemAt(event->pos());
            if (getObject(item)) initItem(after, item);
        }
    }
    else
    {
        if (text == QString("Node"))
        {
            Node* node=getNode(event->pos());

            if (node)
            {
                Node::SPtr newNode=addNode(node);
                if (newNode)
                {
                    QTreeWidgetItem *after = graphListener->items[newNode.get()];
                    QTreeWidgetItem *item = itemAt(event->pos());
                    if (getObject(item)) initItem(after,item);
                }
            }
        }
    }
}


Base* GraphModeler::getComponent(QTreeWidgetItem *item) const
{
    if (!item) return NULL;
    std::map<core::objectmodel::Base*, QTreeWidgetItem* >::iterator it;
    for (it = graphListener->items.begin(); it != graphListener->items.end(); ++it)
    {
        if (it->second == item)
        {
            return it->first;
        }
    }
    return NULL;
}

BaseObject *GraphModeler::getObject(QTreeWidgetItem *item) const
{
    Base* component=getComponent(item);
    return dynamic_cast<BaseObject*>(component);
}


Node *GraphModeler::getNode(const QPoint &pos) const
{
    QTreeWidgetItem *item = itemAt(pos);
    if (!item) return NULL;
    return getNode(item);
}



Node *GraphModeler::getNode(QTreeWidgetItem *item) const
{
    if (!item) return NULL;
    sofa::core::objectmodel::Base *component=getComponent(item);

    if (Node *node=dynamic_cast<Node*>(component)) return node;
    else
    {
        item = item->parent();
        component=getComponent(item);
        if (Node *node=dynamic_cast<Node*>(component)) return node;
        return NULL;
    }
}

QTreeWidgetItem *GraphModeler::getItem(Base *component) const
{
    if (!component) return NULL;
    std::map<core::objectmodel::Base*, QTreeWidgetItem* >::iterator it;
    for (it = graphListener->items.begin(); it != graphListener->items.end(); ++it)
    {
        if (it->first == component)
        {
            return it->second;
        }
    }
    return NULL;
}

void GraphModeler::openModifyObject()
{
    helper::vector<QTreeWidgetItem*> selection;
    getSelectedItems(selection);
    for (unsigned int i=0; i<selection.size(); ++i)
        openModifyObject(selection[i]);
}

void GraphModeler::openModifyObject(QTreeWidgetItem *item)
{
    if (!item) return;

    Base* object = graphListener->findObject(item);
    BaseData* data = graphListener->findData(item);
    if( data == NULL && object == NULL)
    {
        assert(0);
    }

    ModifyObjectFlags dialogFlags = ModifyObjectFlags();
    dialogFlags.setFlagsForModeler();

    if (data)       //user clicked on a data
    {
        current_Id_modifyDialog = data;
    }
    else
    {
        if(object)
        {
            current_Id_modifyDialog = object;
            if (object->toConfigurationSetting()) dialogFlags.HIDE_FLAG=true;
        }
        else
        {
            assert(0);
        }
    }

    //Unicity and identification of the windows

    std::map< void*, QDialog* >::iterator testWindow =  map_modifyObjectWindow.find( current_Id_modifyDialog);
    if ( testWindow != map_modifyObjectWindow.end())
    {
        //Object already being modified: no need to open a new window
        (*testWindow).second->raise();
        return;
    }


    ModifyObject *dialogModify = new ModifyObject( current_Id_modifyDialog,item,this,dialogFlags,item->text(0).toStdString().c_str());

    connect(dialogModify, SIGNAL(beginObjectModification(sofa::core::objectmodel::Base*)), historyManager, SLOT(beginModification(sofa::core::objectmodel::Base*)));
    connect(dialogModify, SIGNAL(endObjectModification(sofa::core::objectmodel::Base*)),   historyManager, SLOT(endModification(sofa::core::objectmodel::Base*)));

    if(data)
    {
        dialogModify->createDialog(data);
    }
    if(object)
    {
        dialogModify->createDialog(object);
    }

    if(object && propertyWidget)
        propertyWidget->addComponent(object->getName().c_str(), object, item);

    map_modifyObjectWindow.insert( std::make_pair(current_Id_modifyDialog, dialogModify));
    //If the item clicked is a node, we add it to the list of the element modified

    map_modifyDialogOpened.insert ( std::make_pair ( current_Id_modifyDialog, item ) );
    connect ( dialogModify, SIGNAL( dialogClosed(void *) ) , this, SLOT( modifyUnlock(void *)));
    dialogModify->show();
    dialogModify->raise();
}

void GraphModeler::addInPropertyWidget()
{
    helper::vector<QTreeWidgetItem*> selection;
    getSelectedItems(selection);

    bool clear = true;
    for (unsigned int i=0; i<selection.size(); ++i)
    {
        addInPropertyWidget(selection[i], clear);
        clear = false;
    }
}

void GraphModeler::addInPropertyWidget(QTreeWidgetItem *item, bool clear)
{
    if(!item)
        return;

    Base* object = graphListener->findObject(item);
    if(object == NULL)
        return;

	if(propertyWidget)
		propertyWidget->addComponent(object->getName().c_str(), object, item, clear);
}

void GraphModeler::doubleClick(QTreeWidgetItem *item, int /* column */)
{
    if (!item) return;
    item->setExpanded(!item->isExpanded());
    openModifyObject(item);

}

void GraphModeler::leftClick(QTreeWidgetItem *item, const QPoint & /*point*/, int /*index*/)
{
    if (!item) return;
    item->setExpanded(!item->isExpanded() );
    addInPropertyWidget(item);
}

void GraphModeler::rightClick(const QPoint& p /*, int  index */)
{
    QTreeWidgetItem* item = this->itemAt(p);
    std::cout << p.x() << " " << p.y()  << std::endl;
    if (!item) return;

    bool isNode=true;
    helper::vector<QTreeWidgetItem*> selection; getSelectedItems(selection);
    bool isSingleSelection= (selection.size() == 1);
    for (unsigned int i=0; i<selection.size(); ++i)
    {
        if (getObject(item)!=NULL)
        {
            isNode = false;
            break;
        }
    }

    bool isLoader = false;
    if (dynamic_cast<sofa::core::loader::BaseLoader*>(getComponent(item)) != NULL)
        isLoader = true;

    QMenu *contextMenu = new QMenu ( this );
    contextMenu->setObjectName("ContextMenu");

    if (isNode)
    {
        contextMenu->addAction("Collapse", this, SLOT( collapseNode()));
        contextMenu->addAction("Expand"  , this, SLOT( expandNode()));
        if (isSingleSelection)
        {
            contextMenu->addSeparator();
            contextMenu->addAction("Load"  , this, SLOT( loadNode()));
            preset->setTitle(QString(tr( "Preset")));
            contextMenu->addMenu(preset);
        }
    }
    contextMenu->addAction("Save"  , this, SLOT( saveComponents()));

    contextMenu->addAction("Delete"  , this, SLOT( deleteComponent()));

    contextMenu->addAction("Modify"  , this, SLOT( openModifyObject()));
    contextMenu->addAction("GlobalModification"  , this, SLOT( globalModification()));

    if (!isNode && !isLoader)
        contextMenu->addAction("Link"  , this, SLOT( linkComponent()));

    contextMenu->exec(this->mapToGlobal(p));

}


void GraphModeler::collapseNode()
{
    helper::vector<QTreeWidgetItem*> selection;
    getSelectedItems(selection);
    for (unsigned int i=0; i<selection.size(); ++i)
        collapseNode(selection[i]);
}

void GraphModeler::collapseNode(QTreeWidgetItem* item)
{
    if (!item) return;

    for(int i=0; i<item->childCount();i++)
    {
        QTreeWidgetItem* child = item->child(i);
        child->setExpanded( false );
    }
    item->setExpanded( true );
}

void GraphModeler::expandNode()
{
    helper::vector<QTreeWidgetItem*> selection;
    getSelectedItems(selection);
    for (unsigned int i=0; i<selection.size(); ++i)
        expandNode(selection[i]);
}

void GraphModeler::expandNode(QTreeWidgetItem* item)
{
    if (!item) return;

    item->setExpanded( true );
    if ( item != NULL )
    {
        for(int i=0; i<item->childCount();i++)
        {
            QTreeWidgetItem* child = item->child(i);
            child->setExpanded( true );
            expandNode(item);
        }
    }
}

Node::SPtr GraphModeler::loadNode()
{
    return loadNode(currentItem());
}

Node::SPtr GraphModeler::loadNode(QTreeWidgetItem* item, std::string filename, bool saveHistory)
{
    Node::SPtr node;
    if (!item) node=NULL;
    else node=getNode(item);

    if (filename.empty())
    {
        QString s = getOpenFileName ( this, NULL,"Scenes (*.scn *.xml *.simu *.pscn)", "open file dialog",  "Choose a file to open" );
        if (s.length() >0)
        {
            filename = s.toStdString();
        }
        else return NULL;
    }
    return loadNode(node,filename, saveHistory);
}


void GraphModeler::globalModification()
{
    //Get all the components which can be modified
    helper::vector< QTreeWidgetItem* > selection;
    getSelectedItems(selection);

    helper::vector< QTreeWidgetItem* > hierarchySelection;
    for (size_t i=0; i<selection.size(); ++i) getComponentHierarchy(selection[i], hierarchySelection);

    helper::vector< Base* > allComponentsSelected;
    for (size_t i=0; i<hierarchySelection.size(); ++i) allComponentsSelected.push_back(getComponent(hierarchySelection[i]));

    sofa::gui::qt::GlobalModification *window=new sofa::gui::qt::GlobalModification(allComponentsSelected, historyManager);

    connect(window, SIGNAL(displayMessage(const std::string&)), this, SIGNAL(displayMessage(const std::string&)));

    window->show();
}

void GraphModeler::linkComponent()
{
    // get the selected component
    helper::vector< QTreeWidgetItem* > selection;
    getSelectedItems(selection);

    // a component must be selected
    if(selection.empty())
        return;

    QTreeWidgetItem *fromItem = *selection.begin();
    BaseObject *fromObject = getObject(fromItem);

    // the object must exist
    if(!fromObject)
        return;

    // the object must not be a loader
    if(dynamic_cast<sofa::core::loader::BaseLoader*>(fromObject))
        return;

    // store the partial component hierarchy i.e we store the parent Nodes only
    std::vector<QTreeWidgetItem*> items;
    for(QTreeWidgetItem *item = fromItem->parent(); item != NULL; item = item->parent())
        items.push_back(item);

    // create and show the LinkComponent dialog box
    sofa::gui::qt::LinkComponent *window=new sofa::gui::qt::LinkComponent(this, items, fromItem);

    if(window->loaderNumber() == 0)
    {
        window->close();

        QMessageBox* messageBox = new QMessageBox(QMessageBox::Warning, "No loaders", "This tree branch does not contain any loader to link with.");
        messageBox->show();

        return;
    }

    connect(window, SIGNAL(displayMessage(const std::string&)), this, SIGNAL(displayMessage(const std::string&)));

    window->show();
}


Node::SPtr GraphModeler::buildNodeFromBaseElement(Node::SPtr node,xml::BaseElement *elem, bool saveHistory)
{
    const bool displayWarning=true;
    Node::SPtr newNode = Node::create("");
    //Configure the new Node
    configureElement(newNode.get(), elem);

    if (newNode->getName() == "Group")
    {
        //We can't use the parent node, as it is null
        if (!node) return NULL;
        //delete newNode;
        newNode = node;
    }
    else
    {
//          if (node)
        {
            //Add as a child
            addNode(node,newNode,saveHistory);
        }
    }


    typedef xml::BaseElement::child_iterator<> elem_iterator;
    for (elem_iterator it=elem->begin(); it != elem->end(); ++it)
    {
        if (std::string(it->getClass()) == std::string("Node"))
        {
            buildNodeFromBaseElement(newNode, it,true);
        }
        else
        {
            const ComponentLibrary *component = sofaLibrary->getComponent(it->getType());
            //Configure the new Component
            const std::string templateAttribute("template");
            std::string templatename;
            templatename = it->getAttribute(templateAttribute, "");


            const ClassEntry::SPtr info = component->getEntry();
            BaseObject::SPtr newComponent=addComponent(newNode, info, templatename, saveHistory,displayWarning);
            if (!newComponent) continue;
            configureElement(newComponent.get(), it);
            QTreeWidgetItem* itemGraph = graphListener->items[newComponent.get()];

            std::string name=itemGraph->text(0).toStdString();
            std::string::size_type pos = name.find(' ');
            if (pos != std::string::npos)  name.resize(pos);
            name += "  ";

            name+=newComponent->getName();
            itemGraph->setText(0,name.c_str());
        }
    }
    newNode->clearWarnings();

    return newNode;
}

void GraphModeler::configureElement(Base* b, xml::BaseElement *elem)
{
    //Init the Attributes of the object
    typedef xml::BaseElement::child_iterator<xml::AttributeElement> attr_iterator;
    typedef xml::BaseElement::child_iterator<xml::DataElement> data_iterator;
    for (attr_iterator itAttribute=elem->begin<xml::AttributeElement>(); itAttribute != elem->end<xml::AttributeElement>(); ++itAttribute)
    {
        for (data_iterator itData=itAttribute->begin<xml::DataElement>(); itData != itAttribute->end<xml::DataElement>(); ++itData)  itData->initNode();
        const std::string nameAttribute = itAttribute->getAttribute("type","");
        const std::string valueAttribute = itAttribute->getValue();
        elem->setAttribute(nameAttribute, valueAttribute.c_str());
    }


    const sofa::core::objectmodel::Base::VecData& vecDatas=b->getDataFields();
    for (unsigned int i=0; i<vecDatas.size(); ++i)
    {
        std::string result = elem->getAttribute(vecDatas[i]->getName(), "");

        if (!result.empty())
        {
            if (result[0] == '@')
                vecDatas[i]->setParent(result);
            else
                vecDatas[i]->read(result);
        }
    }
}

Node::SPtr GraphModeler::loadNode(Node::SPtr node, std::string path, bool saveHistory)
{
    xml::BaseElement* newXML=NULL;

    newXML = xml::loadFromFile (path.c_str() );
    if (newXML == NULL) return NULL;

    //-----------------------------------------------------------------
    //Add the content of a xml file
    Node::SPtr newNode = buildNodeFromBaseElement(node, newXML, saveHistory);
    //-----------------------------------------------------------------

    return newNode;
}

void GraphModeler::loadPreset(std::string presetName)
{
    xml::BaseElement* newXML = xml::loadFromFile (presetName.c_str() );
    if (newXML == NULL) return;

    xml::BaseElement::child_iterator<> it(newXML->begin());
    bool elementPresent[3]= {false,false,false};
    for (; it!=newXML->end(); ++it)
    {
        if (it->getName() == std::string("VisualNode") || it->getType() == std::string("OglModel")) elementPresent[1] = true;
        else if (it->getName() == std::string("CollisionNode")) elementPresent[2] = true;
        else if (it->getType() == std::string("MeshObjLoader") || it->getType() == std::string("SparseGrid")) elementPresent[0] = true;
    }

    if (!DialogAdd)
    {
        DialogAdd = new AddPreset(this);
        DialogAdd->setPath(sofa::helper::system::DataRepository.getFirstPath());
    }

    DialogAdd->setRelativePath(sofa::helper::system::SetDirectory::GetParentDir(filenameXML.c_str()));

    DialogAdd->setElementPresent(elementPresent);
    DialogAdd->setPresetFile(presetName);
    Node *node=getNode(currentItem());
    DialogAdd->setParentNode(node);

    DialogAdd->show();
    DialogAdd->raise();
}


void GraphModeler::loadPreset(Node *parent, std::string presetFile,
        std::string *filenames,
        std::string translation,
        std::string rotation,
        std::string scale)
{


    xml::BaseElement* newXML=NULL;

    newXML = xml::loadFromFile (presetFile.c_str() );
    if (newXML == NULL) return;

    //bool collisionNodeFound=false;
    //xml::BaseElement *meshMecha=NULL;
    xml::BaseElement::child_iterator<> it(newXML->begin());
    for (; it!=newXML->end(); ++it)
    {

        if (it->getType() == std::string("MechanicalObject"))
        {
            updatePresetNode(*it, std::string(), translation, rotation, scale);
        }
        if (it->getType() == std::string("MeshObjLoader") || it->getType() == std::string("SparseGrid"))
        {
            updatePresetNode(*it, filenames[0], translation, rotation, scale);
            //meshMecha = it;
        }
        if (it->getType() == std::string("OglModel"))
        {
            updatePresetNode(*it, filenames[1], translation, rotation, scale);
        }

        if (it->getName() == std::string("VisualNode"))
        {
            xml::BaseElement* visualXML = it;

            xml::BaseElement::child_iterator<> it_visual(visualXML->begin());
            for (; it_visual!=visualXML->end(); ++it_visual)
            {
                if (it_visual->getType() == std::string("OglModel"))
                {
                    updatePresetNode(*it_visual, filenames[1], translation, rotation, scale);
                }
            }
        }
        if (it->getName() == std::string("CollisionNode"))
        {
            //collisionNodeFound=true;
            xml::BaseElement* collisionXML = it;

            xml::BaseElement::child_iterator<> it_collision(collisionXML->begin());
            for (; it_collision!=collisionXML->end(); ++it_collision)
            {

                if (it_collision->getType() == std::string("MechanicalObject"))
                {
                    updatePresetNode(*it_collision, std::string(), translation, rotation, scale);
                }
                if (it_collision->getType() == std::string("MeshObjLoader"))
                {
                    updatePresetNode(*it_collision, filenames[2], translation, rotation, scale);
                }
            }
        }
    }



    if (!newXML->init()) std::cerr<< "Objects initialization failed.\n";
    Node *presetNode = dynamic_cast<Node*> ( newXML->getObject() );
    if (presetNode) addNode(parent,presetNode);
}

void GraphModeler::updatePresetNode(xml::BaseElement &elem, std::string meshFile, std::string translation, std::string rotation, std::string scale)
{
    if (elem.presenceAttribute(std::string("filename")))     elem.setAttribute(std::string("filename"),     meshFile.c_str());
    if (elem.presenceAttribute(std::string("fileMesh")))     elem.setAttribute(std::string("fileMesh"),     meshFile.c_str());
    if (elem.presenceAttribute(std::string("fileTopology"))) elem.setAttribute(std::string("fileTopology"), meshFile.c_str());

    if (elem.presenceAttribute(std::string("translation")))  elem.setAttribute(std::string("translation"), translation.c_str());
    if (elem.presenceAttribute(std::string("rotation")))     elem.setAttribute(std::string("rotation"),    rotation.c_str());
    if (elem.presenceAttribute(std::string("scale3d")))      elem.setAttribute(std::string("scale3d"),     scale.c_str());
}

bool GraphModeler::getSaveFilename(std::string &filename)
{
    QString s = sofa::gui::qt::getSaveFileName ( this, NULL, "Scenes (*.scn *.xml)", "save file dialog", "Choose where the scene will be saved" );
    if ( s.length() >0 )
    {
        std::string extension=sofa::helper::system::SetDirectory::GetExtension(s.toStdString().c_str());
        if (extension.empty()) s+=QString(".scn");
        filename = s.toStdString();
        return true;
    }
    return false;
}

void GraphModeler::save(const std::string &filename)
{
    Node *node = getNode(this->topLevelItem(0));
    simulation::getSimulation()->exportXML(node, filename.c_str());
    emit graphClean();
}

void GraphModeler::saveComponents()
{
    helper::vector<QTreeWidgetItem*> selection;
    getSelectedItems(selection);
    if (selection.empty()) return;
    std::string filename;
    if ( getSaveFilename(filename) )  saveComponents(selection, filename);

}

void GraphModeler::saveComponents(helper::vector<QTreeWidgetItem*> items, const std::string &file)
{
    std::ofstream out(file.c_str());
    simulation::XMLPrintVisitor print(sofa::core::ExecParams::defaultInstance() /* PARAMS FIRST */, out);
    print.setLevel(1);
    out << "<Node name=\"Group\">\n";
    for (unsigned int i=0; i<items.size(); ++i)
    {
        if (BaseObject* object=getObject(items[i]))
            print.processBaseObject(object);
        else if (Node *node=getNode(items[i]))
            print.execute(node);
    }
    out << "</Node>\n";
}

void GraphModeler::clearGraph(bool saveHistory)
{
    deleteComponent(this->topLevelItem(0), saveHistory);
}

void GraphModeler::deleteComponent(QTreeWidgetItem* item, bool saveHistory)
{
    if (!item) return;

    Node *parent = getNode(item->parent());
    bool isNode   = getObject(item)==NULL;
    if (!isNode && isObjectErasable(getObject(item)))
    {
        BaseObject* object = getObject(item);
        if (saveHistory)
        {
            GraphHistoryManager::Operation removal(getObject(item),GraphHistoryManager::Operation::DELETE_OBJECT);
            removal.parent=getNode(item);
            removal.above=getComponentAbove(item);

            removal.info=std::string("Removing Object ") + object->getClassName();
            emit operationPerformed(removal);

        }
        getNode(item)->removeObject(getObject(item));
    }
    else
    {
        if (!isNodeErasable(getNode(item))) return;
        Node *node = getNode(item);

        if (saveHistory)
        {
            GraphHistoryManager::Operation removal(node,GraphHistoryManager::Operation::DELETE_Node);
            removal.parent = parent;
            removal.above=getComponentAbove(item);
            removal.info=std::string("Removing Node ") + node->getClassName();
            emit operationPerformed(removal);
        }
        if (!parent)
            graphListener->removeChild(parent, node);
        else
            parent->removeChild((Node*)node);
        if (!parent && this->children().size() == 0) addNode(NULL);
    }

}

void GraphModeler::changeComponentDataValue(const std::string &name, const std::string &value, Base* component) const
{
    if (!component) return;
    historyManager->beginModification(component);
    const std::vector< BaseData* > &data=component->findGlobalField(name);
    if (data.empty()) //this data is not present in the current component
        return;

    std::string v(value);
    for (unsigned int i=0; i<data.size(); ++i) data[i]->read(v);

    historyManager->endModification(component);

}


Base *GraphModeler::getComponentAbove(QTreeWidgetItem *item)
{
    if (!item) return NULL;
    QTreeWidgetItem* itemAbove = this->itemAbove(item);
    while (itemAbove && itemAbove->parent() != item->parent() )
    {
        itemAbove = itemAbove->parent();
    }
    Base *result=getObject(itemAbove);
    if (!result) result=getNode(itemAbove);
    return result;
}

void GraphModeler::deleteComponent()
{
    helper::vector<QTreeWidgetItem*> selection;
    getSelectedItems(selection);
    for (unsigned int i=0; i<selection.size(); ++i)
        deleteComponent(selection[i]);
}

void GraphModeler::modifyUnlock ( void *Id )
{
    map_modifyDialogOpened.erase( Id );
    map_modifyObjectWindow.erase( Id );
}


void GraphModeler::keyPressEvent ( QKeyEvent * e )
{
    switch ( e->key() )
    {
    case Qt::Key_Delete :
    {
        deleteComponent();
        break;
    }
    case Qt::Key_Return :
    case Qt::Key_Enter :
    {
        openModifyObject();
        break;
    }
    default:
    {
        QTreeWidget::keyPressEvent(e);
        break;
    }
    }
}
/*****************************************************************************************************************/
// Test if a node can be erased in the graph : the condition is that none of its children has a menu modify opened
bool GraphModeler::isNodeErasable ( BaseNode* node)
{
    QTreeWidgetItem* item = graphListener->items[node];
    if(item == NULL)
    {
        return false;
    }
    // check if there is already a dialog opened for that item in the graph
    std::map< void*, QTreeWidgetItem*>::iterator it;
    for (it = map_modifyDialogOpened.begin(); it != map_modifyDialogOpened.end(); ++it)
    {
        if (it->second == item) return false;
    }

    //check the item childs

    for(int i=0 ; i<item->childCount() ; i++)
    {
         QTreeWidgetItem *child = item->child(i);
        for( it = map_modifyDialogOpened.begin(); it != map_modifyDialogOpened.end(); ++it)
        {
            if( it->second == child) return false;
        }
    }

    return true;

}

bool GraphModeler::isObjectErasable ( core::objectmodel::Base* element )
{
    QTreeWidgetItem* item = graphListener->items[element];
    std::map< void*, QTreeWidgetItem*>::iterator it;
    for (it = map_modifyDialogOpened.begin(); it != map_modifyDialogOpened.end(); ++it)
    {
        if (it->second == item) return false;
    }

    return true;
}

void GraphModeler::initItem(QTreeWidgetItem *item, QTreeWidgetItem *above)
{
    moveItem(item, above);
    moveItem(above,item);
}

void GraphModeler::moveItem(QTreeWidgetItem *item, QTreeWidgetItem *above)
{
    if (item)
    {
        if (above)
        {
            //item->moveItem(above);
            item->parent()->removeChild(item);
            QTreeWidgetItem *parent = above->parent();
            parent->insertChild(parent->indexOfChild(above), item );


            //Move the object in the Sofa Node.
            if (above && getObject(item))
            {
                Node *n=getNode(item);
                Node::ObjectIterator A=n->object.end();
                Node::ObjectIterator B=n->object.end();
                BaseObject* objectA=getObject(above);
                BaseObject* objectB=getObject(item);
                bool inversion=false;
                //Find the two objects in the Sequence of the Node
                for (Node::ObjectIterator it=n->object.begin(); it!=n->object.end(); ++it)
                {
                    if( *it == objectA)
                    {
                        A=it;
                        if (B!=n->object.end()) inversion=true;
                    }
                    else if ( *it == objectB) B=it;
                }
                //One has not been found: should not happen
                if (A==n->object.end() || B==n->object.end()) return;

                //Invert the elements
                Node::ObjectIterator it;
                if (inversion) n->object.swap(A,B);
                else
                {
                    for (it=B; it!=A+1; --it) n->object.swap(it,it-1);
                }
            }
        }
        else
        {
            //Object
            if (getObject(item))
            {
                QTreeWidgetItem *nodeQt = graphListener->items[getNode(item)];
                QTreeWidgetItem *firstComp=nodeQt->child(0);
                if (firstComp != item) initItem(item, firstComp);
            }
            //Node
            else
            {
                QTreeWidgetItem *nodeQt = graphListener->items[getNode(item->parent())];
                QTreeWidgetItem *firstComp=nodeQt->child(0);
                if (firstComp != item) initItem(item, firstComp);
            }
        }
    }
}


/// Drag Management
void GraphModeler::dragEnterEvent( QDragEnterEvent* event)
{
    event->accept(event->answerRect());
}

/// Drag Management
void GraphModeler::dragMoveEvent( QDragMoveEvent* event)
{
    QString text;

    if (event->mimeData()->hasText())
        text = event->mimeData()->text();

    if (!text.isEmpty())
    {
        std::string filename(text.toStdString());
        std::string test = filename; test.resize(4);
        if (test == "file") {event->accept(event->answerRect());}
    }
    else
    {
        if ( getNode(event->pos()))
            event->accept(event->answerRect());
        else
            event->ignore(event->answerRect());
    }
}

void GraphModeler::closeDialogs()
{
    std::map< void*, QDialog* >::iterator it;    ;
    for (it=map_modifyObjectWindow.begin();
            it!=map_modifyObjectWindow.end();
            ++it)
    {
        delete it->second;
    }
    map_modifyObjectWindow.clear();
}

/*****************************************************************************************************************/
//History of operations management
//TODO: not use the factory to create the elements!
bool GraphModeler::cut(std::string path)
{
    bool resultCopy=copy(path);
    if (resultCopy)
    {
        deleteComponent();
        return true;
    }
    return false;
}

bool GraphModeler::copy(std::string path)
{
    helper::vector< QTreeWidgetItem*> items; getSelectedItems(items);

    if (!items.empty())
    {
        saveComponents(items,path);
        return true;
    }
    return false;
}

bool GraphModeler::paste(std::string path)
{
    helper::vector< QTreeWidgetItem*> items;
    getSelectedItems(items);
    if (!items.empty())
    {
        //Get the last item of the node: the new items will be inserted AFTER this last item.
        QTreeWidgetItem *last=items.front();
        //while(last->nextSibling()) last=last->nextSibling();
        last = last->child(last->childCount()-1);

        Node *node = getNode(items.front());
        //Load the paste buffer
        loadNode(node, path);
        QTreeWidgetItem *pasteItem=items.front();

        //Find all the QListViewItem inserted
        helper::vector< QTreeWidgetItem* > insertedItems;
//        QTreeWidgetItem *insertedItem=last;
        for(int i=0 ; i<last->parent()->childCount() ; i++)
        {
            QTreeWidgetItem *insertedItem = last->parent()->child(i);
            if(insertedItem != last)
                insertedItems.push_back(insertedItem);
        }

        //Initialize their position in the node
        for (unsigned int i=0; i<insertedItems.size(); ++i) initItem(insertedItems[i], pasteItem);
    }

    return !items.empty();
}
}
}
}
