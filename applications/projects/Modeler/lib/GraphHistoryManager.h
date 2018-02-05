/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_GRAPHHISTORYMANAGER_H
#define SOFA_GRAPHHISTORYMANAGER_H


#include <sofa/core/objectmodel/Base.h>
#include <sofa/simulation/Node.h>

#include <QObject>
#include <vector>

namespace sofa
{

namespace gui
{

namespace qt
{

using sofa::core::objectmodel::Base;
using sofa::simulation::Node;

class GraphModeler;

class GraphHistoryManager: public QObject
{
    Q_OBJECT
public:
    //-----------------------------------------------------------------------------//
    //Historic of actions: management of the undo/redo actions
    ///Basic class storing information about the operation done
    class Operation
    {
    public:
        Operation() {}
        enum op {DELETE_OBJECT,DELETE_Node, ADD_OBJECT,ADD_Node, NODE_MODIFICATION, COMPONENT_MODIFICATION};
        Operation(Base::SPtr sofaComponent_,  op ID_): sofaComponent(sofaComponent_), above(NULL), ID(ID_)
        {}

        Base::SPtr sofaComponent;
        Node::SPtr parent;
        Base::SPtr above;
        op ID;
        std::string info;
    };

    GraphHistoryManager(GraphModeler *);
    ~GraphHistoryManager();

    bool isUndoEnabled() const {return !historyOperation.empty();}
    bool isRedoEnabled() const {return !historyUndoOperation.empty();}

public slots:
    void operationPerformed(GraphHistoryManager::Operation&);
    void undo();
    void redo();
    void graphClean();
    void beginModification(sofa::core::objectmodel::Base* object);
    void endModification(sofa::core::objectmodel::Base* object);
signals:
    void graphModified(bool);
    void undoEnabled(bool);
    void redoEnabled(bool);
    void displayMessage(const std::string&);
protected:
    void clearHistoryUndo();
    void clearHistory();

    void undoOperation(Operation &);
    std::string componentState(Base *base) const;
    std::string setComponentState(Base *base, const std::string &datasStr);

    std::vector< Operation > historyOperation;
    std::vector< Operation > historyUndoOperation;

    std::map<Base*, std::string> componentPriorModificationState;

    GraphModeler *graph;
};

}
}
}

#endif
