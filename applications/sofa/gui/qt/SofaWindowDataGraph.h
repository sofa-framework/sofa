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
#ifndef SOFA_WINDOW_DATAGRAPH
#define SOFA_WINDOW_DATAGRAPH

#include <QDialog>
#include <sofa/simulation/Node.h>

#define NODE_EDITOR_SHARED

namespace QtNodes
{
    class FlowScene;
    class FlowView;
}


namespace sofa
{

namespace gui
{

namespace qt
{
 /**
 * This Class provide an interface with the library QtNodes to display Data Graph connection inside a QDialog.
 * It will take a SOFA simulation scene and create Graph nodes for each Sofa component and display connections between Data.
 */   
class SofaWindowDataGraph : public QDialog
{
    Q_OBJECT
public:
    /// Default constructor of the Widget, given a QWidget as parent and a pointer to the current simulation scene.
    SofaWindowDataGraph(QWidget *parent, sofa::simulation::Node* scene);

    ~SofaWindowDataGraph();

    /// Method to be called when graph need to be recomputed (like reloading scene). Take a pointer to the root node of the scene.
    void resetNodeGraph(sofa::simulation::Node* scene);
   
protected:
    /// Internal method to parse all Sofa component inside a Sofa simulation Node. Will call @sa addSimulationObject for each compoenent then will iterate on children nodes.
    void parseSimulationNode(sofa::simulation::Node* node, int posX = 0);

    /// Internal method to create a Node for this sofa BaseObject.
    size_t addSimulationObject(sofa::core::objectmodel::BaseObject* bObject);

    /// Internal method to create all connection between component on the graph.
    void connectNodeData();

    /// Internal method to clear the graph structures
    void clearNodeData();       

protected:
    /// Pointer to the graphScene used to store nodes.
    QtNodes::FlowScene* m_graphScene;
    /// Pointer to the view of the graph.
    QtNodes::FlowView* m_graphView;

    /// Point to the root node of the current simulation.
    sofa::simulation::Node* m_rootNode;
    
    /// List of component name not to be display in the graph.
    std::vector<std::string> m_exceptions;

    SReal m_scaleX; ///< Scale paramater to apply between nodes for display in abscissa.    
    SReal m_scaleY; ///< Scale paramater to apply between nodes for display in ordinate.

    int m_posX; ///< Increment position counter on abscissa for Node display.
    int m_posY; ///< Increment position counter on ordinate for Node display.

    bool debugNodeGraph; ///< parameter to activate graph logs. False by default.
};

} // namespace qt

} // namespace gui

} // namespace sofa

#endif // SOFA_WINDOW_DATAGRAPH
