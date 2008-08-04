/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "TopologicalChangeManager.h"

#include <sofa/component/collision/TriangleModel.h>

#include <sofa/component/MechanicalObject.h>
#include <sofa/simulation/tree/GNode.h>

#include <sofa/core/componentmodel/topology/TopologicalMapping.h>

#include <sofa/component/topology/EdgeSetTopologyContainer.h>
#include <sofa/component/topology/TriangleSetTopologyContainer.h>
#include <sofa/component/topology/TriangleSetTopologyModifier.h>
#include <sofa/component/topology/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/QuadSetTopologyContainer.h>
#include <sofa/component/topology/HexahedronSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::component::topology;

TopologicalChangeManager::TopologicalChangeManager()
{
}

TopologicalChangeManager::~TopologicalChangeManager()
{
}

void TopologicalChangeManager::removeItemsFromTriangleModel(sofa::core::CollisionElementIterator elem2) const
{
    TriangleModel* my_triangle_model = (dynamic_cast<TriangleModel*>(elem2.getCollisionModel()));
    if (my_triangle_model)
    {
        sofa::core::componentmodel::topology::BaseMeshTopology* topo_curr;
        topo_curr = elem2.getCollisionModel()->getContext()->getMeshTopology();

        unsigned int ind_curr = elem2.getIndex();

        simulation::tree::GNode *node_curr = dynamic_cast<simulation::tree::GNode*>(topo_curr->getContext());

        std::set< unsigned int > items;
        items.insert(ind_curr);

        bool is_topoMap = true;

        while(is_topoMap)
        {

            is_topoMap = false;
            for(simulation::tree::GNode::ObjectIterator it = node_curr->object.begin(); it != node_curr->object.end(); ++it)
            {

                sofa::core::componentmodel::topology::TopologicalMapping *topoMap = dynamic_cast<sofa::core::componentmodel::topology::TopologicalMapping *>(*it);
                if(topoMap != NULL)
                {

                    is_topoMap = true;
                    //unsigned int ind_glob = topoMap->getGlobIndex(ind_curr);
                    //ind_curr = topoMap->getFromIndex(ind_glob);
                    std::set< unsigned int > loc_items = items;
                    items.clear();
                    for (std::set< unsigned int >::const_iterator it=loc_items.begin(); it != loc_items.end(); ++it)
                    {
                        unsigned int ind_glob = topoMap->getGlobIndex(*it);
                        unsigned int ind = topoMap->getFromIndex(ind_glob);
                        //std::cout << *it << " -> "<<ind_glob << " -> "<<ind<<std::endl;
                        items.insert(ind);
                    }

                    topo_curr = topoMap->getFrom()->getContext()->getMeshTopology();
                    node_curr = dynamic_cast<simulation::tree::GNode*>(topo_curr->getContext());

                    break;
                }
            }
        }
        sofa::helper::vector<unsigned int> vitems;
        vitems.reserve(items.size());
        vitems.insert(vitems.end(), items.rbegin(), items.rend());

        sofa::core::componentmodel::topology::TopologyModifier* topoMod;
        topo_curr->getContext()->get(topoMod);

        topoMod->removeItems(vitems);

        topoMod->notifyEndingEvent();


        topoMod->propagateTopologicalChanges();


        /*
        //// For APPLICATION 1 : renumber the mesh vertices with the optimal vertex permutation according to the Reverse CuthillMckee algorithm

        sofa::helper::vector<int> init_inverse_permutation;
        sofa::helper::vector<int>& inverse_permutation = init_inverse_permutation;

        sofa::component::topology::EdgeSetTopologyModifier* edgeMod;
        topo_curr->getContext()->get(edgeMod);

        edgeMod->resortCuthillMckee(inverse_permutation);

        //std::cout << "inverse_permutation : " << std::endl;
        sofa::helper::vector<int> permutation;
        permutation.resize(inverse_permutation.size());
        for (unsigned int i=0; i<inverse_permutation.size(); ++i){
        	permutation[inverse_permutation[i]]=i;
        	//std::cout << i << " -> " << inverse_permutation[i] << std::endl;
        }

        //std::cout << "permutation : " << std::endl;
        //for (unsigned int i=0; i<permutation.size(); ++i){
        	//std::cout << i << " -> " << permutation[i] << std::endl;
        //}

        sofa::core::componentmodel::topology::TopologyModifier* topoMod;
        topo_curr->getContext()->get(topoMod);
        topoMod->renumberPoints((const sofa::helper::vector<unsigned int> &) inverse_permutation, (const sofa::helper::vector<unsigned int> &) permutation);

        */
    }
}

// Handle Removing of topological element (from any type of topology)
void TopologicalChangeManager::removeItemsFromCollisionModel(sofa::core::CollisionElementIterator elem2) const
{
    if(dynamic_cast<TriangleModel*>(elem2.getCollisionModel())!= NULL)
    {
        removeItemsFromTriangleModel(elem2);
    }
}


// Intermediate method to handle cutting
bool TopologicalChangeManager::incisionTriangleSetTopology(sofa::core::componentmodel::topology::BaseMeshTopology* _topology)
{
    const Vec<3,double>& a= (const Vec<3,double>) incision.a_init;
    const Vec<3,double>& b= (const Vec<3,double>) incision.b_init;

    const unsigned int &ind_ta = (const unsigned int) incision.ind_ta_init;
    const unsigned int &ind_tb = (const unsigned int) incision.ind_tb_init;

    unsigned int& a_last = incision.a_last_init;
    sofa::helper::vector< unsigned int >& a_p12_last = incision.a_p12_last_init;
    sofa::helper::vector< unsigned int >& a_i123_last = incision.a_i123_last_init;

    unsigned int& b_last = incision.b_last_init;
    sofa::helper::vector< unsigned int >& b_p12_last = incision.b_p12_last_init;
    sofa::helper::vector< unsigned int >& b_i123_last = incision.b_i123_last_init;

    bool is_prepared=!((a[0]==b[0] && a[1]==b[1] && a[2]==b[2]) || (incision.ind_ta_init == incision.ind_tb_init));

    if(is_prepared)
    {
        sofa::helper::vector< sofa::helper::vector<unsigned int> > new_points_init;
        sofa::helper::vector< sofa::helper::vector<unsigned int> > closest_vertices_init;
        sofa::helper::vector< sofa::helper::vector<unsigned int> > &new_points = new_points_init;
        sofa::helper::vector< sofa::helper::vector<unsigned int> > &closest_vertices = closest_vertices_init;

        sofa::component::topology::TriangleSetTopologyModifier* triangleMod;
        _topology->getContext()->get(triangleMod);

        sofa::component::topology::TriangleSetTopologyAlgorithms<Vec3Types>* triangleAlg;
        _topology->getContext()->get(triangleAlg);

        bool is_fully_cut = triangleAlg->InciseAlongPointsList(incision.is_first_cut, a, b, ind_ta, ind_tb, a_last, a_p12_last, a_i123_last, b_last, b_p12_last, b_i123_last, new_points, closest_vertices);

        // notify the end for the current sequence of topological change events
        triangleMod->notifyEndingEvent();

        triangleMod->propagateTopologicalChanges();

        incision.is_first_cut = false;

        return is_fully_cut;
    }
    else
    {
        return false;
    }
}

// Handle Cutting (activated only for a triangular topology), using global variables to register the two last input points
bool TopologicalChangeManager::incisionCollisionModel(sofa::core::CollisionElementIterator elem2, Vector3& pos,
        const bool firstInput, const bool isCut)
{
    if(dynamic_cast<TriangleModel*>(elem2.getCollisionModel())!= NULL)
    {
        return incisionTriangleModel(elem2, pos, firstInput, isCut);
    }

    return false;
}

bool TopologicalChangeManager::incisionTriangleModel(sofa::core::CollisionElementIterator elem2, Vector3& pos,
        const bool firstInput, const bool isCut)
{
    Triangle triangle(elem2);
    TriangleModel* model2 = triangle.getCollisionModel();

    // Test if a TopologicalMapping (by default from TetrahedronSetTopology to TriangleSetTopology) exists :

    bool is_TopologicalMapping = false;

    sofa::core::componentmodel::topology::BaseMeshTopology* topo_curr;
    topo_curr = elem2.getCollisionModel()->getContext()->getMeshTopology();

    simulation::tree::GNode* parent2 = dynamic_cast<simulation::tree::GNode*>(model2->getContext());

    for (simulation::tree::GNode::ObjectIterator it = parent2->object.begin(); it != parent2->object.end(); ++it)
    {
        //std::cout << "INFO : name of GNode = " << (*it)->getName() <<  std::endl;

        if (dynamic_cast<sofa::core::componentmodel::topology::TopologicalMapping *>(*it)!= NULL)
        {
            is_TopologicalMapping=true;
        }

    }

    // try to catch the topology associated to the detected object (a TriangleSetTopology is expected)

    sofa::component::topology::TriangleSetTopologyContainer* triangleCont;
    topo_curr->getContext()->get(triangleCont);

    if(!is_TopologicalMapping)
    {
        // no TopologicalMapping

        if (triangleCont) // TriangleSetTopologyContainer
        {
            if (firstInput)
            {
                incision.a_init[0] = pos[0];
                incision.a_init[1] = pos[1];
                incision.a_init[2] = pos[2];
                incision.ind_ta_init = elem2.getIndex();

                incision.is_first_cut = true;
            }
            else if (isCut)
            {
                incision.b_init[0] = pos[0];
                incision.b_init[1] = pos[1];
                incision.b_init[2] = pos[2];

                incision.ind_tb_init = elem2.getIndex();

                if(incisionTriangleSetTopology(topo_curr))
                {
                    // full cut
                    incision.a_init[0] = pos[0];
                    incision.a_init[1] = pos[1];
                    incision.a_init[2] = pos[2];
                    incision.ind_ta_init = elem2.getIndex();

                    sofa::helper::vector<unsigned int> components_init;
                    sofa::helper::vector<unsigned int>& components = components_init;

                    sofa::component::topology::EdgeSetTopologyContainer* edgeCont;
                    topo_curr->getContext()->get(edgeCont);

                    int num = edgeCont->getNumberConnectedComponents(components);
                    std::cout << "Number of connected components : " << num << endl;
                    //sofa::helper::vector<int>::size_type i;
                    //for (i = 0; i != components.size(); ++i)
                    //  std::cout << "Vertex " << i <<" is in component " << components[i] << endl;
                }
                else
                {
                    sofa::helper::vector<unsigned int> components_init;
                    sofa::helper::vector<unsigned int>& components = components_init;

                    sofa::component::topology::EdgeSetTopologyContainer* edgeCont;
                    topo_curr->getContext()->get(edgeCont);

                    int num = edgeCont->getNumberConnectedComponents(components);
                    std::cout << "Number of connected components : " << num << endl;
                    //sofa::helper::vector<int>::size_type i;
                    //for (i = 0; i != components.size(); ++i)
                    //  std::cout << "Vertex " << i <<" is in component " << components[i] << endl;
                    return true; // change state to ATTACHED;
                }
            }
        }
    }
    else
    {
        // there may be a TetrahedronSetTopology over the TriangleSetTopology

    }

    return false;
}


} // namespace collision

} // namespace component

} // namespace sofa
