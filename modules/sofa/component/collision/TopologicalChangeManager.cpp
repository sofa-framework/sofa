/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/component/collision/SphereModel.h>

#include <sofa/component/container/MechanicalObject.h>
#include <sofa/simulation/common/Node.h>

#include <sofa/core/componentmodel/topology/TopologicalMapping.h>

#include <sofa/component/topology/PointSetTopologyContainer.h>
#include <sofa/component/topology/EdgeSetTopologyContainer.h>
#include <sofa/component/topology/TriangleSetTopologyContainer.h>
#include <sofa/component/topology/TriangleSetTopologyModifier.h>
#include <sofa/component/topology/TriangleSetTopologyAlgorithms.h>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.h>
#include <sofa/component/topology/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/QuadSetTopologyContainer.h>
#include <sofa/component/topology/HexahedronSetTopologyContainer.h>

#include <sofa/defaulttype/VecTypes.h>

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
    incision.a_last_init = (unsigned int)-1;
    incision.b_last_init = (unsigned int)-1;
    incision.ind_ta_init = (unsigned int)-1;
    incision.ind_tb_init = (unsigned int)-1;
}

TopologicalChangeManager::~TopologicalChangeManager()
{
}

void TopologicalChangeManager::removeItemsFromTriangleModel(sofa::component::collision::TriangleModel* model, const std::vector<int>& indices) const
{
    sofa::core::componentmodel::topology::BaseMeshTopology* topo_curr;
    topo_curr = model->getContext()->getMeshTopology();

    if(topo_curr == NULL)
        return;

    std::set< unsigned int > items;

    simulation::Node *node_curr = dynamic_cast<simulation::Node*>(topo_curr->getContext());

    if (topo_curr->getNbTetras() > 0)
    {
        // get the index of the tetra linked to each triangle
        for (unsigned int i=0; i<indices.size(); ++i)
            items.insert(topo_curr->getTetraTriangleShell(indices[i])[0]);
    }
    else if (topo_curr->getNbHexas() > 0)
    {
        // get the index of the hexa linked to each quad
        for (unsigned int i=0; i<indices.size(); ++i)
            items.insert(topo_curr->getHexaQuadShell(indices[i]/2)[0]);
    }
    else
    {
        int nbt = topo_curr->getNbTriangles();
        for (unsigned int i=0; i<indices.size(); ++i)
        {
            items.insert(indices[i] < nbt ? indices[i] : (indices[i]+nbt)/2);
            //std::cout << indices[i] <<std::endl;
        }
    }

    bool is_topoMap = true;

    while(is_topoMap)
    {
        is_topoMap = false;
        std::vector< core::objectmodel::BaseObject * > listObject;
        node_curr->get<core::objectmodel::BaseObject>(&listObject, core::objectmodel::BaseContext::Local);
        for(unsigned int i=0; i<listObject.size(); ++i)
        {
            sofa::core::componentmodel::topology::TopologicalMapping *topoMap = dynamic_cast<sofa::core::componentmodel::topology::TopologicalMapping *>(listObject[i]);
            if(topoMap != NULL && !topoMap->propagateFromOutputToInputModel())
            {
                is_topoMap = true;
                //unsigned int ind_glob = topoMap->getGlobIndex(ind_curr);
                //ind_curr = topoMap->getFromIndex(ind_glob);
                std::set< unsigned int > loc_items = items;
                items.clear();
                if( topoMap->isTheOutputTopologySubdividingTheInputOne())
                {
                    for (std::set< unsigned int >::const_iterator it=loc_items.begin(); it != loc_items.end(); ++it)
                    {
                        unsigned int ind_glob = topoMap->getGlobIndex(*it);
                        unsigned int ind = topoMap->getFromIndex(ind_glob);
                        //std::cout << *it << " -> "<<ind_glob << " -> "<<ind<<std::endl;
                        items.insert(ind);
                    }
                }
                else
                {
                    for (std::set< unsigned int >::const_iterator it=loc_items.begin(); it != loc_items.end(); ++it)
                    {
                        vector<unsigned int> indices;
                        topoMap->getFromIndex( indices, *it);
                        for( vector<unsigned int>::const_iterator itIndices = indices.begin(); itIndices != indices.end(); itIndices++)
                        {
                            //std::cout << *it << " -> " << *itIndices << std::endl;
                            items.insert( *itIndices );
                        }
                    }
                }
                topo_curr = topoMap->getFrom()->getContext()->getMeshTopology();
                node_curr = dynamic_cast<simulation::Node*>(topo_curr->getContext());

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
}

void TopologicalChangeManager::removeItemsFromSphereModel(sofa::component::collision::SphereModel* model, const std::vector<int>& indices) const
{
    sofa::core::componentmodel::topology::BaseMeshTopology* topo_curr;
    topo_curr = model->getContext()->getMeshTopology();

    if(dynamic_cast<PointSetTopologyContainer*>(topo_curr) == NULL)
        return;

    std::set< unsigned int > items;

    simulation::Node *node_curr = dynamic_cast<simulation::Node*>(topo_curr->getContext());

    for (unsigned int i=0; i<indices.size(); ++i)
        items.insert(indices[i]);

    bool is_topoMap = true;

    while(is_topoMap)
    {
        is_topoMap = false;

        std::vector< core::objectmodel::BaseObject * > listObject;
        node_curr->get<core::objectmodel::BaseObject>(&listObject, core::objectmodel::BaseContext::Local);
        for(unsigned int i=0; i<listObject.size(); ++i)
        {
            sofa::core::componentmodel::topology::TopologicalMapping *topoMap = dynamic_cast<sofa::core::componentmodel::topology::TopologicalMapping *>(listObject[i]);
            if(topoMap != NULL && !topoMap->propagateFromOutputToInputModel())
            {
                is_topoMap = true;
                std::set< unsigned int > loc_items = items;
                items.clear();
                if( topoMap->isTheOutputTopologySubdividingTheInputOne())
                {
                    for (std::set< unsigned int >::const_iterator it=loc_items.begin(); it != loc_items.end(); ++it)
                    {
                        unsigned int ind_glob = topoMap->getGlobIndex(*it);
                        unsigned int ind = topoMap->getFromIndex(ind_glob);
                        //sout << *it << " -> "<<ind_glob << " -> "<<ind<<sendl;
                        items.insert(ind);
                    }
                }
                else
                {
                    for (std::set< unsigned int >::const_iterator it=loc_items.begin(); it != loc_items.end(); ++it)
                    {
                        vector<unsigned int> indices;
                        topoMap->getFromIndex( indices, *it);
                        for( vector<unsigned int>::const_iterator itIndices = indices.begin(); itIndices != indices.end(); itIndices++)
                        {
                            //std::cout << *it << " -> " << *itIndices << std::endl;
                            items.insert( *itIndices );
                        }
                    }
                }
                topo_curr = topoMap->getFrom()->getContext()->getMeshTopology();
                node_curr = dynamic_cast<simulation::Node*>(topo_curr->getContext());

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
}

// Handle Removing of topological element (from any type of topology)
void TopologicalChangeManager::removeItemsFromCollisionModel(sofa::core::CollisionElementIterator elem2) const
{
    std::vector<int> id;
    id.push_back(elem2.getIndex());
    removeItemsFromCollisionModel(elem2.getCollisionModel(), id);
}

void TopologicalChangeManager::removeItemsFromCollisionModel(sofa::core::CollisionModel* model, const std::vector<int>& indices) const
{
    if(dynamic_cast<TriangleModel*>(model)!= NULL)
    {
        removeItemsFromTriangleModel(static_cast<TriangleModel*>(model), indices);
    }
    else if(dynamic_cast<SphereModel*>(model)!= NULL)
    {
        removeItemsFromSphereModel(static_cast<SphereModel*>(model), indices);
    }
}

// Intermediate method to handle cutting
bool TopologicalChangeManager::incisionTriangleSetTopology(sofa::core::componentmodel::topology::BaseMeshTopology* _topology)
{

    //Incision in triangles from point a to point b

    // Mechanical coord of points a & b:
    Vec<3,double>& a= incision.a_init;
    Vec<3,double>& b= incision.b_init;

    // Point Indices
    unsigned int& a_last = incision.a_last_init;
    if(incision.is_first_cut)
        a_last = (unsigned int)-1;
    else
    {
        core::componentmodel::behavior::MechanicalState<Vec3Types>* mstate = _topology->getContext()->get<core::componentmodel::behavior::MechanicalState<Vec3Types> >();
        helper::vector<Vector3> &v_coords =  *mstate->getX();
        a = v_coords[a_last];
    }

    unsigned int& b_last = incision.b_last_init;

    // Triangle Indices
    unsigned int &ind_ta = incision.ind_ta_init;
    unsigned int &ind_tb = incision.ind_tb_init;

    bool is_prepared=!((a[0]==b[0] && a[1]==b[1] && a[2]==b[2]) || incision.is_cut_completed);

    if(is_prepared)
    {

        sofa::component::topology::TriangleSetTopologyModifier* triangleMod;
        _topology->getContext()->get(triangleMod);

        sofa::component::topology::TriangleSetTopologyAlgorithms<Vec3Types>* triangleAlg;
        _topology->getContext()->get(triangleAlg);

        sofa::component::topology::TriangleSetGeometryAlgorithms<Vec3Types>* triangleGeo;
        _topology->getContext()->get(triangleGeo);

        // Output declarations
        sofa::helper::vector< sofa::core::componentmodel::topology::TopologyObjectType> topoPath_list;
        sofa::helper::vector<unsigned int> indices_list;
        sofa::helper::vector< Vec<3, double> > coords2_list;
        //bool is_on_boundary = false;


        /*
        std::cout << "*********************" << std::endl;
        std::cout << "a: " << a_last << " => " << a << " in triangle: " << ind_ta << std::endl;
        std::cout << "b: " << b_last << " => " << b << " in triangle: " << ind_tb << std::endl;
        std::cout << "*********************" << std::endl;
        */


        //    bool ok = triangleGeo->computeIntersectedPointsList(a, b, ind_ta, ind_tb, triangles_list, edges_list, coords_list, is_on_boundary);
        bool ok = triangleGeo->computeIntersectedObjectsList(a_last, a, b, ind_ta, ind_tb, topoPath_list, indices_list, coords2_list);

        //std::cout << "NEW PATH" << std::endl;
        //std::cout << "theenum:  " << topoPath_list << std::endl;
        //std::cout << "indices: " << indices_list << std::endl;
        //std::cout << "barycoord: " << coords2_list << std::endl;

        //triangles_list.push_back(ind_tb);

        if (!ok)
        {
            std::cout << "ERROR in computeIntersectedPointsList" << std::endl;
            return false;
        }


        sofa::helper::vector< unsigned int > new_edges;

        //triangleAlg->SplitAlongPath(a_last, a, b_last, b, triangles_list, edges_list, coords_list, new_edges);
        triangleAlg->SplitAlongPath(a_last, a, b_last, b, topoPath_list, indices_list, coords2_list, new_edges, 0.1, 0.25);
        //std::cout << "** split along path done **" << std::endl;

        //std::cout << "new edges : " << new_edges << std::endl;

        sofa::helper::vector<unsigned int> new_points;
        sofa::helper::vector<unsigned int> end_points;

        bool is_fully_cut = triangleAlg->InciseAlongEdgeList(new_edges, new_points, end_points);
        //std::cout << "** incise along path done **" << std::endl;

        if (!end_points.empty())
        {
            incision.a_last_init = end_points.back();
            incision.is_first_cut = false;
        }
        else
        {
            incision.is_cut_completed = true;
            incision.is_first_cut = true;
        }



        triangleMod->propagateTopologicalChanges();

        // notify the end for the current sequence of topological change events
        triangleMod->notifyEndingEvent();

        triangleMod->propagateTopologicalChanges();



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

    simulation::Node* parent2 = dynamic_cast<simulation::Node*>(model2->getContext());

    std::vector< core::objectmodel::BaseObject * > listObject;
    parent2->get<core::objectmodel::BaseObject>(&listObject, core::objectmodel::BaseContext::Local);
    for(unsigned int i=0; i<listObject.size(); ++i)
    {
        //sout << "INFO : name of Node = " << (listObject[i])->getName() <<  sendl;
        if (dynamic_cast<sofa::core::componentmodel::topology::TopologicalMapping *>(listObject[i])!= NULL)
        {
            is_TopologicalMapping=true;
        }
    }

    // try to catch the topology associated to the detected object (a TriangleSetTopology is expected)
    sofa::component::topology::TriangleSetTopologyContainer* triangleCont;
    topo_curr->getContext()->get(triangleCont);

    if(!is_TopologicalMapping) // no TopologicalMapping
    {
        if (triangleCont) // TriangleSetTopologyContainer
        {
            if (firstInput) // initialise first point of contact from the incisionCollisionModel
            {
                incision.a_init[0] = pos[0];
                incision.a_init[1] = pos[1];
                incision.a_init[2] = pos[2];
                incision.ind_ta_init = elem2.getIndex();

                incision.is_first_cut = true; // first incision                      //encore necessaire??
                incision.is_cut_completed = false;
            }
            else if (isCut) // if it is not the first contact
            {
                incision.b_init[0] = pos[0];
                incision.b_init[1] = pos[1];
                incision.b_init[2] = pos[2];

                incision.ind_tb_init = elem2.getIndex();

                bool incision_ok = incisionTriangleSetTopology(topo_curr);

                // Compute the number of connected components
                sofa::helper::vector<unsigned int> components_init;
                sofa::helper::vector<unsigned int>& components = components_init;

                sofa::component::topology::EdgeSetTopologyContainer* edgeCont;
                topo_curr->getContext()->get(edgeCont);

                int num = edgeCont->getNumberConnectedComponents(components);
                std::cout << "Number of connected components : " << num << std::endl;
                //sofa::helper::vector<int>::size_type i;
                //for (i = 0; i != components.size(); ++i)
                //  sout << "Vertex " << i <<" is in component " << components[i] << endl;


                if(incision_ok)  //switch data b to a, in order to continue incision.
                {
                    // full cut
                    incision.a_init[0] = pos[0];
                    incision.a_init[1] = pos[1];
                    incision.a_init[2] = pos[2];
                    incision.ind_ta_init = elem2.getIndex();
                }
                else
                {
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
