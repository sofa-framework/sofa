/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/gui/component/performer/TopologicalChangeManager.h>


#include <sofa/simulation/Node.h>

#include <sofa/core/topology/TopologicalMapping.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/component/topology/container/dynamic/PointSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/EdgeSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TriangleSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TriangleSetTopologyModifier.h>
#include <sofa/component/topology/container/dynamic/TriangleSetGeometryAlgorithms.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/QuadSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyContainer.h>
#include <sofa/component/topology/mapping/Hexa2TetraTopologicalMapping.h>
#include <sofa/component/collision/geometry/SphereCollisionModel.h>
#include <sofa/component/collision/geometry/PointCollisionModel.h>
#include <sofa/component/collision/geometry/TriangleCollisionModel.h>

namespace sofa::gui::component::performer
{

using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace sofa::component::topology::container::dynamic;
using namespace sofa::component::collision::geometry;
using type::vector;

TopologicalChangeManager::TopologicalChangeManager()
{
    incision.firstCut = true;
    incision.indexPoint = sofa::InvalidID;
}

TopologicalChangeManager::~TopologicalChangeManager()
{
}

Index TopologicalChangeManager::removeItemsFromTriangleModel(TriangleCollisionModel<sofa::defaulttype::Vec3Types>* model, const type::vector<Index>& indices) const
{
    sofa::core::topology::BaseMeshTopology* topo_curr;
    topo_curr = model->getCollisionTopology();

    if(topo_curr == nullptr)
        return 0;

    std::set< unsigned int > items;

    const simulation::Node *node_curr = dynamic_cast<simulation::Node*>(topo_curr->getContext());

    if (topo_curr->getNbTetrahedra() > 0)
    {
        // get the index of the tetra linked to each triangle
        for (unsigned int i=0; i<indices.size(); ++i)
            items.insert(topo_curr->getTetrahedraAroundTriangle(indices[i])[0]);
    }
    else if (topo_curr->getNbHexahedra() > 0)
    {
        // get the index of the hexa linked to each quad
        for (unsigned int i=0; i<indices.size(); ++i)
            items.insert(topo_curr->getHexahedraAroundQuad(indices[i]/2)[0]);
    }
    else
    {
        //Quick HACK for Hexa2TetraMapping
        sofa::component::topology::mapping::Hexa2TetraTopologicalMapping* badMapping;
        model->getContext()->get(badMapping, sofa::core::objectmodel::BaseContext::SearchUp);
        if(badMapping) //stop process
        {
            msg_warning("TopologicalChangeManager") << " Removing element is not handle by Hexa2TetraTopologicalMapping. Stopping process." ;
            return 0;
        }

        const auto nbt = topo_curr->getNbTriangles();
        for (unsigned int i=0; i<indices.size(); ++i)
        {
            items.insert(indices[i] < nbt ? indices[i] : (indices[i]+nbt)/2);
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
            sofa::core::topology::TopologicalMapping *topoMap = dynamic_cast<sofa::core::topology::TopologicalMapping *>(listObject[i]);
            if(topoMap != nullptr && !topoMap->propagateFromOutputToInputModel())
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
                        const unsigned int ind_glob = topoMap->getGlobIndex(*it);
                        unsigned int ind = topoMap->getFromIndex(ind_glob);
                        items.insert(ind);
                    }
                }
                else
                {
                    for (std::set< unsigned int >::const_iterator it=loc_items.begin(); it != loc_items.end(); ++it)
                    {
                        vector<Index> indices;
                        topoMap->getFromIndex( indices, *it);
                        for( auto itIndices = indices.begin(); itIndices != indices.end(); ++itIndices)
                        {
                            items.insert( *itIndices );
                        }
                    }
                }
                topo_curr = topoMap->getFrom();
                node_curr = dynamic_cast<simulation::Node*>(topo_curr->getContext());

                break;
            }
        }
    }

    sofa::type::vector<Index> vitems;
    vitems.reserve(items.size());
    vitems.insert(vitems.end(), items.rbegin(), items.rend());

    const Index res = vitems.size();

    sofa::core::topology::TopologyModifier* topoMod;
    topo_curr->getContext()->get(topoMod);

    topoMod->removeItems(vitems);

    topoMod->notifyEndingEvent();


    topoMod->propagateTopologicalChanges();

    return res;
}


Index TopologicalChangeManager::removeItemsFromPointModel(PointCollisionModel<sofa::defaulttype::Vec3Types>* model, const type::vector<Index>& indices) const
{
    sofa::core::topology::BaseMeshTopology* topo_curr;
    topo_curr = model->getCollisionTopology();

    if (topo_curr == nullptr)
        return 0;

    sofa::type::vector<unsigned int> tItems;
    for (const auto i : indices)
    {
        const sofa::core::topology::BaseMeshTopology::TrianglesAroundVertex& triAV = topo_curr->getTrianglesAroundVertex(i);        
        for (auto j : triAV)
        {
            bool found = false;
            for (const auto k : tItems)
            {
                if (j == k)
                {
                    found = true;
                    break;
                }
            }

            if (!found)
                tItems.push_back(j);
        }
    }

    std::set< unsigned int > items;

    const simulation::Node *node_curr = dynamic_cast<simulation::Node*>(topo_curr->getContext());

    if (topo_curr->getNbTetrahedra() > 0)
    {
        // get the index of the tetra linked to each triangle
        for (unsigned int i = 0; i<tItems.size(); ++i)
            items.insert(topo_curr->getTetrahedraAroundTriangle(tItems[i])[0]);
    }
    else if (topo_curr->getNbHexahedra() > 0)
    {
        // get the index of the hexa linked to each quad
        for (unsigned int i = 0; i<tItems.size(); ++i)
            items.insert(topo_curr->getHexahedraAroundQuad(tItems[i] / 2)[0]);
    }
    else
    {
        //Quick HACK for Hexa2TetraMapping
        sofa::component::topology::mapping::Hexa2TetraTopologicalMapping* badMapping;
        model->getContext()->get(badMapping, sofa::core::objectmodel::BaseContext::SearchUp);
        if (badMapping) //stop process
        {
            msg_warning("TopologicalChangeManager") << " Removing element is not handle by Hexa2TetraTopologicalMapping. Stopping process.";
            return 0;
        }

        const size_t nbt = topo_curr->getNbTriangles();
        for (unsigned int i = 0; i<tItems.size(); ++i)
        {
            items.insert(tItems[i] < nbt ? tItems[i] : (tItems[i] + nbt) / 2);
        }
    }

    bool is_topoMap = true;

    while (is_topoMap)
    {
        is_topoMap = false;
        std::vector< core::objectmodel::BaseObject * > listObject;
        node_curr->get<core::objectmodel::BaseObject>(&listObject, core::objectmodel::BaseContext::Local);
        for (unsigned int i = 0; i<listObject.size(); ++i)
        {
            sofa::core::topology::TopologicalMapping *topoMap = dynamic_cast<sofa::core::topology::TopologicalMapping *>(listObject[i]);
            if (topoMap != nullptr && !topoMap->propagateFromOutputToInputModel())
            {
                is_topoMap = true;
                std::set< unsigned int > loc_items = items;
                items.clear();
                if (topoMap->isTheOutputTopologySubdividingTheInputOne())
                {
                    for (std::set< unsigned int >::const_iterator it = loc_items.begin(); it != loc_items.end(); ++it)
                    {
                        const unsigned int ind_glob = topoMap->getGlobIndex(*it);
                        unsigned int ind = topoMap->getFromIndex(ind_glob);
                        items.insert(ind);
                    }
                }
                else
                {
                    for (auto it = loc_items.begin(); it != loc_items.end(); ++it)
                    {
                        vector<Index> indices;
                        topoMap->getFromIndex(indices, *it);
                        for (auto itIndices = indices.begin(); itIndices != indices.end(); ++itIndices)
                        {
                            items.insert(*itIndices);
                        }
                    }
                }
                topo_curr = topoMap->getFrom();
                node_curr = dynamic_cast<simulation::Node*>(topo_curr->getContext());
                break;
            }
        }
    }

    sofa::type::vector<Index> vitems;
    vitems.reserve(items.size());
    vitems.insert(vitems.end(), items.rbegin(), items.rend());

    const Index res = vitems.size();


    sofa::core::topology::TopologyModifier* topoMod;
    topo_curr->getContext()->get(topoMod);

    topoMod->removeItems(vitems);
    topoMod->notifyEndingEvent();
    topoMod->propagateTopologicalChanges();

    return res;
}

Index TopologicalChangeManager::removeItemsFromLineModel(LineCollisionModel<sofa::defaulttype::Vec3Types>* model, const type::vector<Index>& indices) const
{
    // EdgeSetTopologyContainer
    sofa::core::topology::BaseMeshTopology* topo_curr;
    topo_curr = model->getCollisionTopology();

    if(dynamic_cast<EdgeSetTopologyContainer*>(topo_curr) == nullptr){
        msg_warning("TopologicalChangeManager") << "Topology is not an EdgeSetTopologyContainer. Only EdgeSetTopologyContainer implemented.";
        return 0;
    }

    // copy indices to have a mutable version of the vector
    type::vector<Index> unique_indices = indices;
    // sort followed by unique, to remove all duplicates
    std::sort(unique_indices.begin(), unique_indices.end());
    unique_indices.erase(std::unique(unique_indices.begin(), unique_indices.end()), unique_indices.end());

    EdgeSetTopologyModifier* topo_mod;
    topo_curr->getContext()->get(topo_mod);
    if(topo_mod  == nullptr){
        msg_warning("TopologicalChangeManager") << "Cannot find an EdgeSetTopologyModifier to perform the changes.";
        return 0;
    }

    topo_mod->removeItems(unique_indices);

    topo_mod->notifyEndingEvent();

    return indices.size();
}

Index TopologicalChangeManager::removeItemsFromSphereModel(SphereCollisionModel<sofa::defaulttype::Vec3Types>* model, const type::vector<Index>& indices) const
{
    sofa::core::topology::BaseMeshTopology* topo_curr;
    topo_curr = model->getCollisionTopology();

    if(dynamic_cast<PointSetTopologyContainer*>(topo_curr) == nullptr)
        return 0;

    std::set< unsigned int > items;

    const simulation::Node *node_curr = dynamic_cast<simulation::Node*>(topo_curr->getContext());

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
            sofa::core::topology::TopologicalMapping *topoMap = dynamic_cast<sofa::core::topology::TopologicalMapping *>(listObject[i]);
            if(topoMap != nullptr && !topoMap->propagateFromOutputToInputModel())
            {
                is_topoMap = true;
                std::set< unsigned int > loc_items = items;
                items.clear();
                if( topoMap->isTheOutputTopologySubdividingTheInputOne())
                {
                    for (std::set< unsigned int >::const_iterator it=loc_items.begin(); it != loc_items.end(); ++it)
                    {
                        const unsigned int ind_glob = topoMap->getGlobIndex(*it);
                        unsigned int ind = topoMap->getFromIndex(ind_glob);
                        items.insert(ind);
                    }
                }
                else
                {
                    for (auto it=loc_items.begin(); it != loc_items.end(); ++it)
                    {
                        vector<Index> tmpindices;
                        topoMap->getFromIndex( tmpindices, *it);
                        for(auto itIndices = tmpindices.begin(); itIndices != tmpindices.end(); ++itIndices)
                        {
                            items.insert( *itIndices );
                        }
                    }
                }
                topo_curr = topoMap->getFrom();
                node_curr = dynamic_cast<simulation::Node*>(topo_curr->getContext());

                break;
            }
        }
    }

    sofa::type::vector<Index> vitems;
    vitems.reserve(items.size());
    vitems.insert(vitems.end(), items.rbegin(), items.rend());

    const Index res = vitems.size();

    sofa::core::topology::TopologyModifier* topoMod;
    topo_curr->getContext()->get(topoMod);

    topoMod->removeItems(vitems);

    topoMod->notifyEndingEvent();

    topoMod->propagateTopologicalChanges();

    return res;
}

// Handle Removing of topological element (from any type of topology)
Index TopologicalChangeManager::removeItemsFromCollisionModel(sofa::core::CollisionElementIterator elem2) const
{
    type::vector<Index> id;
    id.push_back(elem2.getIndex());
    return removeItemsFromCollisionModel(elem2.getCollisionModel(), id);
}

Index TopologicalChangeManager::removeItemsFromCollisionModel(sofa::core::CollisionModel* model, const Index& indices) const
{
    type::vector<Index> id;
    id.push_back(indices);
    return removeItemsFromCollisionModel(model, id);
}


Index TopologicalChangeManager::removeItemsFromCollisionModel(sofa::core::CollisionModel* model, const type::vector<Index>& indices) const
{
    if(dynamic_cast<TriangleCollisionModel<sofa::defaulttype::Vec3Types>*>(model)!= nullptr)
    {
        return removeItemsFromTriangleModel(static_cast<TriangleCollisionModel<sofa::defaulttype::Vec3Types>*>(model), indices);
    }
    if (dynamic_cast<PointCollisionModel<sofa::defaulttype::Vec3Types>*>(model) != nullptr)
    {
        return removeItemsFromPointModel(static_cast<PointCollisionModel<sofa::defaulttype::Vec3Types>*>(model), indices);
    }
    else if(dynamic_cast<SphereCollisionModel<sofa::defaulttype::Vec3Types>*>(model)!= nullptr)
    {
        return removeItemsFromSphereModel(static_cast<SphereCollisionModel<sofa::defaulttype::Vec3Types>*>(model), indices);
    }
    else if(dynamic_cast<LineCollisionModel<sofa::defaulttype::Vec3Types>*>(model)!= nullptr)
    {
        return removeItemsFromLineModel(static_cast<LineCollisionModel<sofa::defaulttype::Vec3Types>*>(model), indices);
    }
    else
        return 0;
}



// Handle Cutting (activated only for a triangular topology), using global variables to register the two last input points
bool TopologicalChangeManager::incisionCollisionModel(sofa::core::CollisionElementIterator elem, sofa::type::Vec3& pos, const bool firstInput, int snapingValue, int snapingBorderValue)
{
    const sofa::component::collision::geometry::Triangle triangle(elem);
    TriangleCollisionModel<sofa::defaulttype::Vec3Types>* model = triangle.getCollisionModel();

    if (model != nullptr)
    {

        if (firstInput) // initialise first point of contact from the incisionCollisionModel
        {
            incision.coordPoint[0] = pos[0];
            incision.coordPoint[1] = pos[1];
            incision.coordPoint[2] = pos[2];

            incision.indexTriangle = elem.getIndex();

            return true;
        }
        else // if it is not the first contact, cut
        {
            const bool isCut = this->incisionTriangleModel (model, incision.indexTriangle, incision.coordPoint, model, elem.getIndex(), pos, snapingValue, snapingBorderValue);

            if (isCut && !incision.firstCut) // cut has been reached, and will possible be continue. Stocking information.
            {
                incision.coordPoint[0] = pos[0];
                incision.coordPoint[1] = pos[1];
                incision.coordPoint[2] = pos[2];

                incision.indexTriangle = elem.getIndex();
            }

            return isCut;
        }
    }
    else
        return false;
}

void TopologicalChangeManager::setIncisionFirstCut(bool b)
{
    incision.firstCut = b;
}

// Handle Cutting for general model (only Triangle for the moment)
bool TopologicalChangeManager::incisionCollisionModel(sofa::core::CollisionModel *firstModel , Index idxA, const sofa::type::Vec3& firstPoint,
        sofa::core::CollisionModel *secondModel, Index idxB, const sofa::type::Vec3& secondPoint,
        int snapingValue, int snapingBorderValue)
{

    TriangleCollisionModel<sofa::defaulttype::Vec3Types>* firstCollisionModel = dynamic_cast< TriangleCollisionModel<sofa::defaulttype::Vec3Types>* >(firstModel);
    TriangleCollisionModel<sofa::defaulttype::Vec3Types>* secondCollisionModel = dynamic_cast< TriangleCollisionModel<sofa::defaulttype::Vec3Types>* >(secondModel);
    if (!firstCollisionModel || firstCollisionModel != secondCollisionModel) return false;
    return incisionTriangleModel(firstCollisionModel,  idxA, firstPoint,
            secondCollisionModel, idxB, secondPoint,
            snapingValue, snapingBorderValue);
}



// Perform incision in triangulation
bool TopologicalChangeManager::incisionTriangleModel(TriangleCollisionModel<sofa::defaulttype::Vec3Types> *firstModel , core::topology::BaseMeshTopology::TriangleID idxA, const sofa::type::Vec3& firstPoint,
        TriangleCollisionModel<sofa::defaulttype::Vec3Types> *secondModel, core::topology::BaseMeshTopology::TriangleID idxB, const sofa::type::Vec3& secondPoint,
        int snapingValue, int snapingBorderValue)
{

    // -- STEP 1: looking for collision model and topology components

    TriangleCollisionModel<sofa::defaulttype::Vec3Types>* firstCollisionModel = dynamic_cast< TriangleCollisionModel<sofa::defaulttype::Vec3Types>* >(firstModel);
    TriangleCollisionModel<sofa::defaulttype::Vec3Types>* secondCollisionModel = dynamic_cast< TriangleCollisionModel<sofa::defaulttype::Vec3Types>* >(secondModel);

    sofa::component::collision::geometry::Triangle firstTriangle(firstCollisionModel, idxA);
    sofa::component::collision::geometry::Triangle secondTriangle(secondCollisionModel, idxB);

    if (firstCollisionModel != secondCollisionModel)
    {
        msg_warning("TopologicalChangeManager") << "Incision involving different models is not supported yet!" ;
        return false;
    }


    sofa::core::topology::BaseMeshTopology* currentTopology = firstCollisionModel->getCollisionTopology();
    const simulation::Node* collisionNode = dynamic_cast<simulation::Node*>(firstCollisionModel->getContext());

    // Test if a TopologicalMapping (by default from TetrahedronSetTopology to TriangleSetTopology) exists :
    std::vector< sofa::core::topology::TopologicalMapping *> listTopologicalMapping;
    collisionNode->get<sofa::core::topology::TopologicalMapping>(&listTopologicalMapping, core::objectmodel::BaseContext::Local);
    const bool isTopologicalMapping = !(listTopologicalMapping.empty());

    if (!isTopologicalMapping) // mapping not handle for the moment
    {

        // -- STEP 2: Try to catch the topology associated to the detected object (a TriangleSetTopology is expected)
        TriangleSetTopologyContainer* triangleContainer;
        currentTopology->getContext()->get(triangleContainer);

        TriangleSetTopologyModifier* triangleModifier;
        currentTopology->getContext()->get(triangleModifier);

        TriangleSetGeometryAlgorithms<Vec3Types>* triangleGeometry;
        currentTopology->getContext()->get(triangleGeometry);


        // -- STEP 3: Initialization

        // Mechanical coord of points a & b:
        auto coord_a = (firstPoint);
        auto coord_b = (secondPoint);


        // Path first point Indice. This might be useful if topology is in movement. (coord of point "a" doesn't belongs anymore to triangle of index : "idxA" since pickhandler)
        const core::topology::BaseMeshTopology::PointID& last_indexPoint = incision.indexPoint;

        if(!incision.firstCut) //Not the first cut, look for new coord of a
        {
            const core::behavior::MechanicalState<Vec3Types>* mstate = currentTopology->getContext()->get<core::behavior::MechanicalState<Vec3Types> >();
            const auto &v_coords =  mstate->read(core::vec_id::read_access::position)->getValue();
            coord_a = v_coords[last_indexPoint];
        }


        // Output declarations
        sofa::type::vector< sofa::geometry::ElementType> topoPath_list;
        sofa::type::vector<Index> indices_list;
        sofa::type::vector< Vec3 > coords2_list;

        // Snapping value: input are percentages, we need to transform it as real epsilon value;
        const double epsilonSnap = (double)snapingValue/200;
        const double epsilonBorderSnap = (double)snapingBorderValue/210; // magic number (0.5 is max value and must not be reached, as threshold is compared to barycoord value)


        // -- STEP 4: Creating path through different elements
        const bool path_ok = triangleGeometry->computeIntersectedObjectsList(last_indexPoint, coord_a, coord_b, idxA, idxB, topoPath_list, indices_list, coords2_list);

        if (!path_ok)
        {
            dmsg_error("TopologicalChangeManager") << " in computeIntersectedObjectsList" ;
            return false;
        }


        // -- STEP 5: Splitting elements along path (incision path is stored inside "new_edges")
        sofa::type::vector< Index > new_edges;
        const int result = triangleGeometry->SplitAlongPath(last_indexPoint, coord_a, sofa::InvalidID, coord_b, topoPath_list, indices_list, coords2_list, new_edges, epsilonSnap, epsilonBorderSnap);

        if (result == -1)
        {
            incision.indexPoint = last_indexPoint;
            return false;
        }

        // -- STEP 6: Incise along new_edges path (i.e duplicating edges to create an incision)
        sofa::type::vector<Index> new_points;
        sofa::type::vector<Index> end_points;
        bool reachBorder = false;
        const bool incision_ok = triangleGeometry->InciseAlongEdgeList(new_edges, new_points, end_points, reachBorder);

        if (!incision_ok)
        {
            dmsg_error("TopologicalChangeManager") << " in InciseAlongEdgeList" ;
            return false;
        }


        // -- STEP 7: Updating information if incision has reached a border.
        if (reachBorder)
            incision.firstCut = true;
        else
        {
            incision.firstCut = false;
            // updating triangle index for second function case!!
            if (!end_points.empty())
                incision.indexPoint = end_points.back();
        }
        if (!end_points.empty())
            incision.indexPoint = end_points.back();

        // -- STEP 8: Propagating topological events.
        triangleModifier->notifyEndingEvent();

        return true;
    }
    else
        return false;
}


} //namespace sofa::gui::component::performer
