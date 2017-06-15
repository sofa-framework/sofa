/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "TopologicalChangeManager.h"

#include <SofaMeshCollision/TriangleModel.h>
#if 0
#include <SofaMiscCollision/TetrahedronModel.h>
#endif
#include <SofaBaseCollision/SphereModel.h>

#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/simulation/Node.h>

#include <sofa/core/topology/TopologicalMapping.h>

#include <SofaBaseTopology/PointSetTopologyContainer.h>
#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetTopologyModifier.h>
#include <SofaBaseTopology/TriangleSetTopologyAlgorithms.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
#include <SofaBaseTopology/QuadSetTopologyContainer.h>
#include <SofaBaseTopology/HexahedronSetTopologyContainer.h>

#include <sofa/defaulttype/VecTypes.h>

#include <SofaTopologyMapping/Hexa2TetraTopologicalMapping.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::component::topology;
using helper::vector;

TopologicalChangeManager::TopologicalChangeManager()
{
    incision.firstCut = true;
    incision.indexPoint = core::topology::BaseMeshTopology::InvalidID;
}

TopologicalChangeManager::~TopologicalChangeManager()
{
}

int TopologicalChangeManager::removeItemsFromTriangleModel(sofa::component::collision::TriangleModel* model, const helper::vector<int>& indices) const
{
    sofa::core::topology::BaseMeshTopology* topo_curr;
    topo_curr = model->getContext()->getMeshTopology();

    if(topo_curr == NULL)
        return 0;

    std::set< unsigned int > items;

    simulation::Node *node_curr = dynamic_cast<simulation::Node*>(topo_curr->getContext());

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
        sofa::component::topology::Hexa2TetraTopologicalMapping* badMapping;
        model->getContext()->get(badMapping, sofa::core::objectmodel::BaseContext::SearchRoot);
        if(badMapping) //stop process
        {
            msg_warning("TopologicalChangeManager") << " Removing element is not handle by Hexa2TetraTopologicalMapping. Stopping process." ;
            return 0;
        }

        int nbt = topo_curr->getNbTriangles();
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
                        items.insert(ind);
                    }
                }
                else
                {
                    for (std::set< unsigned int >::const_iterator it=loc_items.begin(); it != loc_items.end(); ++it)
                    {
                        vector<unsigned int> indices;
                        topoMap->getFromIndex( indices, *it);
                        for( vector<unsigned int>::const_iterator itIndices = indices.begin(); itIndices != indices.end(); ++itIndices)
                        {
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

    int res = vitems.size();

    sofa::core::topology::TopologyModifier* topoMod;
    topo_curr->getContext()->get(topoMod);

    topoMod->removeItems(vitems);

    topoMod->notifyEndingEvent();


    topoMod->propagateTopologicalChanges();

    return res;
}

#if 0

int TopologicalChangeManager::removeItemsFromTetrahedronModel(sofa::component::collision::TetrahedronModel* model, const helper::vector<int>& indices) const
{
    sofa::core::topology::BaseMeshTopology* topo_curr;
    topo_curr = model->getContext()->getMeshTopology();

    if(dynamic_cast<PointSetTopologyContainer*>(topo_curr) == NULL)
        return 0;

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
            sofa::core::topology::TopologicalMapping *topoMap = dynamic_cast<sofa::core::topology::TopologicalMapping *>(listObject[i]);
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

    int res = vitems.size();

    sofa::core::topology::TopologyModifier* topoMod;
    topo_curr->getContext()->get(topoMod);

    topoMod->removeItems(vitems);

    topoMod->notifyEndingEvent();

    topoMod->propagateTopologicalChanges();

    return res;
}

#endif	// if 0

int TopologicalChangeManager::removeItemsFromSphereModel(sofa::component::collision::SphereModel* model, const helper::vector<int>& indices) const
{
    sofa::core::topology::BaseMeshTopology* topo_curr;
    topo_curr = model->getContext()->getMeshTopology();

    if(dynamic_cast<PointSetTopologyContainer*>(topo_curr) == NULL)
        return 0;

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
            sofa::core::topology::TopologicalMapping *topoMap = dynamic_cast<sofa::core::topology::TopologicalMapping *>(listObject[i]);
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
                        items.insert(ind);
                    }
                }
                else
                {
                    for (std::set< unsigned int >::const_iterator it=loc_items.begin(); it != loc_items.end(); ++it)
                    {
                        vector<unsigned int> indices;
                        topoMap->getFromIndex( indices, *it);
                        for( vector<unsigned int>::const_iterator itIndices = indices.begin(); itIndices != indices.end(); ++itIndices)
                        {
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

    int res = vitems.size();

    sofa::core::topology::TopologyModifier* topoMod;
    topo_curr->getContext()->get(topoMod);

    topoMod->removeItems(vitems);

    topoMod->notifyEndingEvent();

    topoMod->propagateTopologicalChanges();

    return res;
}

// Handle Removing of topological element (from any type of topology)
int TopologicalChangeManager::removeItemsFromCollisionModel(sofa::core::CollisionElementIterator elem2) const
{
    helper::vector<int> id;
    id.push_back(elem2.getIndex());
    return removeItemsFromCollisionModel(elem2.getCollisionModel(), id);
}

int TopologicalChangeManager::removeItemsFromCollisionModel(sofa::core::CollisionModel* model, const int& indices) const
{
    helper::vector<int> id;
    id.push_back(indices);
    return removeItemsFromCollisionModel(model, id);
}


int TopologicalChangeManager::removeItemsFromCollisionModel(sofa::core::CollisionModel* model, const helper::vector<int>& indices) const
{
    if(dynamic_cast<TriangleModel*>(model)!= NULL)
    {
        return removeItemsFromTriangleModel(static_cast<TriangleModel*>(model), indices);
    }
#if 0
    else if(dynamic_cast<TetrahedronModel*>(model)!= NULL)
    {
        return removeItemsFromTetrahedronModel(static_cast<TetrahedronModel*>(model), indices);
    }
#endif // if 0
    else if(dynamic_cast<SphereModel*>(model)!= NULL)
    {
        return removeItemsFromSphereModel(static_cast<SphereModel*>(model), indices);
    }
    else
        return 0;
}



// Handle Cutting (activated only for a triangular topology), using global variables to register the two last input points
bool TopologicalChangeManager::incisionCollisionModel(sofa::core::CollisionElementIterator elem, defaulttype::Vector3& pos, const bool firstInput, int snapingValue, int snapingBorderValue)
{
    Triangle triangle(elem);
    TriangleModel* model = triangle.getCollisionModel();

    if (model != NULL)
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
            bool isCut = this->incisionTriangleModel (model, incision.indexTriangle, incision.coordPoint, model, elem.getIndex(), pos, snapingValue, snapingBorderValue);

            if (isCut && !incision.firstCut) // cut has been reached, and will possible be continue. Stocking informations.
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
bool TopologicalChangeManager::incisionCollisionModel(sofa::core::CollisionModel *firstModel , unsigned int idxA, const Vector3& firstPoint,
        sofa::core::CollisionModel *secondModel, unsigned int idxB, const Vector3& secondPoint,
        int snapingValue, int snapingBorderValue)
{

    TriangleModel* firstCollisionModel = dynamic_cast< TriangleModel* >(firstModel);
    TriangleModel* secondCollisionModel = dynamic_cast< TriangleModel* >(secondModel);
    if (!firstCollisionModel || firstCollisionModel != secondCollisionModel) return false;
    return incisionTriangleModel(firstCollisionModel,  idxA, firstPoint,
            secondCollisionModel, idxB, secondPoint,
            snapingValue, snapingBorderValue);
}



// Perform incision in triangulation
bool TopologicalChangeManager::incisionTriangleModel(TriangleModel *firstModel , core::topology::BaseMeshTopology::TriangleID idxA, const Vector3& firstPoint,
        TriangleModel *secondModel, core::topology::BaseMeshTopology::TriangleID idxB, const Vector3& secondPoint,
        int snapingValue, int snapingBorderValue)
{

    // -- STEP 1: looking for collision model and topology components

    TriangleModel* firstCollisionModel = dynamic_cast< TriangleModel* >(firstModel);
    TriangleModel* secondCollisionModel = dynamic_cast< TriangleModel* >(secondModel);

    Triangle firstTriangle(firstCollisionModel, idxA);
    Triangle secondTriangle(secondCollisionModel, idxB);

    if (firstCollisionModel != secondCollisionModel)
    {
        msg_warning("TopologicalChangeManager") << "Incision involving different models is not supported yet!" ;
        return false;
    }


    sofa::core::topology::BaseMeshTopology* currentTopology = firstCollisionModel->getContext()->getMeshTopology();
    simulation::Node* collisionNode = dynamic_cast<simulation::Node*>(firstCollisionModel->getContext());

    // Test if a TopologicalMapping (by default from TetrahedronSetTopology to TriangleSetTopology) exists :
    std::vector< sofa::core::topology::TopologicalMapping *> listTopologicalMapping;
    collisionNode->get<sofa::core::topology::TopologicalMapping>(&listTopologicalMapping, core::objectmodel::BaseContext::Local);
    const bool isTopologicalMapping = !(listTopologicalMapping.empty());

    if (!isTopologicalMapping) // mapping not handle for the moment
    {

        // -- STEP 2: Try to catch the topology associated to the detected object (a TriangleSetTopology is expected)
        sofa::component::topology::TriangleSetTopologyContainer* triangleContainer;
        currentTopology->getContext()->get(triangleContainer);

        sofa::component::topology::TriangleSetTopologyModifier* triangleModifier;
        currentTopology->getContext()->get(triangleModifier);

        sofa::component::topology::TriangleSetTopologyAlgorithms<Vec3Types>* triangleAlgorithm;
        currentTopology->getContext()->get(triangleAlgorithm);

        sofa::component::topology::TriangleSetGeometryAlgorithms<Vec3Types>* triangleGeometry;
        currentTopology->getContext()->get(triangleGeometry);


        // -- STEP 3: Initialization

        // Mechanical coord of points a & b:
        Vector3 coord_a(firstPoint);
        Vector3 coord_b(secondPoint);


        // Path first point Indice. This might be useful if topology is in movement. (coord of point "a" doesn't belongs anymore to triangle of index : "idxA" since pickhandler)
        core::topology::BaseMeshTopology::PointID& last_indexPoint = incision.indexPoint;

        if(!incision.firstCut) //Not the first cut, look for new coord of a
        {
            core::behavior::MechanicalState<Vec3Types>* mstate = currentTopology->getContext()->get<core::behavior::MechanicalState<Vec3Types> >();
            const helper::vector<Vector3> &v_coords =  mstate->read(core::ConstVecCoordId::position())->getValue();
            coord_a = v_coords[last_indexPoint];
        }


        // Output declarations
        sofa::helper::vector< sofa::core::topology::TopologyObjectType> topoPath_list;
        sofa::helper::vector<unsigned int> indices_list;
        sofa::helper::vector< Vec<3, double> > coords2_list;

        // Snaping value: input are percentages, we need to transform it as real epsilon value;
        double epsilonSnap = (double)snapingValue/200;
        double epsilonBorderSnap = (double)snapingBorderValue/210; // magic number (0.5 is max value and must not be reached, as threshold is compared to barycoord value)


        // -- STEP 4: Creating path through different elements
        bool path_ok = triangleGeometry->computeIntersectedObjectsList(last_indexPoint, coord_a, coord_b, idxA, idxB, topoPath_list, indices_list, coords2_list);

        if (!path_ok)
        {
            dmsg_error("TopologicalChangeManager") << " in computeIntersectedObjectsList" ;
            return false;
        }


        // -- STEP 5: Splitting elements along path (incision path is stored inside "new_edges")
        sofa::helper::vector< unsigned int > new_edges;
        int result = triangleAlgorithm->SplitAlongPath(last_indexPoint, coord_a, core::topology::BaseMeshTopology::InvalidID, coord_b, topoPath_list, indices_list, coords2_list, new_edges, epsilonSnap, epsilonBorderSnap);

        if (result == -1)
        {
            incision.indexPoint = last_indexPoint;
            return false;
        }

        // -- STEP 6: Incise along new_edges path (i.e duplicating edges to create an incision)
        sofa::helper::vector<unsigned int> new_points;
        sofa::helper::vector<unsigned int> end_points;
        bool reachBorder = false;
        bool incision_ok =  triangleAlgorithm->InciseAlongEdgeList(new_edges, new_points, end_points, reachBorder);

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
        triangleModifier->propagateTopologicalChanges();
        triangleModifier->notifyEndingEvent();
        triangleModifier->propagateTopologicalChanges(); // needed?

        return true;
    }
    else
        return false;
}


} // namespace collision

} // namespace component

} // namespace sofa
