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
#include <sofa/component/topology/mapping/Tetra2TriangleTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/component/topology/container/dynamic/TriangleSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TriangleSetTopologyModifier.h>

#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyModifier.h>
#include <sofa/helper/AdvancedTimer.h>

#include <sofa/core/topology/TopologyChange.h>

#include <sofa/type/Vec.h>
#include <map>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/ScopedAdvancedTimer.h>


namespace sofa::component::topology::mapping
{

using namespace sofa::defaulttype;

using namespace sofa::component::topology::mapping;
using namespace sofa::core::topology;

// Register in the Factory
int Tetra2TriangleTopologicalMappingClass = core::RegisterObject("Special case of mapping where TetrahedronSetTopology is converted to TriangleSetTopology")
        .add< Tetra2TriangleTopologicalMapping >()

        ;

// Implementation

Tetra2TriangleTopologicalMapping::Tetra2TriangleTopologicalMapping()
    : sofa::core::topology::TopologicalMapping()
    , flipNormals(initData(&flipNormals, bool(false), "flipNormals", "Flip Normal ? (Inverse point order when creating triangle)"))
    , noNewTriangles(initData(&noNewTriangles, bool(false), "noNewTriangles", "If true no new triangles are being created"))
    , noInitialTriangles(initData(&noInitialTriangles, bool(false), "noInitialTriangles", "If true the list of initial triangles is initially empty. Only additional triangles will be added in the list"))
    , m_outTopoModifier(nullptr)
{
    m_inputType = geometry::ElementType::TETRAHEDRON;
    m_outputType = geometry::ElementType::TRIANGLE;
}

void Tetra2TriangleTopologicalMapping::init()
{
    if (!this->checkTopologyInputTypes()) // method will display error message if false
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    container::dynamic::TriangleSetTopologyModifier* to_tstm;
    toModel->getContext()->get(to_tstm);
    if (!to_tstm) 
    {
        msg_error() << "No TriangleSetTopologyModifier found in the output topology node '"
            << toModel->getContext()->getName() << "'.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
    else {
        m_outTopoModifier = to_tstm;
    }
    

    // INITIALISATION of Triangle mesh from Tetrahedral mesh :
    // Clear output topology
    toModel->clear();

    // Set the same number of points
    toModel->setNbPoints(fromModel->getNbPoints());

    // if no init triangle option (set output topology to empty)
    if (noInitialTriangles.getValue()){
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
        return;
    }

    // create topology maps and add triangle on border into output topology
    const auto & triangleArray = fromModel->getTriangles();
    const bool flipN = flipNormals.getValue();

    auto Loc2GlobVec = sofa::helper::getWriteOnlyAccessor(Loc2GlobDataVec);
    Loc2GlobVec.clear();
    Glob2LocMap.clear();

    for (Topology::TriangleID triId=0; triId<triangleArray.size(); ++triId)
    {
        if (fromModel->getTetrahedraAroundTriangle(triId).size() == 1)
        {
            const core::topology::BaseMeshTopology::Triangle& t = triangleArray[triId];
            if(flipN)
                toModel->addTriangle(t[0], t[2], t[1]);
            else
                toModel->addTriangle(t[0], t[1], t[2]);

            Loc2GlobVec.push_back(triId);
            Glob2LocMap[triId]= Loc2GlobVec.size() - 1;
        }
    }
    // Need to fully init the target topology
    toModel->init();

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}


Index Tetra2TriangleTopologicalMapping::getFromIndex(Index ind)
{

    if(fromModel->getTetrahedraAroundTriangle(ind).size()==1)
    {
        return fromModel->getTetrahedraAroundTriangle(ind)[0];
    }
    else
    {
        return 0;
    }
}

void Tetra2TriangleTopologicalMapping::updateTopologicalMappingTopDown()
{
    if (this->d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid)
        return;

    SCOPED_TIMER("Update Tetra2TriangleTopologicalMapping");

    auto itBegin=fromModel->beginChange();
    auto itEnd=fromModel->endChange();

    auto Loc2GlobVec = sofa::helper::getWriteAccessor(Loc2GlobDataVec);

    while( itBegin != itEnd )
    {
        TopologyChangeType changeType = (*itBegin)->getChangeType();
        std::string topoChangeType = "Tetra2TriangleTopologicalMapping - " + parseTopologyChangeTypeToString(changeType);
        helper::ScopedAdvancedTimer topoChangetimer(topoChangeType);

        switch( changeType )
        {

        case core::topology::ENDING_EVENT:
        {
            m_outTopoModifier->notifyEndingEvent();
            break;
        }

        case core::topology::TRIANGLESREMOVED:
        {
            const auto &triIDtoRemove = ( static_cast< const TrianglesRemoved *>( *itBegin ) )->getArray();

            // search for the list of triangles to remove in mapped topology
            sofa::type::vector< Index > triangles_to_remove;

            for (auto globTriId : triIDtoRemove)
            {
                auto iter_1 = Glob2LocMap.find(globTriId);
                if (iter_1 != Glob2LocMap.end())
                    triangles_to_remove.push_back(iter_1->second);
            }
            std::sort(triangles_to_remove.begin(), triangles_to_remove.end(), std::greater<unsigned int>());

            unsigned int lastGlobId = (unsigned int)fromModel->getNbTriangles() - 1;
            unsigned int nbrGlobElem = fromModel->getNbTriangles() - triIDtoRemove.size();
            unsigned int nbrBotElem = toModel->getNbTriangles() - triangles_to_remove.size();

            // update Glob2LocMap from fromModel changes
            for (auto oldGlobTriId : triIDtoRemove)
            {
                auto iter_last = Glob2LocMap.find(lastGlobId);
                if (iter_last != Glob2LocMap.end())
                {
                    // swap and pop glob map like fromModel triangle container
                    unsigned int lastLocId = iter_last->second;
                    Glob2LocMap[oldGlobTriId] = lastLocId;

                    if (lastLocId < Loc2GlobVec.size()) // could be maped to an already removed element in loc2Glob
                        Loc2GlobVec[lastLocId] = oldGlobTriId;
                }
                else
                {
                    iter_last = Glob2LocMap.find(oldGlobTriId);
                    if(iter_last == Glob2LocMap.end())
                    {
                        if (!noNewTriangles.getValue() && !noInitialTriangles.getValue()) // otherwise it is normal
                            msg_warning() << "Could not find last triangle id in Glob2LocMap: " << lastGlobId << " nor removed triangle id: " << oldGlobTriId;
                        lastGlobId--;
                        continue;
                    }
                }
                Glob2LocMap.erase(iter_last);
                lastGlobId--;
            }


            // update the maps from toModel changes
            for (auto oldLocTriId : triangles_to_remove)
            {
                //unsigned int oldGlobTriId = tmpLoc2Glob[oldLocTriId];
                unsigned int newGlobTriId = Loc2GlobVec.back();
                Loc2GlobVec[oldLocTriId] = newGlobTriId; // swap loc2Glob map like normal triangle removal

                if (newGlobTriId < nbrGlobElem && oldLocTriId < nbrBotElem){
                    Glob2LocMap[newGlobTriId] = oldLocTriId; // update Glob2LocMap of new loc ids
                }
                Loc2GlobVec.pop_back(); //pop last
            }

            m_outTopoModifier->removeTriangles(triangles_to_remove, true, false);

            break;
        }
        /**
        case core::topology::TRIANGLESADDED:
        {
            This case doesn't need to be handle here as TetraAdded case is emit first and handle new triangle to be added to output topology.
            break;
        }
        */
        case core::topology::TETRAHEDRAADDED:
        {
            if (noNewTriangles.getValue())
                break;

            const auto * tetraAdded = static_cast< const TetrahedraAdded *>( *itBegin );

            sofa::type::vector< BaseMeshTopology::Triangle > triangles_to_create;
            sofa::type::vector< BaseMeshTopology::TriangleID > triangleId_to_create;
            sofa::type::vector< BaseMeshTopology::TriangleID > triangleId_to_remove;

            // Need to first add all the new triangles before removing the old one.
            for (auto tetraId : tetraAdded->tetrahedronIndexArray)
            {
                const auto & triIntetra = fromModel->getTrianglesInTetrahedron(tetraId);
                for (auto triGlobId : triIntetra)
                {
                    auto iter_1 = Glob2LocMap.find(triGlobId);
                    const auto & tetraATri = fromModel->getTetrahedraAroundTriangle(triGlobId);
                    if (iter_1 != Glob2LocMap.end()) // in the map
                    {
                        if (tetraATri.size() != 1) // already in the map but not anymore on border, add it for later removal.
                            triangleId_to_remove.push_back(triGlobId);
                    }
                    else
                    {
                        if (tetraATri.size() > 1) // not in the map and not on border, nothing to do.
                            continue;

                        // not in the map but on border. Need to add this it.
                        const auto triangle = fromModel->getTriangle(triGlobId);
                        triangles_to_create.push_back(triangle);
                        triangleId_to_create.push_back((unsigned int)Loc2GlobVec.size());

                        Loc2GlobVec.push_back(triGlobId);
                        Glob2LocMap[triGlobId] = (unsigned int)Loc2GlobVec.size() - 1;
                    }
                }
            }

            // add new elements to output topology
            m_outTopoModifier->addTriangles(triangles_to_create);

            // remove elements not anymore on part of the border
            sofa::type::vector< BaseMeshTopology::TriangleID > local_triangleId_to_remove;
            for (auto triGlobId : triangleId_to_remove)
            {
                auto iter_1 = Glob2LocMap.find(triGlobId);
                if (iter_1 == Glob2LocMap.end())
                {
                    msg_error() << " in TETRAHEDRAADDED process, triangle id " << triGlobId << " not found in Glob2LocMap";
                    continue;
                }
                // add triangle for output topology update
                local_triangleId_to_remove.push_back(iter_1->second);
                Glob2LocMap.erase(iter_1);
            }

            // sort local triangle to remove
            std::sort(local_triangleId_to_remove.begin(), local_triangleId_to_remove.end(), std::greater<BaseMeshTopology::TriangleID>());

            for (auto triLocId : local_triangleId_to_remove)
            {
                BaseMeshTopology::TriangleID lastGlobId = Loc2GlobVec.back();

                // udpate loc2glob array
                Loc2GlobVec[triLocId] = lastGlobId;
                Loc2GlobVec.pop_back();

                // redirect glob2loc map
                Glob2LocMap[lastGlobId] = triLocId;
            }

            // remove old triangles
            m_outTopoModifier->removeTriangles(local_triangleId_to_remove, true, false);
            break;
        }
        case core::topology::TETRAHEDRAREMOVED:
        {
            if (noNewTriangles.getValue())
                break;

            const auto & tetrahedronArray=fromModel->getTetrahedra();
            const auto & tetraIds2Remove = ( static_cast< const TetrahedraRemoved *>( *itBegin ) )->getArray();

            sofa::type::vector< core::topology::BaseMeshTopology::Triangle > triangles_to_create;
            sofa::type::vector< unsigned int > trianglesIndexList;
            size_t nb_elems = toModel->getNbTriangles();
            const bool flipN = flipNormals.getValue();

            // For each tetrahedron removed inside the tetra2Remove array. Will look for each face if it shared with another tetrahedron.
            // If none, it means it will be added to the triangle border topoloy.
            // NB: doesn't check if triangle is inbetween 2 tetrahedra removed. This will be handle in TriangleRemoved event.
            for (Topology::TetrahedronID i = 0; i < tetraIds2Remove.size(); ++i)
            {
                Topology::TetrahedronID tetraId = tetraIds2Remove[i];
                const auto & triInTetra = fromModel->getTrianglesInTetrahedron(tetraId);

                // get each triangle of the tetrahedron involved
                for (auto triangleId : triInTetra)
                {
                    const auto & tetraATriangle = fromModel->getTetrahedraAroundTriangle(triangleId);

                    if (tetraATriangle.size() != 2) // means either more than 2 tetra sharing the triangle (so will not be on border) or only one, will be removed by TriangleRemoved later.
                        continue;

                    // Id of the opposite tetrahedron
                    Topology::TetrahedronID idOtherTetra;
                    if (tetraId == tetraATriangle[0])
                        idOtherTetra = tetraATriangle[1];
                    else
                        idOtherTetra = tetraATriangle[0];

                    // check if tetrahedron already processed in a previous iteration
                    bool is_present = false;
                    for (unsigned int k=0; k<i; ++k)
                        if (idOtherTetra == tetraIds2Remove[k])
                        {
                            is_present = true;
                            break;
                        }
                    if (is_present) // already done, continue.
                        continue;

                    core::topology::BaseMeshTopology::Triangle tri;
                    const auto & otherTetra = tetrahedronArray[idOtherTetra];
                    const auto & triInOtherTetra = fromModel->getTrianglesInTetrahedron(idOtherTetra);

                    int posInTetra = fromModel->getTriangleIndexInTetrahedron(triInOtherTetra, triangleId);

                    for (int ii = 0; ii < 3; ++ii)
                    {
                        unsigned int vIdInTetra = trianglesOrientationInTetrahedronArray[posInTetra][ii];
                        tri[ii] = otherTetra[vIdInTetra];
                    }

                    if(flipN)
                    {
                        unsigned int temp=tri[2];
                        tri[2]=tri[1];
                        tri[1]=temp;
                    }

                    // sort triangle such that tri[0] is the smallest one
                    while ((tri[0]>tri[1]) || (tri[0]>tri[2]))
                    {
                        int val=tri[0]; tri[0]=tri[1]; tri[1]=tri[2]; tri[2]=val;
                    }

                    // Add triangle to creation buffers
                    triangles_to_create.push_back(tri);
                    trianglesIndexList.push_back(nb_elems);
                    nb_elems+=1;

                    // update topology maps
                    Loc2GlobVec.push_back(triangleId);

                    // check if already exist
                    auto iter_1 = Glob2LocMap.find(triangleId);
                    if(iter_1 != Glob2LocMap.end() )
                        Glob2LocMap.erase(iter_1);

                    Glob2LocMap[triangleId] = (unsigned int)Loc2GlobVec.size()-1;
                }
            }

            m_outTopoModifier->addTriangles(triangles_to_create);
            break;
        }

        case core::topology::EDGESADDED:
        {
            const auto* edgeAdded = static_cast<const sofa::core::topology::EdgesAdded*>(*itBegin);
            m_outTopoModifier->addEdges(edgeAdded->edgeArray, edgeAdded->ancestorsList, edgeAdded->coefs);
            break;
        }

        case core::topology::POINTSADDED:
        {
            const auto* pointAdded = static_cast<const sofa::core::topology::PointsAdded*>(*itBegin);
            m_outTopoModifier->addPoints(sofa::Size(pointAdded->getNbAddedVertices()), pointAdded->ancestorsList, pointAdded->coefs, true);
            break;
        }

        case core::topology::POINTSREMOVED:
        {
            const auto pointRemoved = ( static_cast< const sofa::core::topology::PointsRemoved * >( *itBegin ) )->getArray();

            sofa::type::vector<Index> indices;

            for(unsigned int i = 0; i < pointRemoved.size(); ++i)
            {

                indices.push_back(pointRemoved[i]);
            }

            Topology::SetIndices & tab_indices = indices;

            m_outTopoModifier->removePoints(tab_indices, false);
            break;
        }

        case core::topology::POINTSRENUMBERING:
        {
            const Topology::SetIndices &tab = ( static_cast< const PointsRenumbering * >( *itBegin ) )->getIndexArray();
            const Topology::SetIndices &inv_tab = ( static_cast< const PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

            Topology::SetIndices indices;
            Topology::SetIndices inv_indices;


            for(unsigned int i = 0; i < tab.size(); ++i)
            {
                indices.push_back(tab[i]);
                inv_indices.push_back(inv_tab[i]);
            }

            Topology::SetIndices & tab_indices = indices;
            Topology::SetIndices & inv_tab_indices = inv_indices;

            m_outTopoModifier->renumberPoints(tab_indices, inv_tab_indices, false);
            break;
        }
        default:
            // Ignore events that are not Triangle  related.
            break;
        };

        ++itBegin;
    }    

}


bool Tetra2TriangleTopologicalMapping::checkTopologies()
{
    if (this->d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid)
        return false;

    // result of the method to be changed in case of error encountered
    bool allOk = true;

    const auto & triangleArray_top = fromModel->getTriangles();
    const auto & triangleArray_bot = toModel->getTriangles();
    const auto & buffer = Loc2GlobDataVec.getValue();

    dmsg_info() << "################# checkTopologies #####################";
    dmsg_info() << "triangleArray_bot.size(): " << triangleArray_bot.size();
    dmsg_info() << "Glob2LocMap.size(): " << Glob2LocMap.size();
    dmsg_info() << "Loc2GlobDataVec.size(): " << buffer.size();

    std::map<Index, Index>::iterator itM;
    for (size_t i=0; i<triangleArray_top.size(); i++)
    {
        const auto & tri = triangleArray_top[i];
        const auto & tetraATri = fromModel->getTetrahedraAroundTriangle(i);
        if (tetraATri.size() != 1)
            continue;

        itM = Glob2LocMap.find(i);
        if (itM == Glob2LocMap.end()){
            msg_error() << "Top triangle: " << i << " -> " << tri[0] << " " << tri[1] << " " << tri[2] << " NOT FOUND in Glob2LocMap";
            for (unsigned int k=0; k<triangleArray_bot.size(); k++)
            {
                const auto & triBot = triangleArray_bot[k];
                int cptFound = 0;
                for (unsigned int j=0; j<3; j++)
                {
                    if (triBot[j] == tri[0] || triBot[j] == tri[1] || triBot[j] == tri[2])
                        cptFound++;
                }

                if (cptFound == 3){
                    msg_error() << "Top triangle: " << i << " -> " << tri << " found at bottom id: " << k << " -> " << triBot << " | Loc2GlobDataVec: " << buffer[k];
                    break;
                }
            }

            allOk = false;
            continue;
        }

        unsigned int triLocID = (*itM).second;
        if (triLocID >= triangleArray_bot.size()){
            msg_error() << "## Glob2LocMap out of bounds: " << i << " - " << triLocID;
            allOk = false;
            continue;
        }
        const auto & tri2 = triangleArray_bot[triLocID];



        bool ok = false;
        for (int j=0; j<3; ++j)
        {
            ok = false;
            for (int k=0; k<3; ++k)
                if (tri[j] == tri2[k])
                {
                    ok = true;
                    break;
                }
            if (!ok)
                break;
        }

        if (!ok){
            msg_error() << "## Top Triangle Not same as bottom Triangle: ";
            msg_error() << "Top Triangle: " << i << " -> " << tri[0] << " " << tri[1] << " " << tri[2] << " --> Bottom Triangle: " << triLocID << " -> " << tri2[0] << " " << tri2[1] << " " << tri2[2];
            allOk = false;
        }

        if (buffer[triLocID] != i) {
            msg_error() << "## Maps no coherent: Loc2Glob: " << triLocID << " -> " << buffer[triLocID];
            allOk = false;
        }
    }

    dmsg_info() << "###############################################";
    return allOk;
}

} //namespace sofa::component::topology::mapping
