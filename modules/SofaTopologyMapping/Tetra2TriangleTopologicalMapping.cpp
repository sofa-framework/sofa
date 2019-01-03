/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaTopologyMapping/Tetra2TriangleTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>

#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetTopologyModifier.h>

#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
#include <SofaBaseTopology/TetrahedronSetTopologyModifier.h>
#include <sofa/helper/AdvancedTimer.h>

#include <sofa/core/topology/TopologyChange.h>

#include <sofa/defaulttype/Vec.h>
#include <map>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;

using namespace sofa::component::topology;
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
    , m_outTopoModifier(NULL)
{
}

Tetra2TriangleTopologicalMapping::~Tetra2TriangleTopologicalMapping()
{
}

void Tetra2TriangleTopologicalMapping::init()
{
    // recheck models
    bool modelsOk = true;
    if (!fromModel)
    {
        msg_error() << "Pointer to input topology is invalid.";
        modelsOk = false;
    }

    if (!toModel)
    {
        msg_error() << "Pointer to output topology is invalid.";
        modelsOk = false;
    }

    toModel->getContext()->get(m_outTopoModifier);
    if (!m_outTopoModifier)
    {
        msg_error() << "No TriangleSetTopologyModifier found in the Edge topology Node.";
        modelsOk = false;
    }

    if (!modelsOk)
    {
        this->m_componentstate = sofa::core::objectmodel::ComponentState::Invalid;
        return;
    }


    // INITIALISATION of Triangle mesh from Tetrahedral mesh :
    // Clear output topology
    toModel->clear();

    // Set the same number of points
    toModel->setNbPoints(fromModel->getNbPoints());

    // if no init triangle option (set output topology to empty)
    if (noInitialTriangles.getValue()){
        this->m_componentstate = sofa::core::objectmodel::ComponentState::Valid;
        return;
    }

    // create topology maps and add triangle on border into output topology
    const sofa::helper::vector<core::topology::BaseMeshTopology::Triangle>& triangleArray = fromModel->getTriangles();
    const bool flipN = flipNormals.getValue();

    sofa::helper::vector <unsigned int>& Loc2GlobVec = *(Loc2GlobDataVec.beginEdit());

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

    Loc2GlobDataVec.endEdit();
    this->m_componentstate = sofa::core::objectmodel::ComponentState::Valid;
}


unsigned int Tetra2TriangleTopologicalMapping::getFromIndex(unsigned int ind)
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

const unsigned int trianglesInTetrahedronArray[4][3]= {{1,2,3}, {0,3,2}, {1,3,0},{0,2,1}};

void Tetra2TriangleTopologicalMapping::updateTopologicalMappingTopDown()
{
    if (this->m_componentstate != sofa::core::objectmodel::ComponentState::Valid)
        return;

    sofa::helper::AdvancedTimer::stepBegin("Update Tetra2TriangleTopologicalMapping");

    std::list<const TopologyChange *>::const_iterator itBegin=fromModel->beginChange();
    std::list<const TopologyChange *>::const_iterator itEnd=fromModel->endChange();

    sofa::helper::vector <unsigned int>& Loc2GlobVec = *(Loc2GlobDataVec.beginEdit());

    while( itBegin != itEnd )
    {
        TopologyChangeType changeType = (*itBegin)->getChangeType();
        std::string topoChangeType = "Triangle2EdgeTopologicalMapping - " + parseTopologyChangeTypeToString(changeType);
        sofa::helper::AdvancedTimer::stepBegin(topoChangeType);

        switch( changeType )
        {

        case core::topology::ENDING_EVENT:
        {
            m_outTopoModifier->propagateTopologicalChanges();
            m_outTopoModifier->notifyEndingEvent();
            m_outTopoModifier->propagateTopologicalChanges();
            break;
        }

        case core::topology::TRIANGLESREMOVED:
        {
            const sofa::helper::vector<unsigned int> &triIDtoRemove = ( static_cast< const TrianglesRemoved *>( *itBegin ) )->getArray();

            // search for the list of triangles to remove in mapped topology
            sofa::helper::vector< unsigned int > triangles_to_remove;
            std::map <unsigned int, unsigned int> tmpLoc2Glob;
            for (auto globTriId : triIDtoRemove)
            {
                std::map<unsigned int, unsigned int>::iterator iter_1 = Glob2LocMap.find(globTriId);
                if (iter_1 != Glob2LocMap.end())
                    triangles_to_remove.push_back(iter_1->second);
                tmpLoc2Glob.insert(std::pair<unsigned int, unsigned int>(iter_1->second, globTriId));
            }

            std::sort(triangles_to_remove.begin(), triangles_to_remove.end(), std::greater<unsigned int>());
            m_outTopoModifier->removeTriangles(triangles_to_remove, true, false);

            // update the maps from toModel changes
            for (auto oldLocTriId : triangles_to_remove)
            {
                //unsigned int oldGlobTriId = tmpLoc2Glob[oldLocTriId];
                unsigned int newGlobTriId = Loc2GlobVec.back();
                Loc2GlobVec[oldLocTriId] = newGlobTriId; // swap loc2Glob map
                Loc2GlobVec.pop_back(); //pop last

                Glob2LocMap[newGlobTriId] = oldLocTriId; // update Glob2LocMap of new loc ids
            }

            unsigned int lastGlobId = (unsigned int)fromModel->getNbTriangles() - 1;
            // update Glob2LocMap from fromModel changes
            for (auto oldGlobTriId : triIDtoRemove)
            {
                std::map<unsigned int, unsigned int>::iterator iter_last = Glob2LocMap.find(lastGlobId);
                if (iter_last != Glob2LocMap.end())
                {
                    // swap and pop glob map like fromModel triangle container
                    unsigned int lastLocId = iter_last->second;
                    Glob2LocMap[oldGlobTriId] = lastLocId;

                    if (lastLocId < Loc2GlobVec.size()) // could map to already removed element in loc2Glob
                        Loc2GlobVec[iter_last->second] = oldGlobTriId;
                }
                else
                {
                    iter_last = Glob2LocMap.find(oldGlobTriId);
                    if(iter_last == Glob2LocMap.end())
                    {
                        msg_error() << "Could not find last triangle id in Glob2LocMap: " << lastGlobId << " nor removed triangle id: " << oldGlobTriId;
                        continue;
                    }
                }

                Glob2LocMap.erase(iter_last);
                lastGlobId--;
            }

            break;
        }

        case core::topology::TRIANGLESADDED:
        {
            const sofa::helper::vector<unsigned int> &tab = ( static_cast< const TrianglesAdded *>( *itBegin ) )->getArray();

            const sofa::helper::vector<core::topology::BaseMeshTopology::Tetrahedron> &tetrahedronArray=fromModel->getTetrahedra();
            sofa::helper::vector< core::topology::BaseMeshTopology::Triangle > triangles_to_create;
            sofa::helper::vector< unsigned int > trianglesIndexList;
            unsigned int nb_elems = (unsigned int)toModel->getNbTriangles();
            const bool flipN = flipNormals.getValue();

            for (unsigned int i = 0; i <tab.size(); ++i)
            {
                core::topology::BaseMeshTopology::Triangle t;
                t=fromModel->getTriangle(tab[i]);
                const core::topology::BaseMeshTopology::TetrahedraAroundTriangle tetraId=fromModel->getTetrahedraAroundTriangle(tab[i]);

                if(tetraId.size()==1)
                {
                    std::map<unsigned int, unsigned int>::iterator iter_1 = Glob2LocMap.find(tab[i]);
                    if(iter_1 != Glob2LocMap.end() )
                    {
                        msg_error() << "updateTopologicalMappingTopDown::TRIANGLESADDED - fail to add triangle " << tab[i] << "which already exists";
                    }
                    else
                    {
                        core::topology::BaseMeshTopology::Tetrahedron te=tetrahedronArray[tetraId[0]];

                        for(int j=0; j<4; j++)
                        {
                            bool flag=true;
                            for(int k=0; k<3; k++)
                            {
                                if(t[k]==te[j])
                                {
                                    flag=false;
                                    break;
                                }
                            }
                            if(flag)
                            {
                                if ((j%2))
                                {
                                    t[0]=(int)(te[(j+1)%4]); t[1]=(int)(te[(j+2)%4]); t[2]=(int)(te[(j+3)%4]);
                                }
                                else
                                {
                                    t[0]=(int)(te[(j+1)%4]); t[2]=(int)(te[(j+2)%4]); t[1]=(int)(te[(j+3)%4]);
                                }
                                if(flipN)
                                {
                                    unsigned int temp=t[2];
                                    t[2]=t[1];
                                    t[1]=temp;
                                }
                            }
                        }

                        // sort t such that t[0] is the smallest one
                        while ((t[0]>t[1]) || (t[0]>t[2]))
                        {
                            int val=t[0]; t[0]=t[1]; t[1]=t[2]; t[2]=val;
                        }

                        triangles_to_create.push_back(t);
                        trianglesIndexList.push_back(nb_elems);
                        //addedTriangleIndex.push_back(nb_elems);
                        nb_elems+=1;

                        Loc2GlobVec.push_back(tab[i]);
                        Glob2LocMap[tab[i]]= (unsigned int)Loc2GlobVec.size()-1;
                    }
                }
            }

            m_outTopoModifier->addTrianglesProcess(triangles_to_create) ;
            m_outTopoModifier->addTrianglesWarning(triangles_to_create.size(), triangles_to_create, trianglesIndexList) ;
            break;
        }

        case core::topology::TETRAHEDRAADDED:
        {
            if (noNewTriangles.getValue()==false)
            {
                //const sofa::helper::vector<Tetrahedron> &tetrahedronArray=fromModel->getTetrahedra();
                const sofa::helper::vector<unsigned int> &tab = ( static_cast< const TetrahedraAdded *>( *itBegin ) )->getArray();

                sofa::helper::vector< core::topology::BaseMeshTopology::TriangleID > triangles_to_remove;
                //int nb_elems = toModel->getNbTriangles();

                for (unsigned int i = 0; i < tab.size(); ++i)
                {
                    for (unsigned int j = 0; j < 4; ++j)
                    {
                        unsigned int k = (fromModel->getTrianglesInTetrahedron(tab[i]))[j];
                        if (fromModel->getTetrahedraAroundTriangle(k).size()==1)
                        {
                            //do nothing
                        }
                        else
                        {
                            bool flag=true;
                            for(unsigned int m=0; m<triangles_to_remove.size(); m++)
                            {
                                if(k==triangles_to_remove[m])
                                {
                                    flag=false;
                                    break;
                                }
                            }
                            if(flag)
                                triangles_to_remove.push_back(k);
                        }
                    }
                }

                unsigned int ind_last = (unsigned int)toModel->getNbTriangles();
                unsigned int ind_tmp;

                for (unsigned int i = 0; i <triangles_to_remove.size(); ++i)
                {
                    unsigned int k = triangles_to_remove[i];

                    std::map<unsigned int, unsigned int>::iterator iter_1 = Glob2LocMap.find(k);
                    if(iter_1 != Glob2LocMap.end())
                    {

                        ind_last = ind_last - 1;
                        unsigned int ind_k = Glob2LocMap[k];

                        if (ind_k != ind_last)
                        {

                            Glob2LocMap.erase(Glob2LocMap.find(Loc2GlobVec[ind_last]));
                            Glob2LocMap[Loc2GlobVec[ind_last]] = ind_k;

                            Glob2LocMap.erase(Glob2LocMap.find(Loc2GlobVec[ind_k]));
                            Glob2LocMap[Loc2GlobVec[ind_k]] = ind_last;

                            ind_tmp = Loc2GlobVec[ind_k];
                            Loc2GlobVec[ind_k] = Loc2GlobVec[ind_last];
                            Loc2GlobVec[ind_last] = ind_tmp;

                        }

                        Glob2LocMap.erase(Glob2LocMap.find(Loc2GlobVec[Loc2GlobVec.size() - 1]));
                        Loc2GlobVec.resize( Loc2GlobVec.size() - 1 );

                        sofa::helper::vector< unsigned int > triangle_to_remove;
                        triangle_to_remove.push_back(ind_k);

                        TriangleSetTopologyModifier *triangleMod;
                        toModel->getContext()->get(triangleMod);
                        triangleMod->removeTriangles(triangle_to_remove, true, false);

                        for(unsigned int i=0; i<addedTriangleIndex.size(); i++)
                        {
                            if(addedTriangleIndex[i]==ind_k)
                                addedTriangleIndex[i]=0;
                            if(addedTriangleIndex[i]>ind_k)
                                addedTriangleIndex[i]-=1;
                        }

                    }
                    else
                    {
                        if(CHECK_TOPOLOGY)
                        {
                            msg_info() << "updateTopologicalMappingTopDown::TETRAHEDRAADDED - Glob2LocMap should have the visible triangle " << triangles_to_remove[i];
                            msg_info() << "updateTopologicalMappingTopDown::TETRAHEDRAADDED - nb triangles = " << ind_last;
                        }
                    }
                }
            }
            break;
        }

        case core::topology::TETRAHEDRAREMOVED:
        {
            if (noNewTriangles.getValue())
                break;

            const sofa::helper::vector<core::topology::BaseMeshTopology::Tetrahedron>& tetrahedronArray=fromModel->getTetrahedra();
            const sofa::helper::vector<Topology::TetrahedronID> &tetra2Remove = ( static_cast< const TetrahedraRemoved *>( *itBegin ) )->getArray();

            sofa::helper::vector< core::topology::BaseMeshTopology::Triangle > triangles_to_create;
            sofa::helper::vector< unsigned int > trianglesIndexList;
            size_t nb_elems = toModel->getNbTriangles();
            const bool flipN = flipNormals.getValue();

            // For each tetrahedron removed inside the tetra2Remove array. Will look for each face if it shared with another tetrahedron.
            // If none, it means it will be added to the triangle border topoloy.
            // NB: doesn't check if triangle is inbetween 2 tetrahedra removed. This will be handle in TriangleRemoved event.
            for (Topology::TetrahedronID i = 0; i < tetra2Remove.size(); ++i)
            {
                Topology::TetrahedronID tetraId = tetra2Remove[i];
                const BaseMeshTopology::TrianglesInTetrahedron& triInTetra = fromModel->getTrianglesInTetrahedron(tetraId);

                // get each triangle of the tetrahedron involved
                for (auto triangleId : triInTetra)
                {
                    const BaseMeshTopology::TetrahedraAroundTriangle& tetraATriangle = fromModel->getTetrahedraAroundTriangle(triangleId);

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
                        if (idOtherTetra == tetra2Remove[k])
                        {
                            is_present = true;
                            break;
                        }
                    if (is_present) // already done, continue.
                        continue;

                    core::topology::BaseMeshTopology::Triangle tri;
                    const core::topology::BaseMeshTopology::Tetrahedron& otherTetra = tetrahedronArray[idOtherTetra];
                    const BaseMeshTopology::TrianglesInTetrahedron& triInOtherTetra = fromModel->getTrianglesInTetrahedron(idOtherTetra);

                    int posInTetra = fromModel->getTriangleIndexInTetrahedron(triInOtherTetra, triangleId);

                    for (int i=0; i<3; i++)
                    {
                        unsigned int vIdInTetra = trianglesInTetrahedronArray[posInTetra][i];
                        tri[i] = otherTetra[vIdInTetra];
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
                    std::map<unsigned int, unsigned int>::iterator iter_1 = Glob2LocMap.find(triangleId);
                    if(iter_1 != Glob2LocMap.end() )
                    {
                        msg_error() << "Fail to add triangle " << triangleId << "which already exists with value: " << (*iter_1).second;
                        Glob2LocMap.erase(iter_1);
                    }
                    Glob2LocMap[triangleId] = (unsigned int)Loc2GlobVec.size()-1;
                }
            }

            m_outTopoModifier->addTrianglesProcess(triangles_to_create) ;
            m_outTopoModifier->addTrianglesWarning(triangles_to_create.size(), triangles_to_create, trianglesIndexList) ;
            m_outTopoModifier->propagateTopologicalChanges();

            break;
        }

        case core::topology::EDGESADDED:
        {
            const EdgesAdded *ea=static_cast< const EdgesAdded * >( *itBegin );
            m_outTopoModifier->addEdgesProcess(ea->edgeArray);
            m_outTopoModifier->addEdgesWarning(ea->nEdges,ea->edgeArray,ea->edgeIndexArray);
            break;
        }

        case core::topology::POINTSADDED:
        {
            size_t nbAddedPoints = ( static_cast< const sofa::component::topology::PointsAdded * >( *itBegin ) )->getNbAddedVertices();
            m_outTopoModifier->addPointsProcess(nbAddedPoints);
            m_outTopoModifier->addPointsWarning(nbAddedPoints, true);
            break;
        }

        case core::topology::POINTSREMOVED:
        {
            const sofa::helper::vector<unsigned int> tab = ( static_cast< const sofa::component::topology::PointsRemoved * >( *itBegin ) )->getArray();

            sofa::helper::vector<unsigned int> indices;

            for(unsigned int i = 0; i < tab.size(); ++i)
            {

                indices.push_back(tab[i]);
            }

            sofa::helper::vector<unsigned int>& tab_indices = indices;

            m_outTopoModifier->removePointsWarning(tab_indices, false);

            m_outTopoModifier->propagateTopologicalChanges();
            m_outTopoModifier->removePointsProcess(tab_indices, false);
            break;
        }

        case core::topology::POINTSRENUMBERING:
        {
            const sofa::helper::vector<unsigned int> &tab = ( static_cast< const PointsRenumbering * >( *itBegin ) )->getIndexArray();
            const sofa::helper::vector<unsigned int> &inv_tab = ( static_cast< const PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

            sofa::helper::vector<unsigned int> indices;
            sofa::helper::vector<unsigned int> inv_indices;

            for(unsigned int i = 0; i < tab.size(); ++i)
            {

                //msg_info() << " : Hexa2QuadTopologicalMapping - point = " << tab[i];
                indices.push_back(tab[i]);
                inv_indices.push_back(inv_tab[i]);
            }

            sofa::helper::vector<unsigned int>& tab_indices = indices;
            sofa::helper::vector<unsigned int>& inv_tab_indices = inv_indices;

            m_outTopoModifier->renumberPointsWarning(tab_indices, inv_tab_indices, false);
            m_outTopoModifier->propagateTopologicalChanges();
            m_outTopoModifier->renumberPointsProcess(tab_indices, inv_tab_indices, false);

            break;
        }
        default:
            // Ignore events that are not Triangle  related.
            break;
        };

        sofa::helper::AdvancedTimer::stepEnd(topoChangeType);
        ++itBegin;
    }
    m_outTopoModifier->propagateTopologicalChanges();
    Loc2GlobDataVec.endEdit();

    sofa::helper::AdvancedTimer::stepEnd("Update Tetra2TriangleTopologicalMapping");
}


} // namespace topology

} // namespace component

} // namespace sofa
