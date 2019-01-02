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

    if (!modelsOk)
    {
        this->m_componentstate = sofa::core::objectmodel::ComponentState::Invalid;
        return;
    }


    TriangleSetTopologyContainer *to_tstc;
    toModel->getContext()->get(to_tstc);
    to_tstc->clear();

    toModel->setNbPoints(fromModel->getNbPoints());

    TriangleSetTopologyModifier *to_tstm;
    toModel->getContext()->get(to_tstm);

    const sofa::helper::vector<core::topology::BaseMeshTopology::Triangle> &triangleArray=fromModel->getTriangles();
    const bool flipN = flipNormals.getValue();

    /// only initialize with border triangles if necessary
    if (noInitialTriangles.getValue()==false)
    {

        sofa::helper::vector <unsigned int>& Loc2GlobVec = *(Loc2GlobDataVec.beginEdit());

        Loc2GlobVec.clear();
        Glob2LocMap.clear();

        for (unsigned int i=0; i<triangleArray.size(); ++i)
        {

            if (fromModel->getTetrahedraAroundTriangle(i).size()==1)
            {
                if(flipN)
                {
                    core::topology::BaseMeshTopology::Triangle t = triangleArray[i];
                    unsigned int tmp = t[2];
                    t[2] = t[1];
                    t[1] = tmp;
                    to_tstm->addTriangleProcess(t);
                }
                else
                    to_tstm->addTriangleProcess(triangleArray[i]);

                Loc2GlobVec.push_back(i);
                Glob2LocMap[i]= (unsigned int)Loc2GlobVec.size()-1;
            }
        }

        //to_tstm->propagateTopologicalChanges();
        to_tstm->notifyEndingEvent();
        //to_tstm->propagateTopologicalChanges();
        Loc2GlobDataVec.endEdit();
    }
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

void Tetra2TriangleTopologicalMapping::updateTopologicalMappingTopDown()
{
    if (this->m_componentstate != sofa::core::objectmodel::ComponentState::Valid)
        return;

    sofa::helper::AdvancedTimer::stepBegin("Update Tetra2TriangleTopologicalMapping");

    TriangleSetTopologyModifier *to_tstm;
    toModel->getContext()->get(to_tstm);

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
            to_tstm->propagateTopologicalChanges();
            to_tstm->notifyEndingEvent();
            to_tstm->propagateTopologicalChanges();
            break;
        }

        case core::topology::TRIANGLESREMOVED:
        {
            unsigned int last = (unsigned int)fromModel->getNbTriangles() - 1;
            unsigned int ind_last = (unsigned int)toModel->getNbTriangles() - 1;

            const sofa::helper::vector<unsigned int> &triIDtoRemove = ( static_cast< const TrianglesRemoved *>( *itBegin ) )->getArray();

            unsigned int ind_tmp;
            unsigned int ind_real_last;

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
            to_tstm->removeTriangles(triangles_to_remove, true, false);

            // update the maps from toModel changes
            for (auto oldLocTriId : triangles_to_remove)
            {
                //unsigned int oldGlobTriId = tmpLoc2Glob[oldLocTriId];

                unsigned int newGlobTriId = Loc2GlobVec.back();
                Loc2GlobVec[oldLocTriId] = newGlobTriId; // swap loc2Glob map
                Loc2GlobVec.pop_back(); //pop last

                Glob2LocMap[newGlobTriId] = oldLocTriId; // update Glob2LocMap of new loc ids
            }

            // update Glob2LocMap from fromModel changes
            for (auto oldGlobTriId : triIDtoRemove)
            {
                std::map<unsigned int, unsigned int>::iterator iter_1 = Glob2LocMap.find(oldGlobTriId);
                Glob2LocMap.erase(iter_1); // pop last of glob map
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

            to_tstm->addTrianglesProcess(triangles_to_create) ;
            to_tstm->addTrianglesWarning(triangles_to_create.size(), triangles_to_create, trianglesIndexList) ;
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

            if (noNewTriangles.getValue()==false)
            {
                const sofa::helper::vector<core::topology::BaseMeshTopology::Tetrahedron> &tetrahedronArray=fromModel->getTetrahedra();
                const sofa::helper::vector<unsigned int> &tab = ( static_cast< const TetrahedraRemoved *>( *itBegin ) )->getArray();

                sofa::helper::vector< core::topology::BaseMeshTopology::Triangle > triangles_to_create;
                sofa::helper::vector< unsigned int > trianglesIndexList;
                unsigned int nb_elems = (unsigned int)toModel->getNbTriangles();
                const bool flipN = flipNormals.getValue();

                for (unsigned int i = 0; i < tab.size(); ++i)
                {

                    for (unsigned int j = 0; j < 4; ++j)
                    {
                        unsigned int k = (fromModel->getTrianglesInTetrahedron(tab[i]))[j];

                        if (fromModel->getTetrahedraAroundTriangle(k).size()==1)   // remove as visible the triangle indexed by k
                        {

                            // do nothing

                        }
                        else if(fromModel->getTetrahedraAroundTriangle(k).size()==2)
                        {

                            unsigned int ind_test;
                            if(tab[i] == fromModel->getTetrahedraAroundTriangle(k)[0])
                            {

                                ind_test = fromModel->getTetrahedraAroundTriangle(k)[1];

                            }
                            else   // tab[i] == fromModel->getTetrahedraAroundTriangle(k)[1]
                            {

                                ind_test = fromModel->getTetrahedraAroundTriangle(k)[0];
                            }

                            bool is_present = false;
                            unsigned int k0 = 0;

                            // HD may be a buf here k0<tab.size()
                            while((!is_present) && k0 < i)
                            {
                                is_present = (ind_test == tab[k0]);
                                k0+=1;
                            }

                            int nb=0;
                            for(k0=0; k0<tab.size(); k0++)
                            {
                                if(ind_test==tab[k0])
                                    nb++;
                                if(nb==2)
                                {
                                    is_present=false;
                                    break;
                                }
                            }
                            if(!is_present)
                            {

                                core::topology::BaseMeshTopology::Triangle t;
                                const core::topology::BaseMeshTopology::Tetrahedron &te=tetrahedronArray[ind_test];

                                int h = fromModel->getTriangleIndexInTetrahedron(fromModel->getTrianglesInTetrahedron(ind_test),k);

                                if ((h%2) && (!flipN))
                                {
                                    t[0]=(int)(te[(h+1)%4]); t[1]=(int)(te[(h+2)%4]); t[2]=(int)(te[(h+3)%4]);
                                }
                                else
                                {
                                    t[0]=(int)(te[(h+1)%4]); t[2]=(int)(te[(h+2)%4]); t[1]=(int)(te[(h+3)%4]);
                                }

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

                                nb_elems+=1;

                                Loc2GlobVec.push_back(k);
                                std::map<unsigned int, unsigned int>::iterator iter_1 = Glob2LocMap.find(k);
                                if(iter_1 != Glob2LocMap.end() )
                                {
                                    msg_error() << "updateTopologicalMappingTopDown::TETRAHEDRAREMOVED - fail to add triangle " << k << "which already exists.";
                                    Glob2LocMap.erase(Glob2LocMap.find(k));
                                }
                                else
                                {
                                    addedTriangleIndex.push_back(nb_elems-1);
                                }
                                Glob2LocMap[k]= (unsigned int)Loc2GlobVec.size()-1;
                            }
                        }
                    }
                }

                to_tstm->addTrianglesProcess(triangles_to_create) ;
                to_tstm->addTrianglesWarning(triangles_to_create.size(), triangles_to_create, trianglesIndexList) ;
            }

            break;

        }

        case core::topology::EDGESADDED:
        {
            const EdgesAdded *ea=static_cast< const EdgesAdded * >( *itBegin );
            to_tstm->addEdgesProcess(ea->edgeArray);
            to_tstm->addEdgesWarning(ea->nEdges,ea->edgeArray,ea->edgeIndexArray);
            break;
        }

        case core::topology::POINTSADDED:
        {
            size_t nbAddedPoints = ( static_cast< const sofa::component::topology::PointsAdded * >( *itBegin ) )->getNbAddedVertices();
            to_tstm->addPointsProcess(nbAddedPoints);
            to_tstm->addPointsWarning(nbAddedPoints, true);
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

            to_tstm->removePointsWarning(tab_indices, false);

            to_tstm->propagateTopologicalChanges();
            to_tstm->removePointsProcess(tab_indices, false);
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

            to_tstm->renumberPointsWarning(tab_indices, inv_tab_indices, false);
            to_tstm->propagateTopologicalChanges();
            to_tstm->renumberPointsProcess(tab_indices, inv_tab_indices, false);

            break;
        }
        default:
            // Ignore events that are not Triangle  related.
            break;
        };

        sofa::helper::AdvancedTimer::stepEnd(topoChangeType);
        ++itBegin;
    }
    to_tstm->propagateTopologicalChanges();
    Loc2GlobDataVec.endEdit();

    sofa::helper::AdvancedTimer::stepEnd("Update Tetra2TriangleTopologicalMapping");
}


} // namespace topology

} // namespace component

} // namespace sofa
