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
#include <SofaTopologyMapping/Triangle2EdgeTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>

#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <SofaBaseTopology/EdgeSetTopologyModifier.h>

#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetTopologyModifier.h>

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

SOFA_DECL_CLASS(Triangle2EdgeTopologicalMapping)

// Register in the Factory
int Triangle2EdgeTopologicalMappingClass = core::RegisterObject("Special case of mapping where TriangleSetTopology is converted to EdgeSetTopology")
        .add< Triangle2EdgeTopologicalMapping >()

        ;

Triangle2EdgeTopologicalMapping::Triangle2EdgeTopologicalMapping()
{
}


Triangle2EdgeTopologicalMapping::~Triangle2EdgeTopologicalMapping()
{
}

void Triangle2EdgeTopologicalMapping::init()
{
    sout << "INFO_print : init Triangle2EdgeTopologicalMapping" << sendl;

    // INITIALISATION of EDGE mesh from TRIANGULAR mesh :

    if (fromModel)
    {

        sout << "INFO_print : Triangle2EdgeTopologicalMapping - from = triangle" << sendl;

        if (toModel)
        {

            sout << "INFO_print : Triangle2EdgeTopologicalMapping - to = edge" << sendl;

            EdgeSetTopologyContainer *to_tstc;
            toModel->getContext()->get(to_tstc);
            to_tstc->clear();

            toModel->setNbPoints(fromModel->getNbPoints());


            EdgeSetTopologyModifier *to_tstm;
            toModel->getContext()->get(to_tstm);

            const sofa::helper::vector<core::topology::BaseMeshTopology::Edge> &edgeArray=fromModel->getEdges();

            sofa::helper::vector <unsigned int>& Loc2GlobVec = *(Loc2GlobDataVec.beginEdit());

            Loc2GlobVec.clear();
            Glob2LocMap.clear();

            for (unsigned int i=0; i<edgeArray.size(); ++i)
            {
                if (fromModel->getTrianglesAroundEdge(i).size()==1)
                {

                    to_tstm->addEdgeProcess(edgeArray[i]);

                    Loc2GlobVec.push_back(i);
                    Glob2LocMap[i]=Loc2GlobVec.size()-1;
                }
            }

            //to_tstm->propagateTopologicalChanges();
            to_tstm->notifyEndingEvent();
            //to_tstm->propagateTopologicalChanges();
            Loc2GlobDataVec.endEdit();
        }
    }
}

unsigned int Triangle2EdgeTopologicalMapping::getFromIndex(unsigned int ind)
{

    if(fromModel->getTrianglesAroundEdge(ind).size()==1)
    {
        return fromModel->getTrianglesAroundEdge(ind)[0];
    }
    else
    {
        return 0;
    }
}

void Triangle2EdgeTopologicalMapping::updateTopologicalMappingTopDown()
{

    // INITIALISATION of EDGE mesh from TRIANGULAR mesh :

    if (fromModel)
    {

        EdgeSetTopologyModifier *to_tstm;
        toModel->getContext()->get(to_tstm);

        if (toModel)
        {

            std::list<const TopologyChange *>::const_iterator itBegin=fromModel->beginChange();
            std::list<const TopologyChange *>::const_iterator itEnd=fromModel->endChange();

            sofa::helper::vector <unsigned int>& Loc2GlobVec = *(Loc2GlobDataVec.beginEdit());

            while( itBegin != itEnd )
            {
                TopologyChangeType changeType = (*itBegin)->getChangeType();

                switch( changeType )
                {

                case core::topology::ENDING_EVENT:
                {
                    //sout << "INFO_print : Triangle2EdgeTopologicalMapping - ENDING_EVENT" << sendl;
                    to_tstm->propagateTopologicalChanges();
                    to_tstm->notifyEndingEvent();
                    to_tstm->propagateTopologicalChanges();
                    break;
                }

                case core::topology::EDGESREMOVED:
                {
                    //sout << "INFO_print : Triangle2EdgeTopologicalMapping - EDGESREMOVED" << sendl;

                    int last;
                    int ind_last;

                    last= fromModel->getNbEdges() - 1;

                    const sofa::helper::vector<unsigned int> &tab = ( static_cast< const EdgesRemoved *>( *itBegin ) )->getArray();

                    unsigned int ind_tmp;

                    unsigned int ind_real_last;
                    ind_last=toModel->getNbEdges();

                    for (unsigned int i = 0; i <tab.size(); ++i)
                    {
                        unsigned int k = tab[i];

                        std::map<unsigned int, unsigned int>::iterator iter_1 = Glob2LocMap.find(k);
                        if(iter_1 != Glob2LocMap.end())
                        {

                            ind_last = ind_last - 1;

                            unsigned int ind_k = Glob2LocMap[k];
//                            ind_real_last = ind_k;

                            std::map<unsigned int, unsigned int>::iterator iter_2 = Glob2LocMap.find(last);
                            if(iter_2 != Glob2LocMap.end())
                            {

                                ind_real_last = Glob2LocMap[last];

                                if((int) k != last)
                                {

                                    Glob2LocMap.erase(Glob2LocMap.find(k));
                                    Glob2LocMap[k] = ind_real_last;

                                    Glob2LocMap.erase(Glob2LocMap.find(last));
                                    Glob2LocMap[last] = ind_k;

                                    ind_tmp = Loc2GlobVec[ind_real_last];
                                    Loc2GlobVec[ind_real_last] = Loc2GlobVec[ind_k];
                                    Loc2GlobVec[ind_k] = ind_tmp;
                                }
                            }

                            if((int) ind_k != ind_last)
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

                            sofa::helper::vector< unsigned int > edges_to_remove;
                            edges_to_remove.push_back(ind_k);
                            to_tstm->removeEdges(edges_to_remove, false);

                        }
                        else
                        {

                            sout << "INFO_print : Triangle2EdgeTopologicalMapping - Glob2LocMap should have the visible edge " << tab[i] << sendl;
                            sout << "INFO_print : Triangle2EdgeTopologicalMapping - nb edges = " << ind_last << sendl;
                        }

                        --last;
                    }

                    //to_tstm->propagateTopologicalChanges();

                    break;
                }

                case core::topology::TRIANGLESREMOVED:
                {

                    //sout << "INFO_print : Triangle2EdgeTopologicalMapping - TRIANGLESREMOVED" << sendl;

                    if (fromModel)
                    {

                        const sofa::helper::vector<core::topology::BaseMeshTopology::Triangle> &triangleArray=fromModel->getTriangles();

                        const sofa::helper::vector<unsigned int> &tab = ( static_cast< const TrianglesRemoved *>( *itBegin ) )->getArray();

                        sofa::helper::vector< core::topology::BaseMeshTopology::Edge > edges_to_create;
                        sofa::helper::vector< unsigned int > edgesIndexList;
                        int nb_elems = toModel->getNbEdges();

                        for (unsigned int i = 0; i < tab.size(); ++i)
                        {
                            for (unsigned int j = 0; j < 3; ++j)
                            {
                                unsigned int k = (fromModel->getEdgesInTriangle(tab[i]))[j];

                                if (fromModel->getTrianglesAroundEdge(k).size()!= 2)   // remove as visible the edge indexed by k // ==1
                                {

                                    // do nothing

                                }
                                else   // fromModel->getTrianglesAroundEdge(k).size()==2 // add as visible the edge indexed by k
                                {

                                    unsigned int ind_test;
                                    if(tab[i] == fromModel->getTrianglesAroundEdge(k)[0])
                                    {

                                        ind_test = fromModel->getTrianglesAroundEdge(k)[1];

                                    }
                                    else   // tab[i] == fromModel->getTrianglesAroundEdge(k)[1]
                                    {

                                        ind_test = fromModel->getTrianglesAroundEdge(k)[0];
                                    }

                                    bool is_present = false;
                                    unsigned int k0 = 0;
                                    while((!is_present) && k0 < i)
                                    {
                                        is_present = (ind_test == tab[k0]);
                                        k0+=1;
                                    }
                                    if(!is_present)
                                    {

                                        core::topology::BaseMeshTopology::Edge t;

                                        const core::topology::BaseMeshTopology::Triangle &te=triangleArray[ind_test];
                                        int h = fromModel->getEdgeIndexInTriangle(fromModel->getEdgesInTriangle(ind_test),k);

                                        t[0]=(int)(te[(h+1)%3]); t[1]=(int)(te[(h+2)%3]);

                                        // sort t such that t[0] is the smallest one
                                        if ((t[0]>t[1]))
                                        {
                                            int val=t[0]; t[0]=t[1]; t[1]=val;
                                        }

                                        edges_to_create.push_back(t);
                                        edgesIndexList.push_back(nb_elems);
                                        nb_elems+=1;

                                        Loc2GlobVec.push_back(k);
                                        std::map<unsigned int, unsigned int>::iterator iter_1 = Glob2LocMap.find(k);
                                        if(iter_1 != Glob2LocMap.end() )
                                        {
                                            sout << "INFO_print : Triangle2EdgeTopologicalMapping - fail to add edge " << k << "which already exists" << sendl;
                                            Glob2LocMap.erase(Glob2LocMap.find(k));
                                        }
                                        Glob2LocMap[k]=Loc2GlobVec.size()-1;
                                    }
                                }
                            }
                        }

                        to_tstm->addEdgesProcess(edges_to_create) ;
                        to_tstm->addEdgesWarning(edges_to_create.size(), edges_to_create, edgesIndexList) ;
                        //to_tstm->propagateTopologicalChanges();

                    }

                    break;
                }

                case core::topology::POINTSREMOVED:
                {
                    //sout << "INFO_print : Triangle2EdgeTopologicalMapping - POINTSREMOVED" << sendl;

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
                    //sout << "INFO_print : Hexa2TriangleTopologicalMapping - POINTSREMOVED" << sendl;

                    const sofa::helper::vector<unsigned int> &tab = ( static_cast< const PointsRenumbering * >( *itBegin ) )->getIndexArray();
                    const sofa::helper::vector<unsigned int> &inv_tab = ( static_cast< const PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

                    sofa::helper::vector<unsigned int> indices;
                    sofa::helper::vector<unsigned int> inv_indices;

                    for(unsigned int i = 0; i < tab.size(); ++i)
                    {

                        //sout << "INFO_print : Hexa2TriangleTopologicalMapping - point = " << tab[i] << sendl;
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

                /*
                case core::topology::EDGESADDED:
                {

                	//sout << "INFO_print : Triangle2EdgeTopologicalMapping - EDGESADDED" << sendl;

                	if (fromModel) {

                		const sofa::component::topology::EdgesAdded *ta=static_cast< const sofa::component::topology::EdgesAdded * >( *itBegin );

                		sofa::helper::vector< Edge > edges_to_create;
                		sofa::helper::vector< unsigned int > edgesIndexList;
                		int nb_elems = toModel->getNbEdges();

                		for (unsigned int i=0;i<ta->getNbAddedEdges();++i)
                		{
                			Edge t = ta->edgeArray[i];
                			if ((t[0]>t[1])) {
                				int val=t[0]; t[0]=t[1]; t[1]=val;
                			}
                			unsigned int k = ta->edgeIndexArray[i];

                			edges_to_create.push_back(t);
                			edgesIndexList.push_back(Loc2GlobVec.size());
                			nb_elems+=1;

                			Loc2GlobVec.push_back(k);
                			std::map<unsigned int, unsigned int>::iterator iter_1 = Glob2LocMap.find(k);
                			if(iter_1 != Glob2LocMap.end() ) {
                				sout << "INFO_print : Triangle2EdgeTopologicalMapping - fail to add edge " << k << "which already exists" << sendl;
                				Glob2LocMap.erase(Glob2LocMap.find(k));
                			}
                			Glob2LocMap[k]=Loc2GlobVec.size()-1;
                		}

                		to_tstm->addEdgesProcess(edges_to_create) ;
                		to_tstm->addEdgesWarning(edges_to_create.size(), edges_to_create, edgesIndexList) ;
                		//toModel->propagateTopologicalChanges();

                	}

                	break;
                }
                */

                case core::topology::TRIANGLESADDED:
                {

                    //sout << "INFO_print : Triangle2EdgeTopologicalMapping - TRIANGLESADDED" << sendl;

                    if (fromModel)
                    {

                        //const sofa::helper::vector<Triangle> &triangleArray=fromModel->getTriangles();

                        const sofa::component::topology::TrianglesAdded *ta=static_cast< const sofa::component::topology::TrianglesAdded * >( *itBegin );

                        sofa::helper::vector< core::topology::BaseMeshTopology::Edge > edges_to_create;
                        sofa::helper::vector< unsigned int > edgesIndexList;

                        for (unsigned int i=0; i<ta->getNbAddedTriangles(); ++i)
                        {
                            unsigned int ind_elem = ta->triangleIndexArray[i];
                            for (unsigned int j = 0; j < 3; ++j)
                            {
                                unsigned int k = (fromModel->getEdgesInTriangle(ind_elem))[j];

                                std::map<unsigned int, unsigned int>::iterator iter_1 = Glob2LocMap.find(k);
                                bool is_present = (iter_1 != Glob2LocMap.end()) && fromModel->getTrianglesAroundEdge(k).size()>1;

                                if (is_present)   // remove as visible the edge indexed by k
                                {

                                    unsigned int ind_k = Glob2LocMap[k];
                                    unsigned int ind_last = Loc2GlobVec.size() - 1;
                                    unsigned int ind_real_last = Loc2GlobVec[ind_last];

                                    Glob2LocMap.erase(Glob2LocMap.find(ind_real_last));
                                    Glob2LocMap[ind_real_last] = ind_k;

                                    Loc2GlobVec[ind_k] = ind_real_last;

                                    Glob2LocMap.erase(Glob2LocMap.find(k));
                                    Loc2GlobVec.resize(Loc2GlobVec.size() - 1);

                                    sofa::helper::vector< unsigned int > edges_to_remove;
                                    edges_to_remove.push_back(ind_k);
                                    to_tstm->removeEdges(edges_to_remove, false);

                                }
                                else   // add as visible the edge indexed by k
                                {

                                    if((iter_1 == Glob2LocMap.end()) && (fromModel->getTrianglesAroundEdge(k).size()==1))
                                    {

                                        //sofa::helper::vector< Edge > edges_to_create;
                                        //sofa::helper::vector< unsigned int > edgesIndexList;

                                        core::topology::BaseMeshTopology::Edge t = fromModel->getEdge(k);
                                        /*
                                        if ((t[0]>t[1])) {
                                        	int val=t[0]; t[0]=t[1]; t[1]=val;
                                        }
                                        */

                                        edges_to_create.push_back(t);
                                        edgesIndexList.push_back(Loc2GlobVec.size());

                                        Loc2GlobVec.push_back(k);
                                        std::map<unsigned int, unsigned int>::iterator iter_1 = Glob2LocMap.find(k);
                                        if(iter_1 != Glob2LocMap.end() )
                                        {
                                            sout << "INFO_print : Triangle2EdgeTopologicalMapping - fail to add edge " << k << "which already exists" << sendl;
                                            Glob2LocMap.erase(Glob2LocMap.find(k));
                                        }
                                        Glob2LocMap[k]=Loc2GlobVec.size()-1;

                                        //to_tstm->addEdgesProcess(edges_to_create) ;
                                        //to_tstm->addEdgesWarning(edges_to_create.size(), edges_to_create, edgesIndexList) ;
                                        //toModel->propagateTopologicalChanges();
                                    }

                                }
                            }
                        }

                        to_tstm->addEdgesProcess(edges_to_create) ;
                        to_tstm->addEdgesWarning(edges_to_create.size(), edges_to_create, edgesIndexList) ;
                        to_tstm->propagateTopologicalChanges();

                    }

                    break;
                }


                case core::topology::POINTSADDED:
                {
                    //sout << "INFO_print : Triangle2EdgeTopologicalMapping - POINTSADDED" << sendl;

                    const sofa::component::topology::PointsAdded *ta=static_cast< const sofa::component::topology::PointsAdded * >( *itBegin );
                    to_tstm->addPointsProcess(ta->getNbAddedVertices());
                    to_tstm->addPointsWarning(ta->getNbAddedVertices(), ta->ancestorsList, ta->coefs, false);
                    to_tstm->propagateTopologicalChanges();

                    break;
                }


                default:
                    // Ignore events that are not Edge  related.
                    break;
                };

                ++itBegin;
            }
            to_tstm->propagateTopologicalChanges();
            Loc2GlobDataVec.endEdit();

        }
    }

    return;
}

} // namespace topology

} // namespace component

} // namespace sofa

