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
#include <SofaTopologyMapping/Quad2TriangleTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>

#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetTopologyModifier.h>

#include <SofaBaseTopology/QuadSetTopologyContainer.h>
#include <SofaBaseTopology/QuadSetTopologyModifier.h>

#include <sofa/core/topology/TopologyChange.h>
#include <SofaBaseTopology/GridTopology.h>

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

/// Input Topology
typedef BaseMeshTopology In;
/// Output Topology
typedef BaseMeshTopology Out;

SOFA_DECL_CLASS(Quad2TriangleTopologicalMapping)

// Register in the Factory
int Quad2TriangleTopologicalMappingClass = core::RegisterObject("Special case of mapping where QuadSetTopology is converted to TriangleSetTopology")
        .add< Quad2TriangleTopologicalMapping >()

        ;

// Implementation

Quad2TriangleTopologicalMapping::Quad2TriangleTopologicalMapping()
{
}


Quad2TriangleTopologicalMapping::~Quad2TriangleTopologicalMapping()
{
}

void Quad2TriangleTopologicalMapping::init()
{
    //sout << "INFO_print : init Quad2TriangleTopologicalMapping" << sendl;

    // INITIALISATION of TRIANGULAR mesh from QUADULAR mesh :

    if (fromModel)
    {

        sout << "INFO_print : Quad2TriangleTopologicalMapping - from = quad" << sendl;

        if (toModel)
        {

            sout << "INFO_print : Quad2TriangleTopologicalMapping - to = triangle" << sendl;

            TriangleSetTopologyContainer *to_tstc;
            toModel->getContext()->get(to_tstc);
            to_tstc->clear();

            toModel->setNbPoints(fromModel->getNbPoints());

            TriangleSetTopologyModifier *to_tstm;
            toModel->getContext()->get(to_tstm);

            const sofa::helper::vector<core::topology::BaseMeshTopology::Quad> &quadArray=fromModel->getQuads();

            sofa::helper::vector <unsigned int>& Loc2GlobVec = *(Loc2GlobDataVec.beginEdit());

            Loc2GlobVec.clear();
            In2OutMap.clear();

            // These values are only correct if the mesh is a grid topology
            int nx = 2;
            int ny = 1;
            //int nz = 1;

            {
                topology::GridTopology* grid = dynamic_cast<topology::GridTopology*>(fromModel.get());
                if (grid != NULL)
                {
                    nx = grid->getNx()-1;
                    ny = grid->getNy()-1;
                    //nz = grid->getNz()-1;
                }
            }

            int scale = nx;

            if (nx == 0)
                scale = ny;

            if (nx == 0 && ny == 0)
            {
                serr<<"Error: Input topology is only 1D, this topology can't be mapped into a triangulation." <<sendl;
                return;
            }


            for (unsigned int i=0; i<quadArray.size(); ++i)
            {
                unsigned int p0 = quadArray[i][0];
                unsigned int p1 = quadArray[i][1];
                unsigned int p2 = quadArray[i][2];
                unsigned int p3 = quadArray[i][3];
                if (((i%scale) ^ (i/scale)) & 1)
                {
                    to_tstm->addTriangleProcess(core::topology::BaseMeshTopology::Triangle((unsigned int) p0, (unsigned int) p1, (unsigned int) p3));
                    to_tstm->addTriangleProcess(core::topology::BaseMeshTopology::Triangle((unsigned int) p2, (unsigned int) p3, (unsigned int) p1));
                }
                else
                {
                    to_tstm->addTriangleProcess(core::topology::BaseMeshTopology::Triangle((unsigned int) p1, (unsigned int) p2, (unsigned int) p0));
                    to_tstm->addTriangleProcess(core::topology::BaseMeshTopology::Triangle((unsigned int) p3, (unsigned int) p0, (unsigned int) p2));
                }

                Loc2GlobVec.push_back(i);
                Loc2GlobVec.push_back(i);
                sofa::helper::vector<unsigned int> out_info;
                out_info.push_back(Loc2GlobVec.size()-2);
                out_info.push_back(Loc2GlobVec.size()-1);
                In2OutMap[i]=out_info;
            }

            //to_tstm->propagateTopologicalChanges();
            to_tstm->notifyEndingEvent();
            //to_tstm->propagateTopologicalChanges();
            Loc2GlobDataVec.endEdit();

        }

    }
}

unsigned int Quad2TriangleTopologicalMapping::getFromIndex(unsigned int ind)
{
    return ind; // identity
}

void Quad2TriangleTopologicalMapping::updateTopologicalMappingTopDown()
{

    // INITIALISATION of TRIANGULAR mesh from QUADULAR mesh :

    if (fromModel)
    {

        TriangleSetTopologyModifier *to_tstm;
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
                    //sout << "INFO_print : TopologicalMapping - ENDING_EVENT" << sendl;
                    to_tstm->propagateTopologicalChanges();
                    to_tstm->notifyEndingEvent();
                    to_tstm->propagateTopologicalChanges();
                    break;
                }

                case core::topology::QUADSADDED:
                {
                    //sout << "INFO_print : TopologicalMapping - QUADSADDED" << sendl;
                    if (fromModel)
                    {

                        const sofa::helper::vector<core::topology::BaseMeshTopology::Quad> &quadArray=fromModel->getQuads();

                        const sofa::helper::vector<unsigned int> &tab = ( static_cast< const QuadsAdded *>( *itBegin ) )->getArray();

                        sofa::helper::vector< core::topology::BaseMeshTopology::Triangle > triangles_to_create;
                        sofa::helper::vector< unsigned int > trianglesIndexList;
                        int nb_elems = toModel->getNbTriangles();

                        for (unsigned int i = 0; i < tab.size(); ++i)
                        {
                            unsigned int k = tab[i];

                            unsigned int p0 = quadArray[k][0];
                            unsigned int p1 = quadArray[k][1];
                            unsigned int p2 = quadArray[k][2];
                            unsigned int p3 = quadArray[k][3];
                            core::topology::BaseMeshTopology::Triangle t1 = core::topology::BaseMeshTopology::Triangle((unsigned int) p0, (unsigned int) p1, (unsigned int) p2);
                            core::topology::BaseMeshTopology::Triangle t2 = core::topology::BaseMeshTopology::Triangle((unsigned int) p0, (unsigned int) p2, (unsigned int) p3);

                            triangles_to_create.push_back(t1);
                            trianglesIndexList.push_back(nb_elems);
                            triangles_to_create.push_back(t2);
                            trianglesIndexList.push_back(nb_elems+1);
                            nb_elems+=2;

                            Loc2GlobVec.push_back(k);
                            Loc2GlobVec.push_back(k);
                            sofa::helper::vector<unsigned int> out_info;
                            out_info.push_back(Loc2GlobVec.size()-2);
                            out_info.push_back(Loc2GlobVec.size()-1);
                            In2OutMap[k]=out_info;

                        }

                        to_tstm->addTrianglesProcess(triangles_to_create) ;
                        to_tstm->addTrianglesWarning(triangles_to_create.size(), triangles_to_create, trianglesIndexList) ;
                        to_tstm->propagateTopologicalChanges();
                    }
                    break;
                }
                case core::topology::QUADSREMOVED:
                {
                    //sout << "INFO_print : TopologicalMapping - QUADSREMOVED" << sendl;

                    if (fromModel)
                    {

                        const sofa::helper::vector<unsigned int> &tab = ( static_cast< const QuadsRemoved *>( *itBegin ) )->getArray();

                        int last= fromModel->getNbQuads() - 1;

                        int ind_tmp;

                        sofa::helper::vector<unsigned int> ind_real_last;
                        int ind_last=toModel->getNbTriangles();

                        for (unsigned int i = 0; i < tab.size(); ++i)
                        {
                            //sout << "INFO_print : Quad2TriangleTopologicalMapping - remove quad " << tab[i] << sendl;

                            unsigned int k = tab[i];
                            sofa::helper::vector<unsigned int> ind_k;

                            std::map<unsigned int, sofa::helper::vector<unsigned int> >::iterator iter_1 = In2OutMap.find(k);
                            if(iter_1 != In2OutMap.end())
                            {

                                unsigned int t1 = In2OutMap[k][0];
                                unsigned int t2 = In2OutMap[k][1];

                                ind_last = ind_last - 1;

                                ind_k = In2OutMap[k];
                                ind_real_last = ind_k;

                                std::map<unsigned int, sofa::helper::vector<unsigned int> >::iterator iter_2 = In2OutMap.find(last);
                                if(iter_2 != In2OutMap.end())
                                {

                                    ind_real_last = In2OutMap[last];

                                    if((int) k != last)
                                    {

                                        In2OutMap.erase(In2OutMap.find(k));
                                        In2OutMap[k] = ind_real_last;

                                        In2OutMap.erase(In2OutMap.find(last));
                                        In2OutMap[last] = ind_k;

                                        ind_tmp = Loc2GlobVec[ind_real_last[0]];
                                        Loc2GlobVec[ind_real_last[0]] = Loc2GlobVec[ind_k[0]];
                                        Loc2GlobVec[ind_k[0]] = ind_tmp;

                                        ind_tmp = Loc2GlobVec[ind_real_last[1]];
                                        Loc2GlobVec[ind_real_last[1]] = Loc2GlobVec[ind_k[1]];
                                        Loc2GlobVec[ind_k[1]] = ind_tmp;
                                    }
                                }
                                else
                                {
                                    sout << "INFO_print : Quad2TriangleTopologicalMapping - In2OutMap should have the quad " << last << sendl;
                                }

                                if((int) ind_k[1] != ind_last)
                                {

                                    In2OutMap.erase(In2OutMap.find(Loc2GlobVec[ind_last]));
                                    In2OutMap[Loc2GlobVec[ind_last]] = ind_k;

                                    sofa::helper::vector<unsigned int> out_info;
                                    out_info.push_back(ind_last);
                                    out_info.push_back(ind_last-1);

                                    In2OutMap.erase(In2OutMap.find(Loc2GlobVec[ind_k[1]]));
                                    In2OutMap[Loc2GlobVec[ind_k[1]]] = out_info;

                                    ind_tmp = Loc2GlobVec[ind_k[1]];
                                    Loc2GlobVec[ind_k[1]] = Loc2GlobVec[ind_last];
                                    Loc2GlobVec[ind_last] = ind_tmp;

                                }

                                ind_last = ind_last-1;

                                if((int) ind_k[0] != ind_last)
                                {

                                    ind_tmp = Loc2GlobVec[ind_k[0]];
                                    Loc2GlobVec[ind_k[0]] = Loc2GlobVec[ind_last];
                                    Loc2GlobVec[ind_last] = ind_tmp;

                                }

                                In2OutMap.erase(In2OutMap.find(Loc2GlobVec[Loc2GlobVec.size() - 1]));

                                Loc2GlobVec.resize( Loc2GlobVec.size() - 2 );

                                sofa::helper::vector< unsigned int > triangles_to_remove;
                                triangles_to_remove.push_back(t1);
                                triangles_to_remove.push_back(t2);

                                to_tstm->removeTriangles(triangles_to_remove, true, false);

                            }
                            else
                            {
                                sout << "INFO_print : Quad2TriangleTopologicalMapping - In2OutMap should have the quad " << k << sendl;
                            }

                            --last;
                        }
                    }

                    break;
                }

                case core::topology::POINTSREMOVED:
                {
                    //sout << "INFO_print : TopologicalMapping - POINTSREMOVED" << sendl;

                    const sofa::helper::vector<unsigned int> tab = ( static_cast< const sofa::component::topology::PointsRemoved * >( *itBegin ) )->getArray();

                    sofa::helper::vector<unsigned int> indices;

                    for(unsigned int i = 0; i < tab.size(); ++i)
                    {

                        //sout << "INFO_print : Quad2TriangleTopologicalMapping - point = " << tab[i] << sendl;
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
                    //sout << "INFO_print : Hexa2QuadTopologicalMapping - POINTSREMOVED" << sendl;

                    const sofa::helper::vector<unsigned int> &tab = ( static_cast< const PointsRenumbering * >( *itBegin ) )->getIndexArray();
                    const sofa::helper::vector<unsigned int> &inv_tab = ( static_cast< const PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

                    sofa::helper::vector<unsigned int> indices;
                    sofa::helper::vector<unsigned int> inv_indices;

                    for(unsigned int i = 0; i < tab.size(); ++i)
                    {

                        //sout << "INFO_print : Hexa2QuadTopologicalMapping - point = " << tab[i] << sendl;
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


                case core::topology::POINTSADDED:
                {
                    //sout << "INFO_print : Quad2TriangleTopologicalMapping - POINTSADDED" << sendl;

                    const sofa::component::topology::PointsAdded *ta=static_cast< const sofa::component::topology::PointsAdded * >( *itBegin );

                    to_tstm->addPointsProcess(ta->getNbAddedVertices());
                    to_tstm->addPointsWarning(ta->getNbAddedVertices(), ta->ancestorsList, ta->coefs, false);
                    to_tstm->propagateTopologicalChanges();

                    break;
                }



                default:
                    // Ignore events that are not Triangle  related.
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

