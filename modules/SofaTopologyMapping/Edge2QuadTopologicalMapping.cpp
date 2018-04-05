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
#include <SofaTopologyMapping/Edge2QuadTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>

#include <SofaBaseTopology/QuadSetTopologyModifier.h>
#include <SofaBaseTopology/QuadSetTopologyContainer.h>

#include <sofa/core/topology/TopologyChange.h>

#include <sofa/defaulttype/Vec.h>
#include <map>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/core/behavior/MechanicalState.h>
#include <SofaBaseMechanics/MechanicalObject.h>

#include <math.h>
#include <sofa/defaulttype/Vec.h>

#include <sofa/defaulttype/RigidTypes.h>


namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;

using namespace sofa::component::topology;
using namespace sofa::core::topology;

SOFA_DECL_CLASS(Edge2QuadTopologicalMapping)

// Register in the Factory
int Edge2QuadTopologicalMappingClass = core::RegisterObject("Special case of mapping where EdgeSetTopology is converted to QuadSetTopology")
        .add< Edge2QuadTopologicalMapping >()

        ;

// Implementation

void Edge2QuadTopologicalMapping::init()
{

    if (!m_radius.isSet())
    {
        this->getContext()->get(m_radiusContainer);

        if(!m_radiusContainer)
            sout << "No radius defined" << sendl;
    }
 

    unsigned int N = m_nbPointsOnEachCircle.getValue();
    double rho = m_radius.getValue();

    

    //sout << "INFO_print : init Edge2QuadTopologicalMapping" << sendl;

    // INITIALISATION of QUADULAR mesh from EDGE mesh :

    core::behavior::MechanicalState<Rigid3Types>* from_mstate = dynamic_cast<core::behavior::MechanicalState<Rigid3Types>*>(fromModel->getContext()->getMechanicalState());
    core::behavior::MechanicalState<Vec3Types>* to_mstate = dynamic_cast<core::behavior::MechanicalState<Vec3Types>*>(toModel->getContext()->getMechanicalState());

    if (fromModel)
    {

        sout << "INFO_print : Edge2QuadTopologicalMapping - from = edge" << sendl;

        if (toModel)
        {

            sout << "INFO_print : Edge2QuadTopologicalMapping - to = quad" << sendl;

            QuadSetTopologyModifier *to_tstm;
            toModel->getContext()->get(to_tstm);

            QuadSetTopologyContainer *to_tstc;
            toModel->getContext()->get(to_tstc);

            const sofa::helper::vector<Edge> &edgeArray=fromModel->getEdges();

            sofa::helper::vector <unsigned int>& Loc2GlobVec = *(Loc2GlobDataVec.beginEdit());

            Loc2GlobVec.clear();
            In2OutMap.clear();

            // CREATION of the points (new DOFs for the output topology) along the circles around each point of the input topology

            Vec Y0;
            Vec Z0;
            Y0[0] = (Real) (0.0); Y0[1] = (Real) (1.0); Y0[2] = (Real) (0.0);
            Z0[0] = (Real) (0.0); Z0[1] = (Real) (0.0); Z0[2] = (Real) (1.0);

            if (to_mstate)
            {
                to_mstate->resize(fromModel->getNbPoints() * N);
            }

            to_tstc->clear();

            toModel->setNbPoints(fromModel->getNbPoints() * N);

            if (to_mstate)
            {

                for (unsigned int i=0; i<(unsigned int) fromModel->getNbPoints(); ++i)
                {
                    unsigned int p0=i;

                    Mat rotation;
                    (from_mstate->read(core::ConstVecCoordId::position())->getValue())[p0].writeRotationMatrix(rotation);

                    Vec t;
                    t=(from_mstate->read(core::ConstVecCoordId::position())->getValue())[p0].getCenter();

                    Vec Y;
                    Vec Z;

                    Y = rotation * Y0;
                    Z = rotation * Z0;

                    helper::WriteAccessor< Data< Vec3Types::VecCoord > > to_x = *to_mstate->write(core::VecCoordId::position());

                    for(unsigned int j=0; j<N; ++j)
                    {
                        if(m_radiusContainer) rho = m_radiusContainer->getPointRadius(j);

                        Vec x = t + (Y*cos((Real) (2.0*j*M_PI/N)) + Z*sin((Real) (2.0*j*M_PI/N)))*((Real) rho);
                        to_x[p0*N+j] = x;
                    }
                }
            }


            // CREATION of the quads based on the the circles
            sofa::helper::vector< Quad > quads_to_create;
            sofa::helper::vector< unsigned int > quadsIndexList;
            if(edgeList.getValue().size()==0)
            {

                int nb_elems = toModel->getNbQuads();

                for (unsigned int i=0; i<edgeArray.size(); ++i)
                {

                    unsigned int p0 = edgeArray[i][0];
                    unsigned int p1 = edgeArray[i][1];

                    sofa::helper::vector<unsigned int> out_info;

                    for(unsigned int j=0; j<N; ++j)
                    {

                        unsigned int q0 = p0*N+j;
                        unsigned int q1 = p1*N+j;
                        unsigned int q2 = p1*N+((j+1)%N);
                        unsigned int q3 = p0*N+((j+1)%N);

                        if (flipNormals.getValue())
                        {
                            Quad q = Quad((unsigned int) q3, (unsigned int) q2, (unsigned int) q1, (unsigned int) q0);
                            quads_to_create.push_back(q);
                            quadsIndexList.push_back(nb_elems);
                        }

                        else
                        {
                            Quad q = Quad((unsigned int) q0, (unsigned int) q1, (unsigned int) q2, (unsigned int) q3);
                            quads_to_create.push_back(q);
                            quadsIndexList.push_back(nb_elems);
                        }

                        Loc2GlobVec.push_back(i);
                        out_info.push_back(Loc2GlobVec.size()-1);
                    }


                    nb_elems++;

                    In2OutMap[i]=out_info;
                }
            }
            else
            {
                for (unsigned int j=0; j<edgeList.getValue().size(); ++j)
                {
                    unsigned int i=edgeList.getValue()[j];

                    unsigned int p0 = edgeArray[i][0];
                    unsigned int p1 = edgeArray[i][1];

                    sofa::helper::vector<unsigned int> out_info;

                    for(unsigned int j=0; j<N; ++j)
                    {

                        unsigned int q0 = p0*N+j;
                        unsigned int q1 = p1*N+j;
                        unsigned int q2 = p1*N+((j+1)%N);
                        unsigned int q3 = p0*N+((j+1)%N);

                        if(flipNormals.getValue())
                            to_tstm->addQuadProcess(Quad((unsigned int) q0, (unsigned int) q3, (unsigned int) q2, (unsigned int) q1));
                        else
                            to_tstm->addQuadProcess(Quad((unsigned int) q0, (unsigned int) q1, (unsigned int) q2, (unsigned int) q3));
                        Loc2GlobVec.push_back(i);
                        out_info.push_back(Loc2GlobVec.size()-1);
                    }

                    In2OutMap[i]=out_info;
                }

            }

            to_tstm->addQuadsProcess(quads_to_create);
            to_tstm->addQuadsWarning(quads_to_create.size(), quads_to_create, quadsIndexList);

            //to_tstm->notifyEndingEvent();
            to_tstm->propagateTopologicalChanges();
            Loc2GlobDataVec.endEdit();
        }

    }
}


unsigned int Edge2QuadTopologicalMapping::getFromIndex(unsigned int ind)
{
    return ind; // identity
}

void Edge2QuadTopologicalMapping::updateTopologicalMappingTopDown()
{

    unsigned int N = m_nbPointsOnEachCircle.getValue();

    // INITIALISATION of QUADULAR mesh from EDGE mesh :

    if (fromModel)
    {

        QuadSetTopologyModifier *to_tstm;
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

                case core::topology::EDGESADDED:
                {
                    //sout << "INFO_print : TopologicalMapping - EDGESADDED" << sendl;
                    if (fromModel)
                    {

                        const sofa::helper::vector<Edge> &edgeArray=fromModel->getEdges();

                        const sofa::helper::vector<unsigned int> &tab = ( static_cast< const EdgesAdded *>( *itBegin ) )->edgeIndexArray;

                        sofa::helper::vector< Quad > quads_to_create;
                        sofa::helper::vector< unsigned int > quadsIndexList;
                        int nb_elems = toModel->getNbQuads();

                        for (unsigned int i = 0; i < tab.size(); ++i)
                        {
                            unsigned int k = tab[i];

                            unsigned int p0 = edgeArray[k][0];
                            unsigned int p1 = edgeArray[k][1];

                            sofa::helper::vector<unsigned int> out_info;

                            for(unsigned int j=0; j<N; ++j)
                            {

                                unsigned int q0 = p0*N+j;
                                unsigned int q1 = p1*N+j;
                                unsigned int q2 = p1*N+((j+1)%N);
                                unsigned int q3 = p0*N+((j+1)%N);

                                Quad t = Quad((unsigned int) q0, (unsigned int) q1, (unsigned int) q2, (unsigned int) q3);

                                //quads_to_create.clear();
                                //quadsIndexList.clear();

                                quads_to_create.push_back(t);
                                quadsIndexList.push_back(nb_elems);
                                nb_elems+=1;

                                Loc2GlobVec.push_back(k);
                                out_info.push_back(Loc2GlobVec.size()-1);

                                //to_tstm->addQuadsProcess(quads_to_create) ;
                                //to_tstm->addQuadsWarning(quads_to_create.size(), quads_to_create, quadsIndexList) ;
                                //to_tstm->propagateTopologicalChanges();
                            }

                            In2OutMap[k]=out_info;
                        }

                        to_tstm->addQuadsProcess(quads_to_create);
                        to_tstm->addQuadsWarning(quads_to_create.size(), quads_to_create, quadsIndexList) ;
                        to_tstm->propagateTopologicalChanges();
                    }
                    break;
                }
                case core::topology::EDGESREMOVED:
                {
                    //sout << "INFO_print : TopologicalMapping - EDGESREMOVED" << sendl;

                    if (fromModel)
                    {

                        const sofa::helper::vector<unsigned int> &tab = ( static_cast< const EdgesRemoved *>( *itBegin ) )->getArray();

                        int last= fromModel->getNbEdges() - 1;

                        unsigned int ind_tmp;

                        sofa::helper::vector<unsigned int> ind_real_last;
                        int ind_last=toModel->getNbQuads();

                        for (unsigned int i = 0; i < tab.size(); ++i)
                        {
                            unsigned int k = tab[i];
                            sofa::helper::vector<unsigned int> ind_k;

                            std::map<unsigned int, sofa::helper::vector<unsigned int> >::iterator iter_1 = In2OutMap.find(k);
                            if(iter_1 != In2OutMap.end())
                            {

                                sofa::helper::vector<unsigned int> ind_list;
                                for(unsigned int j=0; j<N; ++j)
                                {
                                    ind_list.push_back(In2OutMap[k][j]);
                                }

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

                                        for(unsigned int j=0; j<N; ++j)
                                        {

                                            ind_tmp = Loc2GlobVec[ind_real_last[j]];
                                            Loc2GlobVec[ind_real_last[j]] = Loc2GlobVec[ind_k[j]];
                                            Loc2GlobVec[ind_k[j]] = ind_tmp;
                                        }
                                    }
                                }
                                else
                                {
                                    sout << "INFO_print : Edge2QuadTopologicalMapping - In2OutMap should have the edge " << last << sendl;
                                }

                                if( (int) ind_k[N-1] != ind_last)
                                {

                                    In2OutMap.erase(In2OutMap.find(Loc2GlobVec[ind_last]));
                                    In2OutMap[Loc2GlobVec[ind_last]] = ind_k;

                                    sofa::helper::vector<unsigned int> out_info;
                                    for(unsigned int j=0; j<N; ++j)
                                    {
                                        out_info.push_back(ind_last-j);
                                    }

                                    In2OutMap.erase(In2OutMap.find(Loc2GlobVec[ind_k[N-1]]));
                                    In2OutMap[Loc2GlobVec[ind_k[N-1]]] = out_info;

                                    ind_tmp = Loc2GlobVec[ind_k[N-1]];
                                    Loc2GlobVec[ind_k[N-1]] = Loc2GlobVec[ind_last];
                                    Loc2GlobVec[ind_last] = ind_tmp;

                                }

                                for(unsigned int j=1; j<N; ++j)
                                {

                                    ind_last = ind_last-1;

                                    if( (int) ind_k[N-1-j] != ind_last)
                                    {

                                        ind_tmp = Loc2GlobVec[ind_k[N-1-j]];
                                        Loc2GlobVec[ind_k[N-1-j]] = Loc2GlobVec[ind_last];
                                        Loc2GlobVec[ind_last] = ind_tmp;
                                    }
                                }

                                In2OutMap.erase(In2OutMap.find(Loc2GlobVec[Loc2GlobVec.size() - 1]));

                                Loc2GlobVec.resize( Loc2GlobVec.size() - N );

                                sofa::helper::vector< unsigned int > quads_to_remove;
                                for(unsigned int j=0; j<N; ++j)
                                {
                                    quads_to_remove.push_back(ind_list[j]);
                                }

                                to_tstm->removeQuads(quads_to_remove, true, true);

                            }
                            else
                            {
                                sout << "INFO_print : Edge2QuadTopologicalMapping - In2OutMap should have the edge " << k << sendl;
                            }

                            --last;
                        }
                    }

                    break;
                }

                case core::topology::POINTSRENUMBERING:
                {
                    //sout << "INFO_print : Edge2QuadTopologicalMapping - POINTSRENUMBERING" << sendl;

                    const sofa::helper::vector<unsigned int> &tab = ( static_cast< const PointsRenumbering * >( *itBegin ) )->getIndexArray();
                    const sofa::helper::vector<unsigned int> &inv_tab = ( static_cast< const PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

                    sofa::helper::vector<unsigned int> indices;
                    sofa::helper::vector<unsigned int> inv_indices;

                    for(unsigned int i = 0; i < tab.size(); ++i)
                    {

                        for(unsigned int j=0; j<N; ++j)
                        {
                            indices.push_back(tab[i]*N + j);
                            inv_indices.push_back(inv_tab[i]*N + j);
                        }

                        //sout << "INFO_print : Edge2QuadTopologicalMapping - renumber point = " << tab[i] << sendl;
                    }

                    sofa::helper::vector<unsigned int>& tab_indices = indices;
                    sofa::helper::vector<unsigned int>& inv_tab_indices = inv_indices;

                    to_tstm->renumberPointsWarning(tab_indices, inv_tab_indices, true);
                    to_tstm->propagateTopologicalChanges();
                    to_tstm->renumberPointsProcess(tab_indices, inv_tab_indices, true);

                    break;
                }

                case core::topology::POINTSADDED:
                {
                    //sout << "INFO_print : Edge2QuadTopologicalMapping - POINTSADDED" << sendl;

                    const sofa::component::topology::PointsAdded *ta=static_cast< const sofa::component::topology::PointsAdded * >( *itBegin );

                    unsigned int to_nVertices = ta->getNbAddedVertices() * N;
                    sofa::helper::vector< sofa::helper::vector< unsigned int > > to_ancestorsList;
                    sofa::helper::vector< sofa::helper::vector< double > > to_coefs;

                    for(unsigned int i =0; i < ta->getNbAddedVertices(); i++)
                    {

                        sofa::helper::vector< unsigned int > my_ancestors;
                        sofa::helper::vector< double > my_coefs;

                        for(unsigned int j =0; j < N; j++)
                        {

                            for(unsigned int k = 0; k < ta->ancestorsList[i].size(); k++)
                            {
                                my_ancestors.push_back(ta->ancestorsList[i][k]*N + j);
                            }
                            for(unsigned int k = 0; k < ta->coefs[i].size(); k++)
                            {
                                my_coefs.push_back(ta->coefs[i][k]*N + j);
                            }

                            to_ancestorsList.push_back(my_ancestors);
                            to_coefs.push_back(my_coefs);
                        }
                    }

                    to_tstm->addPointsProcess(to_nVertices);
                    to_tstm->addPointsWarning(to_nVertices, to_ancestorsList, to_coefs, true);
                    to_tstm->propagateTopologicalChanges();

                    break;
                }

                default:
                    // Ignore events that are not Quad  related.
                    break;
                }

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

