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
#include <sofa/component/topology/mapping/Hexa2QuadTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/component/topology/container/dynamic/QuadSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/QuadSetTopologyModifier.h>

#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyModifier.h>

#include <sofa/core/topology/TopologyChange.h>

#include <sofa/type/Vec.h>
#include <map>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>


namespace sofa::component::topology::mapping
{

using namespace sofa::defaulttype;

using namespace sofa::component::topology;
using namespace sofa::core::topology;

// Register in the Factory
int Hexa2QuadTopologicalMappingClass = core::RegisterObject("Special case of mapping where HexahedronSetTopology is converted to QuadSetTopology")
        .add< Hexa2QuadTopologicalMapping >()

        ;

// Implementation

Hexa2QuadTopologicalMapping::Hexa2QuadTopologicalMapping()
    : sofa::core::topology::TopologicalMapping()
    , flipNormals(initData(&flipNormals, bool(false), "flipNormals", "Flip Normal ? (Inverse point order when creating triangle)"))
{
    m_inputType = geometry::ElementType::HEXAHEDRON;
    m_outputType = geometry::ElementType::QUAD;
}

void Hexa2QuadTopologicalMapping::init()
{
    using namespace container::dynamic;

    if (!this->checkTopologyInputTypes()) // method will display error message if false
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    // Making sure a topology modifier exists at the same level as the output topology
    QuadSetTopologyModifier *to_tstm;
    toModel->getContext()->get(to_tstm);
    if (!to_tstm) {
        msg_error() << "No QuadSetTopologyModifier found in the output topology node '"
                    << toModel->getContext()->getName() << "'.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    // Clear output topology
    toModel->clear();

    // Set the same number of points
    toModel->setNbPoints(fromModel->getNbPoints());

    const sofa::type::vector<core::topology::BaseMeshTopology::Quad> &quadArray=fromModel->getQuads();
    auto Loc2GlobVec = sofa::helper::getWriteOnlyAccessor(Loc2GlobDataVec);
    Loc2GlobVec.clear();
    Glob2LocMap.clear();

    const bool flipN = flipNormals.getValue();

    for (unsigned int i=0; i<quadArray.size(); ++i)
    {
        if (fromModel->getHexahedraAroundQuad(i).size()==1)
        {
            core::topology::BaseMeshTopology::Quad q = quadArray[i];

            if(flipN)
                toModel->addQuad(q[3], q[2], q[1], q[0]);
            else
                toModel->addQuad(q[0], q[1], q[2], q[3]);

            Loc2GlobVec.push_back(i);
            Glob2LocMap[i]= (unsigned int)Loc2GlobVec.size()-1;
        }
    }

    //to_tstm->propagateTopologicalChanges();

    // Need to fully init the target topology
    toModel->init();

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

Index Hexa2QuadTopologicalMapping::getFromIndex(Index ind)
{

    if(fromModel->getHexahedraAroundQuad(ind).size()==1)
    {
        return fromModel->getHexahedraAroundQuad(ind)[0];
    }
    else
    {
        return 0;
    }
}

void Hexa2QuadTopologicalMapping::updateTopologicalMappingTopDown()
{
    if (this->d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid)
        return;

    SCOPED_TIMER("Update Hexa2QuadTopologicalMapping");
    container::dynamic::QuadSetTopologyModifier *to_tstm;
    toModel->getContext()->get(to_tstm);

    auto itBegin=fromModel->beginChange();
    const auto itEnd=fromModel->endChange();

    auto Loc2GlobVec = sofa::helper::getWriteAccessor(Loc2GlobDataVec);

    while( itBegin != itEnd )
    {
        const TopologyChangeType changeType = (*itBegin)->getChangeType();
        std::string topoChangeType = "Hexa2QuadTopologicalMapping - " + parseTopologyChangeTypeToString(changeType);
        helper::ScopedAdvancedTimer topoChangetimer(topoChangeType);

        switch( changeType )
        {

        case core::topology::ENDING_EVENT:
        {
            to_tstm->notifyEndingEvent();
            break;
        }

        case core::topology::QUADSREMOVED:
        {
            auto last = (unsigned int)fromModel->getNbQuads() - 1;
            auto ind_last = (unsigned int)toModel->getNbQuads();

            const Topology::SetIndices & tab = ( static_cast< const QuadsRemoved *>( *itBegin ) )->getArray();

            unsigned int ind_tmp;

            unsigned int ind_real_last;

            for (unsigned int i = 0; i <tab.size(); ++i)
            {
                unsigned int k = tab[i];

                auto iter_1 = Glob2LocMap.find(k);
                if(iter_1 != Glob2LocMap.end())
                {

                    ind_last = ind_last - 1;

                    unsigned int ind_k = Glob2LocMap[k];
//                            ind_real_last = ind_k;

                    auto iter_2 = Glob2LocMap.find(last);
                    if(iter_2 != Glob2LocMap.end())
                    {

                        ind_real_last = Glob2LocMap[last];

                        if(k != last)
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

                    if(ind_k != ind_last)
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

                    sofa::type::vector< Index > quads_to_remove;
                    quads_to_remove.push_back(ind_k);

                    to_tstm->removeQuads(quads_to_remove, true, false);

                }
                else
                {
                    msg_warning() << "Glob2LocMap should have the visible quad " << tab[i] << " - nb quads = " << ind_last;
                }

                --last;
            }
            break;
        }

        case core::topology::HEXAHEDRAREMOVED:
        {
            const sofa::type::vector<core::topology::BaseMeshTopology::Hexahedron> &hexahedronArray=fromModel->getHexahedra();

            const Topology::SetIndices & tab = ( static_cast< const HexahedraRemoved *>( *itBegin ) )->getArray();

            sofa::type::vector< core::topology::BaseMeshTopology::Quad > quads_to_create;
            sofa::type::vector< Index > quadsIndexList;
            auto nb_elems = toModel->getNbQuads();
            const bool flipN = flipNormals.getValue();

            for (unsigned int i = 0; i < tab.size(); ++i)
            {

                for (unsigned int j = 0; j < 6; ++j)
                {
                    unsigned int k = (fromModel->getQuadsInHexahedron(tab[i]))[j];

                    if (fromModel->getHexahedraAroundQuad(k).size()==1)   // remove as visible the quad indexed by k
                    {

                        // do nothing

                    }
                    else   // fromModel->getHexahedraAroundQuad(k).size()==2 // add as visible the quad indexed by k
                    {

                        unsigned int ind_test;
                        if(tab[i] == fromModel->getHexahedraAroundQuad(k)[0])
                        {

                            ind_test = fromModel->getHexahedraAroundQuad(k)[1];

                        }
                        else   // tab[i] == fromModel->getHexahedraAroundQuad(k)[1]
                        {

                            ind_test = fromModel->getHexahedraAroundQuad(k)[0];
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

                            core::topology::BaseMeshTopology::Quad q;

                            const core::topology::BaseMeshTopology::Hexahedron &he=hexahedronArray[ind_test];
                            const int h = fromModel->getQuadIndexInHexahedron(fromModel->getQuadsInHexahedron(ind_test),k);
                            //unsigned int hh = (fromModel->getQuadsInHexahedron(ind_test))[h];

                            //t=from_qstc->getQuad(hh);
                            static const int hexaQuads[6][4] = { { 0,3,2,1 }, {4,5,6,7}, {0,1,5,4}, {1,2,6,5}, {2,3,7,6}, {3,0,4,7} };
                            q[0] = he[hexaQuads[h][0]];
                            q[1] = he[hexaQuads[h][1]];
                            q[2] = he[hexaQuads[h][2]];
                            q[3] = he[hexaQuads[h][3]];

                            if(flipN)
                            {
                                const unsigned int tmp3 = q[3];
                                const unsigned int tmp2 = q[2];
                                q[3] = q[0];
                                q[2] = q[1];
                                q[1] = tmp2;
                                q[0] = tmp3;
                            }
                            quads_to_create.push_back(q);
                            quadsIndexList.push_back(nb_elems);
                            nb_elems+=1;

                            Loc2GlobVec.push_back(k);
                            auto iter_1 = Glob2LocMap.find(k);
                            if(iter_1 != Glob2LocMap.end() )
                            {
                                msg_warning() << "Failed to add quad " << k << " which already exists";
                                Glob2LocMap.erase(Glob2LocMap.find(k));
                            }
                            Glob2LocMap[k]=(unsigned int)Loc2GlobVec.size()-1;
                        }
                    }
                }
            }

            to_tstm->addQuads(quads_to_create) ;
            break;
        }

        case core::topology::POINTSREMOVED:
        {
            const Topology::SetIndices & tab = ( static_cast< const sofa::core::topology::PointsRemoved * >( *itBegin ) )->getArray();

            Topology::SetIndices indices;

            for(unsigned int i = 0; i < tab.size(); ++i)
            {
                indices.push_back(tab[i]);
            }

            Topology::SetIndices & tab_indices = indices;

            to_tstm->removePoints(tab_indices, false);
            break;
        }

        case core::topology::POINTSRENUMBERING:
        {
            const Topology::SetIndices & tab = ( static_cast< const PointsRenumbering * >( *itBegin ) )->getIndexArray();
            const Topology::SetIndices & inv_tab = ( static_cast< const PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

            sofa::type::vector<Index> indices;
            sofa::type::vector<Index> inv_indices;

            for(unsigned int i = 0; i < tab.size(); ++i)
            {
                indices.push_back(tab[i]);
                inv_indices.push_back(inv_tab[i]);
            }

            Topology::SetIndices & tab_indices = indices;
            Topology::SetIndices & inv_tab_indices = inv_indices;

            to_tstm->renumberPoints(tab_indices, inv_tab_indices, false);

            break;
        }

        default:
            // Ignore events that are not Quad  related.
            break;
        };

        ++itBegin;
    }
}

} // namespace sofa::component::topology::mapping
