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
#include <sofa/component/topology/mapping/Quad2TriangleTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/component/topology/container/dynamic/TriangleSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TriangleSetTopologyModifier.h>

#include <sofa/component/topology/container/dynamic/QuadSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/QuadSetTopologyModifier.h>

#include <sofa/core/topology/TopologyChange.h>
#include <sofa/component/topology/container/grid/GridTopology.h>

#include <sofa/type/Vec.h>
#include <map>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>


namespace sofa::component::topology::mapping
{

using namespace sofa::defaulttype;

using namespace sofa::component::topology::mapping;
using namespace sofa::core::topology;

/// Input Topology
typedef BaseMeshTopology In;
/// Output Topology
typedef BaseMeshTopology Out;

// Register in the Factory
int Quad2TriangleTopologicalMappingClass = core::RegisterObject("Special case of mapping where QuadSetTopology is converted to TriangleSetTopology")
        .add< Quad2TriangleTopologicalMapping >()

        ;

// Implementation

Quad2TriangleTopologicalMapping::Quad2TriangleTopologicalMapping()
    : sofa::core::topology::TopologicalMapping()
{
    m_inputType = geometry::ElementType::QUAD;
    m_outputType = geometry::ElementType::TRIANGLE;
}


Quad2TriangleTopologicalMapping::~Quad2TriangleTopologicalMapping()
{
}

void Quad2TriangleTopologicalMapping::init()
{
    using namespace container::dynamic;

    if (!this->checkTopologyInputTypes()) // method will display error message if false
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

        
    // Making sure a topology modifier exists at the same level as the output topology
    TriangleSetTopologyModifier *to_tstm;
    toModel->getContext()->get(to_tstm);
    if (!to_tstm)
    {
        msg_error() << "No TriangleSetTopologyModifier found in the output topology node '"
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
    In2OutMap.clear();

    // These values are only correct if the mesh is a grid topology
    int nx = 2;
    int ny = 1;

    {
        const auto * grid = dynamic_cast<const container::grid::GridTopology*>(fromModel.get());
        if (grid != nullptr)
        {
            nx = grid->getNx()-1;
            ny = grid->getNy()-1;
        }
    }

    int scale = nx;

    if (nx == 0)
        scale = ny;

    if (nx == 0 && ny == 0)
    {
        msg_error() << "Input topology is only 1D, this topology can't be mapped into a triangulation.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }


    for (unsigned int i=0; i<quadArray.size(); ++i)
    {
        const auto & p0 = quadArray[i][0];
        const auto & p1 = quadArray[i][1];
        const auto & p2 = quadArray[i][2];
        const auto & p3 = quadArray[i][3];
        if (((i%scale) ^ (i/scale)) & 1)
        {
            toModel->addTriangle(p0, p1, p3);
            toModel->addTriangle(p2, p3, p1);
        }
        else
        {
            toModel->addTriangle(p1, p2, p0);
            toModel->addTriangle(p3, p0, p2);
        }

        Loc2GlobVec.push_back(i);
        Loc2GlobVec.push_back(i);
        sofa::type::vector<Index> out_info;
        out_info.push_back((Index)Loc2GlobVec.size()-2);
        out_info.push_back((Index)Loc2GlobVec.size()-1);
        In2OutMap[i]=out_info;
    }

    // Need to fully init the target topology
    toModel->init();

    //to_tstm->propagateTopologicalChanges();

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

Index Quad2TriangleTopologicalMapping::getFromIndex(Index ind)
{
    return ind; // identity
}

void Quad2TriangleTopologicalMapping::updateTopologicalMappingTopDown()
{
    using namespace container::dynamic;

    if (this->d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid)
        return;

    SCOPED_TIMER("Update Quad2TriangleTopologicalMapping");

    TriangleSetTopologyModifier *to_tstm;
    toModel->getContext()->get(to_tstm);

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
            to_tstm->notifyEndingEvent();
            break;
        }

        case core::topology::QUADSADDED:
        {
            const auto & quadArray=fromModel->getQuads();

            const Topology::SetIndices & tab = ( static_cast< const QuadsAdded *>( *itBegin ) )->getArray();

            sofa::type::vector< core::topology::BaseMeshTopology::Triangle > triangles_to_create;
            sofa::type::vector< Index > trianglesIndexList;
            auto nb_elems = toModel->getNbTriangles();

            for (unsigned int i = 0; i < tab.size(); ++i)
            {
                unsigned int k = tab[i];

                const auto & p0 = quadArray[k][0];
                const auto & p1 = quadArray[k][1];
                const auto & p2 = quadArray[k][2];
                const auto & p3 = quadArray[k][3];
                const auto tri1 = core::topology::BaseMeshTopology::Triangle((unsigned int) p0, (unsigned int) p1, (unsigned int) p2);
                const auto tri2 = core::topology::BaseMeshTopology::Triangle((unsigned int) p0, (unsigned int) p2, (unsigned int) p3);

                triangles_to_create.push_back(tri1);
                trianglesIndexList.push_back(nb_elems);
                triangles_to_create.push_back(tri2);
                trianglesIndexList.push_back(nb_elems+1);
                nb_elems+=2;

                Loc2GlobVec.push_back(k);
                Loc2GlobVec.push_back(k);
                sofa::type::vector<Index> out_info;
                out_info.push_back((Index)Loc2GlobVec.size()-2);
                out_info.push_back((Index)Loc2GlobVec.size()-1);
                In2OutMap[k]=out_info;

            }

            to_tstm->addTriangles(triangles_to_create) ;
            break;
        }
        case core::topology::QUADSREMOVED:
        {
            const Topology::SetIndices & tab = ( static_cast< const QuadsRemoved *>( *itBegin ) )->getArray();

            unsigned int last = (unsigned int)fromModel->getNbQuads() - 1;

            int ind_tmp;

            sofa::type::vector<Index> ind_real_last;
            auto ind_last = toModel->getNbTriangles();

            for (unsigned int i = 0; i < tab.size(); ++i)
            {
                unsigned int k = tab[i];
                sofa::type::vector<Index> ind_k;

                auto iter_1 = In2OutMap.find(k);
                if(iter_1 != In2OutMap.end())
                {

                    unsigned int t1 = In2OutMap[k][0];
                    unsigned int t2 = In2OutMap[k][1];

                    ind_last = ind_last - 1;

                    ind_k = In2OutMap[k];
                    ind_real_last = ind_k;

                    auto iter_2 = In2OutMap.find(last);
                    if(iter_2 != In2OutMap.end())
                    {

                        ind_real_last = In2OutMap[last];

                        if (k != last)
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
                        msg_warning() << "Quad2TriangleTopologicalMapping - In2OutMap should have the quad " << last;
                    }

                    if (ind_k[1] != ind_last)
                    {

                        In2OutMap.erase(In2OutMap.find(Loc2GlobVec[ind_last]));
                        In2OutMap[Loc2GlobVec[ind_last]] = ind_k;

                        sofa::type::vector<Index> out_info;
                        out_info.push_back(ind_last);
                        out_info.push_back(ind_last-1);

                        In2OutMap.erase(In2OutMap.find(Loc2GlobVec[ind_k[1]]));
                        In2OutMap[Loc2GlobVec[ind_k[1]]] = out_info;

                        ind_tmp = Loc2GlobVec[ind_k[1]];
                        Loc2GlobVec[ind_k[1]] = Loc2GlobVec[ind_last];
                        Loc2GlobVec[ind_last] = ind_tmp;

                    }

                    ind_last = ind_last-1;

                    if (ind_k[0] != ind_last)
                    {

                        ind_tmp = Loc2GlobVec[ind_k[0]];
                        Loc2GlobVec[ind_k[0]] = Loc2GlobVec[ind_last];
                        Loc2GlobVec[ind_last] = ind_tmp;

                    }

                    In2OutMap.erase(In2OutMap.find(Loc2GlobVec[Loc2GlobVec.size() - 1]));

                    Loc2GlobVec.resize( Loc2GlobVec.size() - 2 );

                    sofa::type::vector< Index > triangles_to_remove;
                    triangles_to_remove.push_back(t1);
                    triangles_to_remove.push_back(t2);

                    to_tstm->removeTriangles(triangles_to_remove, true, false);

                }
                else
                {
                    msg_info() << "In2OutMap should have the quad " << k ;
                }

                --last;
            }

            break;
        }

        case core::topology::POINTSREMOVED:
        {
            const Topology::SetIndices & tab = ( static_cast< const sofa::core::topology::PointsRemoved * >( *itBegin ) )->getArray();

            sofa::type::vector<Index> indices;

            for(unsigned int i = 0; i < tab.size(); ++i)
            {
                indices.push_back(tab[i]);
            }

            sofa::type::vector<Index>& tab_indices = indices;

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

            Topology::SetIndices& tab_indices = indices;
            Topology::SetIndices& inv_tab_indices = inv_indices;

            to_tstm->renumberPoints(tab_indices, inv_tab_indices, false);
            break;
        }


        case core::topology::POINTSADDED:
        {
            const auto * ta=static_cast< const sofa::core::topology::PointsAdded * >( *itBegin );

            to_tstm->addPoints(ta->getNbAddedVertices(), ta->ancestorsList, ta->coefs, false);
            break;
        }
        default:
            break;
        };

        ++itBegin;
    }
}


} //namespace sofa::component::topology::mapping
