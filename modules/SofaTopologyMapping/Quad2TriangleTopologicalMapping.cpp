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
#include <sofa/helper/AdvancedTimer.h>

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
    bool modelsOk = true;
    if (!fromModel)
    {
        // If the input topology link isn't set by the user, the TopologicalMapping::create method tries to find it.
        // If it is null at this point, it means no input mesh topology could be found.
        msg_error() << "No input mesh topology found. Consider setting the '" << fromModel.getName() << "' data attribute.";
        modelsOk = false;
    }

    if (!toModel)
    {
        // If the output topology link isn't set by the user, the TopologicalMapping::create method tries to find it.
        // If it is null at this point, it means no output mesh topology could be found.
        msg_error() << "No output mesh topology found. Consider setting the '" << toModel.getName() << "' data attribute.";
        modelsOk = false;
    }

    // Making sure the output topology is derived from the triangle topology container
    if (!dynamic_cast<TriangleSetTopologyContainer *>(toModel.get())) {
        msg_error() << "The output topology '" << toModel.getPath() << "' is not a derived class of TriangleSetTopologyContainer. "
                    << "Consider setting the '" << toModel.getName() << "' data attribute to a valid"
                                                                        " TriangleSetTopologyContainer derived object.";
        modelsOk = false;
    } else {
        // Making sure a topology modifier exists at the same level as the output topology
        TriangleSetTopologyModifier *to_tstm;
        toModel->getContext()->get(to_tstm);
        if (!to_tstm)
        {
            msg_error() << "No TriangleSetTopologyModifier found in the output topology node '"
                        << toModel->getContext()->getName() << "'.";
            modelsOk = false;
        }
    }

    if (!modelsOk)
    {
        this->m_componentstate = sofa::core::objectmodel::ComponentState::Invalid;
        return;
    }


    // Clear output topology
    toModel->clear();

    // Set the same number of points
    toModel->setNbPoints(fromModel->getNbPoints());


    const sofa::helper::vector<core::topology::BaseMeshTopology::Quad> &quadArray=fromModel->getQuads();
    sofa::helper::vector <unsigned int>& Loc2GlobVec = *(Loc2GlobDataVec.beginEdit());

    Loc2GlobVec.clear();
    In2OutMap.clear();

    // These values are only correct if the mesh is a grid topology
    int nx = 2;
    int ny = 1;

    {
        const auto * grid = dynamic_cast<const topology::GridTopology*>(fromModel.get());
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
        this->m_componentstate = sofa::core::objectmodel::ComponentState::Invalid;
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
        sofa::helper::vector<unsigned int> out_info;
        out_info.push_back((unsigned int)Loc2GlobVec.size()-2);
        out_info.push_back((unsigned int)Loc2GlobVec.size()-1);
        In2OutMap[i]=out_info;
    }

    // Need to fully init the target topology
    toModel->init();

    //to_tstm->propagateTopologicalChanges();
    Loc2GlobDataVec.endEdit();

    this->m_componentstate = sofa::core::objectmodel::ComponentState::Valid;
}

unsigned int Quad2TriangleTopologicalMapping::getFromIndex(unsigned int ind)
{
    return ind; // identity
}

void Quad2TriangleTopologicalMapping::updateTopologicalMappingTopDown()
{

    if (this->m_componentstate != sofa::core::objectmodel::ComponentState::Valid)
        return;

    sofa::helper::AdvancedTimer::stepBegin("Update Quad2TriangleTopologicalMapping");

    TriangleSetTopologyModifier *to_tstm;
    toModel->getContext()->get(to_tstm);

    auto itBegin=fromModel->beginChange();
    auto itEnd=fromModel->endChange();

    sofa::helper::vector <unsigned int>& Loc2GlobVec = *(Loc2GlobDataVec.beginEdit());

    while( itBegin != itEnd )
    {
        TopologyChangeType changeType = (*itBegin)->getChangeType();
        std::string topoChangeType = "Tetra2TriangleTopologicalMapping - " + parseTopologyChangeTypeToString(changeType);
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

        case core::topology::QUADSADDED:
        {
            const auto & quadArray=fromModel->getQuads();

            const Topology::SetIndices & tab = ( static_cast< const QuadsAdded *>( *itBegin ) )->getArray();

            sofa::helper::vector< core::topology::BaseMeshTopology::Triangle > triangles_to_create;
            sofa::helper::vector< unsigned int > trianglesIndexList;
            auto nb_elems = (unsigned int)toModel->getNbTriangles();

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
                sofa::helper::vector<unsigned int> out_info;
                out_info.push_back((unsigned int)Loc2GlobVec.size()-2);
                out_info.push_back((unsigned int)Loc2GlobVec.size()-1);
                In2OutMap[k]=out_info;

            }

            to_tstm->addTrianglesProcess(triangles_to_create) ;
            to_tstm->addTrianglesWarning(triangles_to_create.size(), triangles_to_create, trianglesIndexList) ;
            to_tstm->propagateTopologicalChanges();
            break;
        }
        case core::topology::QUADSREMOVED:
        {
            const Topology::SetIndices & tab = ( static_cast< const QuadsRemoved *>( *itBegin ) )->getArray();

            unsigned int last = (unsigned int)fromModel->getNbQuads() - 1;

            int ind_tmp;

            sofa::helper::vector<unsigned int> ind_real_last;
            auto ind_last = (unsigned int)toModel->getNbTriangles();

            for (unsigned int i = 0; i < tab.size(); ++i)
            {
                unsigned int k = tab[i];
                sofa::helper::vector<unsigned int> ind_k;

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

                    if (ind_k[0] != ind_last)
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

            break;
        }

        case core::topology::POINTSREMOVED:
        {
            const Topology::SetIndices & tab = ( static_cast< const sofa::component::topology::PointsRemoved * >( *itBegin ) )->getArray();

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
            const Topology::SetIndices & tab = ( static_cast< const PointsRenumbering * >( *itBegin ) )->getIndexArray();
            const Topology::SetIndices & inv_tab = ( static_cast< const PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

            sofa::helper::vector<unsigned int> indices;
            sofa::helper::vector<unsigned int> inv_indices;

            for(unsigned int i = 0; i < tab.size(); ++i)
            {
                indices.push_back(tab[i]);
                inv_indices.push_back(inv_tab[i]);
            }

            Topology::SetIndices& tab_indices = indices;
            Topology::SetIndices& inv_tab_indices = inv_indices;

            to_tstm->renumberPointsWarning(tab_indices, inv_tab_indices, false);
            to_tstm->propagateTopologicalChanges();
            to_tstm->renumberPointsProcess(tab_indices, inv_tab_indices, false);

            break;
        }


        case core::topology::POINTSADDED:
        {
            const auto * ta=static_cast< const sofa::component::topology::PointsAdded * >( *itBegin );

            to_tstm->addPointsProcess(ta->getNbAddedVertices());
            to_tstm->addPointsWarning(ta->getNbAddedVertices(), ta->ancestorsList, ta->coefs, false);
            to_tstm->propagateTopologicalChanges();

            break;
        }
        default:
            break;
        };

        sofa::helper::AdvancedTimer::stepEnd(topoChangeType);
        ++itBegin;
    }
    to_tstm->propagateTopologicalChanges();
    Loc2GlobDataVec.endEdit();

    sofa::helper::AdvancedTimer::stepEnd("Update Quad2TriangleTopologicalMapping");
}


} // namespace topology

} // namespace component

} // namespace sofa

