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
#pragma once
#include <sofa/component/collision/geometry/LineCollisionModel.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/component/collision/geometry/PointCollisionModel.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/collision/geometry/CubeCollisionModel.h>
#include <sofa/core/CollisionElement.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/topology/TopologyChange.h>
#include <vector>

namespace sofa::component::collision::geometry
{

using core::topology::BaseMeshTopology;

template<class DataTypes>
LineCollisionModel<DataTypes>::LineCollisionModel()
    : d_bothSide(initData(&d_bothSide, false, "bothSide", "activate collision on both side of the line model (when surface normals are defined on these lines)") )
    , d_displayFreePosition(initData(&d_displayFreePosition, false, "displayFreePosition", "Display Collision Model Points free position(in green)") )
    , l_topology(initLink("topology", "link to the topology container"))
    , mstate(nullptr), topology(nullptr), meshRevision(-1)
{
    enum_type = LINE_TYPE;
}


template<class DataTypes>
void LineCollisionModel<DataTypes>::resize(sofa::Size size)
{
    this->core::CollisionModel::resize(size);
    elems.resize(size);
}

template<class DataTypes>
void LineCollisionModel<DataTypes>::init()
{
    this->CollisionModel::init();
    mstate = dynamic_cast< core::behavior::MechanicalState<DataTypes>* > (getContext()->getMechanicalState());
    this->getContext()->get(mpoints);

    if (mstate==nullptr)
    {
        msg_error() << "LineModel requires a Vec3 Mechanical Model";
        return;
    }

    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    core::topology::BaseMeshTopology *bmt = l_topology.get();
    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    if (!bmt)
    {
        msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name << ". LineCollisionModel<sofa::defaulttype::Vec3Types> requires a MeshTopology";
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
    resize( bmt->getNbEdges() );

    for(core::topology::BaseMeshTopology::EdgeID i = 0; i < bmt->getNbEdges(); i++)
    {
        elems[i].p[0] = bmt->getEdge(i)[0];
        elems[i].p[1] = bmt->getEdge(i)[1];
    }

    updateFromTopology();
}

template<class DataTypes>
void LineCollisionModel<DataTypes>::handleTopologyChange()
{
    core::topology::BaseMeshTopology *bmt = l_topology.get();
    if (bmt)
    {
        resize(bmt->getNbEdges());

        for(sofa::Size i = 0; i < bmt->getNbEdges(); i++)
        {
            elems[i].p[0] = bmt->getEdge(i)[0];
            elems[i].p[1] = bmt->getEdge(i)[1];
        }

        needsUpdate = true;
    }
    if (bmt)
    {
        std::list<const sofa::core::topology::TopologyChange *>::const_iterator itBegin = bmt->beginChange();
        const std::list<const sofa::core::topology::TopologyChange *>::const_iterator itEnd = bmt->endChange();

        while( itBegin != itEnd )
        {
            const core::topology::TopologyChangeType changeType = (*itBegin)->getChangeType();

            switch( changeType )
            {
            case core::topology::ENDING_EVENT :
            {
                needsUpdate = true;
                break;
            }


            case core::topology::EDGESADDED :
            {
                const core::topology::EdgesAdded *ta = static_cast< const core::topology::EdgesAdded * >( *itBegin );

                const sofa::Size elemsSize = sofa::Size(elems.size());

                for (sofa::Size i = 0; i < ta->getNbAddedEdges(); ++i)
                {
                    elems[elemsSize - ta->getNbAddedEdges() + i].p[0] = (ta->edgeArray[i])[0];
                    elems[elemsSize - ta->getNbAddedEdges() + i].p[1] = (ta->edgeArray[i])[1];
                }

                resize(elemsSize);
                needsUpdate = true;

                break;
            }

            case core::topology::EDGESREMOVED :
            {
                sofa::Index last;
                sofa::Index ind_last;

                if (bmt)
                {
                    last = bmt->getNbEdges() - 1;
                }
                else
                {
                    last = sofa::Size(elems.size()) -1;
                }

                const auto &tab = ( static_cast< const core::topology::EdgesRemoved *>( *itBegin ) )->getArray();

                LineData tmp;
                for (sofa::Size i = 0; i < tab.size(); ++i)
                {
                    sofa::Index ind_k = tab[i];

                    tmp = elems[ind_k];
                    elems[ind_k] = elems[last];
                    elems[last] = tmp;

                    ind_last = sofa::Size(elems.size()) - 1;

                    if(last != ind_last)
                    {
                        tmp = elems[last];
                        elems[last] = elems[ind_last];
                        elems[ind_last] = tmp;
                    }
                    resize(sofa::Size(elems.size()) - 1 );

                    --last;
                }

                needsUpdate=true;
                break;
            }

            case core::topology::POINTSREMOVED :
            {
                if (bmt)
                {
                    sofa::Index last = bmt->getNbPoints() - 1;

                    sofa::Index i,j;
                    const auto& tab = ( static_cast< const core::topology::PointsRemoved * >( *itBegin ) )->getArray();

                    sofa::type::vector<sofa::Index> lastIndexVec;
                    for(sofa::Index i_init = 0; i_init < tab.size(); ++i_init)
                    {
                        lastIndexVec.push_back(last - i_init);
                    }

                    for ( i = 0; i < tab.size(); ++i)
                    {
                        sofa::Index i_next = i;
                        bool is_reached = false;
                        while( (!is_reached) && (i_next < lastIndexVec.size() - 1))
                        {
                            i_next += 1 ;
                            is_reached = is_reached || (lastIndexVec[i_next] == tab[i]);
                        }

                        if(is_reached)
                        {
                            lastIndexVec[i_next] = lastIndexVec[i];
                        }

                        const auto &shell = bmt->getEdgesAroundVertex(lastIndexVec[i]);

                        for (j = 0; j < shell.size(); ++j)
                        {
                            sofa::Index ind_j = shell[j];

                            if (elems[ind_j].p[0] == last)
                            {
                                elems[ind_j].p[0] = tab[i];
                            }
                            else if (elems[ind_j].p[1] == last)
                            {
                                elems[ind_j].p[1] = tab[i];
                            }
                        }

                        --last;
                    }
                }

                needsUpdate=true;

                break;
            }

            case core::topology::POINTSRENUMBERING:
            {
                if (bmt)
                {
                    sofa::Index i;

                    const auto& tab = ( static_cast< const core::topology::PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

                    for ( i = 0; i < elems.size(); ++i)
                    {
                        elems[i].p[0]  = tab[elems[i].p[0]];
                        elems[i].p[1]  = tab[elems[i].p[1]];
                    }
                }

                break;
            }

            default:
                // Ignore events that are not Edge  related.
                break;
            };

            ++itBegin;
        }
    }
}

template<class DataTypes>
void LineCollisionModel<DataTypes>::updateFromTopology()
{
    core::topology::BaseMeshTopology *bmt = l_topology.get();
    if (bmt)
    {
        const int revision = bmt->getRevision();
        if (revision == meshRevision)
            return;

        needsUpdate = true;

        const sofa::Size nbPoints = mstate->getSize();
        const sofa::Size nbLines = bmt->getNbEdges();

        resize( nbLines );
        sofa::Index index = 0;

        for (sofa::Size i = 0; i < nbLines; i++)
        {
            core::topology::BaseMeshTopology::Line idx = bmt->getEdge(i);

            if (idx[0] >= nbPoints || idx[1] >= nbPoints)
            {
                msg_error() << "Out of range index in Line " << i << ": " << idx[0] << " " << idx[1] << " : total points (size of the MState) = " << nbPoints;
                continue;
            }

            elems[index].p[0] = idx[0];
            elems[index].p[1] = idx[1];
            ++index;
        }

        meshRevision = revision;
    }
}

template<class DataTypes>
void LineCollisionModel<DataTypes>::drawCollisionModel(const core::visual::VisualParams* vparams)
{
    if (vparams->displayFlags().getShowWireFrame())
    {
        vparams->drawTool()->setPolygonMode(0, true);
    }

    std::vector<helper::visual::DrawTool::Vec3> points;
    points.reserve(size * 2);
    for (sofa::Size i = 0; i < size; i++)
    {
        TLine<DataTypes> l(this, i);
        if (l.isActive())
        {
            // note the conversion if !std::is_same_v<helper::visual::DrawTool::Vec3, Coord>
            points.emplace_back(helper::visual::DrawTool::Vec3{l.p1()});
            points.emplace_back(helper::visual::DrawTool::Vec3{l.p2()});
        }
    }

    const auto c = getColor4f();
    vparams->drawTool()->drawLines(points, 1, sofa::type::RGBAColor(c[0], c[1], c[2], c[3]));

    if (d_displayFreePosition.getValue())
    {
        std::vector<type::Vec3> pointsFree;
        for (sofa::Size i = 0; i < size; i++)
        {
            TLine<DataTypes> l(this, i);
            if (l.isActive())
            {
                pointsFree.push_back(l.p1Free());
                pointsFree.push_back(l.p2Free());
            }
        }

        vparams->drawTool()->drawLines(pointsFree, 1,
                                       sofa::type::RGBAColor(0.0f, 1.0f, 0.2f, 1.0f));
    }

    if (vparams->displayFlags().getShowWireFrame())
    {
        vparams->drawTool()->setPolygonMode(0, false);
    }
}

template<class DataTypes>
bool LineCollisionModel<DataTypes>::canCollideWithElement(sofa::Index index, CollisionModel* model2, sofa::Index index2)
{
    if (!this->bSelfCollision.getValue()) return true;
    if (this->getContext() != model2->getContext()) return true;
    core::topology::BaseMeshTopology *topology = l_topology.get();
    /*
        TODO : separate 2 case: the model is only composed of lines or is composed of triangles
    */
    sofa::Index p11 = elems[index].p[0];
    sofa::Index p12 = elems[index].p[1];


    if (!topology)
    {
        msg_error() << "no topology found";
        return true;
    }
    const auto& EdgesAroundVertex11 =topology->getEdgesAroundVertex(p11);
    const auto& EdgesAroundVertex12 =topology->getEdgesAroundVertex(p12);

    if (model2 == this)
    {
        // if point in common, return false:
        const sofa::Index p21 = elems[index2].p[0];
        const sofa::Index p22 = elems[index2].p[1];

        if (p11==p21 || p11==p22 || p12==p21 || p12==p22)
            return false;


        // in the neighborhood, if we find a segment in common, we cancel the collision
        const auto& EdgesAroundVertex21 =topology->getEdgesAroundVertex(p21);
        const auto& EdgesAroundVertex22 =topology->getEdgesAroundVertex(p22);

        for (sofa::Size i1=0; i1<EdgesAroundVertex11.size(); i1++)
        {
            const sofa::Index e11 = EdgesAroundVertex11[i1];
            sofa::Size i2;
            for (i2=0; i2<EdgesAroundVertex21.size(); i2++)
            {
                if (e11==EdgesAroundVertex21[i2])
                    return false;
            }
            for (i2=0; i2<EdgesAroundVertex22.size(); i2++)
            {
                if (e11==EdgesAroundVertex22[i2])
                    return false;
            }
        }

        for (sofa::Size i1=0; i1<EdgesAroundVertex12.size(); i1++)
        {
            const sofa::Index e11 = EdgesAroundVertex12[i1];
            sofa::Size i2;
            for (i2=0; i2<EdgesAroundVertex21.size(); i2++)
            {
                if (e11==EdgesAroundVertex21[i2])
                    return false;
            }
            for (i2=0; i2<EdgesAroundVertex22.size(); i2++)
            {
                if (e11==EdgesAroundVertex22[i2])
                    return false;
            }

        }
        return true;



    }
    else if (model2 == mpoints)
    {
        // if point belong to the segment, return false
        if (index2==p11 || index2==p12)
            return false;

        // if the point belong to the a segment in the neighborhood, return false
        for (sofa::Size i1=0; i1<EdgesAroundVertex11.size(); i1++)
        {
            sofa::Index e11 = EdgesAroundVertex11[i1];
            p11 = elems[e11].p[0];
            p12 = elems[e11].p[1];
            if (index2==p11 || index2==p12)
                return false;
        }
        for (sofa::Size i1=0; i1<EdgesAroundVertex12.size(); i1++)
        {
            sofa::Index e12 = EdgesAroundVertex12[i1];
            p11 = elems[e12].p[0];
            p12 = elems[e12].p[1];
            if (index2==p11 || index2==p12)
                return false;
        }
        return true;

        // only removes collision with the two vertices of the segment
        // TODO: neighborhood search !
    }
    else
        return model2->canCollideWithElement(index2, this, index);
}

template<class DataTypes>
void LineCollisionModel<DataTypes>::computeBoundingTree(int maxDepth)
{
    CubeCollisionModel* cubeModel = createPrevious<CubeCollisionModel>();
    updateFromTopology();
    if (needsUpdate) cubeModel->resize(0);
    if (!isMoving() && !cubeModel->empty() && !needsUpdate) return; // No need to recompute BBox if immobile

    needsUpdate = false;

    cubeModel->resize(size);
    if (!empty())
    {
        const SReal distance = this->d_contactDistance.getValue();
        const auto& positions = this->mstate->read(core::vec_id::read_access::position)->getValue();
        for (sofa::Size i=0; i<size; i++)
        {
            type::Vec3 minElem, maxElem;
            const type::Vec3& pt1 = positions[this->elems[i].p[0]];
            const type::Vec3& pt2 = positions[this->elems[i].p[1]];

            for (int c = 0; c < 3; c++)
            {
                minElem[c] = pt1[c];
                maxElem[c] = pt1[c];
                if (pt2[c] > maxElem[c]) maxElem[c] = pt2[c];
                else if (pt2[c] < minElem[c]) minElem[c] = pt2[c];
                minElem[c] -= distance;
                maxElem[c] += distance;
            }

            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

template<class DataTypes>
void LineCollisionModel<DataTypes>::computeContinuousBoundingTree(SReal dt, int maxDepth)
{
    CubeCollisionModel* cubeModel = createPrevious<CubeCollisionModel>();
    updateFromTopology();
    if (needsUpdate) cubeModel->resize(0);
    if (!isMoving() && !cubeModel->empty() && !needsUpdate) return; // No need to recompute BBox if immobile

    needsUpdate=false;
    type::Vec3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        const SReal distance = (SReal)this->d_contactDistance.getValue();
        for (sofa::Size i=0; i<size; i++)
        {
            TLine<DataTypes> t(this,i);
            const type::Vec3& pt1 = t.p1();
            const type::Vec3& pt2 = t.p2();
            const type::Vec3 pt1v = pt1 + t.v1()*dt;
            const type::Vec3 pt2v = pt2 + t.v2()*dt;

            for (int c = 0; c < 3; c++)
            {
                minElem[c] = pt1[c];
                maxElem[c] = pt1[c];
                if (pt2[c] > maxElem[c]) maxElem[c] = pt2[c];
                else if (pt2[c] < minElem[c]) minElem[c] = pt2[c];

                if (pt1v[c] > maxElem[c]) maxElem[c] = pt1v[c];
                else if (pt1v[c] < minElem[c]) minElem[c] = pt1v[c];
                if (pt2v[c] > maxElem[c]) maxElem[c] = pt2v[c];
                else if (pt2v[c] < minElem[c]) minElem[c] = pt2v[c];
                minElem[c] -= distance;
                maxElem[c] += distance;
            }
            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

template<class DataTypes>
int LineCollisionModel<DataTypes>::getLineFlags(sofa::Index i)
{
    int f = 0;
    if (topology)
    {
        sofa::core::topology::BaseMeshTopology::Edge e(elems[i].p[0], elems[i].p[1]);
        i = getElemEdgeIndex(i);
        if (i < topology->getNbEdges())
        {
            for (sofa::Index j=0; j<2; ++j)
            {
                const auto& eav = topology->getEdgesAroundVertex(e[j]);
                if (eav[0] == (sofa::core::topology::BaseMeshTopology::EdgeID)i)
                    f |= (FLAG_P1 << j);
                if (eav.size() == 1)
                    f |= (FLAG_BP1 << j);
            }
        }
    }
    return f;
}

template<class DataTypes>
void LineCollisionModel<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    SOFA_UNUSED(params);

    if( onlyVisible && !sofa::core::visual::VisualParams::defaultInstance()->displayFlags().getShowCollisionModels())
        return;

    const auto& positions = this->mstate->read(core::vec_id::read_access::position)->getValue();
    type::BoundingBox bbox;

    for (sofa::Size i=0; i<size; i++)
    {
        const Element e(this,i);
        const Coord& pt1 = positions[this->elems[i].p[0]];
        const Coord& pt2 = positions[this->elems[i].p[1]];

        bbox.include(pt1);
        bbox.include(pt2);
    }

    this->f_bbox.setValue(bbox);
}

template<class DataTypes>
inline sofa::Index TLine<DataTypes>::i1() const { return this->model->elems[this->index].p[0]; }

template<class DataTypes>
inline sofa::Index TLine<DataTypes>::i2() const { return this->model->elems[this->index].p[1]; }

template<class DataTypes>
inline const typename DataTypes::Coord& TLine<DataTypes>::p1() const { return this->model->mstate->read(core::vec_id::read_access::position)->getValue()[this->model->elems[this->index].p[0]]; }

template<class DataTypes>
inline const typename DataTypes::Coord& TLine<DataTypes>::p2() const { return this->model->mstate->read(core::vec_id::read_access::position)->getValue()[this->model->elems[this->index].p[1]]; }

template<class DataTypes>
inline const typename DataTypes::Coord& TLine<DataTypes>::p(Index i) const {
    return this->model->mstate->read(core::vec_id::read_access::position)->getValue()[this->model->elems[this->index].p[i]];
}

template<class DataTypes>
inline const typename DataTypes::Coord& TLine<DataTypes>::p1Free() const
{
    if (hasFreePosition())
        return this->model->mstate->read(core::vec_id::read_access::freePosition)->getValue()[this->model->elems[this->index].p[0]];
    else
        return p1();
}

template<class DataTypes>
inline const typename DataTypes::Coord& TLine<DataTypes>::p2Free() const
{
    if (hasFreePosition())
        return this->model->mstate->read(core::vec_id::read_access::freePosition)->getValue()[this->model->elems[this->index].p[1]];
    else
        return p2();
}

template<class DataTypes>
inline const typename DataTypes::Deriv& TLine<DataTypes>::v1() const { return this->model->mstate->read(core::vec_id::read_access::velocity)->getValue()[this->model->elems[this->index].p[0]]; }

template<class DataTypes>
inline const typename DataTypes::Deriv& TLine<DataTypes>::v2() const { return this->model->mstate->read(core::vec_id::read_access::velocity)->getValue()[this->model->elems[this->index].p[1]]; }

template<class DataTypes>
inline typename DataTypes::Deriv TLine<DataTypes>::n() const {return (this->model->mpoints->getNormal(this->i1()) + this->model->mpoints->getNormal( this->i2())).normalized();}

template<class DataTypes>
inline typename LineCollisionModel<DataTypes>::Deriv LineCollisionModel<DataTypes>::velocity(sofa::Index index) const { return (mstate->read(core::vec_id::read_access::velocity)->getValue()[elems[index].p[0]] + mstate->read(core::vec_id::read_access::velocity)->getValue()[elems[index].p[1]])/((Real)(2.0)); }

template<class DataTypes>
inline int TLine<DataTypes>::flags() const { return this->model->getLineFlags(this->index); }

template<class DataTypes>
inline bool TLine<DataTypes>::hasFreePosition() const { return this->model->mstate->read(core::vec_id::read_access::freePosition)->isSet(); }


} //namespace sofa::component::collision::geometry
