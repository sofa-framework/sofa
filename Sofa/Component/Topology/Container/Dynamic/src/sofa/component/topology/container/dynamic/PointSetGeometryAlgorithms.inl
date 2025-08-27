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
#include <sofa/component/topology/container/dynamic/PointSetGeometryAlgorithms.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/objectmodel/Tag.h>
#include <sofa/simulation/fwd.h>
#include <sofa/simulation/Simulation.h>

namespace sofa::component::topology::container::dynamic
{

using sofa::core::objectmodel::ComponentState;

template <class DataTypes>
 PointSetGeometryAlgorithms< DataTypes >::PointSetGeometryAlgorithms()        
    : GeometryAlgorithms()
    , d_showIndicesScale (core::objectmodel::Base::initData(&d_showIndicesScale, (float) 0.02, "showIndicesScale", "Debug : scale for view topology indices"))
    , d_showPointIndices (core::objectmodel::Base::initData(&d_showPointIndices, (bool) false, "showPointIndices", "Debug : view Point indices"))
    , d_tagMechanics( initData(&d_tagMechanics,std::string(),"tagMechanics","Tag of the Mechanical Object"))
    , l_topology(initLink("topology", "link to the topology container"))
{
}

template <class DataTypes>
void PointSetGeometryAlgorithms< DataTypes >::init()
{
    this->d_componentState.setValue(ComponentState::Invalid);
    if ( this->d_tagMechanics.getValue().size()>0) {
        const sofa::core::objectmodel::Tag mechanicalTag(this->d_tagMechanics.getValue());
        object = this->getContext()->core::objectmodel::BaseContext::template get< core::State< DataTypes > >(mechanicalTag,sofa::core::objectmodel::BaseContext::SearchUp);
    } else {
        object = this->getContext()->core::objectmodel::BaseContext::template get< core::State< DataTypes > >();
    }
    core::topology::GeometryAlgorithms::init();

    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    this->m_topology = l_topology.get();
    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    if (!m_topology)
    {
        msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name << ". TriangleCollisionModel<sofa::defaulttype::Vec3Types> requires a Triangular Topology";
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    if(this->object ==nullptr)
    {
        msg_error() << "Unable to get a valid state from the context";
        return;
    }
    this->d_componentState.setValue(ComponentState::Valid);
}

template <class DataTypes>
void PointSetGeometryAlgorithms< DataTypes >::reinit()
{
}

template <class DataTypes>
float PointSetGeometryAlgorithms< DataTypes >::getIndicesScale() const
{
    const sofa::type::BoundingBox& bbox = this->getContext()->f_bbox.getValue();
    const float bbDiff = float((bbox.maxBBox() - bbox.minBBox()).norm());
    if (std::isinf(bbDiff))
        return d_showIndicesScale.getValue();
    else
        return bbDiff * d_showIndicesScale.getValue();
}


template <class DataTypes>
typename DataTypes::Coord PointSetGeometryAlgorithms<DataTypes>::getPointSetCenter() const
{
    typename DataTypes::Coord center;
    // get current positions
    const typename DataTypes::VecCoord& p =(object->read(core::vec_id::read_access::position)->getValue());

    const int numVertices = this->m_topology->getNbPoints();
    for(int i=0; i<numVertices; ++i)
    {
        center += p[i];
    }

    center /= numVertices;
    return center;
}

template<class DataTypes>
void  PointSetGeometryAlgorithms<DataTypes>::getEnclosingSphere(typename DataTypes::Coord &center,
        typename DataTypes::Real &radius) const
{
    // get current positions
    const typename DataTypes::VecCoord& p =(object->read(core::vec_id::read_access::position)->getValue());

    const unsigned int numVertices = this->m_topology->getNbPoints();
    for(unsigned int i=0; i<numVertices; ++i)
    {
        center += p[i];
    }
    center /= numVertices;
    radius = (Real) 0;

    for(unsigned int i=0; i<numVertices; ++i)
    {
        const CPos dp = DataTypes::getCPos(center)-DataTypes::getCPos(p[i]);
        const Real val = dot(dp,dp);
        if(val > radius)
            radius = val;
    }
    radius = (Real)sqrt((double) radius);
}

template<class DataTypes>
void  PointSetGeometryAlgorithms<DataTypes>::getAABB(typename DataTypes::Real bb[6] ) const
{
    CPos minCoord, maxCoord;
    getAABB(minCoord, maxCoord);

    bb[0] = (NC>0) ? minCoord[0] : (Real)0;
    bb[1] = (NC>1) ? minCoord[1] : (Real)0;
    bb[2] = (NC>2) ? minCoord[2] : (Real)0;
    bb[3] = (NC>0) ? maxCoord[0] : (Real)0;
    bb[4] = (NC>1) ? maxCoord[1] : (Real)0;
    bb[5] = (NC>2) ? maxCoord[2] : (Real)0;
}

template<class DataTypes>
void PointSetGeometryAlgorithms<DataTypes>::getAABB(CPos& minCoord, CPos& maxCoord) const
{
    // get current positions
    const VecCoord& p =(object->read(core::vec_id::read_access::position)->getValue());

    minCoord = DataTypes::getCPos(p[0]);
    maxCoord = minCoord;

    for(unsigned int i=1; i<p.size(); ++i)
    {
        CPos pi = DataTypes::getCPos(p[i]);
        for (unsigned int c=0; c<pi.size(); ++c)
            if(minCoord[c] > pi[c]) minCoord[c] = pi[c];
            else if(maxCoord[c] < pi[c]) maxCoord[c] = pi[c];
    }
}

template<class DataTypes>
const typename DataTypes::Coord& PointSetGeometryAlgorithms<DataTypes>::getPointPosition(const PointID pointId) const
{
    // get current positions
    const typename DataTypes::VecCoord& p =(object->read(core::vec_id::read_access::position)->getValue());

    return p[pointId];
}

template<class DataTypes>
const typename DataTypes::Coord& PointSetGeometryAlgorithms<DataTypes>::getPointRestPosition(const PointID pointId) const
{
    // get rest positions
    const typename DataTypes::VecCoord& p = (object->read(core::vec_id::read_access::restPosition)->getValue());

    return p[pointId];
}

template<class DataTypes>
typename PointSetGeometryAlgorithms<DataTypes>::Angle
PointSetGeometryAlgorithms<DataTypes>::computeAngle(PointID ind_p0, PointID ind_p1, PointID ind_p2) const
{
    const double ZERO = 1e-10;
    const typename DataTypes::VecCoord& p =(object->read(core::vec_id::read_access::position)->getValue());
    Coord p0 = p[ind_p0];
    Coord p1 = p[ind_p1];
    Coord p2 = p[ind_p2];
    const double t = (p1 - p0)*(p2 - p0);

    if(fabs(t) < ZERO)
        return RIGHT;
    if(t > 0.0)
        return ACUTE;
    else
        return OBTUSE;
}


template<class DataTypes>
void PointSetGeometryAlgorithms<DataTypes>::initPointsAdded(const type::vector< sofa::Index > &indices, const type::vector< core::topology::PointAncestorElem > &ancestorElems
    , const type::vector< core::VecCoordId >& coordVecs, const type::vector< core::VecDerivId >& derivVecs )
{
    using namespace sofa::core::topology;

    type::vector< VecCoord* > pointsAddedVecCoords;
    type::vector< VecDeriv* > pointsAddedVecDerivs;

    const size_t nbPointCoords = coordVecs.size();
    const size_t nbPointDerivs = derivVecs.size();

    for (size_t i=0; i < nbPointCoords; i++)
    {
        pointsAddedVecCoords.push_back(this->object->write(coordVecs[i])->beginEdit());
    }

    for (size_t i=0; i < nbPointDerivs; i++)
    {
        pointsAddedVecDerivs.push_back(this->object->write(derivVecs[i])->beginEdit());
    }

    for (size_t i=0; i < indices.size(); i++)
    {
        if (ancestorElems[i].index != sofa::InvalidID)
        {
            initPointAdded(indices[i], ancestorElems[i], pointsAddedVecCoords, pointsAddedVecDerivs);
        }
    }

    for (size_t i=0; i < nbPointCoords; i++)
    {
        this->object->write(coordVecs[i])->endEdit();
    }

    for (size_t i=0; i < nbPointDerivs; i++)
    {
        this->object->write(derivVecs[i])->endEdit();
    }
}


template<class DataTypes>
void PointSetGeometryAlgorithms<DataTypes>::initPointAdded(PointID index, const core::topology::PointAncestorElem &ancestorElem
    , const type::vector< VecCoord* >& coordVecs, const type::vector< VecDeriv* >& /*derivVecs*/)

{
    using namespace sofa::core::topology;

    for (unsigned int i = 0; i < coordVecs.size(); i++)
    {
        (*coordVecs[i])[index] = (*coordVecs[i])[ancestorElem.index];
        DataTypes::add((*coordVecs[i])[index], ancestorElem.localCoords[0], ancestorElem.localCoords[1], ancestorElem.localCoords[2]);
    }
}

template <class DataTypes>
bool PointSetGeometryAlgorithms<DataTypes>::mustComputeBBox() const
{
    return this->m_topology->getNbPoints() != 0 && d_showPointIndices.getValue();
}

template<class DataTypes>
void PointSetGeometryAlgorithms<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->disableLighting();

    if (d_showPointIndices.getValue())
    {
        const VecCoord& coords =(this->object->read(core::vec_id::read_access::position)->getValue());
        constexpr auto color4 = sofa::type::RGBAColor::white();
        const float scale = getIndicesScale();

        std::vector<type::Vec3> positions;
        for (unsigned int i =0; i<coords.size(); i++)
        {
            const type::Vec3 center = type::toVec3(DataTypes::getCPos(coords[i]));
            positions.push_back(center);

        }
        vparams->drawTool()->draw3DText_Indices(positions, scale, color4);
    }


}

template <class DataTypes>
void PointSetGeometryAlgorithms<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    SOFA_UNUSED(params);

    if (!onlyVisible) return;
    if (!this->object) return;
    if (!this->m_topology) return;

    if (mustComputeBBox())
    {
        const auto bbox = this->object->computeBBox(); //this may compute twice the mstate bbox, but there is no way to determine if the bbox has already been computed
        this->object->f_bbox.setValue(std::move(bbox));
    }
    this->f_bbox.setValue(type::BoundingBox());
}

} //namespace sofa::component::topology::container::dynamic
