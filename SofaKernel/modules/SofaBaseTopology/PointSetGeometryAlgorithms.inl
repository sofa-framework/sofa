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
#ifndef SOFA_COMPONENT_TOPOLOGY_POINTSETGEOMETRYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_POINTSETGEOMETRYALGORITHMS_INL

#include <SofaBaseTopology/PointSetGeometryAlgorithms.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/core/objectmodel/Tag.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>

namespace sofa
{

namespace component
{

namespace topology
{

template <class DataTypes>
 PointSetGeometryAlgorithms< DataTypes >::PointSetGeometryAlgorithms()        : GeometryAlgorithms()
        ,d_showIndicesScale (core::objectmodel::Base::initData(&d_showIndicesScale, (float) 0.02, "showIndicesScale", "Debug : scale for view topology indices"))
        ,d_showPointIndices (core::objectmodel::Base::initData(&d_showPointIndices, (bool) false, "showPointIndices", "Debug : view Point indices"))
        ,d_tagMechanics( initData(&d_tagMechanics,std::string(),"tagMechanics","Tag of the Mechanical Object"))
    {
    }
template <class DataTypes>
void PointSetGeometryAlgorithms< DataTypes >::init()
{
    if ( this->d_tagMechanics.getValue().size()>0) {
        sofa::core::objectmodel::Tag mechanicalTag(this->d_tagMechanics.getValue());
        object = this->getContext()->core::objectmodel::BaseContext::template get< core::behavior::MechanicalState< DataTypes > >(mechanicalTag,sofa::core::objectmodel::BaseContext::SearchUp);
    } else {
        object = this->getContext()->core::objectmodel::BaseContext::template get< core::behavior::MechanicalState< DataTypes > >();
    }
    core::topology::GeometryAlgorithms::init();
    this->m_topology = this->getContext()->getMeshTopology();
}

template <class DataTypes>
void PointSetGeometryAlgorithms< DataTypes >::reinit()
{
}

template <class DataTypes>
float PointSetGeometryAlgorithms< DataTypes >::getIndicesScale() const
{
    const sofa::defaulttype::BoundingBox& bbox = this->getContext()->f_bbox.getValue();
    return (float)((bbox.maxBBox() - bbox.minBBox()).norm() * d_showIndicesScale.getValue());
}


template <class DataTypes>
typename DataTypes::Coord PointSetGeometryAlgorithms<DataTypes>::getPointSetCenter() const
{
    typename DataTypes::Coord center;
    // get current positions
    const typename DataTypes::VecCoord& p =(object->read(core::ConstVecCoordId::position())->getValue());

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
    const typename DataTypes::VecCoord& p =(object->read(core::ConstVecCoordId::position())->getValue());

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
    const VecCoord& p =(object->read(core::ConstVecCoordId::position())->getValue());

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
    const typename DataTypes::VecCoord& p =(object->read(core::ConstVecCoordId::position())->getValue());

    return p[pointId];
}

template<class DataTypes>
const typename DataTypes::Coord& PointSetGeometryAlgorithms<DataTypes>::getPointRestPosition(const PointID pointId) const
{
    // get rest positions
    const typename DataTypes::VecCoord& p = (object->read(core::ConstVecCoordId::restPosition())->getValue());

    return p[pointId];
}

template<class DataTypes>
typename PointSetGeometryAlgorithms<DataTypes>::Angle
PointSetGeometryAlgorithms<DataTypes>::computeAngle(PointID ind_p0, PointID ind_p1, PointID ind_p2) const
{
    const double ZERO = 1e-10;
    const typename DataTypes::VecCoord& p =(object->read(core::ConstVecCoordId::position())->getValue());
    Coord p0 = p[ind_p0];
    Coord p1 = p[ind_p1];
    Coord p2 = p[ind_p2];
    double t = (p1 - p0)*(p2 - p0);

    if(fabs(t) < ZERO)
        return RIGHT;
    if(t > 0.0)
        return ACUTE;
    else
        return OBTUSE;
}


template<class DataTypes>
void PointSetGeometryAlgorithms<DataTypes>::initPointsAdded(const helper::vector< unsigned int > &indices, const helper::vector< core::topology::PointAncestorElem > &ancestorElems
    , const helper::vector< core::VecCoordId >& coordVecs, const helper::vector< core::VecDerivId >& derivVecs )
{
    using namespace sofa::core::topology;

    helper::vector< VecCoord* > pointsAddedVecCoords;
    helper::vector< VecDeriv* > pointsAddedVecDerivs;

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
        if (ancestorElems[i].index != BaseMeshTopology::InvalidID)
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
void PointSetGeometryAlgorithms<DataTypes>::initPointAdded(unsigned int index, const core::topology::PointAncestorElem &ancestorElem
    , const helper::vector< VecCoord* >& coordVecs, const helper::vector< VecDeriv* >& /*derivVecs*/)

{
    using namespace sofa::core::topology;

    for (unsigned int i = 0; i < coordVecs.size(); i++)
    {
        (*coordVecs[i])[index] = (*coordVecs[i])[ancestorElem.index];
        DataTypes::add((*coordVecs[i])[index], ancestorElem.localCoords[0], ancestorElem.localCoords[1], ancestorElem.localCoords[2]);
    }
}


template<class DataTypes>
void PointSetGeometryAlgorithms<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (d_showPointIndices.getValue())
    {
        sofa::defaulttype::Vec<3, SReal> sceneMinBBox, sceneMaxBBox;
        const VecCoord& coords =(this->object->read(core::ConstVecCoordId::position())->getValue());

        sofa::simulation::Node* context = dynamic_cast<sofa::simulation::Node*>(this->getContext());
        defaulttype::Vec4f color4(1.0, 1.0, 1.0, 1.0);

        sofa::simulation::getSimulation()->computeBBox((sofa::simulation::Node*)context, sceneMinBBox.ptr(), sceneMaxBBox.ptr());

        float scale = getIndicesScale();

        helper::vector<defaulttype::Vector3> positions;
        for (unsigned int i =0; i<coords.size(); i++)
        {
            defaulttype::Vector3 center; center = DataTypes::getCPos(coords[i]);
            positions.push_back(center);

        }
        vparams->drawTool()->draw3DText_Indices(positions, scale, color4);
    }
}


} // namespace topology

} // namespace component

} // namespace sofa

#endif
