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
#include <sofa/component/solidmechanics/spring/SpringForceField.h>
#include <sofa/core/behavior/PairInteractionForceField.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/topology/TopologyChange.h>
#include <sofa/helper/io/XspLoader.h>
#include <cassert>
#include <iostream>
#include <fstream>

namespace sofa::component::solidmechanics::spring
{

template<class DataTypes>
SpringForceField<DataTypes>::SpringForceField(SReal _ks, SReal _kd)
    : SpringForceField(nullptr, nullptr, _ks, _kd)
{
}

template<class DataTypes>
SpringForceField<DataTypes>::SpringForceField(MechanicalState* mstate1, MechanicalState* mstate2, SReal _ks, SReal _kd)
    : Inherit(mstate1, mstate2)
    , ks(initData(&ks,_ks,"stiffness","uniform stiffness for the all springs"))
    , kd(initData(&kd,_kd,"damping","uniform damping for the all springs"))
    , showArrowSize(initData(&showArrowSize,0.01f,"showArrowSize","size of the axis"))
    , drawMode(initData(&drawMode,0,"drawMode","The way springs will be drawn:\n- 0: Line\n- 1:Cylinder\n- 2: Arrow"))
    , springs(initData(&springs,"spring","pairs of indices, stiffness, damping, rest length"))
    , maskInUse(false)
{
    this->addAlias(&fileSprings, "fileSprings");
}

template <class DataTypes>
class SpringForceField<DataTypes>::Loader : public helper::io::XspLoaderDataHook
{
public:
    SpringForceField<DataTypes>* dest;
    Loader(SpringForceField<DataTypes>* dest) : dest(dest) {}
    void addSpring(size_t m1, size_t m2, SReal ks, SReal kd, SReal initpos) override
    {
        type::vector<Spring>& springs = *dest->springs.beginEdit();
        springs.push_back(Spring(sofa::Index(m1), sofa::Index(m2),ks,kd,initpos));
        dest->springs.endEdit();
    }
};

template <class DataTypes>
bool SpringForceField<DataTypes>::load(const char *filename)
{
    bool ret = true;
    if (filename && filename[0])
    {
        Loader loader(this);
        ret &= helper::io::XspLoader::Load(filename, loader);
    }
    else ret = false;
    return ret;
}


template <class DataTypes>
void SpringForceField<DataTypes>::reinit()
{
    for (sofa::Index i=0; i<springs.getValue().size(); ++i)
    {
        (*springs.beginEdit())[i].ks = (Real) ks.getValue();
        (*springs.beginEdit())[i].kd = (Real) kd.getValue();
    }
}

template <class DataTypes>
void SpringForceField<DataTypes>::updateTopologyIndicesFromSprings()
{
    auto& indices1 = *sofa::helper::getWriteOnlyAccessor(d_springsIndices[0]);
    auto& indices2 = *sofa::helper::getWriteOnlyAccessor(d_springsIndices[1]);
    indices1.clear();
    indices2.clear();
    for (const auto& spring : sofa::helper::getReadAccessor(springs))
    {
        indices1.push_back(spring.m1);
        indices2.push_back(spring.m2);
    }
    areSpringIndicesDirty = false;
}

template <class DataTypes>
void SpringForceField<DataTypes>::applyRemovedEdges(const sofa::core::topology::EdgesRemoved* edgesRemoved, sofa::Index mstateId)
{
    if (edgesRemoved == nullptr)
        return;

    const type::vector<sofa::core::topology::Topology::EdgeID>& edges = edgesRemoved->getArray();

    if (edges.empty())
        return;
    
    core::topology::BaseMeshTopology* modifiedTopology;
    if (mstateId == 0)
    {
        modifiedTopology = this->getMState1()->getContext()->getMeshTopology();
    }
    else
    {
        modifiedTopology = this->getMState2()->getContext()->getMeshTopology();
    }

    if (modifiedTopology == nullptr)
        return;

    type::vector<Spring>& springsValue = *sofa::helper::getWriteAccessor(this->springs);
    
    const auto& topologyEdges = modifiedTopology->getEdges();

    type::vector<sofa::Index> springIdsToDelete;

    for (const auto& edgeId : edges) // iterate on the edgeIds to remove and save the respective point pairs
    {
        auto& firstPointId = topologyEdges[edgeId][0];
        auto& secondPointId = topologyEdges[edgeId][1];

        // sane default value to check against, if no spring is found for the edge
        int springIdToDelete = -1;
        sofa::Index i = 0;

        for (const auto& spring : springsValue) // loop on the list of springs to find the spring with targeted pointIds
        {
            auto& firstSpringPointId = mstateId == 0 ? spring.m1 : spring.m2;
            auto& secondSpringPointId = mstateId == 0 ? spring.m2 : spring.m1;

            if (firstSpringPointId == firstPointId && secondSpringPointId == secondPointId)
            {
                dmsg_info() << "Spring " << spring << " has an edge to be removed: REMOVED edgeId: " << edgeId;
                springIdToDelete = i;
                break; // break as soon as the first matching spring is found. TODO is there a valid case for having multiple springs on the same topology edge?
            }
            ++i;
        }
       
        if (springIdToDelete != -1) // if a matching spring was found, add it to the vector of Ids that will be removed
        {
            springIdsToDelete.push_back(springIdToDelete);
        }
    }

    // sort the edges to make sure we detele them from last to first
    std::sort (springIdsToDelete.begin(), springIdsToDelete.end());
    for (auto it = springIdsToDelete.rbegin(); it != springIdsToDelete.rend(); ++it) // delete accumulated springs to be removed
    {
        springsValue.erase(springsValue.begin() + (*it));
    }
}


template <class DataTypes>
void SpringForceField<DataTypes>::applyRemovedPoints(const sofa::core::topology::PointsRemoved* pointsRemoved, sofa::Index mstateId)
{
    if (pointsRemoved == nullptr)
        return;

    const auto& tab = pointsRemoved->getArray();

    if (tab.empty())
        return;
    
    core::topology::BaseMeshTopology* modifiedTopology;
    if (mstateId == 0)
    {
        modifiedTopology = this->getMState1()->getContext()->getMeshTopology();
    }
    else
    {
        modifiedTopology = this->getMState2()->getContext()->getMeshTopology();
    }

    if (modifiedTopology == nullptr)
        return;

    type::vector<Spring>& springsValue = *sofa::helper::getWriteAccessor(this->springs);
    auto nbPoints = modifiedTopology->getNbPoints();

    for (const auto pntId : tab) // iterate on the pointIds to remove
    {
        --nbPoints;

        type::vector<sofa::Index> toDelete;
        sofa::Index i {};
        for (const auto& spring : springsValue) // loop on the list of springs to find springs with targeted pointId
        {
            auto& id = mstateId == 0 ? spring.m1 : spring.m2;
            if (id == pntId)
            {
                dmsg_info() << "Spring " << spring << " has a point to be removed: REMOVED pointId: " << pntId;
                toDelete.push_back(i);
            }
            ++i;
        }

        for (auto it = toDelete.rbegin(); it != toDelete.rend(); ++it) // delete accumulated springs to be removed
        {
            springsValue.erase(springsValue.begin() + (*it));
        }
        
        if (pntId == nbPoints) // no need to renumber springs as last pointId has just been removed
            continue;

        for (auto& spring : springsValue) // renumber spring with last point indices to match the swap-pop_back process
        {
            auto& id = mstateId == 0 ? spring.m1 : spring.m2;
            if (id == nbPoints)
            {
                dmsg_info() << "Spring " << spring << " has a renumbered point: MODIFY from " << id << " to " << pntId;
                id = pntId;
            }
        }

        areSpringIndicesDirty = true;
    }
}

template <class DataTypes>
void SpringForceField<DataTypes>::init()
{
    // Load
    if (!fileSprings.getValue().empty())
        load(fileSprings.getFullPath().c_str());
    this->Inherit::init();

    initializeTopologyHandler(d_springsIndices[0], this->mstate1->getContext()->getMeshTopology(), 0);
    initializeTopologyHandler(d_springsIndices[1], this->mstate2->getContext()->getMeshTopology(), 1);

    updateTopologyIndicesFromSprings();
}

template <class DataTypes>
void SpringForceField<DataTypes>::initializeTopologyHandler(sofa::core::topology::TopologySubsetIndices& indices,
    core::topology::BaseMeshTopology* topology, sofa::Index mstateId)
{
    if (topology)
    {
        indices.createTopologyHandler(topology);

        indices.addTopologyEventCallBack(core::topology::TopologyChangeType::POINTSREMOVED,
            [this, mstateId](const core::topology::TopologyChange* change)
            {
                const auto* pointsRemoved = static_cast<const core::topology::PointsRemoved*>(change);
                msg_info(this) << "Removed points: [" << pointsRemoved->getArray() << "]";
                applyRemovedPoints(pointsRemoved, mstateId);
            });

        if (topology->getTopologyType() == sofa::geometry::ElementType::EDGE)
        {
            indices.linkToEdgeDataArray();  
            indices.addTopologyEventCallBack(core::topology::TopologyChangeType::EDGESREMOVED,
                [this, mstateId](const core::topology::TopologyChange* change)
                {
                    const auto* edgesRemoved = static_cast<const core::topology::EdgesRemoved*>(change);
                    msg_info(this) << "Removed edges: [" << edgesRemoved->getArray() << "]";
                    applyRemovedEdges(edgesRemoved, mstateId);
                });
        }
        
        indices.addTopologyEventCallBack(core::topology::TopologyChangeType::ENDING_EVENT,
            [this](const core::topology::TopologyChange*)
            {
                if (areSpringIndicesDirty)
                {
                    msg_info(this) << "Update topology indices from springs";
                    updateTopologyIndicesFromSprings();
                }
            });
    }
}

template<class DataTypes>
void SpringForceField<DataTypes>::addSpringForce(Real& ener, VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1, VecDeriv& f2, const VecCoord& p2, const VecDeriv& v2, sofa::Index /*i*/, const Spring& spring)
{
    const auto springForce = this->computeSpringForce(p1, v1, p2, v2, spring);

    if (springForce)
    {
        sofa::Index a = spring.m1;
        sofa::Index b = spring.m2;

        DataTypes::setDPos( f1[a], DataTypes::getDPos(f1[a]) + std::get<0>(springForce->force)) ;
        DataTypes::setDPos( f2[b], DataTypes::getDPos(f2[b]) + std::get<1>(springForce->force)) ;

        ener += springForce->energy;
    }
}

template <class DataTypes>
auto SpringForceField<DataTypes>::computeSpringForce(const VecCoord& p1, const VecDeriv& v1, const VecCoord& p2, const VecDeriv& v2, const Spring& spring)
-> std::unique_ptr<SpringForce>
{
    sofa::Index a = spring.m1;
    sofa::Index b = spring.m2;
    typename DataTypes::CPos u = DataTypes::getCPos(p2[b])-DataTypes::getCPos(p1[a]);
    Real d = u.norm();
    if( spring.enabled && d<1.0e-4 ) // null length => no force
        return {};
    std::unique_ptr<SpringForce> springForce = std::make_unique<SpringForce>();

    Real inverseLength = 1.0f/d;
    u *= inverseLength;
    Real elongation = d - spring.initpos;
    springForce->energy = elongation * elongation * spring.ks /2;
    typename DataTypes::DPos relativeVelocity = DataTypes::getDPos(v2[b])-DataTypes::getDPos(v1[a]);
    Real elongationVelocity = dot(u,relativeVelocity);
    Real forceIntensity = spring.ks*elongation+spring.kd*elongationVelocity;
    typename DataTypes::DPos force = u*forceIntensity;
    springForce->force = std::make_pair(force, -force);
    return springForce;
}

template<class DataTypes>
void SpringForceField<DataTypes>::addForce(
    const core::MechanicalParams* /* mparams */, DataVecDeriv& data_f1, DataVecDeriv& data_f2,
    const DataVecCoord& data_x1, const DataVecCoord& data_x2,
    const DataVecDeriv& data_v1, const DataVecDeriv& data_v2)
{
    const VecCoord& x1 = data_x1.getValue();
    const VecDeriv& v1 = data_v1.getValue();

    const VecCoord& x2 = data_x2.getValue();
    const VecDeriv& v2 = data_v2.getValue();

    sofa::helper::WriteOnlyAccessor<sofa::Data<VecDeriv> > f1 = sofa::helper::getWriteOnlyAccessor(data_f1);
    sofa::helper::WriteOnlyAccessor<sofa::Data<VecDeriv> > f2 = sofa::helper::getWriteOnlyAccessor(data_f2);

    const type::vector<Spring>& springs= this->springs.getValue();
    f1.resize(x1.size());
    f2.resize(x2.size());
    this->m_potentialEnergy = 0;
    for (unsigned int i=0; i < springs.size(); i++)
    {
        this->addSpringForce(this->m_potentialEnergy,f1.wref(),x1,v1,f2.wref(),x2,v2, i, springs[i]);
    }
}

template<class DataTypes>
void SpringForceField<DataTypes>::addDForce(const core::MechanicalParams*, DataVecDeriv&, DataVecDeriv&, const DataVecDeriv&, const DataVecDeriv& )
{
    msg_error() << "SpringForceField does not support implicit integration. Use StiffSpringForceField instead.";
}


template<class DataTypes>
SReal SpringForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& data_x1, const DataVecCoord& data_x2) const
{
    const type::vector<Spring>& springs= this->springs.getValue();
    const VecCoord& p1 =  data_x1.getValue();
    const VecCoord& p2 =  data_x2.getValue();

    SReal ener = 0;

    for (sofa::Index i=0; i<springs.size(); i++)
    {
        sofa::Index a = springs[i].m1;
        sofa::Index b = springs[i].m2;
        Coord u = p2[b]-p1[a];
        Real d = u.norm();
        Real elongation = (Real)(d - springs[i].initpos);
        ener += elongation * elongation * springs[i].ks /2;
    }

    return ener;
}


template<class DataTypes>
void SpringForceField<DataTypes>::addKToMatrix(sofa::linearalgebra::BaseMatrix *, SReal, unsigned int &)
{
    msg_error() << "SpringForceField does not support implicit integration. Use StiffSpringForceField instead.";
}

template <class DataTypes>
void SpringForceField<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    SOFA_UNUSED(matrix);
    msg_error() << "SpringForceField does not support implicit integration. Use StiffSpringForceField instead.";
}

template <class DataTypes>
void SpringForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}


template<class DataTypes>
void SpringForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    using namespace sofa::defaulttype;
    using namespace sofa::type;

    if (!((this->mstate1 == this->mstate2) ? vparams->displayFlags().getShowForceFields() : vparams->displayFlags().getShowInteractionForceFields())) return;
    const VecCoord& p1 = this->mstate1->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& p2 = this->mstate2->read(core::ConstVecCoordId::position())->getValue();

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    std::vector< Vec3 > points[4];
    const bool external = (this->mstate1 != this->mstate2);
    const type::vector<Spring>& springs = this->springs.getValue();
    for (sofa::Index i = 0; i < springs.size(); i++)
    {
        if (!springs[i].enabled) continue;
        assert(i < springs.size());
        assert(springs[i].m2 < p2.size());
        assert(springs[i].m1 < p1.size());
        Real d = (p2[springs[i].m2] - p1[springs[i].m1]).norm();
        Vec3 point2, point1;
        point1 = DataTypes::getCPos(p1[springs[i].m1]);
        point2 = DataTypes::getCPos(p2[springs[i].m2]);

        if (external)
        {
            if (d < springs[i].initpos * 0.9999)
            {
                points[0].push_back(point1);
                points[0].push_back(point2);
            }
            else
            {
                points[1].push_back(point1);
                points[1].push_back(point2);
            }
        }
        else
        {
            if (d < springs[i].initpos * 0.9999)
            {
                points[2].push_back(point1);
                points[2].push_back(point2);
            }
            else
            {
                points[3].push_back(point1);
                points[3].push_back(point2);
            }
        }
    }
    constexpr RGBAColor c0 = RGBAColor::red();
    constexpr RGBAColor c1 = RGBAColor::green();
    constexpr RGBAColor c2 {1.0f, 0.5f, 0.0f, 1.0f };
    constexpr RGBAColor c3{ 0.0f, 1.0f, 0.5f, 1.0f };

    if (showArrowSize.getValue()==0 || drawMode.getValue() == 0)
    {
        vparams->drawTool()->drawLines(points[0], 1, c0);
        vparams->drawTool()->drawLines(points[1], 1, c1);
        vparams->drawTool()->drawLines(points[2], 1, c2);
        vparams->drawTool()->drawLines(points[3], 1, c3);
    }
    else if (drawMode.getValue() == 1)
    {
        const auto numLines0=points[0].size()/2;
        const auto numLines1=points[1].size()/2;
        const auto numLines2=points[2].size()/2;
        const auto numLines3=points[3].size()/2;

        for (unsigned int i=0; i<numLines0; ++i) vparams->drawTool()->drawCylinder(points[0][2*i+1], points[0][2*i], showArrowSize.getValue(), c0);
        for (unsigned int i=0; i<numLines1; ++i) vparams->drawTool()->drawCylinder(points[1][2*i+1], points[1][2*i], showArrowSize.getValue(), c1);
        for (unsigned int i=0; i<numLines2; ++i) vparams->drawTool()->drawCylinder(points[2][2*i+1], points[2][2*i], showArrowSize.getValue(), c2);
        for (unsigned int i=0; i<numLines3; ++i) vparams->drawTool()->drawCylinder(points[3][2*i+1], points[3][2*i], showArrowSize.getValue(), c3);

    }
    else if (drawMode.getValue() == 2)
    {
        const auto numLines0=points[0].size()/2;
        const auto numLines1=points[1].size()/2;
        const auto numLines2=points[2].size()/2;
        const auto numLines3=points[3].size()/2;

        for (unsigned int i=0; i<numLines0; ++i) vparams->drawTool()->drawArrow(points[0][2*i+1], points[0][2*i], showArrowSize.getValue(), c0);
        for (unsigned int i=0; i<numLines1; ++i) vparams->drawTool()->drawArrow(points[1][2*i+1], points[1][2*i], showArrowSize.getValue(), c1);
        for (unsigned int i=0; i<numLines2; ++i) vparams->drawTool()->drawArrow(points[2][2*i+1], points[2][2*i], showArrowSize.getValue(), c2);
        for (unsigned int i=0; i<numLines3; ++i) vparams->drawTool()->drawArrow(points[3][2*i+1], points[3][2*i], showArrowSize.getValue(), c3);
    }
    else
    {
        msg_error()<< "No proper drawing mode found!";
    }
}

template <class DataTypes>
void SpringForceField<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    SOFA_UNUSED(params);

    if( !onlyVisible ) return;

    if (!this->mstate1 || !this->mstate2)
    {
        return;
    }

    const auto& springsValue = springs.getValue();
    if (springsValue.empty())
    {
        return;
    }

    const VecCoord& p1 = this->mstate1->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& p2 = this->mstate2->read(core::ConstVecCoordId::position())->getValue();

    constexpr Real max_real = std::numeric_limits<Real>::max();
    constexpr Real min_real = std::numeric_limits<Real>::lowest();

    Real maxBBox[DataTypes::spatial_dimensions];
    Real minBBox[DataTypes::spatial_dimensions];

    for (sofa::Index c = 0; c < DataTypes::spatial_dimensions; ++c)
    {
        maxBBox[c] = min_real;
        minBBox[c] = max_real;
    }

    bool foundSpring = false;

    for (const auto& spring : springsValue)
    {
        if (spring.enabled)
        {
            if (spring.m1 < p1.size() && spring.m2 < p2.size())
            {
                foundSpring = true;

                const auto& a = p1[spring.m1];
                const auto& b = p2[spring.m2];
                for (const auto& p : {a, b})
                {
                    for (sofa::Index c = 0; c < DataTypes::spatial_dimensions; ++c)
                    {
                        if (p[c] > maxBBox[c])
                            maxBBox[c] = p[c];
                        else if (p[c] < minBBox[c])
                            minBBox[c] = p[c];
                    }
                }
            }
        }
    }

    if (foundSpring)
    {
        this->f_bbox.setValue(sofa::type::TBoundingBox<Real>(minBBox,maxBBox));
    }
}

template <class DataTypes>
void SpringForceField<DataTypes>::clear(sofa::Size reserve)
{
    sofa::type::vector<Spring>& springs = *this->springs.beginEdit();
    springs.clear();
    if (reserve) springs.reserve(reserve);
    this->springs.endEdit();
}

template <class DataTypes>
void SpringForceField<DataTypes>::removeSpring(sofa::Index idSpring)
{
    if (idSpring >= (this->springs.getValue()).size())
        return;

    sofa::type::vector<Spring>& springs = *this->springs.beginEdit();
    springs.erase(springs.begin() +idSpring );
    this->springs.endEdit();
}

template <class DataTypes>
void SpringForceField<DataTypes>::addSpring(sofa::Index m1, sofa::Index m2, SReal ks, SReal kd, SReal initlen)
{
    springs.beginEdit()->push_back(Spring(m1,m2,ks,kd,initlen));
    springs.endEdit();

    sofa::helper::getWriteAccessor(d_springsIndices[0]).push_back(m1);
    sofa::helper::getWriteAccessor(d_springsIndices[1]).push_back(m2);
}

template <class DataTypes>
void SpringForceField<DataTypes>::addSpring(const Spring& spring)
{
    springs.beginEdit()->push_back(spring);
    springs.endEdit();

    sofa::helper::getWriteAccessor(d_springsIndices[0]).push_back(spring.m1);
    sofa::helper::getWriteAccessor(d_springsIndices[1]).push_back(spring.m2);
}

template<class DataTypes>
void SpringForceField<DataTypes>::initGnuplot(const std::string path)
{
    if (!this->getName().empty())
    {
        if (m_gnuplotFileEnergy != nullptr)
        {
            m_gnuplotFileEnergy->close();
            delete m_gnuplotFileEnergy;
        }
        m_gnuplotFileEnergy = new std::ofstream( (path+this->getName()+"_PotentialEnergy.txt").c_str() );
    }
}

template<class DataTypes>
void SpringForceField<DataTypes>::exportGnuplot(SReal time)
{
    if (m_gnuplotFileEnergy!=nullptr)
    {
        (*m_gnuplotFileEnergy) << time <<"\t"<< this->m_potentialEnergy << std::endl;
    }
}

} // namespace sofa::component::solidmechanics::spring
