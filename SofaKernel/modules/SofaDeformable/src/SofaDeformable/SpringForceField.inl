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
#include <SofaDeformable/SpringForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/topology/TopologyChange.h>
#include <sofa/helper/io/XspLoader.h>
#include <cassert>
#include <iostream>
#include <fstream>

namespace sofa::component::interactionforcefield
{

template<class DataTypes>
SpringForceField<DataTypes>::SpringForceField(SReal _ks, SReal _kd)
    : SpringForceField(nullptr, _ks, _kd)
{
}

template<class DataTypes>
SpringForceField<DataTypes>::SpringForceField(MechanicalState* mstate, SReal _ks, SReal _kd)
    : Inherit(mstate)
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

        sofa::type::vector<sofa::Index> toDelete;
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

    initializeTopologyHandler(d_springsIndices[0], this->getContext()->getMeshTopology(), 0);
    initializeTopologyHandler(d_springsIndices[1], this->getContext()->getMeshTopology(), 1);

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

    initializeMappingLink();
}

template <class DataTypes>
void SpringForceField<DataTypes>::addForce(const core::MechanicalParams*, DataVecDeriv& f, const DataVecCoord& x,
    const DataVecDeriv& v)
{
    VecDeriv& _f = *sofa::helper::getWriteAccessor(f);
    const auto& _x = x.getValue();
    const auto& _v = v.getValue();

    const type::vector<Spring>& springs= this->springs.getValue();

    _f.resize(_x.size());
    this->m_potentialEnergy = 0;
    for (unsigned int i=0; i<this->springs.getValue().size(); i++)
    {
        this->addSpringForce(this->m_potentialEnergy,_f,_x,_v,_f,_x,_v, i, springs[i]);
    }
}

template <class DataTypes>
void SpringForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df,
    const DataVecDeriv& dx)
{
    msg_error() << "SpringForceField does not support implicit integration. Use StiffSpringForceField instead.";
}

template <class DataTypes>
SReal SpringForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& x) const
{
    const type::vector<Spring>& springs= this->springs.getValue();
    auto _x = sofa::helper::getReadAccessor(x);

    SReal ener = 0;

    for (sofa::Index i=0; i<springs.size(); i++)
    {
        sofa::Index a = springs[i].m1;
        sofa::Index b = springs[i].m2;
        Coord u = _x[b] - _x[a];
        Real d = u.norm();
        Real elongation = (Real)(d - springs[i].initpos);
        ener += elongation * elongation * springs[i].ks /2;
    }

    return ener;
}

template<class DataTypes>
void SpringForceField<DataTypes>::addSpringForce(Real& ener, VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1, VecDeriv& f2, const VecCoord& p2, const VecDeriv& v2, sofa::Index /*i*/, const Spring& spring)
{
    sofa::Index a = spring.m1;
    sofa::Index b = spring.m2;
    typename DataTypes::CPos u = DataTypes::getCPos(p2[b])-DataTypes::getCPos(p1[a]);
    Real d = u.norm();
    if( spring.enabled && d<1.0e-4 ) // null length => no force
        return;
    Real inverseLength = 1.0f/d;
    u *= inverseLength;
    Real elongation = d - spring.initpos;
    ener += elongation * elongation * spring.ks /2;
    typename DataTypes::DPos relativeVelocity = DataTypes::getDPos(v2[b])-DataTypes::getDPos(v1[a]);
    Real elongationVelocity = dot(u,relativeVelocity);
    Real forceIntensity = spring.ks*elongation+spring.kd*elongationVelocity;
    typename DataTypes::DPos force = u*forceIntensity;

    DataTypes::setDPos( f1[a], DataTypes::getDPos(f1[a]) + force ) ;
    DataTypes::setDPos( f2[b], DataTypes::getDPos(f2[b]) - force ) ;
}

template<class DataTypes>
void SpringForceField<DataTypes>::addKToMatrix(sofa::linearalgebra::BaseMatrix *, SReal, unsigned int &)
{
    msg_error() << "SpringForceField does not support implicit integration. Use StiffSpringForceField instead.";
}



template<class DataTypes>
void SpringForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    using namespace sofa::defaulttype;
    using namespace sofa::type;

    if (this->d_componentState.getValue() ==core::objectmodel::ComponentState::Invalid)
        return ;
    if (!this->mstate)
        return;
    if (!vparams->displayFlags().getShowForceFields())
        return;
    const VecCoord& p = this->getMState()->read(core::ConstVecCoordId::position())->getValue();

    std::vector< Vector3 > points[4];
    const type::vector<Spring>& springs = this->springs.getValue();
    for (sofa::Index i = 0; i < springs.size(); i++)
    {
        if (!springs[i].enabled) continue;
        Real d = (p[springs[i].m2] - p[springs[i].m1]).norm();
        Vector3 point2, point1;
        point1 = DataTypes::getCPos(p[springs[i].m1]);
        point2 = DataTypes::getCPos(p[springs[i].m2]);

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
    if( !onlyVisible ) return;

    if (!this->getMState())
    {
        return;
    }

    const auto& springsValue = springs.getValue();
    if (springsValue.empty())
    {
        return;
    }

    const VecCoord& p = this->getMState()->read(core::ConstVecCoordId::position())->getValue();

    constexpr Real max_real = std::numeric_limits<Real>::max();
    constexpr Real min_real = std::numeric_limits<Real>::lowest();
    Real maxBBox[DataTypes::spatial_dimensions];
    Real minBBox[DataTypes::spatial_dimensions];

    for (int c = 0; c < DataTypes::spatial_dimensions; ++c)
    {
        maxBBox[c] = min_real;
        minBBox[c] = max_real;
    }

    bool foundSpring = false;

    for (const auto& spring : springsValue)
    {
        if (spring.enabled)
        {
            if (spring.m1 < p.size())
            {
                foundSpring = true;

                const auto& a = p[spring.m1];
                const auto& b = p[spring.m2];
                for (const auto& point : {a, b})
                {
                    for (int c = 0; c < DataTypes::spatial_dimensions; ++c)
                    {
                        if (point[c] > maxBBox[c])
                            maxBBox[c] = point[c];
                        else if (point[c] < minBBox[c])
                            minBBox[c] = point[c];
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
void SpringForceField<DataTypes>::addSpringBetweenTwoObjects(const Spring& spring)
{
    initializeMappingLink();

    if (!m_mapping)
    {
        msg_error() << "No mapping found. " << this->getClassName() << " must work with a SubsetMultiMapping in order"
            " to define springs between objects";
        return;
    }

    const auto localSpring = updateMappingIndexPairs(spring);
    if (localSpring.second)
    {
        m_mapping->init();
    }

    this->addSpring(localSpring.first);
}

template <class DataTypes>
template <class InputIt>
void SpringForceField<DataTypes>::addSprings(InputIt first, InputIt last)
{
    auto s = sofa::helper::getWriteAccessor(springs);
    while(first != last)
    {
        s->push_back(*first);

        sofa::helper::getWriteAccessor(d_springsIndices[0]).push_back(first->m1);
        sofa::helper::getWriteAccessor(d_springsIndices[1]).push_back(first->m2);

        ++first;
    }
}

template <class DataTypes>
template <class InputIt>
void SpringForceField<DataTypes>::addSpringsBetweenTwoObjects(InputIt first, InputIt last)
{
    initializeMappingLink();

    if (!m_mapping)
    {
        msg_error() << "No mapping found. " << this->getClassName() << " must work with a SubsetMultiMapping in order"
            " to define springs between objects";
        return;
    }

    bool mustReinitMapping = false;
    while(first != last)
    {
        const auto localSpring = updateMappingIndexPairs(*first++);
        mustReinitMapping |= localSpring.second;
        this->addSpring(localSpring.first);
    }

    if (mustReinitMapping)
    {
        m_mapping->init();
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
    addSpring(Spring(m1,m2,ks,kd,initlen));
}

template <class DataTypes>
void SpringForceField<DataTypes>::addSpring(const Spring& spring)
{
    sofa::helper::getWriteAccessor(springs)->push_back(spring);

    sofa::helper::getWriteAccessor(d_springsIndices[0]).push_back(spring.m1);
    sofa::helper::getWriteAccessor(d_springsIndices[1]).push_back(spring.m2);
}

template <class DataTypes>
typename SpringForceField<DataTypes>::MechanicalState* SpringForceField<DataTypes>::getMState1()
{
    initializeMappingLink();

    if (m_mapping)
    {
        const auto models = m_mapping->getMechFrom();
        if (!models.empty())
        {
            if (auto* baseState = models.front())
            {
                return dynamic_cast<MechanicalState*>(baseState);
            }
        }
    }
    return nullptr;
}

template <class DataTypes>
typename SpringForceField<DataTypes>::MechanicalState* SpringForceField<DataTypes>::getMState2()
{
    initializeMappingLink();
    if (m_mapping)
    {
        const auto models = m_mapping->getMechFrom();
        if (models.size() > 1)
        {
            if (auto* baseState = models[1])
            {
                return dynamic_cast<MechanicalState*>(baseState);
            }
        }
    }
    return nullptr;
}

template <class DataTypes>
bool SpringForceField<DataTypes>::isLinkingTwoObjects()
{
    return this->getMState1() != nullptr && this->getMState2() != nullptr;
}

template <class DataTypes>
void SpringForceField<DataTypes>::initializeMappingLink()
{
    if (!m_mapping)
    {
        auto* mapping = this->getContext()->template get<mapping::SubsetMultiMapping<DataTypes, DataTypes> >();
        m_mapping.set(mapping);
    }
}

template <class DataTypes>
std::pair<typename SpringForceField<DataTypes>::Spring, bool>
SpringForceField<DataTypes>::updateMappingIndexPairs(const Spring & spring)
{
    if (m_mapping)
    {
        auto indexPairs = sofa::helper::getWriteAccessor(m_mapping->indexPairs);
        Spring localSpring { spring };

        sofa::type::fixed_array<bool, 2> hasFoundLocalDoF(false, false);

        for (sofa::Index i = 0; i < indexPairs.size(); i+=2)
        {
            const auto currentMStateId = indexPairs[i];
            const auto currentMStateDoFId = indexPairs[i+1];

            for (sofa::Index j = 0; j < 2; ++j)
            {
                if (j == currentMStateId && spring.getIndex(j) == currentMStateDoFId)
                {
                    localSpring.getIndex(j) = i / 2;
                    hasFoundLocalDoF[j] = true;
                }
            }
            if (hasFoundLocalDoF[0] && hasFoundLocalDoF[1])
                break;
        }

        for (sofa::Index j = 0; j < 2; ++j)
        {
            if (!hasFoundLocalDoF[j]) // local index #j is not managed by the mapping. It needs to be added
            {
                indexPairs.push_back(j);
                indexPairs.push_back(spring.getIndex(j));
                localSpring.getIndex(j) = indexPairs.size() / 2;
            }
        }

        return {localSpring, !hasFoundLocalDoF[0] || !hasFoundLocalDoF[1]};
    }
    return {};
}
} // namespace sofa::component::interactionforcefield
