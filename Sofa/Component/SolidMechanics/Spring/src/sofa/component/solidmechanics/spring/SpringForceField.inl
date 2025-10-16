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
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>

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
    , d_showArrowSize(initData(&d_showArrowSize,0.01f,"showArrowSize","size of the axis"))
    , d_drawMode(initData(&d_drawMode,0,"drawMode","The way springs will be drawn:\n- 0: Line\n- 1:Cylinder\n- 2: Arrow"))
    , d_ks(initData(&d_ks,{_ks},"stiffness","uniform stiffness for the all springs"))
    , d_kd(initData(&d_kd,{_kd},"damping","uniform damping for the all springs"))
    , d_springs(initData(&d_springs,"spring","pairs of indices, stiffness, damping, rest length"))
    , d_lengths(initData(&d_lengths, "lengths", "List of lengths to create the springs. Must have the same than indices1 & indices2, or if only one element, it will be applied to all springs. If empty, 0 will be applied everywhere"))
    , d_elongationOnly(initData(&d_elongationOnly, type::vector<bool>{false}, "elongationOnly", "///< List of boolean stating on the fact that the spring should only apply forces on elongations. Must have the same than indices1 & indices2, or if only one element, it will be applied to all springs. If empty, False will be applied everywhere"))
    , d_enabled(initData(&d_enabled, type::vector<bool>{true}, "enabled", "///< List of boolean stating on the fact that the spring is enabled. Must have the same than indices1 & indices2, or if only one element, it will be applied to all springs. If empty, True will be applied everywhere"))
    , maskInUse(false)
{
    this->addAlias(&fileSprings, "fileSprings");
    this->addAlias(&d_lengths, "length");
    this->addAlias(&d_springsIndices[0], "indices1");
    this->addAlias(&d_springsIndices[1], "indices2");

}

template <class DataTypes>
class SpringForceField<DataTypes>::Loader : public helper::io::XspLoaderDataHook
{
public:
    SpringForceField<DataTypes>* dest;
    Loader(SpringForceField<DataTypes>* dest) : dest(dest) {}
    void addSpring(size_t m1, size_t m2, SReal ks, SReal kd, SReal initpos) override
    {
        type::vector<Spring>& springs = *dest->d_springs.beginEdit();
        springs.push_back(Spring(sofa::Index(m1), sofa::Index(m2),ks,kd,initpos));
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
void SpringForceField<DataTypes>::init()
{
    // Load
    if (!fileSprings.getValue().empty())
        load(fileSprings.getFullPath().c_str());
    this->Inherit::init();

    initializeTopologyHandler(d_springsIndices[0], this->mstate1->getContext()->getMeshTopology(), 0);
    initializeTopologyHandler(d_springsIndices[1], this->mstate2->getContext()->getMeshTopology(), 1);


    if (d_springs.isSet())
        updateTopologyIndicesFromSprings();
    else
        updateSpringsFromTopologyIndices();


    this->addUpdateCallback("TopoCallBack",{ &d_springsIndices[0], &d_springsIndices[1], &d_lengths},
                      [this](const sofa::core::DataTracker& ) -> sofa::core::objectmodel::ComponentState
                      {
                          updateSpringsFromTopologyIndices();
                          return sofa::core::objectmodel::ComponentState::Valid;
                      },
                      {&d_springs});
}

template <class DataTypes>
void SpringForceField<DataTypes>::reinit()
{
    if (d_springs.isSet())
        updateTopologyIndicesFromSprings();
    else
        updateSpringsFromTopologyIndices();
}

template <class DataTypes>
void SpringForceField<DataTypes>::updateTopologyIndicesFromSprings()
{
    const auto& springValues = *sofa::helper::getReadAccessor(d_springs);
    auto& indices1 = *sofa::helper::getWriteOnlyAccessor(d_springsIndices[0]);
    auto& indices2 = *sofa::helper::getWriteOnlyAccessor(d_springsIndices[1]);
    auto& lengths = *sofa::helper::getWriteOnlyAccessor(d_lengths);
    auto& kds = *sofa::helper::getWriteOnlyAccessor(d_kd);
    auto& kss = *sofa::helper::getWriteOnlyAccessor(d_ks);
    auto& elongationOnly = *sofa::helper::getWriteOnlyAccessor(d_elongationOnly);
    auto& enabled = *sofa::helper::getWriteOnlyAccessor(d_enabled);

    indices1.resize(springValues.size());
    indices2.resize(springValues.size());
    lengths.resize(springValues.size());
    kds.resize(springValues.size());
    kss.resize(springValues.size());
    elongationOnly.resize(springValues.size());
    enabled.resize(springValues.size());
    for (unsigned i=0; i<springValues.size(); ++i)
    {
        indices1[i] = springValues[i].m1;
        indices2[i] = springValues[i].m2;
        lengths[i] = springValues[i].initpos;
        kds[i] = springValues[i].kd;
        kss[i] = springValues[i].ks;
        elongationOnly[i] = springValues[i].elongationOnly;
        enabled[i] = springValues[i].enabled;
    }

    areSpringIndicesDirty = false;
}


template <class DataTypes>
void SpringForceField<DataTypes>::updateTopologyIndicesFromSprings_springAdded()
{
    const auto& springValues = *sofa::helper::getReadAccessor(d_springs);
    auto& indices1 = *sofa::helper::getWriteOnlyAccessor(d_springsIndices[0]);
    auto& indices2 = *sofa::helper::getWriteOnlyAccessor(d_springsIndices[1]);
    auto& lengths = *sofa::helper::getWriteOnlyAccessor(d_lengths);
    auto& kds = *sofa::helper::getWriteOnlyAccessor(d_kd);
    auto& kss = *sofa::helper::getWriteOnlyAccessor(d_ks);
    auto& elongationOnly = *sofa::helper::getWriteOnlyAccessor(d_elongationOnly);
    auto& enabled = *sofa::helper::getWriteOnlyAccessor(d_enabled);

    const unsigned oldSize = indices1.size();
    const unsigned newSize = springValues.size();

    indices1.resize(springValues.size());
    indices2.resize(springValues.size());
    lengths.resize(springValues.size());
    kds.resize(springValues.size());
    kss.resize(springValues.size());
    elongationOnly.resize(springValues.size());
    enabled.resize(springValues.size());
    for (unsigned i=oldSize; i<newSize; ++i)
    {
        indices1[i] = springValues[i].m1;
        indices2[i] = springValues[i].m2;
        lengths[i] = springValues[i].initpos;
        kds[i] = springValues[i].kd;
        kss[i] = springValues[i].ks;
        elongationOnly[i] = springValues[i].elongationOnly;
        enabled[i] = springValues[i].enabled;
    }

    areSpringIndicesDirty = false;
}



template <class DataTypes>
void SpringForceField<DataTypes>::updateTopologyIndices_springRemoved(unsigned id)
{
    auto& indices1 = *sofa::helper::getWriteOnlyAccessor(d_springsIndices[0]);
    auto& indices2 = *sofa::helper::getWriteOnlyAccessor(d_springsIndices[1]);
    auto& lengths = *sofa::helper::getWriteOnlyAccessor(d_lengths);
    auto& kds = *sofa::helper::getWriteOnlyAccessor(d_kd);
    auto& kss = *sofa::helper::getWriteOnlyAccessor(d_ks);
    auto& elongationOnly = *sofa::helper::getWriteOnlyAccessor(d_elongationOnly);
    auto& enabled = *sofa::helper::getWriteOnlyAccessor(d_enabled);

    indices1.erase(indices1.begin() + id);
    indices2.erase(indices2.begin() + id);
    lengths.erase(lengths.begin() + id);
    kds.erase(kds.begin() + id);
    kss.erase(kss.begin() + id);
    elongationOnly.erase(elongationOnly.begin() + id);
    enabled.erase(enabled.begin() + id);

    areSpringIndicesDirty = false;
}

template <class DataTypes>
void SpringForceField<DataTypes>::updateSpringsFromTopologyIndices()
{
    const auto& indices1 = d_springsIndices[0].getValue();
    const auto& indices2 = d_springsIndices[1].getValue();

    if (indices1.size() != indices2.size())
    {
        msg_error() << "Inputs indices sets sizes are different: d_indices1: " << indices1.size()
                    << " | d_indices2 " << indices2.size()
                    << " . No springs will be created";
        return;
    }

    if (indices1.empty())
        return;

    auto lengths = sofa::helper::getWriteAccessor(d_lengths);
    if (lengths.empty())
    {
        lengths.push_back({0.0});
    }

    if (lengths.size() != indices1.size())
    {
        msg_warning() << "Lengths list has a different size than indices1. The list will be resized to " << indices1.size() << " elements.";
        lengths->resize(indices1.size(), lengths->back());
    }

    auto kds = sofa::helper::getWriteAccessor(d_kd);
    if (kds.size() != indices1.size())
    {
        msg_warning() << "Kd list has a different size than indices1. The list will be resized to " << indices1.size() << " elements.";
        kds->resize(indices1.size(), kds->back());
    }

    auto kss = sofa::helper::getWriteAccessor(d_ks);
    if (kss.size() != indices1.size())
    {
        msg_warning() << "Ks list has a different size than indices1. The list will be resized to " << indices1.size() << " elements.";
        kss->resize(indices1.size(), kss->back());
    }

    auto elongationOnly = sofa::helper::getWriteAccessor(d_elongationOnly);
    if (elongationOnly.size() != indices1.size())
    {
        msg_warning() << "elongationOnly list has a different size than indices1. The list will be resized to " << indices1.size() << " elements.";
        elongationOnly->resize(indices1.size(), elongationOnly->back());
    }


    auto enabled = sofa::helper::getWriteAccessor(d_enabled);
    if (enabled.size() != indices1.size())
    {
        msg_warning() << "enabled list has a different size than indices1. The list will be resized to " << indices1.size() << " elements.";
        enabled->resize(indices1.size(), enabled->back());
    }

    msg_info() << "Inputs have changed, recompute  Springs From Data Inputs";

    type::vector<Spring>& _springs = *this->d_springs.beginEdit();
    _springs.clear();

    for (sofa::Index i = 0; i<indices1.size(); ++i)
        _springs.push_back(Spring(indices1[i], indices2[i], kss[i], kds[i], lengths[i],elongationOnly[i],enabled[i]));
    
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

    type::vector<Spring>& springsValue = *sofa::helper::getWriteAccessor(this->d_springs);
    
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
        areSpringIndicesDirty = true;
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

    type::vector<Spring>& springsValue = *sofa::helper::getWriteAccessor(this->d_springs);
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
                    //We know that changes have been performed on the indices data from the topological changes,
                    //the springs are up-to-date thanks to our callbacks, but not the un-changed indices list,
                    //so we clean dirty on the springs to avoid call to the datacallback when accessing the data
                    d_springs.cleanDirty();
                    //Clean the indices list of the unmodified topology to match the size of the newly modified one
                    updateTopologyIndicesFromSprings();
                    //Clean dirtiness of springs because we just updated the indices lists from the spring data itself
                    d_springs.cleanDirty();
                    areSpringIndicesDirty = false;
                }
            });
    }
}

template<class DataTypes>
void SpringForceField<DataTypes>::addSpringForce(
    Real& potentialEnergy,
    VecDeriv& f1,
    const  VecCoord& p1,
    const VecDeriv& v1,
    VecDeriv& f2,
    const  VecCoord& p2,
    const  VecDeriv& v2,
    sofa::Index i,
    const Spring& spring)
{
    const std::unique_ptr<SpringForce> springForce = this->computeSpringForce(p1, v1, p2, v2, spring);

    if (springForce)
    {

        sofa::Index a = spring.m1;
        sofa::Index b = spring.m2;

        DataTypes::setDPos( f1[a], DataTypes::getDPos(f1[a]) + std::get<0>(springForce->force)) ;
        DataTypes::setDPos( f2[b], DataTypes::getDPos(f2[b]) + std::get<1>(springForce->force)) ;

        potentialEnergy += springForce->energy;

        this->dfdx[i] = springForce->dForce_dX;
    }
    else
    {
        // set derivative to 0
        this->dfdx[i].clear();
    }
}


template <class DataTypes>
auto SpringForceField<DataTypes>::computeSpringForce(
    const VecCoord& p1, const VecDeriv& v1,
    const VecCoord& p2, const VecDeriv& v2,
    const Spring& spring)
    -> std::unique_ptr<SpringForce>
{
    sofa::Index a = spring.m1;
    sofa::Index b = spring.m2;

    /// Get the positional part out of the dofs.
    typename DataTypes::CPos u = DataTypes::getCPos(p2[b])-DataTypes::getCPos(p1[a]);
    Real d = u.norm();
    if( spring.enabled && d>1.0e-9 && (!spring.elongationOnly || d>spring.initpos))
    {
        std::unique_ptr<SpringForce> springForce = std::make_unique<SpringForce>();

        // F =   k_s.(l-l_0 ).U + k_d((V_b - V_a).U).U = f.U   where f is the intensity and U the direction
        Real inverseLength = 1.0f/d;
        u *= inverseLength;
        Real elongation = (Real)(d - spring.initpos);
        springForce->energy = elongation * elongation * spring.ks / 2;
        typename DataTypes::DPos relativeVelocity = DataTypes::getDPos(v2[b])-DataTypes::getDPos(v1[a]);
        Real elongationVelocity = dot(u,relativeVelocity);
        Real forceIntensity = (Real)(spring.ks*elongation+spring.kd*elongationVelocity);
        typename DataTypes::DPos force = u*forceIntensity;

        // Compute stiffness dF/dX
        // The force change dF comes from length change dl and unit vector change dU:
        // dF = k_s.dl.U + f.dU
        // dU = 1/l.(I-U.U^T).dX   where dX = dX_1 - dX_0  and I is the identity matrix
        // dl = U^T.dX
        // dF = k_s.U.U^T.dX + f/l.(I-U.U^T).dX = ((k_s-f/l).U.U^T + f/l.I).dX
        auto& m = springForce->dForce_dX;
        Real tgt = forceIntensity * inverseLength;
        for(sofa::Index j=0; j<N; ++j )
        {
            for(sofa::Index k=0; k<N; ++k )
            {
                m(j,k) = ((Real)spring.ks-tgt) * u[j] * u[k];
            }
            m(j,j) += tgt;
        }

        springForce->force = std::make_pair(force, -force);
        return springForce;
    }

    return {};
}

template<class DataTypes>
void SpringForceField<DataTypes>::addForce(
    const core::MechanicalParams* /* mparams */, DataVecDeriv& data_f1, DataVecDeriv& data_f2,
    const DataVecCoord& data_x1, const DataVecCoord& data_x2,
    const DataVecDeriv& data_v1, const DataVecDeriv& data_v2)
{
    const type::vector<Spring>& _springs = this->d_springs.getValue();
    this->dfdx.resize(_springs.size());

    const VecCoord& x1 = data_x1.getValue();
    const VecDeriv& v1 = data_v1.getValue();

    const VecCoord& x2 = data_x2.getValue();
    const VecDeriv& v2 = data_v2.getValue();

    sofa::helper::WriteOnlyAccessor<sofa::Data<VecDeriv> > f1 = sofa::helper::getWriteOnlyAccessor(data_f1);
    sofa::helper::WriteOnlyAccessor<sofa::Data<VecDeriv> > f2 = sofa::helper::getWriteOnlyAccessor(data_f2);

    f1.resize(x1.size());
    f2.resize(x2.size());
    this->m_potentialEnergy = 0;
    for (unsigned int i=0; i < _springs.size(); i++)
    {
        this->addSpringForce(this->m_potentialEnergy,f1.wref(),x1,v1,f2.wref(),x2,v2, i, _springs[i]);
    }
}

template<class DataTypes>
void SpringForceField<DataTypes>::addSpringDForce(VecDeriv& df1,const  VecDeriv& dx1, VecDeriv& df2,const  VecDeriv& dx2, sofa::Index i, const Spring& spring, SReal kFactor, SReal bFactor)
{
    typename DataTypes::DPos dforce = computeSpringDForce(df1, dx1, df2, dx2, i, spring, kFactor, bFactor);

    const sofa::Index a = spring.m1;
    const sofa::Index b = spring.m2;

    DataTypes::setDPos( df1[a], DataTypes::getDPos(df1[a]) + dforce ) ;
    DataTypes::setDPos( df2[b], DataTypes::getDPos(df2[b]) - dforce ) ;
}

template <class DataTypes>
typename DataTypes::DPos SpringForceField<DataTypes>::computeSpringDForce(VecDeriv& df1, const VecDeriv& dx1, VecDeriv& df2, const VecDeriv& dx2, sofa::Index i, const Spring& spring, SReal kFactor, SReal bFactor)
{
    SOFA_UNUSED(df1);
    SOFA_UNUSED(df2);
    SOFA_UNUSED(bFactor);
    const typename DataTypes::CPos d = DataTypes::getDPos(dx2[spring.m2]) - DataTypes::getDPos(dx1[spring.m1]);
    return this->dfdx[i] * d * kFactor;
}

template<class DataTypes>
void SpringForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& data_df1, DataVecDeriv& data_df2, const DataVecDeriv& data_dx1, const DataVecDeriv& data_dx2)
{
    sofa::helper::WriteOnlyAccessor<sofa::Data<VecDeriv>> df1 = sofa::helper::getWriteOnlyAccessor(data_df1);
    sofa::helper::WriteOnlyAccessor<sofa::Data<VecDeriv>> df2 = sofa::helper::getWriteOnlyAccessor(data_df2);
    const VecDeriv&  dx1 =  data_dx1.getValue();
    const VecDeriv&  dx2 =  data_dx2.getValue();
    Real kFactor       =  (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams,this->rayleighStiffness.getValue());
    Real bFactor       =  (Real)sofa::core::mechanicalparams::bFactor(mparams);

    const type::vector<Spring>& springs = this->d_springs.getValue();
    df1.resize(dx1.size());
    df2.resize(dx2.size());

    for (sofa::Index i=0; i<springs.size(); i++)
    {
        this->addSpringDForce(df1.wref(), dx1,df2.wref(),dx2, i, springs[i], kFactor, bFactor);
    }
}

template<class DataTypes>
SReal SpringForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& data_x1, const DataVecCoord& data_x2) const
{
    const type::vector<Spring>& springs= this->d_springs.getValue();
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


template <class DataTypes>
template<class Matrix>
void SpringForceField<DataTypes>::addToMatrix(Matrix* globalMatrix,
                                                   const unsigned int offsetRow,
                                                   const unsigned int offsetCol,
                                                   const Mat& localMatrix)
{
    if (globalMatrix)
    {
        if constexpr(N == 2 || N == 3 )
        {
            // BaseMatrix::add can accept Mat2x2 and Mat3x3 and it's sometimes faster than the 2 loops
            globalMatrix->add(offsetRow, offsetCol, -localMatrix);
        }
        else
        {
            for(sofa::Index i = 0; i < N; ++i)
            {
                for (sofa::Index j = 0; j < N; ++j)
                {
                    globalMatrix->add(offsetRow + i, offsetCol + j, (Real)localMatrix(i,j));
                }
            }
        }
    }
}

template<class DataTypes>
void SpringForceField<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    const Real kFact = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams,this->rayleighStiffness.getValue());
    if (this->mstate1 == this->mstate2)
    {
        sofa::core::behavior::MultiMatrixAccessor::MatrixRef mat = matrix->getMatrix(this->mstate1);
        if (!mat) return;
        const sofa::type::vector<Spring >& ss = this->d_springs.getValue();
        const sofa::Size n = ss.size() < this->dfdx.size() ? sofa::Size(ss.size()) : sofa::Size(this->dfdx.size());
        for (sofa::Index e = 0; e < n; ++e)
        {
            const Spring& s = ss[e];
            const sofa::Index p1 = mat.offset + Deriv::total_size * s.m1;
            const sofa::Index p2 = mat.offset + Deriv::total_size * s.m2;
            const Mat& m = this->dfdx[e];
            for(sofa::Index i=0; i<N; i++)
            {
                for (sofa::Index j=0; j<N; j++)
                {
                    Real k = (Real)(m(i,j)*kFact);
                    mat.matrix->add(p1+i,p1+j, -k);
                    mat.matrix->add(p1+i,p2+j,  k);
                    mat.matrix->add(p2+i,p1+j,  k);//or mat->add(p1+j,p2+i, k);
                    mat.matrix->add(p2+i,p2+j, -k);
                }
            }
        }
    }
    else
    {
        const sofa::core::behavior::MultiMatrixAccessor::MatrixRef mat11 = matrix->getMatrix(this->mstate1);
        const sofa::core::behavior::MultiMatrixAccessor::MatrixRef mat22 = matrix->getMatrix(this->mstate2);
        const sofa::core::behavior::MultiMatrixAccessor::InteractionMatrixRef mat12 = matrix->getMatrix(this->mstate1, this->mstate2);
        const sofa::core::behavior::MultiMatrixAccessor::InteractionMatrixRef mat21 = matrix->getMatrix(this->mstate2, this->mstate1);

        if (!mat11 && !mat22 && !mat12 && !mat21) return;
        const sofa::type::vector<Spring >& ss = this->d_springs.getValue();
        const sofa::Size n = ss.size() < this->dfdx.size() ? sofa::Size(ss.size()) : sofa::Size(this->dfdx.size());
        for (sofa::Index e = 0; e < n; ++e)
        {
            const Spring& s = ss[e];
            const unsigned p1 = Deriv::total_size * s.m1;
            const unsigned p2 = Deriv::total_size * s.m2;
            const Mat m = this->dfdx[e] * (Real) kFact;

            addToMatrix(mat11.matrix, mat11.offset + p1, mat11.offset + p1, -m);
            addToMatrix(mat12.matrix, mat12.offRow + p1, mat12.offCol + p2,  m);
            addToMatrix(mat21.matrix, mat21.offRow + p2, mat21.offCol + p1,  m);
            addToMatrix(mat22.matrix, mat22.offset + p2, mat22.offset + p2, -m);
        }
    }

}

template <class DataTypes>
void SpringForceField<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    const sofa::type::vector<Spring >& ss = this->d_springs.getValue();
    const auto n = std::min(ss.size(), this->dfdx.size());
    if (this->mstate1 == this->mstate2)
    {
        auto dfdx = matrix->getForceDerivativeIn(this->mstate1.get())
                        .withRespectToPositionsIn(this->mstate1.get());

        for (std::size_t e = 0; e < n; ++e)
        {
            const Spring& s = ss[e];
            const Mat& m = this->dfdx[e];

            const auto p1 = Deriv::total_size * s.m1;
            const auto p2 = Deriv::total_size * s.m2;

            for(sofa::Index i = 0; i < N; ++i)
            {
                for (sofa::Index j = 0; j < N; ++j)
                {
                    const auto k = m(i,j);
                    dfdx(p1+i, p1+j) += -k;
                    dfdx(p1+i, p2+j) +=  k;
                    dfdx(p2+i, p1+j) +=  k;
                    dfdx(p2+i, p2+j) += -k;
                }
            }
        }
    }
    else
    {
        auto* m1 = this->mstate1.get();
        auto* m2 = this->mstate2.get();

        auto df1_dx1 = matrix->getForceDerivativeIn(m1).withRespectToPositionsIn(m1);
        auto df1_dx2 = matrix->getForceDerivativeIn(m1).withRespectToPositionsIn(m2);
        auto df2_dx1 = matrix->getForceDerivativeIn(m2).withRespectToPositionsIn(m1);
        auto df2_dx2 = matrix->getForceDerivativeIn(m2).withRespectToPositionsIn(m2);

        df1_dx1.checkValidity(this);
        df1_dx2.checkValidity(this);
        df2_dx1.checkValidity(this);
        df2_dx2.checkValidity(this);

        for (sofa::Index e = 0; e < n; ++e)
        {
            const Spring& s = ss[e];
            const Mat& m = this->dfdx[e];

            const unsigned p1 = Deriv::total_size * s.m1;
            const unsigned p2 = Deriv::total_size * s.m2;

            df1_dx1(p1, p1) += -m;
            df1_dx2(p1, p2) +=  m;
            df2_dx1(p2, p1) +=  m;
            df2_dx2(p2, p2) += -m;
        }
    }
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
    const VecCoord& p1 = this->mstate1->read(core::vec_id::read_access::position)->getValue();
    const VecCoord& p2 = this->mstate2->read(core::vec_id::read_access::position)->getValue();

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    std::vector< Vec3 > points[4];
    const bool external = (this->mstate1 != this->mstate2);
    const type::vector<Spring>& springs = this->d_springs.getValue();
    for (sofa::Index i = 0; i < springs.size(); i++)
    {
        if (!springs[i].enabled) continue;
        assert(i < springs.size());
        assert(springs[i].m2 < p2.size());
        assert(springs[i].m1 < p1.size());
        Real d = (p2[springs[i].m2] - p1[springs[i].m1]).norm();

        const Vec3 point1 = toVec3(DataTypes::getCPos(p1[springs[i].m1]));
        const Vec3 point2 = toVec3(DataTypes::getCPos(p2[springs[i].m2]));

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

    if (d_showArrowSize.getValue()==0 || d_drawMode.getValue() == 0)
    {
        vparams->drawTool()->drawLines(points[0], 1, c0);
        vparams->drawTool()->drawLines(points[1], 1, c1);
        vparams->drawTool()->drawLines(points[2], 1, c2);
        vparams->drawTool()->drawLines(points[3], 1, c3);
    }
    else if (d_drawMode.getValue() == 1)
    {
        const auto numLines0=points[0].size()/2;
        const auto numLines1=points[1].size()/2;
        const auto numLines2=points[2].size()/2;
        const auto numLines3=points[3].size()/2;

        for (unsigned int i=0; i<numLines0; ++i) vparams->drawTool()->drawCylinder(points[0][2*i+1], points[0][2*i], d_showArrowSize.getValue(), c0);
        for (unsigned int i=0; i<numLines1; ++i) vparams->drawTool()->drawCylinder(points[1][2*i+1], points[1][2*i], d_showArrowSize.getValue(), c1);
        for (unsigned int i=0; i<numLines2; ++i) vparams->drawTool()->drawCylinder(points[2][2*i+1], points[2][2*i], d_showArrowSize.getValue(), c2);
        for (unsigned int i=0; i<numLines3; ++i) vparams->drawTool()->drawCylinder(points[3][2*i+1], points[3][2*i], d_showArrowSize.getValue(), c3);

    }
    else if (d_drawMode.getValue() == 2)
    {
        const auto numLines0=points[0].size()/2;
        const auto numLines1=points[1].size()/2;
        const auto numLines2=points[2].size()/2;
        const auto numLines3=points[3].size()/2;

        for (unsigned int i=0; i<numLines0; ++i) vparams->drawTool()->drawArrow(points[0][2*i+1], points[0][2*i], d_showArrowSize.getValue(), c0);
        for (unsigned int i=0; i<numLines1; ++i) vparams->drawTool()->drawArrow(points[1][2*i+1], points[1][2*i], d_showArrowSize.getValue(), c1);
        for (unsigned int i=0; i<numLines2; ++i) vparams->drawTool()->drawArrow(points[2][2*i+1], points[2][2*i], d_showArrowSize.getValue(), c2);
        for (unsigned int i=0; i<numLines3; ++i) vparams->drawTool()->drawArrow(points[3][2*i+1], points[3][2*i], d_showArrowSize.getValue(), c3);
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

    const auto& springsValue = d_springs.getValue();
    if (springsValue.empty())
    {
        return;
    }

    const VecCoord& p1 = this->mstate1->read(core::vec_id::read_access::position)->getValue();
    const VecCoord& p2 = this->mstate2->read(core::vec_id::read_access::position)->getValue();

    type::BoundingBox bbox;

    bool foundSpring = false;

    type::Vec3 a,b;
    for (const auto& spring : springsValue)
    {
        if (spring.enabled)
        {
            if (spring.m1 < p1.size() && spring.m2 < p2.size())
            {
                foundSpring = true;

                DataTypes::get(a[0], a[1], a[2], p1[spring.m1]);
                DataTypes::get(b[0], b[1], b[2], p2[spring.m2]);

                bbox.include(a);
                bbox.include(b);
            }
        }
    }

    if (foundSpring)
    {
        this->f_bbox.setValue(bbox);
    }
}

template <class DataTypes>
void SpringForceField<DataTypes>::clear(sofa::Size reserve)
{
    sofa::type::vector<Spring>& _springs = *this->d_springs.beginEdit();
    _springs.clear();
    if (reserve) _springs.reserve(reserve);

    this->d_springs.cleanDirty();

    updateTopologyIndicesFromSprings();
}

template <class DataTypes>
void SpringForceField<DataTypes>::removeSpring(sofa::Index idSpring)
{
    if (idSpring >= (this->d_springs.getValue()).size())
        return;

    sofa::type::vector<Spring>& springs = *this->d_springs.beginEdit();
    springs.erase(springs.begin() +idSpring );
    this->d_springs.cleanDirty();

    updateTopologyIndices_springRemoved(idSpring);
}

template <class DataTypes>
void SpringForceField<DataTypes>::addSpring(sofa::Index m1, sofa::Index m2, SReal ks, SReal kd, SReal initlen)
{
    d_springs.beginEdit()->push_back(Spring(m1,m2,ks,kd,initlen));
    d_springs.cleanDirty();

    updateTopologyIndicesFromSprings_springAdded();
}

template <class DataTypes>
void SpringForceField<DataTypes>::addSpring(const Spring& spring)
{
    d_springs.beginEdit()->push_back(spring);
    d_springs.cleanDirty();

    updateTopologyIndicesFromSprings_springAdded();
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
