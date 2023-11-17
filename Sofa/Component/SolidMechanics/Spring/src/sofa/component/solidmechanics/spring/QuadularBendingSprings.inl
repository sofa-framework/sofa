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

#include <sofa/component/solidmechanics/spring/QuadularBendingSprings.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/visual/VisualParams.h>
#include <iostream>
#include <sofa/type/RGBAColor.h>
#include <sofa/core/topology/TopologyData.inl>

#include <sofa/core/topology/TopologyChange.h>

namespace sofa::component::solidmechanics::spring
{

typedef core::topology::BaseMeshTopology::Quad				Quad;
typedef core::topology::BaseMeshTopology::EdgesInQuad			EdgesInQuad;

template< class DataTypes>
void QuadularBendingSprings<DataTypes>::applyEdgeCreation(Index /*edgeIndex*/, EdgeInformation &ei, const core::topology::Edge &,
        const sofa::type::vector<Index> &, const sofa::type::vector<SReal> &)
{
    /// set to zero the edge stiffness matrix
    for (auto& s : ei.springs)
    {
        s.DfDx.clear();
    }

    ei.is_activated=false;
    ei.is_initialized=false;
}


template< class DataTypes>
void QuadularBendingSprings<DataTypes>::applyQuadCreation(const sofa::type::vector<Index> &quadAdded,
        const sofa::type::vector<Quad> &,
        const sofa::type::vector<sofa::type::vector<Index> > &,
        const sofa::type::vector<sofa::type::vector<SReal> > &)
{
    SReal m_ks=getKs();
    SReal m_kd=getKd();

    unsigned int u,v;

    const typename DataTypes::VecCoord& restPosition=this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    helper::WriteOnlyAccessor< Data< type::vector<EdgeInformation> > > edgeData = edgeInfo;

    for (unsigned int i=0; i<quadAdded.size(); ++i)
    {

        /// describe the jth edge index of quad no i
        EdgesInQuad te2 = this->m_topology->getEdgesInQuad(quadAdded[i]);
        /// describe the jth vertex index of quad no i
        Quad t2 = this->m_topology->getQuad(quadAdded[i]);

        for(unsigned int j=0; j<4; ++j)
        {

            EdgeInformation &ei = edgeData[te2[j]]; // ff->edgeInfo
            if(!(ei.is_initialized))
            {
                unsigned int edgeIndex = te2[j];
                ei.is_activated=true;

                /// set to zero the edge stiffness matrix
                for (auto& s : ei.springs)
                {
                    s.DfDx.clear();
                }

                const auto& shell = this->m_topology->getQuadsAroundEdge(edgeIndex);
                if (shell.size()==2)
                {
                    EdgesInQuad te1;
                    Quad t1;

                    if(shell[0] == quadAdded[i])
                    {
                        te1 = this->m_topology->getEdgesInQuad(shell[1]);
                        t1 = this->m_topology->getQuad(shell[1]);
                    }
                    else   // shell[1] == quadAdded[i]
                    {
                        te1 = this->m_topology->getEdgesInQuad(shell[0]);
                        t1 = this->m_topology->getQuad(shell[0]);
                    }

                    const int i1 = this->m_topology->getEdgeIndexInQuad(te1, edgeIndex); //edgeIndex //te1[j]
                    const int i2 = this->m_topology->getEdgeIndexInQuad(te2, edgeIndex); // edgeIndex //te2[j]

                    ei.springs[0].edge[0] = t1[i1]; // i1
                    ei.springs[0].edge[1] = t2[(i2+3)%4]; // i2

                    ei.springs[1].edge[0] = t1[(i1+3)%4]; // (i1+3)%4
                    ei.springs[1].edge[1] = t2[i2]; // (i2+3)%4

                    //QuadularBendingSprings<DataTypes> *fftest= (QuadularBendingSprings<DataTypes> *)param;
                    ei.ks=m_ks; //(fftest->ks).getValue();
                    ei.kd=m_kd; //(fftest->kd).getValue();

                    for (auto& s : ei.springs)
                    {
                        const Coord diff = restPosition[s.edge[0]] - restPosition[s.edge[1]];
                        s.restLength = diff.norm();
                    }

                    ei.is_activated=true;
                }
                else
                {
                    ei.is_activated=false;
                }

                ei.is_initialized = true;
            }
        }

    }
}

template< class DataTypes>
void QuadularBendingSprings<DataTypes>::applyQuadDestruction(const sofa::type::vector<Index> &quadRemoved)
{
    SReal m_ks=getKs();
    SReal m_kd=getKd();

    const typename DataTypes::VecCoord& restPosition= this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    helper::WriteOnlyAccessor< Data< type::vector<EdgeInformation> > > edgeData = edgeInfo;

    for (unsigned int i=0; i<quadRemoved.size(); ++i)
    {

        /// describe the jth edge index of quad no i
        EdgesInQuad te = this->m_topology->getEdgesInQuad(quadRemoved[i]);
        /// describe the jth vertex index of quad no i
        //Quad t =  this->m_topology->getQuad(quadRemoved[i]);


        for(unsigned int j=0; j<4; ++j)
        {

            EdgeInformation &ei = edgeData[te[j]]; // ff->edgeInfo
            if(ei.is_initialized)
            {

                unsigned int edgeIndex = te[j];

                const auto& shell = this->m_topology->getQuadsAroundEdge(edgeIndex);
                if (shell.size()==3)
                {

                    EdgesInQuad te1;
                    Quad t1;
                    EdgesInQuad te2;
                    Quad t2;

                    if(shell[0] == quadRemoved[i])
                    {
                        te1 = this->m_topology->getEdgesInQuad(shell[1]);
                        t1 = this->m_topology->getQuad(shell[1]);
                        te2 = this->m_topology->getEdgesInQuad(shell[2]);
                        t2 = this->m_topology->getQuad(shell[2]);

                    }
                    else
                    {

                        if(shell[1] == quadRemoved[i])
                        {

                            te1 = this->m_topology->getEdgesInQuad(shell[2]);
                            t1 = this->m_topology->getQuad(shell[2]);
                            te2 = this->m_topology->getEdgesInQuad(shell[0]);
                            t2 = this->m_topology->getQuad(shell[0]);

                        }
                        else   // shell[2] == quadRemoved[i]
                        {

                            te1 = this->m_topology->getEdgesInQuad(shell[0]);
                            t1 = this->m_topology->getQuad(shell[0]);
                            te2 = this->m_topology->getEdgesInQuad(shell[1]);
                            t2 = this->m_topology->getQuad(shell[1]);

                        }
                    }

                    const int i1 = this->m_topology->getEdgeIndexInQuad(te1, edgeIndex);
                    const int i2 = this->m_topology->getEdgeIndexInQuad(te2, edgeIndex);

                    ei.springs[0].edge[0] = t1[i1];
                    ei.springs[0].edge[1] = t2[(i2+3)%4];

                    ei.springs[1].edge[0] = t1[(i1+3)%4];
                    ei.springs[1].edge[1] = t2[i2];

                    //QuadularBendingSprings<DataTypes> *fftest= (QuadularBendingSprings<DataTypes> *)param;
                    ei.ks=m_ks; //(fftest->ks).getValue();
                    ei.kd=m_kd; //(fftest->kd).getValue();

                    for (auto& s : ei.springs)
                    {
                        const Coord u = restPosition[s.edge[0]] - restPosition[s.edge[1]];
                        s.restLength = u.norm();
                    }

                    ei.is_activated=true;

                }
                else
                {

                    ei.is_activated=false;
                    ei.is_initialized = false;

                }

            }
            else
            {

                ei.is_activated=false;
                ei.is_initialized = false;

            }
        }

    }
}

template< class DataTypes>
void QuadularBendingSprings<DataTypes>::applyPointDestruction(const sofa::type::vector<Index> &tab)
{
    unsigned int last = this->m_topology->getNbPoints() -1;

    helper::WriteOnlyAccessor< Data< type::vector<EdgeInformation> > > edgeInf = edgeInfo;

    sofa::type::vector<Index> lastIndexVec;
    for(unsigned int i_init = 0; i_init < tab.size(); ++i_init)
    {

        lastIndexVec.push_back(last - i_init);
    }

    for ( unsigned int i = 0; i < tab.size(); ++i)
    {
        unsigned int i_next = i;
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

        const auto &shell= this->m_topology->getQuadsAroundVertex(lastIndexVec[i]);
        for (unsigned int j = 0; j < shell.size(); ++j)
        {

            Quad tj = this->m_topology->getQuad(shell[j]);

            const unsigned int vertexIndex = this->m_topology->getVertexIndexInQuad(tj, lastIndexVec[i]);

            const EdgesInQuad& tej = this->m_topology->getEdgesInQuad(shell[j]);

            for (unsigned int j_edge=vertexIndex; j_edge%4 !=(vertexIndex+2)%4; ++j_edge)
            {
                unsigned int ind_j = tej[j_edge%4];

                for (auto& s : edgeInf[ind_j].springs)
                {
                    for (auto& e : s.edge)
                    {
                        if (e == last)
                        {
                            e = tab[i];
                        }
                    }
                }

            }
        }

        --last;
    }
}



template< class DataTypes>
void QuadularBendingSprings<DataTypes>::applyPointRenumbering(const sofa::type::vector<Index> &tab)
{
    for (auto& edgeInf : sofa::helper::getWriteOnlyAccessor(edgeInfo))
    {
        if (edgeInf.is_activated)
        {
            for (auto& s : edgeInf.springs)
            {
                for (auto& e : s.edge)
                {
                    e = tab[e];
                }
            }
        }
    }
}


template<class DataTypes>
QuadularBendingSprings<DataTypes>::QuadularBendingSprings()
    : f_ks ( initData(&f_ks,(SReal) 100000.0,"stiffness","uniform stiffness for the all springs"))
    , f_kd ( initData(&f_kd,(SReal) 1.0,"damping","uniform damping for the all springs"))
    , l_topology(initLink("topology", "link to the topology container"))
    , edgeInfo ( initData(&edgeInfo, "edgeInfo","Internal edge data"))
    , m_topology(nullptr)
    , updateMatrix(true)
{

}


template<class DataTypes>
QuadularBendingSprings<DataTypes>::~QuadularBendingSprings()
{

}


template<class DataTypes>
void QuadularBendingSprings<DataTypes>::init()
{
    this->Inherited::init();

    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    m_topology = l_topology.get();
    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    if (m_topology == nullptr)
    {
        msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    if (m_topology->getNbQuads()==0)
    {
        msg_warning() << "No Quads found in linked Topology.";
    }

    edgeInfo.createTopologyHandler(m_topology);
    edgeInfo.linkToPointDataArray();
    edgeInfo.linkToQuadDataArray();

    /// prepare to store info in the edge array
    helper::WriteOnlyAccessor< Data< type::vector<EdgeInformation> > > edgeInf = edgeInfo;
    edgeInf.resize(m_topology->getNbEdges());

    // set edge tensor to 0
    for (Index i=0; i<m_topology->getNbEdges(); ++i)
    {
        applyEdgeCreation(i, edgeInf[i],
            m_topology->getEdge(i),  (const sofa::type::vector< Index > )0,
            (const sofa::type::vector< SReal >)0);
    }

    // create edge tensor by calling the quad creation function
    sofa::type::vector<Index> quadAdded;
    for (unsigned int i=0; i<m_topology->getNbQuads(); ++i)
        quadAdded.push_back(i);

    applyQuadCreation(quadAdded,
        (const sofa::type::vector<Quad>)0,
        (const sofa::type::vector<sofa::type::vector<Index> >)0,
        (const sofa::type::vector<sofa::type::vector<SReal> >)0);

    edgeInfo.setCreationCallback([this](Index edgeIndex, EdgeInformation& ei,
        const core::topology::BaseMeshTopology::Edge& edge,
        const sofa::type::vector< Index >& ancestors,
        const sofa::type::vector< SReal >& coefs)
    {
        applyEdgeCreation(edgeIndex, ei, edge, ancestors, coefs);
    });

    edgeInfo.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::QUADSADDED, [this](const core::topology::TopologyChange* eventTopo) {
        const core::topology::QuadsAdded* quadAdd = static_cast<const core::topology::QuadsAdded*>(eventTopo);
        applyQuadCreation(quadAdd->getIndexArray(), quadAdd->getElementArray(), quadAdd->ancestorsList, quadAdd->coefs);
    });

    edgeInfo.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::QUADSREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
        const core::topology::QuadsRemoved* quadRemove = static_cast<const core::topology::QuadsRemoved*>(eventTopo);
        applyQuadDestruction(quadRemove->getArray());
    });

    edgeInfo.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::POINTSREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
        const core::topology::PointsRemoved* pRemove = static_cast<const core::topology::PointsRemoved*>(eventTopo);
        applyPointDestruction(pRemove->getArray());
    });

    edgeInfo.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::POINTSRENUMBERING, [this](const core::topology::TopologyChange* eventTopo) {
        const core::topology::PointsRenumbering* pRenum = static_cast<const core::topology::PointsRenumbering*>(eventTopo);
        applyPointRenumbering(pRenum->getIndexArray());
    });

}

template <class DataTypes>
SReal QuadularBendingSprings<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /* mparams */, const DataVecCoord& /* d_x */) const
{
    msg_error() << "getPotentialEnergy-not-implemented !!!";
    return 0;
}

template <class DataTypes>
auto QuadularBendingSprings<DataTypes>::computeForce(
    const VecDeriv& v,
    const EdgeInformation& einfo, const typename EdgeInformation::Spring& spring,
    Coord direction,
    Real distance) -> ForceOutput
{
    ForceOutput force;

    force.inverseLength = 1 / distance;
    direction *= force.inverseLength;
    const Real elongation = distance - spring.restLength;
    m_potentialEnergy += elongation * elongation * einfo.ks / 2;

    const Deriv relativeVelocity = v[spring.edge[1]] - v[spring.edge[0]];
    const Real elongationVelocity = sofa::type::dot(direction, relativeVelocity);
    force.forceIntensity = einfo.ks * elongation + einfo.kd * elongationVelocity;
    force.force = direction * force.forceIntensity;

    return force;
}

template <class DataTypes>
auto QuadularBendingSprings<DataTypes>::computeLocalJacobian(EdgeInformation& einfo, const Coord& direction, const ForceOutput& force)
-> Mat
{
    const Real tgt = force.forceIntensity * force.inverseLength;
    Mat jacobian = (einfo.ks - tgt) * sofa::type::dyad(direction, direction);
    for (int j = 0; j < N; ++j)
    {
        jacobian[j][j] += tgt;
    }
    return jacobian;
}

template <class DataTypes>
void QuadularBendingSprings<DataTypes>::computeSpringForce(VecDeriv& f, const VecCoord& x,
    const VecDeriv& v, EdgeInformation& einfo, typename EdgeInformation::Spring& spring)
{
    const auto e0 = spring.edge[0];
    const auto e1 = spring.edge[1];

    const Coord difference = x[e1] - x[e0];
    const Real distance = difference.norm();

    if (distance > 1.0e-4)
    {
        const ForceOutput force = computeForce(v, einfo, spring, difference, distance);

        f[e0] += force.force;
        f[e1] -= force.force;

        updateMatrix = true;

        spring.DfDx += computeLocalJacobian(einfo, difference / distance, force);
    }
}

template<class DataTypes>
void QuadularBendingSprings<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    auto f = sofa::helper::getWriteAccessor(d_f);

    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    auto edgeInf = sofa::helper::getWriteAccessor(edgeInfo);

    f.resize(x.size());
    m_potentialEnergy = 0;

    for (auto& einfo : edgeInf)
    {
        if (einfo.is_activated)
        {
            for (auto& s : einfo.springs)
            {
                s.DfDx.clear();
                computeSpringForce(f.wref(), x, v, einfo, s);
            }
        }
    }
}

template<class DataTypes>
void QuadularBendingSprings<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    auto df = sofa::helper::getWriteAccessor(d_df);
    const VecDeriv& dx = d_dx.getValue();
    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    const type::vector<EdgeInformation>& edgeInf = edgeInfo.getValue();

    df.resize(dx.size());

    for (const auto& einfo : edgeInf)
    {
        if (einfo.is_activated)
        {
            for (const auto& s : einfo.springs)
            {
                const auto e0 = s.edge[0];
                const auto e1 = s.edge[1];
                const Coord ddx = dx[e1] - dx[e0];
                const Deriv dforce = s.DfDx * (ddx * kFactor);

                df[e0] += dforce;
                df[e1] -= dforce;
            }

            updateMatrix = false;
        }
    }
}

template <class DataTypes>
void QuadularBendingSprings<DataTypes>::buildStiffnessMatrix(
    core::behavior::StiffnessMatrix* matrix)
{
    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);

    const type::vector<EdgeInformation>& edgeInf = edgeInfo.getValue();

    for (const auto& einfo : edgeInf)
    {
        if (einfo.is_activated) // edge not in middle of 2 triangles
        {
            for (const auto& s : einfo.springs)
            {
                const sofa::Index a = Deriv::total_size * s.edge[0];
                const sofa::Index b = Deriv::total_size * s.edge[1];

                const auto& dfdxLocal = s.DfDx;

                dfdx(a, a) += -dfdxLocal;
                dfdx(a, b) +=  dfdxLocal;
                dfdx(b, a) +=  dfdxLocal;
                dfdx(b, b) += -dfdxLocal;
            }
        }
    }
}

template <class DataTypes>
void QuadularBendingSprings<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}

template<class DataTypes>
void QuadularBendingSprings<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0, true);

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    vparams->drawTool()->disableLighting();

    const type::vector<EdgeInformation>& edgeInf = edgeInfo.getValue();
    std::vector<sofa::type::Vec3> vertices;
    std::vector<sofa::type::RGBAColor> colors;
    constexpr sofa::type::RGBAColor green_color = sofa::type::RGBAColor::green();
    constexpr sofa::type::RGBAColor red_color   = sofa::type::RGBAColor::red();
    constexpr sofa::type::RGBAColor color1 = sofa::type::RGBAColor(1,0.5, 0,1);
    constexpr sofa::type::RGBAColor color2 = sofa::type::RGBAColor(0,1,0.5,1);

    for(unsigned int i=0; i<edgeInf.size(); ++i)
    {
        if(edgeInf[i].is_activated)
        {
            for (const auto& s : edgeInf[i].springs)
            {
                const Real d1 = (x[s.edge[1]] - x[s.edge[0]]).norm();
                if (d1 < s.restLength * 0.9999)
                {
                    colors.push_back(red_color);
                }
                else
                {
                    colors.push_back(green_color);
                }

                vertices.push_back( x[s.edge[0]] );
                vertices.push_back( x[s.edge[1]] );
            }
        }
    }
    vparams->drawTool()->drawLines(vertices, 1, colors);

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0, false);


    vertices.clear();
    for(unsigned int i=0; i<m_topology->getNbQuads(); ++i)
    {
        for(unsigned int j = 0 ; j<4 ; j++)
            vertices.push_back(x[m_topology->getQuad(i)[j]]);
    }
    vparams->drawTool()->drawQuads(vertices, sofa::type::RGBAColor::red());
}


} // namespace sofa::component::solidmechanics::spring
