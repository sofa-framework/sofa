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

#include <sofa/component/solidmechanics/spring/TriangularBendingSprings.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/core/topology/TopologyData.inl>

namespace sofa::component::solidmechanics::spring
{

typedef core::topology::BaseMeshTopology::EdgesInTriangle EdgesInTriangle;

template< class DataTypes>
void TriangularBendingSprings<DataTypes>::applyEdgeCreation(Index , EdgeInformation &ei, const core::topology::Edge &,
    const sofa::type::vector<Index> &, const sofa::type::vector<SReal> &)
{
    /// set to zero the edge stiffness matrix
    for (auto u=0; u<N; ++u)
    {
        for (auto v=0; v<N; ++v)
        {
            ei.DfDx(u,v) = Real(0);
        }
    }

    ei.is_activated=false;
    ei.is_initialized=false;
}


template< class DataTypes>
void TriangularBendingSprings<DataTypes>::applyTriangleCreation(const sofa::type::vector<Index> &triangleAdded, const sofa::type::vector<core::topology::Triangle> &, 
    const sofa::type::vector<sofa::type::vector<Index> > &, const sofa::type::vector<sofa::type::vector<SReal> > &)
{
    using namespace core::topology;
    Real m_ks=getKs();
    Real m_kd=getKd();

    const typename DataTypes::VecCoord& restPosition = this->mstate->read(core::vec_id::read_access::restPosition)->getValue();
    sofa::helper::WriteOnlyAccessor< core::objectmodel::Data< type::vector<EdgeInformation> > > edgeData = d_edgeInfo;

    for (unsigned int i=0; i<triangleAdded.size(); ++i)
    {
        /// describe the jth edge index of triangle no i
        EdgesInTriangle te2 = m_topology->getEdgesInTriangle(triangleAdded[i]);
        /// describe the jth vertex index of triangle no i
        Triangle t2 = m_topology->getTriangle(triangleAdded[i]);

        for(unsigned int j=0; j<3; ++j) // for each edge in triangle
        {
            EdgeInformation &ei = edgeData[te2[j]]; // d_edgeInfo
            if(!(ei.is_initialized))
            {
                ei.is_initialized = true;
                const unsigned int edgeIndex = te2[j];
                const auto& shell = m_topology->getTrianglesAroundEdge(edgeIndex);

                if (shell.size() != 2) // Only Manifold triangulation is handled
                {
                    ei.is_activated = false;
                    continue;
                }


                /// set to zero the edge stiffness matrix
                for (auto u=0; u<N; ++u)
                {
                    for (auto v=0; v<N; ++v)
                    {
                        ei.DfDx(u,v) = Real(0);
                    }
                }

                EdgesInTriangle te1;
                Triangle t1;

                // search for the opposite triangle in shell
                if(shell[0] == triangleAdded[i])
                {
                    te1 = m_topology->getEdgesInTriangle(shell[1]);
                    t1 = m_topology->getTriangle(shell[1]);
                }
                else   // shell[1] == triangleAdded[i]
                {
                    te1 = m_topology->getEdgesInTriangle(shell[0]);
                    t1 = m_topology->getTriangle(shell[0]);
                }

                // get localIndices of the edge in triangle, same index as opposite vertex in triangle
                const int i1 = m_topology->getEdgeIndexInTriangle(te1, edgeIndex);
                const int i2 = m_topology->getEdgeIndexInTriangle(te2, edgeIndex);

                // store vertex indices of the spring extremities, as well as global spring stiffness and damping
                ei.m1 = t1[i1];
                ei.m2 = t2[i2];
                ei.ks = m_ks;
                ei.kd = m_kd;

                Coord u = (restPosition)[ei.m1] - (restPosition)[ei.m2];
                Real d = u.norm();
                ei.restlength = d;
                ei.is_activated = true;
            }
        }
    }
}


template< class DataTypes>
void TriangularBendingSprings<DataTypes>::applyTriangleDestruction(const sofa::type::vector<Index> &triangleRemoved)
{
    using namespace core::topology;

    Real m_ks=getKs();
    Real m_kd=getKd();

    const typename DataTypes::VecCoord& restPosition = this->mstate->read(core::vec_id::read_access::restPosition)->getValue();
    sofa::helper::WriteOnlyAccessor< core::objectmodel::Data< type::vector<EdgeInformation> > > edgeData = d_edgeInfo;

    for (unsigned int i=0; i<triangleRemoved.size(); ++i)
    {
        /// describe the jth edge index of triangle no i
        EdgesInTriangle te = m_topology->getEdgesInTriangle(triangleRemoved[i]);

        for(unsigned int j=0; j<3; ++j) // for each edge of the triangle
        {
            EdgeInformation &ei = edgeData[te[j]]; // d_edgeInfo
            if (!ei.is_initialized) // not init == no spring, nothing to do
            {
                ei.is_activated = false;
                ei.is_initialized = false;
                continue;
            }

            const unsigned int edgeIndex = te[j];

            const auto& shell = m_topology->getTrianglesAroundEdge(edgeIndex);
            if (shell.size()==3) // This case is possible during remeshing phase (adding/removing triangles)
            {
                // search for the 2 opposites triangles in shell To keep only this spring
                EdgesInTriangle te1;
                Triangle t1;
                EdgesInTriangle te2;
                Triangle t2;

                if(shell[0] == triangleRemoved[i])
                {
                    te1 = m_topology->getEdgesInTriangle(shell[1]);
                    t1 = m_topology->getTriangle(shell[1]);
                    te2 = m_topology->getEdgesInTriangle(shell[2]);
                    t2 = m_topology->getTriangle(shell[2]);
                }
                else if (shell[1] == triangleRemoved[i])
                {
                    te1 = m_topology->getEdgesInTriangle(shell[2]);
                    t1 = m_topology->getTriangle(shell[2]);
                    te2 = m_topology->getEdgesInTriangle(shell[0]);
                    t2 = m_topology->getTriangle(shell[0]);
                }
                else   // shell[2] == triangleRemoved[i]
                {
                    te1 = m_topology->getEdgesInTriangle(shell[0]);
                    t1 = m_topology->getTriangle(shell[0]);
                    te2 = m_topology->getEdgesInTriangle(shell[1]);
                    t2 = m_topology->getTriangle(shell[1]);
                }

                const int i1 = m_topology->getEdgeIndexInTriangle(te1, edgeIndex);
                const int i2 = m_topology->getEdgeIndexInTriangle(te2, edgeIndex);

                ei.m1 = t1[i1];
                ei.m2 = t2[i2];
                ei.ks = m_ks;
                ei.kd = m_kd;

                Coord u = (restPosition)[ei.m1] - (restPosition)[ei.m2];
                Real d = u.norm();

                ei.restlength = d;
                ei.is_activated = true;
            }
            else
            {
                ei.is_activated = false;
                ei.is_initialized = false;
            }
        }
    }
}


template<class DataTypes>
void TriangularBendingSprings<DataTypes>::applyPointDestruction(const sofa::type::vector<Index> &tab)
{
    using namespace core::topology;

    sofa::Index last = m_topology->getNbPoints() -1;
    unsigned int i,j;

    sofa::helper::WriteOnlyAccessor< core::objectmodel::Data< type::vector<EdgeInformation> > > edgeInf = d_edgeInfo;

    sofa::type::vector<unsigned int> lastIndexVec;
    for(unsigned int i_init = 0; i_init < tab.size(); ++i_init)
    {
        lastIndexVec.push_back(last - i_init);
    }

    for ( i = 0; i < tab.size(); ++i)
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

        const auto &shell= m_topology->getTrianglesAroundVertex(lastIndexVec[i]);
        int iLast = int(last);
        int iTab = int(tab[i]);
        for (j=0; j<shell.size(); ++j)
        {
            Triangle tj = m_topology->getTriangle(shell[j]);
            const int vertexIndex = m_topology->getVertexIndexInTriangle(tj, lastIndexVec[i]);

            EdgesInTriangle tej = m_topology->getEdgesInTriangle(shell[j]);
            unsigned int ind_j = tej[vertexIndex];

            if (edgeInf[ind_j].m1 == iLast)
            {
                edgeInf[ind_j].m1 = iTab;
            }
            else if (edgeInf[ind_j].m2 == iLast)
            {
                edgeInf[ind_j].m2 = iTab;
            }
        }

        --last;
    }
}


template<class DataTypes>
void TriangularBendingSprings<DataTypes>::applyPointRenumbering(const sofa::type::vector<Index> &tab)
{
    sofa::helper::WriteOnlyAccessor< core::objectmodel::Data< type::vector<EdgeInformation> > > edgeInf = d_edgeInfo;
    for (unsigned int i = 0; i < m_topology->getNbEdges(); ++i)
    {
        if(edgeInf[i].is_activated)
        {
            edgeInf[i].m1  = tab[edgeInf[i].m1];
            edgeInf[i].m2  = tab[edgeInf[i].m2];
        }
    }
}


template<class DataTypes>
TriangularBendingSprings<DataTypes>::TriangularBendingSprings()
    : d_ks(initData(&d_ks, Real(100000.0),"stiffness","uniform stiffness for the all springs"))
    , d_kd(initData(&d_kd, Real(1.0),"damping","uniform damping for the all springs"))
    , d_showSprings(initData(&d_showSprings, true, "showSprings", "option to draw springs"))
    , l_topology(initLink("topology", "link to the topology container"))
    , d_edgeInfo(initData(&d_edgeInfo, "edgeInfo", "Internal edge data"))
    , m_potentialEnergy(0.0)
    , m_topology(nullptr)
{
}

template<class DataTypes>
TriangularBendingSprings<DataTypes>::~TriangularBendingSprings()
{

}


template<class DataTypes>
void TriangularBendingSprings<DataTypes>::init()
{
    this->Inherited::init();

    // checking inputs using setter
    setKs(d_ks.getValue());
    setKd(d_kd.getValue());

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

    if (m_topology->d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Invalid || m_topology->getNbTriangles()==0)
    {
        msg_error() << " object must have a Triangular Set Topology.";
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
    d_edgeInfo.createTopologyHandler(m_topology);
    d_edgeInfo.linkToPointDataArray();
    d_edgeInfo.linkToTriangleDataArray();

    d_edgeInfo.setCreationCallback([this](Index edgeIndex, EdgeInformation& ei,
                                          const core::topology::BaseMeshTopology::Edge& edge,
                                          const sofa::type::vector< Index >& ancestors,
                                          const sofa::type::vector< SReal >& coefs)
    {
        applyEdgeCreation(edgeIndex, ei, edge, ancestors, coefs);
    });

    d_edgeInfo.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TRIANGLESADDED, [this](const core::topology::TopologyChange* eventTopo) {
        const core::topology::TrianglesAdded* triAdd = static_cast<const core::topology::TrianglesAdded*>(eventTopo);
        applyTriangleCreation(triAdd->getIndexArray(), triAdd->getElementArray(), triAdd->ancestorsList, triAdd->coefs);
    });

    d_edgeInfo.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TRIANGLESREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
        const core::topology::TrianglesRemoved* triRemove = static_cast<const core::topology::TrianglesRemoved*>(eventTopo);
        applyTriangleDestruction(triRemove->getArray());
    });

    d_edgeInfo.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::POINTSREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
        const core::topology::PointsRemoved* pRemove = static_cast<const core::topology::PointsRemoved*>(eventTopo);
        applyPointDestruction(pRemove->getArray());
    });

    d_edgeInfo.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::POINTSRENUMBERING, [this](const core::topology::TopologyChange* eventTopo) {
        const core::topology::PointsRenumbering* pRenum = static_cast<const core::topology::PointsRenumbering*>(eventTopo);
        applyPointRenumbering(pRenum->getIndexArray());
    });

    this->reinit();
}


template<class DataTypes>
void TriangularBendingSprings<DataTypes>::reinit()
{
    using namespace core::topology;
    /// prepare to store info in the edge array
    sofa::helper::WriteOnlyAccessor< core::objectmodel::Data< type::vector<EdgeInformation> > > edgeInf = d_edgeInfo;
    edgeInf.resize(m_topology->getNbEdges());
    Index i;
    // set edge tensor to 0
    for (i=0; i<m_topology->getNbEdges(); ++i)
    {

        applyEdgeCreation(i, edgeInf[i],
            m_topology->getEdge(i),  (const sofa::type::vector< Index > )0,
            (const sofa::type::vector< SReal >)0);
    }

    // create edge tensor by calling the triangle creation function
    sofa::type::vector<Index> triangleAdded;
    for (i=0; i<m_topology->getNbTriangles(); ++i)
        triangleAdded.push_back(i);

    applyTriangleCreation(triangleAdded,
        (const sofa::type::vector<Triangle>)0,
        (const sofa::type::vector<sofa::type::vector<Index> >)0,
        (const sofa::type::vector<sofa::type::vector<SReal> >)0);
}

template <class DataTypes>
SReal TriangularBendingSprings<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /* mparams */, const DataVecCoord& /* d_x */) const
{
    msg_error()<<"TriangularBendingSprings::getPotentialEnergy-not-implemented !!!";
    return 0;
}


template<class DataTypes>
void TriangularBendingSprings<DataTypes>::setKs(const Real ks)
{ 
    if (ks < 0)
    {
        msg_warning() << "Input Bending Stiffness is not possible: " << ks << ", setting default value: 100000.0";
    }
    else if (ks != d_ks.getValue())
    {
        d_ks.setValue(ks);
    }
}


template<class DataTypes>
void TriangularBendingSprings<DataTypes>::setKd(const Real kd) 
{ 
    if (kd < 0)
    {
        msg_warning() << "Input Bending damping is not possible: " << kd << ", setting default value: 1.0";
    }
    else if (kd != d_kd.getValue())
    {
        d_kd.setValue(kd);
    }
}


template<class DataTypes>
void TriangularBendingSprings<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    const size_t nbEdges = m_topology->getNbEdges();
    sofa::helper::WriteOnlyAccessor< core::objectmodel::Data< type::vector<EdgeInformation> > > edgeInf = d_edgeInfo;

    f.resize(x.size());
    m_potentialEnergy = 0;

    for(sofa::Index i=0; i<nbEdges; i++ )
    {
        EdgeInformation& einfo = edgeInf[i];

        if (!einfo.is_activated) // edge not in middle of 2 triangles
            continue;

        int a = einfo.m1;
        int b = einfo.m2;
        Coord u = x[b]-x[a];
        Real d = u.norm();
        if( d>1.0e-4 )
        {
            Real inverseLength = 1.0f/d;
            u *= inverseLength;
            Real elongation = (Real)(d - einfo.restlength);
            m_potentialEnergy += elongation * elongation * einfo.ks / 2;

            Deriv relativeVelocity = v[b]-v[a];
            Real elongationVelocity = dot(u,relativeVelocity);
            Real forceIntensity = (Real)(einfo.ks*elongation+einfo.kd*elongationVelocity);
            Deriv force = u*forceIntensity;
            f[a]+=force;
            f[b]-=force;

            Mat& m = einfo.DfDx;
            Real tgt = forceIntensity * inverseLength;
            for( int j=0; j<N; ++j )
            {
                for( int k=0; k<N; ++k )
                {
                    m(j,k) = ((Real)einfo.ks-tgt) * u[j] * u[k];
                }
                m(j,j) += tgt;
            }
        }
        else // null length, no force and no stiffness
        {
            Mat& m = einfo.DfDx;
            for( int j=0; j<N; ++j )
            {
                for( int k=0; k<N; ++k )
                {
                    m(j,k) = 0;
                }
            }
        }
    }

    d_f.endEdit();
}

template<class DataTypes>
void TriangularBendingSprings<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    auto df = sofa::helper::getWriteAccessor(d_df);
    const VecDeriv& dx = d_dx.getValue();
    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    const size_t nbEdges=m_topology->getNbEdges();
    const type::vector<EdgeInformation>& edgeInf = d_edgeInfo.getValue();
    df.resize(dx.size());

    for(sofa::Index i=0; i<nbEdges; i++ )
    {
        const EdgeInformation& einfo = edgeInf[i];

        if (!einfo.is_activated) // edge not in middle of 2 triangles
            continue;

        const int a = einfo.m1;
        const int b = einfo.m2;
        const Coord d = dx[b]-dx[a];
        const Deriv dforce = einfo.DfDx*d;
        df[a]+= dforce * kFactor;
        df[b]-= dforce * kFactor;
    }
}

template <class DataTypes>
void TriangularBendingSprings<DataTypes>::buildStiffnessMatrix(
    core::behavior::StiffnessMatrix* matrix)
{
    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);

    const type::vector<EdgeInformation>& edgeInf = d_edgeInfo.getValue();

    for (sofa::Index i = 0; i < m_topology->getNbEdges(); i++)
    {
        const EdgeInformation& einfo = edgeInf[i];
        if (einfo.is_activated) // edge not in middle of 2 triangles
        {
            const sofa::Index a = Deriv::total_size * einfo.m1;
            const sofa::Index b = Deriv::total_size * einfo.m2;

            const Mat& dfdxLocal = einfo.DfDx;

            dfdx(a, a) += -dfdxLocal;
            dfdx(a, b) +=  dfdxLocal;
            dfdx(b, a) +=  dfdxLocal;
            dfdx(b, b) += -dfdxLocal;
        }
    }
}

template <class DataTypes>
void TriangularBendingSprings<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}


template<class DataTypes>
void TriangularBendingSprings<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!d_showSprings.getValue()) { return; }
    if (!vparams->displayFlags().getShowForceFields()) {return;}
    if (!this->mstate) {return;}

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    if (vparams->displayFlags().getShowWireFrame()){
        vparams->drawTool()->setPolygonMode(0, true);
    }

    const VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();
    std::vector<sofa::type::Vec3> vertices;
    std::vector<sofa::type::RGBAColor> colors;

    vparams->drawTool()->disableLighting();
    const type::vector<EdgeInformation>& edgeInfos = d_edgeInfo.getValue();
    for(auto& edgeInfo : edgeInfos)
    {
        if(edgeInfo.is_activated)
        {
            const bool external=true;
            Real d = (x[edgeInfo.m2] - x[edgeInfo.m1]).norm();
            if (external)
            {
                if (d<edgeInfo.restlength*0.9999)
                {
                    colors.push_back(sofa::type::RGBAColor::red());
                }
                else
                {
                    colors.push_back(sofa::type::RGBAColor::green());
                }
            }
            else
            {
                if (d<edgeInfo.restlength*0.9999)
                {
                    colors.push_back(sofa::type::RGBAColor(1,0.5, 0,1));
                }
                else
                {
                    colors.push_back(sofa::type::RGBAColor(0,1,0.5,1));
                }
            }

            vertices.push_back( x[edgeInfo.m1] );
            vertices.push_back( x[edgeInfo.m2] );
        }
    }
    vparams->drawTool()->drawLines(vertices, 1, colors);

    if (vparams->displayFlags().getShowWireFrame()){
        vparams->drawTool()->setPolygonMode(0, false);
    }


}

} // namespace sofa::component::solidmechanics::spring
