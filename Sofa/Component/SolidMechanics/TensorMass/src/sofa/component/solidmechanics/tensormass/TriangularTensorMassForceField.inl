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

#include <sofa/component/solidmechanics/tensormass/TriangularTensorMassForceField.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/core/topology/TopologyData.inl>

namespace sofa::component::solidmechanics::tensormass
{

typedef core::topology::BaseMeshTopology::EdgesInTriangle EdgesInTriangle;

template< class DataTypes >
void TriangularTensorMassForceField<DataTypes>::applyEdgeCreation(Index /*edgeIndex*/,
        EdgeRestInformation & ei,
        const core::topology::Edge &/*e*/,
        const sofa::type::vector<Index> &,
        const sofa::type::vector<SReal> &)
{
    unsigned int u,v;
    /// set to zero the stiffness matrix
    for (u=0; u<3; ++u)
    {
        for (v=0; v<3; ++v)
        {
            ei.DfDx[u][v]=0;
        }
    }
}

template< class DataTypes >
void TriangularTensorMassForceField<DataTypes>::applyTriangleCreation(const sofa::type::vector<Index> &triangleAdded,
        const sofa::type::vector<core::topology::Triangle> &,
        const sofa::type::vector<sofa::type::vector<Index> > &,
        const sofa::type::vector<sofa::type::vector<SReal> > &)
{
    using namespace core::topology;
    unsigned int i,j,k,l,u,v;

    typename DataTypes::Real val1,area,restSquareLength[3],cotangent[3];
    typename DataTypes::Real lambda=getLambda();
    typename DataTypes::Real mu=getMu();
    typename DataTypes::Real lambdastar, mustar;
    typename DataTypes::Coord point[3],dpk,dpl;
    helper::WriteOnlyAccessor< Data< type::vector<EdgeRestInformation> > > edgeData = edgeInfo;

    const typename DataTypes::VecCoord& restPosition= this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    for (i=0; i<triangleAdded.size(); ++i)
    {

        /// describe the jth edge index of triangle no i
        const EdgesInTriangle &te= this->m_topology->getEdgesInTriangle(triangleAdded[i]);
        /// describe the jth vertex index of triangle no i
        const Triangle &t= this->m_topology->getTriangle(triangleAdded[i]);
        // store points
        for(j=0; j<3; ++j)
            point[j]=(restPosition)[t[j]];
        // store square rest length
        for(j=0; j<3; ++j)
        {
            restSquareLength[j]= (point[(j+1)%3] -point[(j+2)%3]).norm2();
        }
        // compute rest area based on Heron's formula
        area=0;
        for(j=0; j<3; ++j)
        {
            area+=restSquareLength[j]*(restSquareLength[(j+1)%3] +restSquareLength[(j+2)%3]-restSquareLength[j]);
        }
        area=sqrt(area)/4;
        lambdastar=lambda/(4*area);
        mustar=mu/(8*area);

        for(j=0; j<3; ++j)
        {
            cotangent[j]=(restSquareLength[(j+1)%3] +restSquareLength[(j+2)%3]-restSquareLength[j])/(4*area);

            msg_info_when(cotangent[j]<0, this) <<"negative cotangent["
                                                <<triangleAdded[i]<<"]["
                                                <<j<<"]" ;

        }
        for(j=0; j<3; ++j)
        {
            k=(j+1)%3;
            l=(j+2)%3;
            Mat3 &m=edgeData[te[j]].DfDx;
            dpl= point[j]-point[k];
            dpk= point[j]-point[l];
            val1= -cotangent[j]*(lambda+mu)/2;

            if (this->m_topology->getEdge(te[j])[0]==t[l])
            {
                for (u=0; u<3; ++u)
                {
                    for (v=0; v<3; ++v)
                    {
                        m[u][v]+= lambdastar*dpl[u]*dpk[v]+mustar*dpk[u]*dpl[v];
                    }
                    m[u][u]+=val1;
                }
            }
            else
            {
                for (u=0; u<3; ++u)
                {
                    for (v=0; v<3; ++v)
                    {
                        m[v][u]+= lambdastar*dpl[u]*dpk[v]+mustar*dpk[u]*dpl[v];
                    }
                    m[u][u]+=val1;
                }
            }
        }
    }
}

template< class DataTypes>
void TriangularTensorMassForceField<DataTypes>::applyTriangleDestruction(const sofa::type::vector<Index> &triangleRemoved)
{
    using namespace core::topology;
    unsigned int i,j,k,l,u,v;

    typename DataTypes::Real val1,area,restSquareLength[3],cotangent[3];
    typename DataTypes::Real lambda=getLambda();
    typename DataTypes::Real mu=getMu();
    typename DataTypes::Real lambdastar, mustar;
    typename DataTypes::Coord point[3],dpk,dpl;

    helper::WriteOnlyAccessor< Data< type::vector<EdgeRestInformation> > > edgeData = edgeInfo;
    const typename DataTypes::VecCoord& restPosition= this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    for (i=0; i<triangleRemoved.size(); ++i)
    {

        /// describe the jth edge index of triangle no i
        const EdgesInTriangle &te= this->m_topology->getEdgesInTriangle(triangleRemoved[i]);
        /// describe the jth vertex index of triangle no i
        const Triangle &t= this->m_topology->getTriangle(triangleRemoved[i]);
        // store points
        for(j=0; j<3; ++j)
            point[j]=(restPosition)[t[j]];
        // store square rest length
        for(j=0; j<3; ++j)
        {
            restSquareLength[j]= (point[(j+1)%3] -point[(j+2)%3]).norm2();
        }
        // compute rest area based on Heron's formula
        area=0;
        for(j=0; j<3; ++j)
        {
            area+=restSquareLength[j]*(restSquareLength[(j+1)%3] +restSquareLength[(j+2)%3]-restSquareLength[j]);
        }
        area=sqrt(area)/4;
        lambdastar=lambda/(4*area);
        mustar=mu/(8*area);

        for(j=0; j<3; ++j)
        {
            cotangent[j]=(restSquareLength[(j+1)%3] +restSquareLength[(j+2)%3]-restSquareLength[j])/(4*area);

            msg_info_when(cotangent[j]<0, this) << "negative cotangent["
                                                << triangleRemoved[i]<<"]["
                                                << j<<"]"<< msgendl;

        }
        for(j=0; j<3; ++j)
        {
            k=(j+1)%3;
            l=(j+2)%3;
            Mat3 &m=edgeData[te[j]].DfDx;
            dpl= point[j]-point[k];
            dpk= point[j]-point[l];
            val1= -cotangent[j]*(lambda+mu)/2;

            if (this->m_topology->getEdge(te[j])[0]==t[l])
            {
                for (u=0; u<3; ++u)
                {
                    for (v=0; v<3; ++v)
                    {
                        m[u][v]-= lambdastar*dpl[u]*dpk[v]+mustar*dpk[u]*dpl[v];
                    }
                    m[u][u]-=val1;
                }
            }
            else
            {
                for (u=0; u<3; ++u)
                {
                    for (v=0; v<3; ++v)
                    {
                        m[v][u]-= lambdastar*dpl[u]*dpk[v]+mustar*dpk[u]*dpl[v];
                    }
                    m[u][u]-=val1;
                }
            }
        }

    }
}

template <class DataTypes> TriangularTensorMassForceField<DataTypes>::TriangularTensorMassForceField()
    : edgeInfo(initData(&edgeInfo, "edgeInfo", "Internal edge data"))
    , _initialPoints(0)
    , updateMatrix(true)
    , f_poissonRatio(initData(&f_poissonRatio,(Real)0.3,"poissonRatio","Poisson ratio in Hooke's law"))
    , f_youngModulus(initData(&f_youngModulus,(Real)1000.,"youngModulus","Young's modulus in Hooke's law"))
    , l_topology(initLink("topology", "link to the topology container"))
    , lambda(0)
    , mu(0)
    , m_topology(nullptr)
{

}

template <class DataTypes> TriangularTensorMassForceField<DataTypes>::~TriangularTensorMassForceField()
{

}

template <class DataTypes> void TriangularTensorMassForceField<DataTypes>::init()
{
    msg_info() << "initializing TriangularTensorMassForceField";
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

    if (m_topology->getNbTriangles()==0)
    {
        msg_warning() << "No triangles found in linked Topology.";
    }
    updateLameCoefficients();


    helper::WriteOnlyAccessor< Data< type::vector<EdgeRestInformation> > > edgeInf = edgeInfo;

    /// prepare to store info in the edge array
    edgeInf.resize(m_topology->getNbEdges());

    if (_initialPoints.size() == 0)
    {
        // get restPosition
        const VecCoord& p = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
        _initialPoints=p;
    }

    Index i;
    // set edge tensor to 0
    for (i=0; i<m_topology->getNbEdges(); ++i)
    {
        applyEdgeCreation(i,edgeInf[i],m_topology->getEdge(i),
            (const sofa::type::vector<Index>)0,
            (const sofa::type::vector<SReal>)0);
    }
    // create edge tensor by calling the triangle creation function
    sofa::type::vector<Index> triangleAdded;
    for (i=0; i<m_topology->getNbTriangles(); ++i)
        triangleAdded.push_back(i);

    applyTriangleCreation(triangleAdded,
        (const sofa::type::vector<core::topology::Triangle>)0,
        (const sofa::type::vector<sofa::type::vector<Index> >)0,
        (const sofa::type::vector<sofa::type::vector<SReal> >)0
                                      );

    edgeInfo.createTopologyHandler(m_topology);
    edgeInfo.linkToTriangleDataArray();

    edgeInfo.setCreationCallback([this](Index edgeIndex, EdgeRestInformation& ei,
        const core::topology::BaseMeshTopology::Edge& edge,
        const sofa::type::vector< Index >& ancestors,
        const sofa::type::vector< SReal >& coefs)
    {
        applyEdgeCreation(edgeIndex, ei, edge, ancestors, coefs);
    });

    edgeInfo.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TRIANGLESADDED, [this](const core::topology::TopologyChange* eventTopo) {
        const core::topology::TrianglesAdded* triAdd = static_cast<const core::topology::TrianglesAdded*>(eventTopo);
        applyTriangleCreation(triAdd->getIndexArray(), triAdd->getElementArray(), triAdd->ancestorsList, triAdd->coefs);
    });

    edgeInfo.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TRIANGLESREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
        const core::topology::TrianglesRemoved* triRemove = static_cast<const core::topology::TrianglesRemoved*>(eventTopo);
        applyTriangleDestruction(triRemove->getArray());
    });
}

template <class DataTypes>
void TriangularTensorMassForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& /* d_v */)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();

    unsigned int i,v0,v1;
    const unsigned int nbEdges=m_topology->getNbEdges();
    EdgeRestInformation *einfo;

    type::vector<EdgeRestInformation>& edgeInf = *(edgeInfo.beginEdit());

    Coord dp0,dp1,dp;

    for(i=0; i<nbEdges; i++ )
    {
        einfo=&edgeInf[i];
        v0=m_topology->getEdge(i)[0];
        v1=m_topology->getEdge(i)[1];
        dp0=x[v0]-_initialPoints[v0];
        dp1=x[v1]-_initialPoints[v1];
        dp = dp1-dp0;

        f[v1]+=einfo->DfDx*dp;
        f[v0]-=einfo->DfDx.transposeMultiply(dp);
    }

    edgeInfo.endEdit();
    d_f.endEdit();
}


template <class DataTypes>
void TriangularTensorMassForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();
    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    unsigned int v0,v1;
    const size_t nbEdges=m_topology->getNbEdges();
    EdgeRestInformation *einfo;

    type::vector<EdgeRestInformation>& edgeInf = *(edgeInfo.beginEdit());

    Coord dp0,dp1,dp;

    for(unsigned int i=0; i<nbEdges; i++ )
    {
        einfo=&edgeInf[i];
        v0=m_topology->getEdge(i)[0];
        v1=m_topology->getEdge(i)[1];
        dp0=dx[v0];
        dp1=dx[v1];
        dp = dp1-dp0;

        df[v1]+= (einfo->DfDx*dp) * kFactor;
        df[v0]-= (einfo->DfDx.transposeMultiply(dp)) * kFactor;
    }

    edgeInfo.endEdit();
    d_df.endEdit();
}

template <class DataTypes>
void TriangularTensorMassForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}


template<class DataTypes>
void TriangularTensorMassForceField<DataTypes>::updateLameCoefficients()
{
    lambda= f_youngModulus.getValue()*f_poissonRatio.getValue()/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
    mu = f_youngModulus.getValue()*(1-f_poissonRatio.getValue())/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
}


template<class DataTypes>
void TriangularTensorMassForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0, true);

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const size_t nbTriangles=m_topology->getNbTriangles();

    std::vector<sofa::type::Vec3> vertices;
    std::vector<sofa::type::RGBAColor> colors;
    const std::vector<sofa::type::Vec3> normals;

    vparams->drawTool()->disableLighting();

    for(unsigned int i=0; i<nbTriangles; ++i)
    {
        int a = m_topology->getTriangle(i)[0];
        int b = m_topology->getTriangle(i)[1];
        int c = m_topology->getTriangle(i)[2];

        colors.push_back(sofa::type::RGBAColor::green());
        vertices.push_back(x[a]);
        colors.push_back(sofa::type::RGBAColor(0,0.5,0.5,1));
        vertices.push_back(x[b]);
        colors.push_back(sofa::type::RGBAColor::blue());
        vertices.push_back(x[c]);
    }
    vparams->drawTool()->drawTriangles(vertices, normals, colors);

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0, false);


}

} // namespace sofa::component::solidmechanics::tensormass
