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

#include <sofa/component/solidmechanics/tensormass/TetrahedralTensorMassForceField.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/core/topology/TopologyData.inl>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>


namespace sofa::component::solidmechanics::tensormass
{

typedef core::topology::BaseMeshTopology::Tetra				Tetra;
typedef core::topology::BaseMeshTopology::EdgesInTetrahedron		EdgesInTetrahedron;

typedef Tetra			Tetrahedron;
typedef EdgesInTetrahedron		EdgesInTetrahedron;


template< class DataTypes>
void TetrahedralTensorMassForceField<DataTypes>::createEdgeRestInformation(Index, EdgeRestInformation &ei, const core::topology::BaseMeshTopology::Edge &edge, const sofa::type::vector<Index> &, const sofa::type::vector<SReal> &)
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

    ei.vertices[0] =(float) edge[0];
    ei.vertices[1] =(float) edge[1];
}

template< class DataTypes>
void TetrahedralTensorMassForceField<DataTypes>::applyTetrahedronCreation(const sofa::type::vector<Index> &tetrahedronAdded,
        const sofa::type::vector<Tetrahedron> &,
        const sofa::type::vector<sofa::type::vector<Index> > &,
        const sofa::type::vector<sofa::type::vector<SReal> > &)
{
    unsigned int i,j,k,l,u,v;

    typename DataTypes::Real val1,volume;
    typename DataTypes::Real lambda=getLambda();
    typename DataTypes::Real mu=getMu();
    typename DataTypes::Real lambdastar, mustar;
    typename DataTypes::Coord point[4],shapeVector[4];

    const typename DataTypes::VecCoord restPosition=this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    edgeRestInfoVector& edgeData = *(edgeInfo.beginEdit());

    for (i=0; i<tetrahedronAdded.size(); ++i)
    {

        /// get a reference on the edge set of the ith added tetrahedron
        const EdgesInTetrahedron &te= m_topology->getEdgesInTetrahedron(tetrahedronAdded[i]);
        ///get a reference on the vertex set of the ith added tetrahedron
        const Tetrahedron &t= m_topology->getTetrahedron(tetrahedronAdded[i]);
        // store points
        for(j=0; j<4; ++j)
            point[j]=(restPosition)[t[j]];
        /// compute 6 times the rest volume
        volume=dot(cross(point[1]-point[0],point[2]-point[0]),point[0]-point[3]);
        // store shape vectors
        for(j=0; j<4; ++j)
        {
            if ((j%2)==0)
                shapeVector[j]=cross(point[(j+2)%4] - point[(j+1)%4],point[(j+3)%4] - point[(j+1)%4])/volume;
            else
                shapeVector[j]= -cross(point[(j+2)%4] - point[(j+1)%4],point[(j+3)%4] - point[(j+1)%4])/volume;
        }

        lambdastar=lambda*fabs(volume)/6;
        mustar=mu*fabs(volume)/6;


        for(j=0; j<6; ++j)
        {
            /// local indices of the edge
            k = m_topology->getLocalEdgesInTetrahedron(j)[0];
            l = m_topology->getLocalEdgesInTetrahedron(j)[1];

            auto& m = edgeData[te[j]].DfDx;

            val1= dot(shapeVector[k],shapeVector[l])*mustar;

            // print if obtuse tetrahedron along that edge
            msg_info_when(val1<0) << "negative cotangent["<<tetrahedronAdded[i]<<"]["<<j<<"]" ;

            if (m_topology->getEdge(te[j])[0]!=t[l])
            {
                for (u=0; u<3; ++u)
                {
                    for (v=0; v<3; ++v)
                    {
                        m[u][v]+= lambdastar*shapeVector[l][u]*shapeVector[k][v]+mustar*shapeVector[k][u]*shapeVector[l][v];
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
                        m[v][u]+= lambdastar*shapeVector[l][u]*shapeVector[k][v]+mustar*shapeVector[k][u]*shapeVector[l][v];
                    }
                    m[u][u]+=val1;
                }
            }


        }
    }
    edgeInfo.endEdit();
}

template< class DataTypes>
void TetrahedralTensorMassForceField<DataTypes>::applyTetrahedronDestruction(const sofa::type::vector<Index> &tetrahedronRemoved)
{
    unsigned int i,j,k,l,u,v;

    typename DataTypes::Real val1,volume;
    typename DataTypes::Real lambda=getLambda();
    typename DataTypes::Real mu=getMu();
    typename DataTypes::Real lambdastar, mustar;
    typename DataTypes::Coord point[4],shapeVector[4];

    const typename DataTypes::VecCoord restPosition=this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    edgeRestInfoVector& edgeData = *(edgeInfo.beginEdit());

    for (i=0; i<tetrahedronRemoved.size(); ++i)
    {
        /// get a reference on the edge set of the ith added tetrahedron
        const EdgesInTetrahedron &te= m_topology->getEdgesInTetrahedron(tetrahedronRemoved[i]);
        ///get a reference on the vertex set of the ith added tetrahedron
        const Tetrahedron &t= m_topology->getTetrahedron(tetrahedronRemoved[i]);
        // store points
        for(j=0; j<4; ++j)
            point[j]=(restPosition)[t[j]];
        /// compute 6 times the rest volume
        volume=dot(cross(point[1]-point[0],point[2]-point[0]),point[0]-point[3]);
        // store shape vectors
        for(j=0; j<4; ++j)
        {
            if ((j%2)==0)
                shapeVector[j]=cross(point[(j+2)%4] - point[(j+1)%4],point[(j+3)%4] - point[(j+1)%4])/volume;
            else
                shapeVector[j]= -cross(point[(j+2)%4] - point[(j+1)%4],point[(j+3)%4] - point[(j+1)%4])/volume;
        }

        lambdastar=lambda*fabs(volume)/6;
        mustar=mu*fabs(volume)/6;


        for(j=0; j<6; ++j)
        {
            /// local indices of the edge
            k = m_topology->getLocalEdgesInTetrahedron(j)[0];
            l = m_topology->getLocalEdgesInTetrahedron(j)[1];

            auto& m = edgeData[te[j]].DfDx;

            val1= dot(shapeVector[k],shapeVector[l])*mustar;
            // print if obtuse tetrahedron along that edge
            msg_info_when(val1<0) <<"negative cotangent["<<tetrahedronRemoved[i]<<"]["<<j<<"]" ;

            if (m_topology->getEdge(te[j])[0]!=t[l])
            {
                for (u=0; u<3; ++u)
                {
                    for (v=0; v<3; ++v)
                    {
                        m[u][v]-= lambdastar*shapeVector[l][u]*shapeVector[k][v]+mustar*shapeVector[k][u]*shapeVector[l][v];
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
                        m[v][u]-= lambdastar*shapeVector[l][u]*shapeVector[k][v]+mustar*shapeVector[k][u]*shapeVector[l][v];
                    }
                    m[u][u]-=val1;
                }
            }


        }

    }
    edgeInfo.endEdit();
}


template <class DataTypes> 
TetrahedralTensorMassForceField<DataTypes>::TetrahedralTensorMassForceField()
    : _initialPoints(0)
    , updateMatrix(true)
    , f_poissonRatio(initData(&f_poissonRatio,(Real)0.3,"poissonRatio","Poisson ratio in Hooke's law"))
    , f_youngModulus(initData(&f_youngModulus,(Real)1000.,"youngModulus","Young's modulus in Hooke's law"))
    , lambda(0)
    , mu(0)
    , l_topology(initLink("topology", "link to the topology container"))
    , edgeInfo(initData(&edgeInfo, "edgeInfo", "Internal edge data"))
    , m_topology(nullptr)
{

}

template <class DataTypes> 
TetrahedralTensorMassForceField<DataTypes>::~TetrahedralTensorMassForceField()
{

}

template <class DataTypes> void 
TetrahedralTensorMassForceField<DataTypes>::init()
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

    edgeInfo.createTopologyHandler(m_topology);
    edgeInfo.linkToTetrahedronDataArray();

    if (m_topology->getNbTetrahedra()==0)
    {
        msg_warning() << "No tetrahedra found in linked Topology.";
    }
    updateLameCoefficients();

    edgeRestInfoVector& edgeInf = *(edgeInfo.beginEdit());


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
        createEdgeRestInformation(i, edgeInf[i],
                m_topology->getEdge(i),  (const sofa::type::vector< Index > )0,
                (const sofa::type::vector< SReal >)0);
    }
    // create edge tensor by calling the tetrahedron creation function
    sofa::type::vector<Index> tetrahedronAdded;
    for (i=0; i<m_topology->getNbTetrahedra(); ++i)
        tetrahedronAdded.push_back(i);

    applyTetrahedronCreation(tetrahedronAdded,
        (const sofa::type::vector<Tetrahedron>)0,
        (const sofa::type::vector<sofa::type::vector<Index> >)0,
        (const sofa::type::vector<sofa::type::vector<SReal> >)0);


    edgeInfo.setCreationCallback([this](Index edgeIndex, EdgeRestInformation& ei,
        const core::topology::BaseMeshTopology::Edge& edge,
        const sofa::type::vector< Index >& ancestors,
        const sofa::type::vector< SReal >& coefs)
    {
        createEdgeRestInformation(edgeIndex, ei, edge, ancestors, coefs);
    });

    edgeInfo.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TETRAHEDRAADDED, [this](const core::topology::TopologyChange* eventTopo)
    {
        const core::topology::TetrahedraAdded* tAdd = static_cast<const core::topology::TetrahedraAdded*>(eventTopo);
        applyTetrahedronCreation(tAdd->getIndexArray(), tAdd->getElementArray(), tAdd->ancestorsList, tAdd->coefs);
    });

    edgeInfo.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TETRAHEDRAREMOVED, [this](const core::topology::TopologyChange* eventTopo)
    {
        const core::topology::TetrahedraRemoved* tRemove = static_cast<const core::topology::TetrahedraRemoved*>(eventTopo);
        applyTetrahedronDestruction(tRemove->getArray());
    });

    edgeInfo.endEdit();

    /// FOR CUDA
    /// Save the neighbourhood for points (in case of CudaTypes)
    this->initNeighbourhoodPoints();
}

template <class DataTypes>
void TetrahedralTensorMassForceField<DataTypes>::initNeighbourhoodPoints() {}


template <class DataTypes>
SReal  TetrahedralTensorMassForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /* mparams */) const
{
    SCOPED_TIMER("getPotentialEnergy");

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    SReal energy=0;

    unsigned int v0,v1;
    const int nbEdges=m_topology->getNbEdges();

    const EdgeRestInformation *einfo;

    const edgeRestInfoVector edgeInf = edgeInfo.getValue();
    Deriv force,dp;
    Deriv dp0,dp1;

    for(int i=0; i<nbEdges; i++ )
    {
        einfo=&edgeInf[i];
        v0=m_topology->getEdge(i)[0];
        v1=m_topology->getEdge(i)[1];
        dp0=x[v0]-_initialPoints[v0];
        dp1=x[v1]-_initialPoints[v1];
        dp = dp1-dp0;
        force=einfo->DfDx*dp;
        energy+=dot(force,dp1);
        force=einfo->DfDx.multTranspose(dp);
        energy-=dot(force,dp0);
    }

    energy/=-2.0;

    msg_info() << "energy="<<energy ;

    return(energy);
}

template <class DataTypes>
void TetrahedralTensorMassForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& /* d_v */)
{
    SCOPED_TIMER("addForceTetraTensorMass");

    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();

    unsigned int v0,v1;
    const int nbEdges=m_topology->getNbEdges();

    EdgeRestInformation *einfo;

    edgeRestInfoVector& edgeInf = *(edgeInfo.beginEdit());

    Coord dp0,dp1,dp;

    for(int i=0; i<nbEdges; i++ )
    {
        einfo=&edgeInf[i];
        v0=m_topology->getEdge(i)[0];
        v1=m_topology->getEdge(i)[1];
        dp0=x[v0]-_initialPoints[v0];
        dp1=x[v1]-_initialPoints[v1];
        dp = dp1-dp0;

        f[v1]+=einfo->DfDx*dp;
        f[v0]-=einfo->DfDx.multTranspose(dp);
    }

    edgeInfo.endEdit();
    d_f.endEdit();
}


template <class DataTypes>
void TetrahedralTensorMassForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    SCOPED_TIMER("addDForceTetraTensorMass");

    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();
    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    unsigned int v0,v1;
    const int nbEdges=m_topology->getNbEdges();

    EdgeRestInformation *einfo;

    edgeRestInfoVector& edgeInf = *(edgeInfo.beginEdit());

    Coord dp0,dp1,dp;

    for(int i=0; i<nbEdges; i++ )
    {
        einfo=&edgeInf[i];
        v0=m_topology->getEdge(i)[0];
        v1=m_topology->getEdge(i)[1];
        dp0=dx[v0];
        dp1=dx[v1];
        dp = dp1-dp0;

        df[v1]+= (einfo->DfDx*dp) * kFactor;
        df[v0]-= (einfo->DfDx.multTranspose(dp)) * kFactor;
    }
    edgeInfo.endEdit();

    d_df.endEdit();
}

template <class DataTypes>
void TetrahedralTensorMassForceField<DataTypes>::buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix)
{
    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);
    const sofa::Size nbEdges = m_topology->getNbEdges();

    const auto edgeInf = sofa::helper::getReadAccessor(edgeInfo);
    const auto edges = m_topology->getEdges();

    for (sofa::Size i = 0; i < nbEdges; ++i)
    {
        const sofa::topology::Edge& edge = edges[i];

        const unsigned p0 = Deriv::total_size * edge[0];
        const unsigned p1 = Deriv::total_size * edge[1];

        const auto& localDfdx = edgeInf[i].DfDx;
        const auto localDfdx_T = edgeInf[i].DfDx.transposed();

        dfdx(p0, p0) +=  localDfdx_T;
        dfdx(p0, p1) += -localDfdx_T;
        dfdx(p1, p0) += -localDfdx;
        dfdx(p1, p1) +=  localDfdx;
    }
}

template <class DataTypes>
void TetrahedralTensorMassForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}


template<class DataTypes>
void TetrahedralTensorMassForceField<DataTypes>::updateLameCoefficients()
{
    lambda= f_youngModulus.getValue()*f_poissonRatio.getValue()/((1-2*f_poissonRatio.getValue())*(1+f_poissonRatio.getValue()));
    mu = f_youngModulus.getValue()/(2*(1+f_poissonRatio.getValue()));
}


template<class DataTypes>
void TetrahedralTensorMassForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->disableLighting();

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0,true);

    constexpr sofa::type::RGBAColor color = sofa::type::RGBAColor::green();
    std::vector<sofa::type::Vec3> vertices;

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const int nbTriangles=m_topology->getNbTriangles();

    for(int i=0;i<nbTriangles; ++i)
    {
        int a = m_topology->getTriangle(i)[0];
        int b = m_topology->getTriangle(i)[1];
        int c = m_topology->getTriangle(i)[2];

        vertices.push_back(sofa::type::Vec3(x[a]));
        vertices.push_back(sofa::type::Vec3(x[b]));
        vertices.push_back(sofa::type::Vec3(x[c]));
    }
    vparams->drawTool()->drawTriangles(vertices,color);

}

} // namespace sofa::component::solidmechanics::tensormass
