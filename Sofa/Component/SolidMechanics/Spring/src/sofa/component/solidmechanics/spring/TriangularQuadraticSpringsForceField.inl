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

#include <sofa/component/solidmechanics/spring/TriangularQuadraticSpringsForceField.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/core/topology/TopologyData.inl>

namespace sofa::component::solidmechanics::spring
{

template< class DataTypes>
void TriangularQuadraticSpringsForceField<DataTypes>::applyEdgeCreation(Index edgeIndex, EdgeRestInformation &ei, const core::topology::Edge &, const sofa::type::vector<Index> &, const sofa::type::vector<SReal> &)
{
    // store the rest length of the edge created
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    const auto& e = this->m_topology->getEdge(edgeIndex);
    const auto& n0 = DataTypes::getCPos(x[e[0]]);
    const auto& n1 = DataTypes::getCPos(x[e[1]]);

    ei.restLength = sofa::geometry::Edge::length(n0, n1);
    ei.stiffness=0;
}



template< class DataTypes>
void TriangularQuadraticSpringsForceField<DataTypes>::applyTriangleCreation(Index triangleIndex, TriangleRestInformation &tinfo,
        const core::topology::Triangle &, const sofa::type::vector<Index> &,
        const sofa::type::vector<SReal> &)
{
    unsigned int j=0,k=0,l=0;

    typename DataTypes::Real area,squareRestLength[3],restLength[3],cotangent[3];
    typename DataTypes::Real lambda=getLambda();
    typename DataTypes::Real mu=getMu();

    helper::WriteOnlyAccessor< Data< type::vector<EdgeRestInformation> > > edgeInf = edgeInfo;

    /// describe the jth edge index of triangle no i
    const core::topology::BaseMeshTopology::EdgesInTriangle &te= this->m_topology->getEdgesInTriangle(triangleIndex);
    // store square rest length
    for(j=0; j<3; ++j)
    {
        restLength[j]=edgeInf[te[j]].restLength;
        squareRestLength[j]= restLength[j]*restLength[j];
    }
    // compute rest area based on Heron's formula
    area=0;
    for(j=0; j<3; ++j)
    {
        area+=squareRestLength[j]*(squareRestLength[(j+1)%3] +squareRestLength[(j+2)%3]-squareRestLength[j]);
    }
    area=sqrt(area)/4;

    for(j=0; j<3; ++j)
    {
        cotangent[j]=(squareRestLength[(j+1)%3] +squareRestLength[(j+2)%3]-squareRestLength[j])/(4*area);
    }
    for(j=0; j<3; ++j)
    {
        k=(j+1)%3;
        l=(j+2)%3;
        tinfo.gamma[j]=restLength[k]*restLength[l]*(2*cotangent[k]*cotangent[l]*(lambda+mu)-mu)/(8*area);
        tinfo.stiffness[j]=restLength[j]*restLength[j]*(2*cotangent[j]*cotangent[j]*(lambda+mu)+mu)/(8*area);
        edgeInf[te[j]].stiffness+=tinfo.stiffness[j];
    }

}


template< class DataTypes>
void TriangularQuadraticSpringsForceField<DataTypes>::applyTriangleDestruction(Index triangleIndex, TriangleRestInformation &tinfo)
{
    unsigned int j;

    helper::WriteOnlyAccessor< Data< type::vector<EdgeRestInformation> > > edgeInf = edgeInfo;

    /// describe the jth edge index of triangle no i
    const core::topology::BaseMeshTopology::EdgesInTriangle &te= this->m_topology->getEdgesInTriangle(triangleIndex);
    // store square rest length
    for(j=0; j<3; ++j)
    {
        edgeInf[te[j]].stiffness -= tinfo.stiffness[j];
    }
}

template <class DataTypes> TriangularQuadraticSpringsForceField<DataTypes>::TriangularQuadraticSpringsForceField()
    : _initialPoints(initData(&_initialPoints,"initialPoints", "Initial Position"))
    , updateMatrix(true)
    , f_poissonRatio(initData(&f_poissonRatio,(Real)0.3,"poissonRatio","Poisson ratio in Hooke's law"))
    , f_youngModulus(initData(&f_youngModulus,(Real)1000.,"youngModulus","Young modulus in Hooke's law"))
    , f_dampingRatio(initData(&f_dampingRatio,(Real)0.,"dampingRatio","Ratio damping/stiffness"))
    , f_useAngularSprings(initData(&f_useAngularSprings,true,"useAngularSprings","If Angular Springs should be used or not"))
    , lambda(0)
    , mu(0)
    , l_topology(initLink("topology", "link to the topology container"))
    , triangleInfo(initData(&triangleInfo, "triangleInfo", "Internal triangle data"))
    , edgeInfo(initData(&edgeInfo, "edgeInfo", "Internal edge data"))
    , m_topology(nullptr)
{

}

template <class DataTypes> TriangularQuadraticSpringsForceField<DataTypes>::~TriangularQuadraticSpringsForceField()
{

}

template <class DataTypes> void TriangularQuadraticSpringsForceField<DataTypes>::init()
{
    msg_info() << "initializing TriangularQuadraticSpringsForceField";
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

    triangleInfo.createTopologyHandler(m_topology);
    edgeInfo.createTopologyHandler(m_topology);

    if (m_topology->getNbTriangles()==0)
    {
        msg_error() << "ERROR(TriangularQuadraticSpringsForceField): object must have a Triangular Set Topology.";
        return;
    }
    updateLameCoefficients();

    /// prepare to store info in the triangle array
    helper::WriteOnlyAccessor< Data< type::vector<TriangleRestInformation> > > triangleInf = triangleInfo;
    triangleInf.resize(m_topology->getNbTriangles());
    /// prepare to store info in the edge array
    helper::WriteOnlyAccessor< Data< type::vector<EdgeRestInformation> > > edgeInf = edgeInfo;
    edgeInf.resize(m_topology->getNbEdges());

    if (_initialPoints.getValue().size() == 0)
    {
        // get restPosition
        const VecCoord& p = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
        _initialPoints.setValue(p);
    }
    unsigned int i;
    for (i=0; i<m_topology->getNbEdges(); ++i)
    {
        applyEdgeCreation(i, edgeInf[i],
            m_topology->getEdge(i),  (const sofa::type::vector< Index > )0,
            (const sofa::type::vector< SReal >)0);
    }
    for (i=0; i<m_topology->getNbTriangles(); ++i)
    {
        applyTriangleCreation(i, triangleInf[i],
            m_topology->getTriangle(i),  (const sofa::type::vector< Index > )0,
            (const sofa::type::vector< SReal >)0);
    }

    edgeInfo.setCreationCallback([this](Index edgeIndex, EdgeRestInformation& ei,
        const core::topology::BaseMeshTopology::Edge& edge,
        const sofa::type::vector< Index >& ancestors,
        const sofa::type::vector< SReal >& coefs)
    {
        applyEdgeCreation(edgeIndex, ei, edge, ancestors, coefs);
    });

    triangleInfo.setCreationCallback([this](Index triangleIndex, TriangleRestInformation& tinfo,
        const core::topology::BaseMeshTopology::Triangle& triangle,
        const sofa::type::vector< Index >& ancestors,
        const sofa::type::vector< SReal >& coefs)
    {
        applyTriangleCreation(triangleIndex, tinfo, triangle, ancestors, coefs);
    });

    triangleInfo.setDestructionCallback([this](Index triangleIndex, TriangleRestInformation& tinfo)
    {
        applyTriangleDestruction(triangleIndex, tinfo);
    });
}

template <class DataTypes>
void TriangularQuadraticSpringsForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    unsigned int j,k,l,v0,v1;
    const size_t nbEdges=m_topology->getNbEdges();
    const size_t nbTriangles=m_topology->getNbTriangles();

    Real val,L;
    TriangleRestInformation *tinfo;
    EdgeRestInformation *einfo;

    type::vector<typename TriangularQuadraticSpringsForceField<DataTypes>::TriangleRestInformation>& triangleInf = *(triangleInfo.beginEdit());

    type::vector<typename TriangularQuadraticSpringsForceField<DataTypes>::EdgeRestInformation>& edgeInf = *(edgeInfo.beginEdit());

    assert(this->mstate);

    Deriv force;
    Coord dp,dv;
    Real _dampingRatio=f_dampingRatio.getValue();


    for(unsigned int i=0; i<nbEdges; i++ )
    {
        einfo=&edgeInf[i];
        v0=m_topology->getEdge(i)[0];
        v1=m_topology->getEdge(i)[1];
        dp=x[v0]-x[v1];
        dv=v[v0]-v[v1];
        L=einfo->currentLength=dp.norm();
        einfo->dl=einfo->currentLength-einfo->restLength +_dampingRatio*dot(dv,dp)/L;

        val=einfo->stiffness*(einfo->dl)/L;
        f[v1]+=dp*val;
        f[v0]-=dp*val;
    }
    if (f_useAngularSprings.getValue()==true)
    {
        for(unsigned int i=0; i<nbTriangles; i++ )
        {
            tinfo=&triangleInf[i];
            /// describe the jth edge index of triangle no i
            const core::topology::BaseMeshTopology::EdgesInTriangle &tea= m_topology->getEdgesInTriangle(i);
            /// describe the jth vertex index of triangle no i
            const core::topology::BaseMeshTopology::Triangle &ta= m_topology->getTriangle(i);

            // store points
            for(j=0; j<3; ++j)
            {
                k=(j+1)%3;
                l=(j+2)%3;
                force=(x[ta[k]] - x[ta[l]])*
                        (edgeInf[tea[k]].dl * tinfo->gamma[l] +edgeInf[tea[l]].dl * tinfo->gamma[k])/edgeInf[tea[j]].currentLength;
                f[ta[l]]+=force;
                f[ta[k]]-=force;
            }
        }

    }
    edgeInfo.endEdit();
    triangleInfo.endEdit();
    updateMatrix=true;
    d_f.endEdit();
}


template <class DataTypes>
void TriangularQuadraticSpringsForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();
    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    unsigned int i,j,k;
    const int nbTriangles = m_topology->getNbTriangles();

    TriangleRestInformation *tinfo;

    type::vector<typename TriangularQuadraticSpringsForceField<DataTypes>::TriangleRestInformation>& triangleInf = *(triangleInfo.beginEdit());
    type::vector<typename TriangularQuadraticSpringsForceField<DataTypes>::EdgeRestInformation>& edgeInf = *(edgeInfo.beginEdit());

    assert(this->mstate);
    const VecDeriv& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();


    Deriv deltax,res;

    if (updateMatrix)
    {
        int u,v;
        Real val1,val2,vali,valj,valk;
        Coord dpj,dpk,dpi;

        updateMatrix=false;
        for(int l=0; l<nbTriangles; l++ )
        {
            tinfo=&triangleInf[l];
            /// describe the jth edge index of triangle no i
            const core::topology::BaseMeshTopology::EdgesInTriangle &tea= m_topology->getEdgesInTriangle(l);
            /// describe the jth vertex index of triangle no i
            const core::topology::BaseMeshTopology::Triangle &ta= m_topology->getTriangle(l);

            // store points
            for(k=0; k<3; ++k)
            {
                i=(k+1)%3;
                j=(k+2)%3;
                Mat3 &m=tinfo->DfDx[k];
                dpk = x[ta[i]]- x[ta[j]];

                if (f_useAngularSprings.getValue()==false)
                {
                    val1 = -tinfo->stiffness[k]*edgeInf[tea[k]].dl;
                    val1/=edgeInf[tea[k]].currentLength;

                    val2= -tinfo->stiffness[k]*edgeInf[tea[k]].restLength;
                    val2/=edgeInf[tea[k]].currentLength*edgeInf[tea[k]].currentLength*edgeInf[tea[k]].currentLength;

                    for (u=0; u<3; ++u)
                    {
                        for (v=0; v<3; ++v)
                        {
                            m[u][v]=dpk[u]*dpk[v]*val2;
                        }
                        m[u][u]+=val1;
                    }

                }
                else
                {
                    dpj = x[ta[i]]- x[ta[k]];
                    dpi = x[ta[j]]- x[ta[k]];

                    val1 = -(tinfo->stiffness[k]*edgeInf[tea[k]].dl+
                            tinfo->gamma[i]*edgeInf[tea[j]].dl+
                            tinfo->gamma[j]*edgeInf[tea[i]].dl);

                    val2= -val1 - tinfo->stiffness[k]*edgeInf[tea[k]].restLength;
                    val1/=edgeInf[tea[k]].currentLength;
                    val2/=edgeInf[tea[k]].currentLength*edgeInf[tea[k]].currentLength*edgeInf[tea[k]].currentLength;
                    valk=tinfo->gamma[k]/(edgeInf[tea[j]].currentLength*
                            edgeInf[tea[i]].currentLength);
                    vali=tinfo->gamma[i]/(edgeInf[tea[j]].currentLength*
                            edgeInf[tea[k]].currentLength);
                    valj=tinfo->gamma[j]/(edgeInf[tea[k]].currentLength*
                            edgeInf[tea[i]].currentLength);


                    for (u=0; u<3; ++u)
                    {
                        for (v=0; v<3; ++v)
                        {
                            m[u][v]=dpk[u]*dpk[v]*val2
                                    +dpj[u]*dpi[v]*valk
                                    -dpj[u]*dpk[v]*vali
                                    +dpk[u]*dpi[v]*valj;

                        }
                        m[u][u]+=val1;
                    }
                }
            }
        }

    }

    for(int l=0; l<nbTriangles; l++ )
    {
        tinfo=&triangleInf[l];
        /// describe the jth vertex index of triangle no l
        const core::topology::BaseMeshTopology::Triangle &ta= m_topology->getTriangle(l);

        // store points
        for(k=0; k<3; ++k)
        {
            i=(k+1)%3;
            j=(k+2)%3;
            deltax= dx[ta[i]] -dx[ta[j]];
            res=tinfo->DfDx[k]*deltax;
            df[ta[i]]+= res * kFactor;
            df[ta[j]]-= (tinfo->DfDx[k].transposeMultiply(deltax)) * kFactor;
        }
    }
    edgeInfo.endEdit();
    triangleInfo.endEdit();
    d_df.endEdit();
}

template <class DataTypes>
void TriangularQuadraticSpringsForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}


template<class DataTypes>
void TriangularQuadraticSpringsForceField<DataTypes>::updateLameCoefficients()
{
    lambda= f_youngModulus.getValue()*f_poissonRatio.getValue()/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
    mu = f_youngModulus.getValue()*(1-f_poissonRatio.getValue())/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
}


template<class DataTypes>
void TriangularQuadraticSpringsForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
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

} // namespace sofa::component::solidmechanics::spring
