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

#include <sofa/component/solidmechanics/spring/TriangularBiquadraticSpringsForceField.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/core/topology/TopologyData.inl>

namespace sofa::component::solidmechanics::spring
{

typedef core::topology::BaseMeshTopology::Triangle			Triangle;
typedef core::topology::BaseMeshTopology::EdgesInTriangle		EdgesInTriangle;

template< class DataTypes >
void TriangularBiquadraticSpringsForceField<DataTypes>::applyTriangleCreation(Index triangleIndex,
        TriangleRestInformation &tinfo,
        const Triangle &,
        const sofa::type::vector<Index> &,
        const sofa::type::vector<SReal> &)
{
    using namespace sofa::defaulttype;

    unsigned int j,k,l;

    typename DataTypes::Real area,restSquareLength[3],cotangent[3];
    typename DataTypes::Real lambda=getLambda();
    typename DataTypes::Real mu=getMu();
    const typename DataTypes::VecCoord restPosition=this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    helper::WriteOnlyAccessor< Data< type::vector<EdgeRestInformation> > > edgeInf = edgeInfo;

    ///describe the indices of the 3 triangle vertices
    const Triangle &t= this->m_topology->getTriangle(triangleIndex);
    /// describe the jth edge index of triangle no i
    const EdgesInTriangle &te= this->m_topology->getEdgesInTriangle(triangleIndex);
    // store square rest length
    for(j=0; j<3; ++j)
    {
        restSquareLength[j]=edgeInf[te[j]].restSquareLength;
    }
    // compute rest area based on Heron's formula
    area=0;
    for(j=0; j<3; ++j)
    {
        area+=restSquareLength[j]*(restSquareLength[(j+1)%3] +restSquareLength[(j+2)%3]-restSquareLength[j]);
    }
    area=sqrt(area)/4;
    tinfo.restArea=area;
    tinfo.currentNormal= cross((restPosition)[t[1]]-(restPosition)[t[0]],(restPosition)[t[2]]-(restPosition)[t[0]])/(2*area);
    tinfo.lastValidNormal=tinfo.currentNormal;

    for(j=0; j<3; ++j)
    {
        cotangent[j]=(restSquareLength[(j+1)%3] +restSquareLength[(j+2)%3]-restSquareLength[j])/(4*area);

        msg_info_when(cotangent[j]<0, this) <<"negative cotangent["
                                            << triangleIndex<<"]["
                                            <<j<<"]" ;
    }
    for(j=0; j<3; ++j)
    {
        k=(j+1)%3;
        l=(j+2)%3;
        tinfo.gamma[j]=(2*cotangent[k]*cotangent[l]*(lambda+mu)-mu)/(16*area);
        tinfo.stiffness[j]=(2*cotangent[j]*cotangent[j]*(lambda+mu)+mu)/(16*area);
        edgeInf[te[j]].stiffness+=tinfo.stiffness[j];
    }

}

template< class DataTypes >
void TriangularBiquadraticSpringsForceField<DataTypes>::applyTriangleDestruction(Index triangleIndex,
        TriangleRestInformation  &tinfo)
{
    unsigned int j;

    helper::WriteOnlyAccessor< Data< type::vector<EdgeRestInformation> > > edgeInf = edgeInfo;

    /// describe the jth edge index of triangle no i
    const EdgesInTriangle &te= this->m_topology->getEdgesInTriangle(triangleIndex);
    // store square rest length
    for(j=0; j<3; ++j)
    {
        edgeInf[te[j]].stiffness -= tinfo.stiffness[j];
    }
}

template< class DataTypes >
void TriangularBiquadraticSpringsForceField<DataTypes>::applyEdgeCreation(Index edgeIndex,
        EdgeRestInformation &ei,
        const core::topology::Edge &,
        const sofa::type::vector<Index> &,
        const sofa::type::vector<SReal> &)

{
    // store the rest length of the edge created
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    const auto& e = this->m_topology->getEdge(edgeIndex);
    const auto& n0 = DataTypes::getCPos(x[e[0]]);
    const auto& n1 = DataTypes::getCPos(x[e[1]]);

    ei.restSquareLength = sofa::geometry::Edge::squaredLength(n0, n1);
    ei.stiffness=0;
}

template <class DataTypes> TriangularBiquadraticSpringsForceField<DataTypes>::TriangularBiquadraticSpringsForceField()
    : triangleInfo(initData(&triangleInfo, "triangleInfo", "Internal triangle data"))
    , edgeInfo(initData(&edgeInfo, "edgeInfo", "Internal edge data"))
    , _initialPoints(initData(&_initialPoints,"initialPoints", "Initial Position"))
    , updateMatrix(true)
    , f_poissonRatio(initData(&f_poissonRatio,(Real)0.3,"poissonRatio","Poisson ratio in Hooke's law"))
    , f_youngModulus(initData(&f_youngModulus,(Real)1000.,"youngModulus","Young modulus in Hooke's law"))
    , f_dampingRatio(initData(&f_dampingRatio,(Real)0.,"dampingRatio","Ratio damping/stiffness"))
    , f_useAngularSprings(initData(&f_useAngularSprings,true,"useAngularSprings","If Angular Springs should be used or not"))
    , f_compressible(initData(&f_compressible,true,"compressible","If additional energy penalizing compressibility should be used"))
    , f_stiffnessMatrixRegularizationWeight(initData(&f_stiffnessMatrixRegularizationWeight,(Real)0.4,"matrixRegularization","Regularization of the Stiffnes Matrix (between 0 and 1)"))
    , lambda(0)
    , mu(0)
    , l_topology(initLink("topology", "link to the topology container"))
    , m_topology(nullptr)
{

}

template <class DataTypes> TriangularBiquadraticSpringsForceField<DataTypes>::~TriangularBiquadraticSpringsForceField()
{

}

template <class DataTypes> void TriangularBiquadraticSpringsForceField<DataTypes>::init()
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

    if (m_topology->getNbTriangles()==0)
    {
        msg_warning() << "No triangles found in linked Topology.";
    }
    updateLameCoefficients();

    /// prepare to store info in the triangle array
    helper::WriteOnlyAccessor< Data< type::vector<TriangleRestInformation> > > triangleInf = triangleInfo;
    triangleInf.resize(m_topology->getNbTriangles());

    /// prepare to store info in the edge array
    helper::WriteOnlyAccessor< Data< type::vector<EdgeRestInformation> > > edgeInf = edgeInfo;
    edgeInf.resize(m_topology->getNbEdges());

    // get restPosition
    if (_initialPoints.getValue().size() == 0)
    {
        const VecCoord& p = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
        _initialPoints.setValue(p);
    }
    unsigned int i;
    for (i=0; i<m_topology->getNbEdges(); ++i)
    {
        applyEdgeCreation(i,edgeInf[i], m_topology->getEdge(i),  
            (const sofa::type::vector< Index > )0,
            (const sofa::type::vector< SReal >)0 );
    }
    for (i=0; i<m_topology->getNbTriangles(); ++i)
    {
        applyTriangleCreation(i, triangleInf[i],
            m_topology->getTriangle(i),  (const sofa::type::vector< Index > )0,
            (const sofa::type::vector< SReal >)0);
    }

    // Edge info
    edgeInfo.createTopologyHandler(m_topology);
    edgeInfo.setCreationCallback([this](Index edgeIndex, EdgeRestInformation& ei,
        const core::topology::BaseMeshTopology::Edge& edge,
        const sofa::type::vector< Index >& ancestors,
        const sofa::type::vector< SReal >& coefs)
    {
        applyEdgeCreation(edgeIndex, ei, edge, ancestors, coefs);
    });

    // Triangle info
    triangleInfo.createTopologyHandler(m_topology);    
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
void TriangularBiquadraticSpringsForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    using namespace sofa::defaulttype;

    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    unsigned int j,k,l,v0,v1;
    const size_t nbEdges=m_topology->getNbEdges();
    const size_t nbTriangles=m_topology->getNbTriangles();
    const bool compressible=f_compressible.getValue();
    Real areaStiffness=(getLambda()+getMu())*3;

    Real val,L;
    TriangleRestInformation *tinfo;
    EdgeRestInformation *einfo;

    type::vector<typename TriangularBiquadraticSpringsForceField<DataTypes>::TriangleRestInformation>& triangleInf = *(triangleInfo.beginEdit());

    type::vector<typename TriangularBiquadraticSpringsForceField<DataTypes>::EdgeRestInformation>& edgeInf = *(edgeInfo.beginEdit());

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
        L=einfo->currentSquareLength=dp.norm2();
        einfo->deltaL2=einfo->currentSquareLength-einfo->restSquareLength +_dampingRatio*dot(dv,dp)/L;

        val=einfo->stiffness*einfo->deltaL2;
        force=dp*val;
        f[v1]+=force;
        f[v0]-=force;
    }
    if (f_useAngularSprings.getValue()==true)
    {
        Real JJ;
        std::vector<int> flippedTriangles ;
        for(unsigned int i=0; i<nbTriangles; i++ )
        {
            tinfo=&triangleInf[i];
            /// describe the jth edge index of triangle no i
            const EdgesInTriangle &tea= m_topology->getEdgesInTriangle(i);
            /// describe the jth vertex index of triangle no i
            const Triangle &ta= m_topology->getTriangle(i);

            // store points
            for(j=0; j<3; ++j)
            {
                k=(j+1)%3;
                l=(j+2)%3;
                tinfo->dp[j]=x[ta[l]] - x[ta[k]];
                force= -tinfo->dp[j]*
                        (edgeInf[tea[k]].deltaL2 * tinfo->gamma[l] +edgeInf[tea[l]].deltaL2 * tinfo->gamma[k]);
                f[ta[l]]+=force;
                f[ta[k]]-=force;
            }
            if (compressible)
            {
                tinfo->currentNormal= -cross(tinfo->dp[2],tinfo->dp[1]);
                tinfo->area=tinfo->currentNormal.norm()/2;
                tinfo->J=(tinfo->area/tinfo->restArea);
                if (tinfo->J<1)   // only apply compressible force if the triangle is compressed
                {
                    JJ=tinfo->J-1;
                    if (tinfo->J<1e-2)
                    {
                        /// if the current area is too small compared to its original value, then the normal is considered to
                        // be non valid. It is replaced by its last reasonable value
                        // this is to cope with very flat triangles
                        tinfo->currentNormal=tinfo->lastValidNormal;
                    }
                    else
                    {
                        tinfo->currentNormal/= (2*tinfo->area);
                        if (dot(tinfo->currentNormal,tinfo->lastValidNormal) >-0.5) // if the normal has suddenly flipped (= angle changed is large)
                            tinfo->lastValidNormal=tinfo->currentNormal;
                        else
                        {
                            flippedTriangles.push_back(i);
                            tinfo->currentNormal*= -1.0;
                        }
                    }
                    val= areaStiffness*JJ*JJ*JJ;
                    // computes area vector
                    for(j=0; j<3; ++j)
                    {
                        tinfo->areaVector[j]=cross(tinfo->currentNormal,tinfo->dp[j])/2;
                        f[ta[j]]-= tinfo->areaVector[j]*val;
                    }
                }
            }
        }
        /// Prints the flipped triangles in a single message to avoid flooding the user.
        /// Only the 50 first indices are shown.
        const Size flippedTrianglesNb = flippedTriangles.size();
        if(flippedTrianglesNb != 0){
            std::stringstream tmp ;
            tmp << "[" ;
            for(Size i=0 ; i<std::min(Size(50), flippedTrianglesNb) ; i++)
            {
                tmp << ", " << flippedTriangles[i] ;
            }
            if(flippedTrianglesNb >=50){
                tmp << ", ..." << flippedTrianglesNb -50 << " more]" ;
            }
            else{
                tmp << "]" ;
            }
            msg_warning() << "The following triangles have flipped: " << tmp.str() ;
        }
    }
    edgeInfo.endEdit();
    triangleInfo.endEdit();
    updateMatrix=true;
    d_f.endEdit();
}


template <class DataTypes>
void TriangularBiquadraticSpringsForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();
    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    unsigned int i,j,k;
    const int nbTriangles=m_topology->getNbTriangles();
    bool compressible=f_compressible.getValue();
    Real areaStiffness=(getLambda()+getMu())*3;
    TriangleRestInformation *tinfo;

    Deriv deltax,res;

    type::vector<typename TriangularBiquadraticSpringsForceField<DataTypes>::TriangleRestInformation>& triangleInf = *(triangleInfo.beginEdit());

    type::vector<typename TriangularBiquadraticSpringsForceField<DataTypes>::EdgeRestInformation>& edgeInf = *(edgeInfo.beginEdit());


    if (updateMatrix)
    {
        int u,v;
        Real val1,val2,vali,valj,valk,JJ,dpij,h,lengthSquare[3],totalLength;
        Coord dpj,dpk,dpi,dp;
        Mat3 m1;

        updateMatrix=false;
        for(int l=0; l<nbTriangles; l++ )
        {
            tinfo=&triangleInf[l];
            /// describe the jth edge index of triangle no i
            const EdgesInTriangle &tea= m_topology->getEdgesInTriangle(l);
            /// describe the jth vertex index of triangle no i

            // store points
            for(k=0; k<3; ++k)
            {
                Mat3 &m=tinfo->DfDx[k];
                dpk = -tinfo->dp[k];

                if (f_useAngularSprings.getValue()==false)
                {
                    val1 = -tinfo->stiffness[k]*edgeInf[tea[k]].deltaL2;
                    val2= -2*tinfo->stiffness[k];

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
                    i=(k+1)%3;
                    j=(k+2)%3;
                    dpj = tinfo->dp[j];//x[ta[i]]- x[ta[k]];
                    dpi = -tinfo->dp[i];//x[ta[j]]- x[ta[k]];

                    val1 = -(tinfo->stiffness[k]*edgeInf[tea[k]].deltaL2+
                            tinfo->gamma[i]*edgeInf[tea[j]].deltaL2+
                            tinfo->gamma[j]*edgeInf[tea[i]].deltaL2);

                    val2= -2*tinfo->stiffness[k];
                    valk=2*tinfo->gamma[k];
                    vali=2*tinfo->gamma[i];
                    valj=2*tinfo->gamma[j];

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
            if ((compressible) && (tinfo->J<1.0))
            {
                JJ=tinfo->J-1;

                h=-JJ;
                val2= 3*areaStiffness*JJ*JJ/tinfo->restArea;
                val1= areaStiffness*JJ*JJ*JJ/2;
                dp= -tinfo->currentNormal*val1; // vector for antisymmetric matrix
                // compute m1 as the tensor product of n X n
                val1/=2*tinfo->area;
                for (u=0; u<3; ++u)
                {
                    for (v=0; v<3; ++v)
                    {
                        m1[u][v]=tinfo->currentNormal[u]*tinfo->currentNormal[v]*val1;
                    }
                }
                lengthSquare[0]=edgeInf[tea[0]].currentSquareLength;
                lengthSquare[1]=edgeInf[tea[1]].currentSquareLength;
                lengthSquare[2]=edgeInf[tea[2]].currentSquareLength;
                totalLength=lengthSquare[0]+lengthSquare[1]+lengthSquare[2];
                for(k=0; k<3; ++k)
                {
                    Mat3 &m=tinfo->DfDx[k];
                    val1= totalLength-2*lengthSquare[k];
                    // add antisymmetric matrix
                    m[0][1]+=dp[2];
                    m[0][2]-=dp[1];
                    m[1][0]-=dp[2];
                    m[1][2]+=dp[0];
                    m[2][0]+=dp[1];
                    m[2][1]-=dp[0];
                    dpi=tinfo->areaVector[(k+1)%3];
                    dpj=tinfo->areaVector[(k+2)%3];
                    /// the trace of the tensor product av[k]^T av[l]
                    dpij=dot(dpi,dpj);
                    /// adds a weighted average between the tensor product of av[k]^T av[l] and the trace* Identity
                    /// if h = 0 matrix is singular if h=1 matrix is proportional to identity
                    for (u=0; u<3; ++u)
                    {
                        for (v=0; v<3; ++v)
                        {
                            m[u][v]+=(1-h)*dpi[u]*dpj[v]*val2;
                            m[u][v]+=m1[u][v]*val1;
                            if (u==v)
                            {
                                m[u][v]+=h*val2*dpij;
                            }
                        }
                    }
                }
            }
        }
    }

    for(int l=0; l<nbTriangles; l++ )
    {
        tinfo=&triangleInf[l];
        /// describe the jth vertex index of triangle no l
        const Triangle &ta= m_topology->getTriangle(l);

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
void TriangularBiquadraticSpringsForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}


template<class DataTypes>
void TriangularBiquadraticSpringsForceField<DataTypes>::updateLameCoefficients()
{
    lambda= f_youngModulus.getValue()*f_poissonRatio.getValue()/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
    mu = f_youngModulus.getValue()*(1-f_poissonRatio.getValue())/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
}


template<class DataTypes>
void TriangularBiquadraticSpringsForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
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
