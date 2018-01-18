/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGULARBIQUADRATICSPRINGSFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_TRIANGULARBIQUADRATICSPRINGSFORCEFIELD_INL

#include <SofaGeneralDeformable/TriangularBiquadraticSpringsForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <sofa/helper/gl/template.h>
#include <SofaBaseTopology/TopologyData.inl>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

typedef core::topology::BaseMeshTopology::Triangle			Triangle;
typedef core::topology::BaseMeshTopology::EdgesInTriangle		EdgesInTriangle;

template< class DataTypes >
void TriangularBiquadraticSpringsForceField<DataTypes>::TRBSTriangleHandler::applyCreateFunction(unsigned int triangleIndex,
        TriangleRestInformation &tinfo,
        const Triangle &,
        const sofa::helper::vector<unsigned int> &,
        const sofa::helper::vector<double> &)
{
    using namespace sofa::defaulttype;
    using namespace	sofa::component::topology;

    if (ff)
    {

        unsigned int j,k,l;

        EdgeData<helper::vector<typename TriangularBiquadraticSpringsForceField<DataTypes>::EdgeRestInformation> > &edgeInfo=ff->getEdgeInfo();
        typename DataTypes::Real area,restSquareLength[3],cotangent[3];
        typename DataTypes::Real lambda=ff->getLambda();
        typename DataTypes::Real mu=ff->getMu();
        const typename DataTypes::VecCoord restPosition=ff->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
        helper::vector<typename TriangularBiquadraticSpringsForceField<DataTypes>::EdgeRestInformation>& edgeInf = *(edgeInfo.beginEdit());

        ///describe the indices of the 3 triangle vertices
        const Triangle &t= ff->_topology->getTriangle(triangleIndex);
        /// describe the jth edge index of triangle no i
        const EdgesInTriangle &te= ff->_topology->getEdgesInTriangle(triangleIndex);
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
        tinfo.currentNormal= cross<Real>((restPosition)[t[1]]-(restPosition)[t[0]],(restPosition)[t[2]]-(restPosition)[t[0]])/(2*area);
        tinfo.lastValidNormal=tinfo.currentNormal;

        for(j=0; j<3; ++j)
        {
            cotangent[j]=(restSquareLength[(j+1)%3] +restSquareLength[(j+2)%3]-restSquareLength[j])/(4*area);

            msg_info_when(cotangent[j]<0, ff) <<"negative cotangent["
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
        edgeInfo.endEdit();
    }

}

template< class DataTypes >
void TriangularBiquadraticSpringsForceField<DataTypes>::TRBSTriangleHandler::applyDestroyFunction(unsigned int triangleIndex,
        TriangleRestInformation  &tinfo)
{
    using namespace	sofa::component::topology;

    if (ff)
    {

        unsigned int j;

        EdgeData<helper::vector<typename TriangularBiquadraticSpringsForceField<DataTypes>::EdgeRestInformation> > &edgeInfo=ff->getEdgeInfo();

        helper::vector<typename TriangularBiquadraticSpringsForceField<DataTypes>::EdgeRestInformation>& edgeInf = *(edgeInfo.beginEdit());

        /// describe the jth edge index of triangle no i
        const EdgesInTriangle &te= ff->_topology->getEdgesInTriangle(triangleIndex);
        // store square rest length
        for(j=0; j<3; ++j)
        {
            edgeInf[te[j]].stiffness -= tinfo.stiffness[j];
        }

    }

}

template< class DataTypes >
void TriangularBiquadraticSpringsForceField<DataTypes>::TRBSEdgeHandler::applyCreateFunction(unsigned int edgeIndex,
        EdgeRestInformation &ei,
        const core::topology::Edge &,
        const sofa::helper::vector<unsigned int> &,
        const sofa::helper::vector<double> &)

{
    if (ff)
    {

        sofa::component::topology::TriangleSetGeometryAlgorithms<DataTypes>* triangleGeo;
        ff->getContext()->get(triangleGeo);

        // store the rest length of the edge created
        ei.restSquareLength=triangleGeo->computeRestSquareEdgeLength(edgeIndex);
        ei.stiffness=0;
    }
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
    , edgeHandler(NULL)
    , triangleHandler(NULL)
{
    edgeHandler = new TRBSEdgeHandler(this,&edgeInfo);
    triangleHandler = new TRBSTriangleHandler(this,&triangleInfo);
}

template <class DataTypes> TriangularBiquadraticSpringsForceField<DataTypes>::~TriangularBiquadraticSpringsForceField()
{
    if(edgeHandler) delete edgeHandler;
    if(triangleHandler) delete triangleHandler;

}

template <class DataTypes> void TriangularBiquadraticSpringsForceField<DataTypes>::init()
{
    this->Inherited::init();
    _topology = this->getContext()->getMeshTopology();

    if (_topology->getNbTriangles()==0)
    {
        msg_error() << "Object must have a Triangular Set Topology." ;
        return;
    }
    updateLameCoefficients();

    /// prepare to store info in the triangle array
    helper::vector<typename TriangularBiquadraticSpringsForceField<DataTypes>::TriangleRestInformation>& triangleInf = *(triangleInfo.beginEdit());

    triangleInf.resize(_topology->getNbTriangles());
    /// prepare to store info in the edge array
    helper::vector<typename TriangularBiquadraticSpringsForceField<DataTypes>::EdgeRestInformation>& edgeInf = *(edgeInfo.beginEdit());

    edgeInf.resize(_topology->getNbEdges());

    // get restPosition
    if (_initialPoints.getValue().size() == 0)
    {
        const VecCoord& p = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
        _initialPoints.setValue(p);
    }
    int i;
    for (i=0; i<_topology->getNbEdges(); ++i)
    {
        edgeHandler->applyCreateFunction(i,edgeInf[i], _topology->getEdge(i),  (const sofa::helper::vector< unsigned int > )0,
                (const sofa::helper::vector< double >)0 );
    }
    for (i=0; i<_topology->getNbTriangles(); ++i)
    {
        triangleHandler->applyCreateFunction(i, triangleInf[i],
                _topology->getTriangle(i),  (const sofa::helper::vector< unsigned int > )0,
                (const sofa::helper::vector< double >)0);
    }

    // Edge info
    edgeInfo.createTopologicalEngine(_topology,edgeHandler);
    edgeInfo.registerTopologicalData();
    edgeInfo.endEdit();

    // Triangle info
    triangleInfo.createTopologicalEngine(_topology,triangleHandler);
    triangleInfo.registerTopologicalData();
    triangleInfo.endEdit();
}

template <class DataTypes>
void TriangularBiquadraticSpringsForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    using namespace sofa::defaulttype;

    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    unsigned int j,k,l,v0,v1;
    int nbEdges=_topology->getNbEdges();
    int nbTriangles=_topology->getNbTriangles();
    bool compressible=f_compressible.getValue();
    Real areaStiffness=(getLambda()+getMu())*3;

    Real val,L;
    TriangleRestInformation *tinfo;
    EdgeRestInformation *einfo;

    helper::vector<typename TriangularBiquadraticSpringsForceField<DataTypes>::TriangleRestInformation>& triangleInf = *(triangleInfo.beginEdit());

    helper::vector<typename TriangularBiquadraticSpringsForceField<DataTypes>::EdgeRestInformation>& edgeInf = *(edgeInfo.beginEdit());

    assert(this->mstate);

    Deriv force;
    Coord dp,dv;
    Real _dampingRatio=f_dampingRatio.getValue();


    for(int i=0; i<nbEdges; i++ )
    {
        einfo=&edgeInf[i];
        v0=_topology->getEdge(i)[0];
        v1=_topology->getEdge(i)[1];
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
        for(int i=0; i<nbTriangles; i++ )
        {
            tinfo=&triangleInf[i];
            /// describe the jth edge index of triangle no i
            const EdgesInTriangle &tea= _topology->getEdgesInTriangle(i);
            /// describe the jth vertex index of triangle no i
            const Triangle &ta= _topology->getTriangle(i);

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
                tinfo->currentNormal= -cross<Real>(tinfo->dp[2],tinfo->dp[1]);
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
        /// Only the 50 first indices are showned.
        if(flippedTriangles.size()!=0){
            std::stringstream tmp ;
            tmp << "[" ;
            for(size_t i=0;i<std::min((size_t)50, flippedTriangles.size());i++)
            {
                tmp << ", " << flippedTriangles[i] ;
            }
            if(flippedTriangles.size()>=50){
                tmp << ", ..." << flippedTriangles.size()-50 << " more]" ;
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
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    unsigned int i,j,k;
    int nbTriangles=_topology->getNbTriangles();
    bool compressible=f_compressible.getValue();
    Real areaStiffness=(getLambda()+getMu())*3;
    TriangleRestInformation *tinfo;

    Deriv deltax,res;

    helper::vector<typename TriangularBiquadraticSpringsForceField<DataTypes>::TriangleRestInformation>& triangleInf = *(triangleInfo.beginEdit());

    helper::vector<typename TriangularBiquadraticSpringsForceField<DataTypes>::EdgeRestInformation>& edgeInf = *(edgeInfo.beginEdit());


    if (updateMatrix)
    {
        int u,v;
        Real val1,val2,vali,valj,valk,JJ,dpij,h,lengthSquare[3],totalLength;
        Coord dpj,dpk,dpi,dp;
        Mat3 m1,m2;

        updateMatrix=false;
        for(int l=0; l<nbTriangles; l++ )
        {
            tinfo=&triangleInf[l];
            /// describe the jth edge index of triangle no i
            const EdgesInTriangle &tea= _topology->getEdgesInTriangle(l);
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
        const Triangle &ta= _topology->getTriangle(l);

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


template<class DataTypes>
void TriangularBiquadraticSpringsForceField<DataTypes>::updateLameCoefficients()
{
    lambda= f_youngModulus.getValue()*f_poissonRatio.getValue()/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
    mu = f_youngModulus.getValue()*(1-f_poissonRatio.getValue())/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
}


template<class DataTypes>
void TriangularBiquadraticSpringsForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    int nbTriangles=_topology->getNbTriangles();

    glDisable(GL_LIGHTING);

    glBegin(GL_TRIANGLES);
    for(int i=0; i<nbTriangles; ++i)
    {
        int a = _topology->getTriangle(i)[0];
        int b = _topology->getTriangle(i)[1];
        int c = _topology->getTriangle(i)[2];

        glColor4f(0,1,0,1);
        helper::gl::glVertexT(x[a]);
        glColor4f(0,0.5,0.5,1);
        helper::gl::glVertexT(x[b]);
        glColor4f(0,0,1,1);
        helper::gl::glVertexT(x[c]);
    }
    glEnd();


    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
#endif /* SOFA_NO_OPENGL */
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_TRIANGULARBIQUADRATICSPRINGSFORCEFIELD_INL
