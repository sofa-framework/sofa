/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGULARQUADRATICSPRINGSFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_TRIANGULARQUADRATICSPRINGSFORCEFIELD_INL

#include <SofaGeneralDeformable/TriangularQuadraticSpringsForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <sofa/helper/gl/template.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TopologyData.inl>

namespace sofa
{

namespace component
{

namespace forcefield
{

template< class DataTypes>
void TriangularQuadraticSpringsForceField<DataTypes>::TRQSEdgeHandler::applyCreateFunction(unsigned int edgeIndex, EdgeRestInformation &ei, const core::topology::Edge &, const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
{
    if (ff)
    {

        sofa::component::topology::TriangleSetGeometryAlgorithms<DataTypes>* triangleGeo=NULL;
        ff->getContext()->get(triangleGeo);

        // store the rest length of the edge created
        ei.restLength=triangleGeo->computeRestEdgeLength(edgeIndex);
        ei.stiffness=0;
    }
}



template< class DataTypes>
void TriangularQuadraticSpringsForceField<DataTypes>::TRQSTriangleHandler::applyCreateFunction(unsigned int triangleIndex, TriangleRestInformation &tinfo,
        const core::topology::Triangle &, const sofa::helper::vector<unsigned int> &,
        const sofa::helper::vector<double> &)
{
    using namespace	sofa::component::topology;

    if (ff)
    {
        unsigned int j=0,k=0,l=0;

        EdgeData<sofa::helper::vector<typename TriangularQuadraticSpringsForceField<DataTypes>::EdgeRestInformation> > &edgeInfo=ff->getEdgeInfo();
        typename DataTypes::Real area,squareRestLength[3],restLength[3],cotangent[3];
        typename DataTypes::Real lambda=ff->getLambda();
        typename DataTypes::Real mu=ff->getMu();

        helper::vector<typename TriangularQuadraticSpringsForceField<DataTypes>::EdgeRestInformation>& edgeInf = *(edgeInfo.beginEdit());

        /// describe the jth edge index of triangle no i
        const core::topology::BaseMeshTopology::EdgesInTriangle &te= ff->_topology->getEdgesInTriangle(triangleIndex);
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
            /*	if (cotangent[j]<0)
            serr<<"negative cotangent["<<i<<"]["<<j<<"]"<<sendl;
            else
            serr<<"cotangent="<<cotangent[j]<<sendl;*/

        }
        for(j=0; j<3; ++j)
        {
            k=(j+1)%3;
            l=(j+2)%3;
            tinfo.gamma[j]=restLength[k]*restLength[l]*(2*cotangent[k]*cotangent[l]*(lambda+mu)-mu)/(8*area);
            tinfo.stiffness[j]=restLength[j]*restLength[j]*(2*cotangent[j]*cotangent[j]*(lambda+mu)+mu)/(8*area);
            edgeInf[te[j]].stiffness+=tinfo.stiffness[j];
        }
        edgeInfo.endEdit();
    }

}


template< class DataTypes>
void TriangularQuadraticSpringsForceField<DataTypes>::TRQSTriangleHandler::applyDestroyFunction(unsigned int triangleIndex, TriangleRestInformation &tinfo)
{
    using namespace	sofa::component::topology;

    if (ff)
    {
        unsigned int j;

        EdgeData<sofa::helper::vector<typename TriangularQuadraticSpringsForceField<DataTypes>::EdgeRestInformation> > &edgeInfo=ff->getEdgeInfo();

        helper::vector<typename TriangularQuadraticSpringsForceField<DataTypes>::EdgeRestInformation>& edgeInf = *(edgeInfo.beginEdit());

        /// describe the jth edge index of triangle no i
        const core::topology::BaseMeshTopology::EdgesInTriangle &te= ff->_topology->getEdgesInTriangle(triangleIndex);
        // store square rest length
        for(j=0; j<3; ++j)
        {
            edgeInf[te[j]].stiffness -= tinfo.stiffness[j];
        }

        edgeInfo.endEdit();
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
    , triangleInfo(initData(&triangleInfo, "triangleInfo", "Internal triangle data"))
    , edgeInfo(initData(&edgeInfo, "edgeInfo", "Internal edge data"))
{
    triangleHandler = new TRQSTriangleHandler(this, &triangleInfo);
    edgeHandler = new TRQSEdgeHandler(this, &edgeInfo);
}

template <class DataTypes> TriangularQuadraticSpringsForceField<DataTypes>::~TriangularQuadraticSpringsForceField()
{
    if(triangleHandler) delete triangleHandler;
    if(edgeHandler) delete edgeHandler;
}

template <class DataTypes> void TriangularQuadraticSpringsForceField<DataTypes>::init()
{
    sout<< "initializing TriangularQuadraticSpringsForceField" << sendl;
    this->Inherited::init();

    _topology = this->getContext()->getMeshTopology();

    triangleInfo.createTopologicalEngine(_topology,triangleHandler);
    triangleInfo.registerTopologicalData();

    edgeInfo.createTopologicalEngine(_topology,edgeHandler);
    edgeInfo.registerTopologicalData();

    if (_topology->getNbTriangles()==0)
    {
        serr << "ERROR(TriangularQuadraticSpringsForceField): object must have a Triangular Set Topology."<<sendl;
        return;
    }
    updateLameCoefficients();

    helper::vector<typename TriangularQuadraticSpringsForceField<DataTypes>::TriangleRestInformation>& triangleInf = *(triangleInfo.beginEdit());

    /// prepare to store info in the triangle array
    triangleInf.resize(_topology->getNbTriangles());
    /// prepare to store info in the edge array
    helper::vector<typename TriangularQuadraticSpringsForceField<DataTypes>::EdgeRestInformation>& edgeInf = *(edgeInfo.beginEdit());

    edgeInf.resize(_topology->getNbEdges());

    if (_initialPoints.getValue().size() == 0)
    {
        // get restPosition
        const VecCoord& p = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
        _initialPoints.setValue(p);
    }
    int i;
    for (i=0; i<_topology->getNbEdges(); ++i)
    {
        edgeHandler->applyCreateFunction(i, edgeInf[i],
                _topology->getEdge(i),  (const sofa::helper::vector< unsigned int > )0,
                (const sofa::helper::vector< double >)0);
    }
    for (i=0; i<_topology->getNbTriangles(); ++i)
    {
        triangleHandler->applyCreateFunction(i, triangleInf[i],
                _topology->getTriangle(i),  (const sofa::helper::vector< unsigned int > )0,
                (const sofa::helper::vector< double >)0);
    }
    /*
            // Edge info
            edgeInfo.createTopologicalEngine(_topology);
            edgeInfo.setCreateFunction(TRQSEdgeCreationFunction);
            edgeInfo.setCreateParameter( (void *) this );
            edgeInfo.setDestroyParameter( (void *) this );
            edgeInfo.registerTopologicalData();
            edgeInfo.endEdit();

            // Triangle info
            triangleInfo.createTopologicalEngine(_topology);
            triangleInfo.setCreateFunction(TRQSTriangleCreationFunction);
            triangleInfo.setDestroyFunction(TRQSTriangleDestroyFunction);
            triangleInfo.setCreateParameter( (void *) this );
            triangleInfo.setDestroyParameter( (void *) this );
            triangleInfo.registerTopologicalData();
            triangleInfo.endEdit();
    */
}

template <class DataTypes>
void TriangularQuadraticSpringsForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    using namespace	sofa::component::topology;

    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    unsigned int j,k,l,v0,v1;
    int nbEdges=_topology->getNbEdges();
    int nbTriangles=_topology->getNbTriangles();

    Real val,L;
    TriangleRestInformation *tinfo;
    EdgeRestInformation *einfo;

    helper::vector<typename TriangularQuadraticSpringsForceField<DataTypes>::TriangleRestInformation>& triangleInf = *(triangleInfo.beginEdit());

    helper::vector<typename TriangularQuadraticSpringsForceField<DataTypes>::EdgeRestInformation>& edgeInf = *(edgeInfo.beginEdit());

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
        L=einfo->currentLength=dp.norm();
        einfo->dl=einfo->currentLength-einfo->restLength +_dampingRatio*dot(dv,dp)/L;
        /*if (i==0) {
        serr << "dl= " <<  einfo->dl<<sendl;
        serr << "damping= " <<  (_dampingRatio*dot(dv,dp)*einfo->restLength/(L*L))<<sendl;
        }*/
        val=einfo->stiffness*(einfo->dl)/L;
        f[v1]+=dp*val;
        f[v0]-=dp*val;
        //	serr << "einfo->stiffness= "<<einfo->stiffness<<sendl;
    }
    if (f_useAngularSprings.getValue()==true)
    {
        for(int i=0; i<nbTriangles; i++ )
        {
            tinfo=&triangleInf[i];
            /// describe the jth edge index of triangle no i
            const core::topology::BaseMeshTopology::EdgesInTriangle &tea= _topology->getEdgesInTriangle(i);
            /// describe the jth vertex index of triangle no i
            const core::topology::BaseMeshTopology::Triangle &ta= _topology->getTriangle(i);

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
        //	serr << "tinfo->gamma[0] "<<tinfo->gamma[0]<<sendl;

    }
    edgeInfo.endEdit();
    triangleInfo.endEdit();
    updateMatrix=true;
    d_f.endEdit();
    //serr << "end addForce" << sendl;
}


template <class DataTypes>
void TriangularQuadraticSpringsForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    using namespace	sofa::component::topology;

    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    unsigned int i,j,k;
    int nbTriangles = _topology->getNbTriangles();

    TriangleRestInformation *tinfo;

    helper::vector<typename TriangularQuadraticSpringsForceField<DataTypes>::TriangleRestInformation>& triangleInf = *(triangleInfo.beginEdit());

    //	serr << "start addDForce" << sendl;
    helper::vector<typename TriangularQuadraticSpringsForceField<DataTypes>::EdgeRestInformation>& edgeInf = *(edgeInfo.beginEdit());

    assert(this->mstate);
    const VecDeriv& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();


    Deriv deltax,res;

    if (updateMatrix)
    {
        int u,v;
        Real val1,val2,vali,valj,valk;
        Coord dpj,dpk,dpi;

        //	serr <<"updating matrix"<<sendl;
        updateMatrix=false;
        for(int l=0; l<nbTriangles; l++ )
        {
            tinfo=&triangleInf[l];
            /// describe the jth edge index of triangle no i
            const core::topology::BaseMeshTopology::EdgesInTriangle &tea= _topology->getEdgesInTriangle(l);
            /// describe the jth vertex index of triangle no i
            const core::topology::BaseMeshTopology::Triangle &ta= _topology->getTriangle(l);

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
        const core::topology::BaseMeshTopology::Triangle &ta= _topology->getTriangle(l);

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
void TriangularQuadraticSpringsForceField<DataTypes>::updateLameCoefficients()
{
    lambda= f_youngModulus.getValue()*f_poissonRatio.getValue()/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
    mu = f_youngModulus.getValue()*(1-f_poissonRatio.getValue())/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
    //	serr << "initialized Lame coef : lambda=" <<lambda<< " mu="<<mu<<sendl;
}


template<class DataTypes>
void TriangularQuadraticSpringsForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
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

#endif //SOFA_COMPONENT_FORCEFIELD_TRIANGULARQUADRATICSPRINGSFORCEFIELD_INL
