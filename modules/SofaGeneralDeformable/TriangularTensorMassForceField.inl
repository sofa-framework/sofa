/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGULARTENSORMASSFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_TRIANGULARTENSORMASSFORCEFIELD_INL

#include <SofaGeneralDeformable/TriangularTensorMassForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <sofa/helper/gl/template.h>
#include <SofaBaseTopology/TopologyData.inl>

namespace sofa
{

namespace component
{

namespace forcefield
{

typedef core::topology::BaseMeshTopology::EdgesInTriangle EdgesInTriangle;

template< class DataTypes >
void TriangularTensorMassForceField<DataTypes>::TriangularTMEdgeHandler::applyCreateFunction(unsigned int /*edgeIndex*/,
        EdgeRestInformation & ei,
        const core::topology::Edge &/*e*/,
        const sofa::helper::vector<unsigned int> &,
        const sofa::helper::vector<double> &)
{
    if(ff)
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
}

template< class DataTypes >
void TriangularTensorMassForceField<DataTypes>::TriangularTMEdgeHandler::applyTriangleCreation(const sofa::helper::vector<unsigned int> &triangleAdded,
        const sofa::helper::vector<core::topology::Triangle> &,
        const sofa::helper::vector<sofa::helper::vector<unsigned int> > &,
        const sofa::helper::vector<sofa::helper::vector<double> > &)
{
    using namespace core::topology;
    if(ff)
    {

        unsigned int i,j,k,l,u,v;

        typename DataTypes::Real val1,area,restSquareLength[3],cotangent[3];
        typename DataTypes::Real lambda=ff->getLambda();
        typename DataTypes::Real mu=ff->getMu();
        typename DataTypes::Real lambdastar, mustar;
        typename DataTypes::Coord point[3],dpk,dpl;
        helper::vector<EdgeRestInformation> &edgeData = *ff->edgeInfo.beginEdit();

        const typename DataTypes::VecCoord& restPosition=ff->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

        for (i=0; i<triangleAdded.size(); ++i)
        {

            /// describe the jth edge index of triangle no i
            const EdgesInTriangle &te= ff->_topology->getEdgesInTriangle(triangleAdded[i]);
            /// describe the jth vertex index of triangle no i
            const Triangle &t= ff->_topology->getTriangle(triangleAdded[i]);
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

                msg_info_when(cotangent[j]<0, ff) <<"negative cotangent["
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

                if (ff->_topology->getEdge(te[j])[0]==t[l])
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
        ff->edgeInfo.endEdit();
    }
}

template< class DataTypes>
void TriangularTensorMassForceField<DataTypes>::TriangularTMEdgeHandler::applyTriangleDestruction(const sofa::helper::vector<unsigned int> &triangleRemoved)
{
    using namespace core::topology;
    if (ff)
    {

        unsigned int i,j,k,l,u,v;

        typename DataTypes::Real val1,area,restSquareLength[3],cotangent[3];
        typename DataTypes::Real lambda=ff->getLambda();
        typename DataTypes::Real mu=ff->getMu();
        typename DataTypes::Real lambdastar, mustar;
        typename DataTypes::Coord point[3],dpk,dpl;

        helper::vector<EdgeRestInformation> &edgeData = *ff->edgeInfo.beginEdit();
        const typename DataTypes::VecCoord& restPosition=ff->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

        for (i=0; i<triangleRemoved.size(); ++i)
        {

            /// describe the jth edge index of triangle no i
            const EdgesInTriangle &te= ff->_topology->getEdgesInTriangle(triangleRemoved[i]);
            /// describe the jth vertex index of triangle no i
            const Triangle &t= ff->_topology->getTriangle(triangleRemoved[i]);
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

                msg_info_when(cotangent[j]<0, ff) << "negative cotangent["
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

                if (ff->_topology->getEdge(te[j])[0]==t[l])
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
        ff->edgeInfo.endEdit();
    }
}

template<class DataTypes>
void TriangularTensorMassForceField<DataTypes>::TriangularTMEdgeHandler::ApplyTopologyChange(const core::topology::TrianglesAdded* e)
{
    const sofa::helper::vector<unsigned int> &triangleAdded = e->getIndexArray();
    const sofa::helper::vector<core::topology::Triangle> &elems = e->getElementArray();
    const sofa::helper::vector<sofa::helper::vector<unsigned int> > & ancestors = e->ancestorsList;
    const sofa::helper::vector<sofa::helper::vector<double> > & coefs = e->coefs;

    applyTriangleCreation(triangleAdded, elems, ancestors, coefs);
}

template<class DataTypes>
void TriangularTensorMassForceField<DataTypes>::TriangularTMEdgeHandler::ApplyTopologyChange(const core::topology::TrianglesRemoved* e)
{
    const sofa::helper::vector<unsigned int> &triangleRemoved = e->getArray();

    applyTriangleDestruction(triangleRemoved);
}

template <class DataTypes> TriangularTensorMassForceField<DataTypes>::TriangularTensorMassForceField()
    : edgeInfo(initData(&edgeInfo, "edgeInfo", "Internal edge data"))
    , _initialPoints(0)
    , updateMatrix(true)
    , f_poissonRatio(initData(&f_poissonRatio,(Real)0.3,"poissonRatio","Poisson ratio in Hooke's law"))
    , f_youngModulus(initData(&f_youngModulus,(Real)1000.,"youngModulus","Young modulus in Hooke's law"))
    , lambda(0)
    , mu(0)
    , edgeHandler(NULL)
{
    edgeHandler = new TriangularTMEdgeHandler(this,&edgeInfo);
}

template <class DataTypes> TriangularTensorMassForceField<DataTypes>::~TriangularTensorMassForceField()
{
    if(edgeHandler) delete edgeHandler;
}

template <class DataTypes> void TriangularTensorMassForceField<DataTypes>::init()
{
    sout << "initializing TriangularTensorMassForceField" << sendl;
    this->Inherited::init();

    _topology = this->getContext()->getMeshTopology();

    if (_topology->getNbTriangles()==0)
    {
        serr << "ERROR(TriangularTensorMassForceField): object must have a Triangular Set Topology."<<sendl;
        return;
    }
    updateLameCoefficients();



    helper::vector<EdgeRestInformation>& edgeInf = *(edgeInfo.beginEdit());

    /// prepare to store info in the edge array
    edgeInf.resize(_topology->getNbEdges());

    if (_initialPoints.size() == 0)
    {
        // get restPosition
        const VecCoord& p = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
        _initialPoints=p;
    }

    int i;
    // set edge tensor to 0
    for (i=0; i<_topology->getNbEdges(); ++i)
    {
        edgeHandler->applyCreateFunction(i,edgeInf[i],_topology->getEdge(i),
                (const sofa::helper::vector<unsigned int>)0,
                (const sofa::helper::vector<double>)0
                                        );
    }
    // create edge tensor by calling the triangle creation function
    sofa::helper::vector<unsigned int> triangleAdded;
    for (i=0; i<_topology->getNbTriangles(); ++i)
        triangleAdded.push_back(i);

    edgeHandler->applyTriangleCreation(triangleAdded,
            (const sofa::helper::vector<core::topology::Triangle>)0,
            (const sofa::helper::vector<sofa::helper::vector<unsigned int> >)0,
            (const sofa::helper::vector<sofa::helper::vector<double> >)0
                                      );

    edgeInfo.createTopologicalEngine(_topology,edgeHandler);
    edgeInfo.linkToTriangleDataArray();
    edgeInfo.registerTopologicalData();
    edgeInfo.endEdit();
}

template <class DataTypes>
void TriangularTensorMassForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& /* d_v */)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();

    unsigned int i,v0,v1;
    unsigned int nbEdges=_topology->getNbEdges();
    EdgeRestInformation *einfo;

    helper::vector<EdgeRestInformation>& edgeInf = *(edgeInfo.beginEdit());

    Deriv force;
    Coord dp0,dp1,dp;

    for(i=0; i<nbEdges; i++ )
    {
        einfo=&edgeInf[i];
        v0=_topology->getEdge(i)[0];
        v1=_topology->getEdge(i)[1];
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
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    unsigned int v0,v1;
    int nbEdges=_topology->getNbEdges();
    EdgeRestInformation *einfo;

    helper::vector<EdgeRestInformation>& edgeInf = *(edgeInfo.beginEdit());

    Deriv force;
    Coord dp0,dp1,dp;

    for(int i=0; i<nbEdges; i++ )
    {
        einfo=&edgeInf[i];
        v0=_topology->getEdge(i)[0];
        v1=_topology->getEdge(i)[1];
        dp0=dx[v0];
        dp1=dx[v1];
        dp = dp1-dp0;

        df[v1]+= (einfo->DfDx*dp) * kFactor;
        df[v0]-= (einfo->DfDx.transposeMultiply(dp)) * kFactor;
    }

    edgeInfo.endEdit();
    d_df.endEdit();
}


template<class DataTypes>
void TriangularTensorMassForceField<DataTypes>::updateLameCoefficients()
{
    lambda= f_youngModulus.getValue()*f_poissonRatio.getValue()/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
    mu = f_youngModulus.getValue()*(1-f_poissonRatio.getValue())/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
    //	serr << "initialized Lame coef : lambda=" <<lambda<< " mu="<<mu<<sendl;
}


template<class DataTypes>
void TriangularTensorMassForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    int i;
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    int nbTriangles=_topology->getNbTriangles();

    glDisable(GL_LIGHTING);

    glBegin(GL_TRIANGLES);
    for(i=0; i<nbTriangles; ++i)
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

#endif //#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGULARTENSORMASSFORCEFIELD_INL
