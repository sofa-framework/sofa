/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/component/forcefield/TriangularTensorMassForceField.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <sofa/helper/gl/template.h>
#include <GL/gl.h>
#include <sofa/component/topology/TriangleData.inl>
#include <sofa/component/topology/EdgeData.inl>


namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using namespace	sofa::component::topology;
using namespace core::componentmodel::topology;

using std::cerr;
using std::cout;
using std::endl;

template< class DataTypes>
void TriangularTensorMassForceField<DataTypes>::TriangularTMEdgeCreationFunction(int edgeIndex, void* param, EdgeRestInformation &ei,
        const Edge& ,  const std::vector< unsigned int > &,
        const std::vector< double >&)
{
    TriangularTensorMassForceField<DataTypes> *ff= (TriangularTensorMassForceField<DataTypes> *)param;
    if (ff)
    {
        TriangleSetTopology<DataTypes> *_mesh=ff->getTriangularTopology();
        assert(_mesh!=0);
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

template< class DataTypes>
void TriangularTensorMassForceField<DataTypes>::TriangularTMTriangleCreationFunction (const std::vector<unsigned int> &triangleAdded,
        void* param, vector<EdgeRestInformation> &edgeData)
{
    TriangularTensorMassForceField<DataTypes> *ff= (TriangularTensorMassForceField<DataTypes> *)param;
    if (ff)
    {
        TriangleSetTopology<DataTypes> *_mesh=ff->getTriangularTopology();
        assert(_mesh!=0);
        TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
        const std::vector< Edge > &edgeArray=container->getEdgeArray() ;
        const std::vector< Triangle > &triangleArray=container->getTriangleArray() ;
        const std::vector< TriangleEdges > &triangleEdgeArray=container->getTriangleEdgeArray() ;

        unsigned int i,j,k,l,u,v;

        typename DataTypes::Real val1,area,restSquareLength[3],cotangent[3];
        typename DataTypes::Real lambda=ff->getLambda();
        typename DataTypes::Real mu=ff->getMu();
        typename DataTypes::Real lambdastar, mustar;
        typename DataTypes::Coord point[3],dpk,dpl;
        const typename DataTypes::VecCoord *restPosition=_mesh->getDOF()->getX0();

        for (i=0; i<triangleAdded.size(); ++i)
        {

            /// describe the jth edge index of triangle no i
            const TriangleEdges &te= triangleEdgeArray[triangleAdded[i]];
            /// describe the jth vertex index of triangle no i
            const Triangle &t= triangleArray[triangleAdded[i]];
            // store points
            for(j=0; j<3; ++j)
                point[j]=(*restPosition)[t[j]];
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
                if (ff->f_printLog.getValue())
                {
                    if (cotangent[j]<0)
                        std::cerr<<"negative cotangent["<<triangleAdded[i]<<"]["<<j<<"]"<<std::endl;
                }
            }
            for(j=0; j<3; ++j)
            {
                k=(j+1)%3;
                l=(j+2)%3;
                Mat3 &m=edgeData[te[j]].DfDx;
                dpl= point[j]-point[k];
                dpk= point[j]-point[l];
                val1= -cotangent[j]*(lambda+mu)/2;

                if (edgeArray[te[j]].first==t[l])
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
}

template< class DataTypes>
void TriangularTensorMassForceField<DataTypes>::TriangularTMTriangleDestructionFunction (const std::vector<unsigned int> &triangleRemoved,
        void* param, vector<EdgeRestInformation> &edgeData)
{
    TriangularTensorMassForceField<DataTypes> *ff= (TriangularTensorMassForceField<DataTypes> *)param;
    if (ff)
    {
        TriangleSetTopology<DataTypes> *_mesh=ff->getTriangularTopology();
        assert(_mesh!=0);
        TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
        const std::vector< Edge > &edgeArray=container->getEdgeArray() ;
        const std::vector< Triangle > &triangleArray=container->getTriangleArray() ;
        const std::vector< TriangleEdges > &triangleEdgeArray=container->getTriangleEdgeArray() ;

        unsigned int i,j,k,l,u,v;

        typename DataTypes::Real val1,area,restSquareLength[3],cotangent[3];
        typename DataTypes::Real lambda=ff->getLambda();
        typename DataTypes::Real mu=ff->getMu();
        typename DataTypes::Real lambdastar, mustar;
        typename DataTypes::Coord point[3],dpk,dpl;
        const typename DataTypes::VecCoord *restPosition=_mesh->getDOF()->getX0();

        for (i=0; i<triangleRemoved.size(); ++i)
        {

            /// describe the jth edge index of triangle no i
            const TriangleEdges &te= triangleEdgeArray[triangleRemoved[i]];
            /// describe the jth vertex index of triangle no i
            const Triangle &t= triangleArray[triangleRemoved[i]];
            // store points
            for(j=0; j<3; ++j)
                point[j]=(*restPosition)[t[j]];
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
                if (ff->f_printLog.getValue())
                {
                    if (cotangent[j]<0)
                        std::cerr<<"negative cotangent["<<triangleRemoved[i]<<"]["<<j<<"]"<<std::endl;
                }
            }
            for(j=0; j<3; ++j)
            {
                k=(j+1)%3;
                l=(j+2)%3;
                Mat3 &m=edgeData[te[j]].DfDx;
                dpl= point[j]-point[k];
                dpk= point[j]-point[l];
                val1= -cotangent[j]*(lambda+mu)/2;

                if (edgeArray[te[j]].first==t[l])
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
}


template <class DataTypes> TriangularTensorMassForceField<DataTypes>::TriangularTensorMassForceField()
    : _mesh(NULL)
    , updateMatrix(true)
    , f_poissonRatio(dataField(&f_poissonRatio,(Real)0.3,"poissonRatio","Poisson ratio in Hooke's law"))
    , f_youngModulus(dataField(&f_youngModulus,(Real)1000.,"youngModulus","Young modulus in Hooke's law"))
    , f_dampingRatio(dataField(&f_dampingRatio,(Real)0.,"dampingRatio","Ratio damping/stiffness"))
    , lambda(0)
    , mu(0)
{
}

template <class DataTypes> void TriangularTensorMassForceField<DataTypes>::handleTopologyChange()
{
    sofa::core::componentmodel::topology::BaseTopology *topology = static_cast<sofa::core::componentmodel::topology::BaseTopology *>(getContext()->getMainTopology());

    std::list<const TopologyChange *>::const_iterator itBegin=topology->firstChange();
    std::list<const TopologyChange *>::const_iterator itEnd=topology->lastChange();

    edgeInfo.handleTopologyEvents(itBegin,itEnd);
}

template <class DataTypes> TriangularTensorMassForceField<DataTypes>::~TriangularTensorMassForceField()
{

}

template <class DataTypes> void TriangularTensorMassForceField<DataTypes>::init()
{
    std::cerr << "initializing TriangularTensorMassForceField" << std::endl;
    this->Inherited::init();
    _mesh =0;
    if (getContext()->getMainTopology()!=0)
        _mesh= dynamic_cast<TriangleSetTopology<DataTypes>*>(getContext()->getMainTopology());

    if ((_mesh==0) || (_mesh->getTriangleSetTopologyContainer()->getNumberOfTriangles()==0))
    {
        std::cerr << "ERROR(TriangularTensorMassForceField): object must have a Triangular Set Topology.\n";
        return;
    }
    updateLameCoefficients();

    TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();

    /// prepare to store info in the edge array
    edgeInfo.resize(container->getNumberOfEdges());

    // get restPosition
    VecCoord& p = *this->mstate->getX();
    _initialPoints = p;

    unsigned int i;
    // set edge tensor to 0
    const std::vector<Edge> &edgeArray=container->getEdgeArray();
    for (i=0; i<container->getNumberOfEdges(); ++i)
    {
        TriangularTMEdgeCreationFunction(i, (void*) this, edgeInfo[i],
                edgeArray[i],  (const std::vector< unsigned int > )0,
                (const std::vector< double >)0);
    }
    // create edge tensor by calling the triangle creation function
    std::vector<unsigned int> triangleAdded;
    for (i=0; i<container->getNumberOfTriangles(); ++i)
        triangleAdded.push_back(i);
    TriangularTMTriangleCreationFunction(triangleAdded,(void*) this,
            edgeInfo);


    edgeInfo.setCreateFunction(TriangularTMEdgeCreationFunction);
    edgeInfo.setCreateTriangleFunction(TriangularTMTriangleCreationFunction);
    edgeInfo.setDestroyTriangleFunction(TriangularTMTriangleDestructionFunction);
    edgeInfo.setCreateParameter( (void *) this );
    edgeInfo.setDestroyParameter( (void *) this );

}


template <class DataTypes>
double TriangularTensorMassForceField<DataTypes>::getPotentialEnergy(const VecCoord& /*x*/)
{
    std::cerr<<"TriangularTensorMassForceField::getPotentialEnergy-not-implemented !!!"<<endl;
    return 0;
}
template <class DataTypes>
void TriangularTensorMassForceField<DataTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    unsigned int i,v0,v1;
    TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
    unsigned int nbEdges=container->getNumberOfEdges();
    const std::vector<Edge> &edgeArray=container->getEdgeArray();

    EdgeRestInformation *einfo;

    Deriv force;
    Coord dp0,dp1,dp;


    for(i=0; i<nbEdges; i++ )
    {
        einfo=&edgeInfo[i];
        v0=edgeArray[i].first;
        v1=edgeArray[i].second;
        dp0=x[v0]-_initialPoints[v0];
        dp1=x[v1]-_initialPoints[v1];
        dp = dp1-dp0;

        f[v1]+=einfo->DfDx*dp;
        f[v0]-=einfo->DfDx.transposeMultiply(dp);
    }

}


template <class DataTypes>
void TriangularTensorMassForceField<DataTypes>::addDForce(VecDeriv& df, const VecDeriv& dx)
{
    unsigned int i,v0,v1;
    TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
    unsigned int nbEdges=container->getNumberOfEdges();
    const std::vector<Edge> &edgeArray=container->getEdgeArray();

    EdgeRestInformation *einfo;


    Deriv force;
    Coord dp0,dp1,dp;

    for(i=0; i<nbEdges; i++ )
    {
        einfo=&edgeInfo[i];
        v0=edgeArray[i].first;
        v1=edgeArray[i].second;
        dp0=dx[v0];
        dp1=dx[v1];
        dp = dp1-dp0;

        df[v1]+=einfo->DfDx*dp;
        df[v0]-=einfo->DfDx.transposeMultiply(dp);
    }

}


template<class DataTypes>
void TriangularTensorMassForceField<DataTypes>::updateLameCoefficients()
{
    lambda= f_youngModulus.getValue()*f_poissonRatio.getValue()/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
    mu = f_youngModulus.getValue()*(1-f_poissonRatio.getValue())/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
//	std::cerr << "initialized Lame coef : lambda=" <<lambda<< " mu="<<mu<<std::endl;
}


template<class DataTypes>
void TriangularTensorMassForceField<DataTypes>::draw()
{
    unsigned int i;
    if (!getContext()->getShowForceFields()) return;
    if (!this->mstate) return;

    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    VecCoord& x = *this->mstate->getX();
    TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
    unsigned int nbTriangles=container->getNumberOfTriangles();
    const std::vector< Triangle> &triangleArray=container->getTriangleArray() ;


    glDisable(GL_LIGHTING);

    glBegin(GL_TRIANGLES);
    for(i=0; i<nbTriangles; ++i)
    {
        int a = triangleArray[i][0];
        int b = triangleArray[i][1];
        int c = triangleArray[i][2];

        glColor4f(0,1,0,1);
        helper::gl::glVertexT(x[a]);
        glColor4f(0,0.5,0.5,1);
        helper::gl::glVertexT(x[b]);
        glColor4f(0,0,1,1);
        helper::gl::glVertexT(x[c]);
    }
    glEnd();


    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

} // namespace forcefield

} // namespace Components

} // namespace Sofa
