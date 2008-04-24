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
#include <sofa/component/forcefield/TriangularBiquadraticSpringsForceField.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <sofa/helper/gl/template.h>
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
void TriangularBiquadraticSpringsForceField<DataTypes>::TRBSEdgeCreationFunction(int edgeIndex, void* param, EdgeRestInformation &ei,
        const Edge& ,  const sofa::helper::vector< unsigned int > &,
        const sofa::helper::vector< double >&)
{
    TriangularBiquadraticSpringsForceField<DataTypes> *ff= (TriangularBiquadraticSpringsForceField<DataTypes> *)param;
    if (ff)
    {
        TriangleSetTopology<DataTypes> *_mesh=ff->getTriangularTopology();
        assert(_mesh!=0);
        TriangleSetGeometryAlgorithms<DataTypes> *ga=_mesh->getTriangleSetGeometryAlgorithms();

        // store the rest length of the edge created
        ei.restSquareLength=ga->computeRestSquareEdgeLength(edgeIndex);
        ei.stiffness=0;
    }
}



template< class DataTypes>
void TriangularBiquadraticSpringsForceField<DataTypes>::TRBSTriangleCreationFunction (int triangleIndex, void* param,
        TriangleRestInformation &tinfo,
        const Triangle& ,
        const sofa::helper::vector< unsigned int > &,
        const sofa::helper::vector< double >&)
{
    TriangularBiquadraticSpringsForceField<DataTypes> *ff= (TriangularBiquadraticSpringsForceField<DataTypes> *)param;
    if (ff)
    {
        TriangleSetTopology<DataTypes> *_mesh=ff->getTriangularTopology();
        assert(_mesh!=0);
        TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
        //const sofa::helper::vector<Edge> &edgeArray=container->getEdgeArray();
        const sofa::helper::vector< TriangleEdges > &triangleEdgeArray=container->getTriangleEdgeArray() ;
        unsigned int j,k,l;

        EdgeData<typename TriangularBiquadraticSpringsForceField<DataTypes>::EdgeRestInformation> &edgeInfo=ff->getEdgeInfo();
        typename DataTypes::Real area,restSquareLength[3],cotangent[3];
        typename DataTypes::Real lambda=ff->getLambda();
        typename DataTypes::Real mu=ff->getMu();

        /// describe the jth edge index of triangle no i
        const TriangleEdges &te= triangleEdgeArray[triangleIndex];
        // store square rest length
        for(j=0; j<3; ++j)
        {
            restSquareLength[j]=edgeInfo[te[j]].restSquareLength;
        }
        // compute rest area based on Heron's formula
        area=0;
        for(j=0; j<3; ++j)
        {
            area+=restSquareLength[j]*(restSquareLength[(j+1)%3] +restSquareLength[(j+2)%3]-restSquareLength[j]);
        }
        area=sqrt(area)/4;

        for(j=0; j<3; ++j)
        {
            cotangent[j]=(restSquareLength[(j+1)%3] +restSquareLength[(j+2)%3]-restSquareLength[j])/(4*area);
            if (ff->f_printLog.getValue())
            {
                if (cotangent[j]<0)
                    std::cerr<<"negative cotangent["<<triangleIndex<<"]["<<j<<"]"<<std::endl;
            }
        }
        for(j=0; j<3; ++j)
        {
            k=(j+1)%3;
            l=(j+2)%3;
            tinfo.gamma[j]=(2*cotangent[k]*cotangent[l]*(lambda+mu)-mu)/(16*area);
            tinfo.stiffness[j]=(2*cotangent[j]*cotangent[j]*(lambda+mu)+mu)/(16*area);
            edgeInfo[te[j]].stiffness+=tinfo.stiffness[j];
        }

    }

}


template< class DataTypes>
void TriangularBiquadraticSpringsForceField<DataTypes>::TRBSTriangleDestroyFunction(int triangleIndex, void* param, typename TriangularBiquadraticSpringsForceField<DataTypes>::TriangleRestInformation &tinfo)
{
    TriangularBiquadraticSpringsForceField<DataTypes> *ff= (TriangularBiquadraticSpringsForceField<DataTypes> *)param;
    if (ff)
    {
        TriangleSetTopology<DataTypes> *_mesh=ff->getTriangularTopology();
        assert(_mesh!=0);
        TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
        const sofa::helper::vector< TriangleEdges > &triangleEdgeArray=container->getTriangleEdgeArray() ;
        unsigned int j;

        EdgeData<typename TriangularBiquadraticSpringsForceField<DataTypes>::EdgeRestInformation> &edgeInfo=ff->getEdgeInfo();

        /// describe the jth edge index of triangle no i
        const TriangleEdges &te= triangleEdgeArray[triangleIndex];
        // store square rest length
        for(j=0; j<3; ++j)
        {
            edgeInfo[te[j]].stiffness -= tinfo.stiffness[j];
        }

    }
}
template <class DataTypes> TriangularBiquadraticSpringsForceField<DataTypes>::TriangularBiquadraticSpringsForceField()
    : _mesh(NULL)
    , _initialPoints(initData(&_initialPoints,"initialPoints", "Initial Position"))
    , updateMatrix(true)
    , f_poissonRatio(initData(&f_poissonRatio,(Real)0.3,"poissonRatio","Poisson ratio in Hooke's law"))
    , f_youngModulus(initData(&f_youngModulus,(Real)1000.,"youngModulus","Young modulus in Hooke's law"))
    , f_dampingRatio(initData(&f_dampingRatio,(Real)0.,"dampingRatio","Ratio damping/stiffness"))
    , f_useAngularSprings(initData(&f_useAngularSprings,true,"useAngularSprings","If Angular Springs should be used or not"))
    , lambda(0)
    , mu(0)
{
}

template <class DataTypes> void TriangularBiquadraticSpringsForceField<DataTypes>::handleTopologyChange()
{
    sofa::core::componentmodel::topology::BaseTopology *topology = static_cast<sofa::core::componentmodel::topology::BaseTopology *>(getContext()->getMainTopology());

    std::list<const TopologyChange *>::const_iterator itBegin=topology->firstChange();
    std::list<const TopologyChange *>::const_iterator itEnd=topology->lastChange();

    edgeInfo.handleTopologyEvents(itBegin,itEnd);
    triangleInfo.handleTopologyEvents(itBegin,itEnd);
}

template <class DataTypes> TriangularBiquadraticSpringsForceField<DataTypes>::~TriangularBiquadraticSpringsForceField()
{

}

template <class DataTypes> void TriangularBiquadraticSpringsForceField<DataTypes>::init()
{
    std::cerr << "initializing TriangularBiquadraticSpringsForceField" << std::endl;
    this->Inherited::init();
    _mesh =0;
    if (getContext()->getMainTopology()!=0)
        _mesh= dynamic_cast<TriangleSetTopology<DataTypes>*>(getContext()->getMainTopology());

    if ((_mesh==0) || (_mesh->getTriangleSetTopologyContainer()->getNumberOfTriangles()==0))
    {
        std::cerr << "ERROR(TriangularBiquadraticSpringsForceField): object must have a Triangular Set Topology.\n";
        return;
    }
    updateLameCoefficients();

    TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();

    /// prepare to store info in the triangle array
    triangleInfo.resize(container->getNumberOfTriangles());
    /// prepare to store info in the edge array
    edgeInfo.resize(container->getNumberOfEdges());

    // get restPosition
    if (_initialPoints.getValue().size() == 0)
    {
        VecCoord& p = *this->mstate->getX0();
        _initialPoints.setValue(p);
    }
    unsigned int i;
    const sofa::helper::vector<Edge> &edgeArray=container->getEdgeArray();
    for (i=0; i<container->getNumberOfEdges(); ++i)
    {
        TRBSEdgeCreationFunction(i, (void*) this, edgeInfo[i],
                edgeArray[i],  (const sofa::helper::vector< unsigned int > )0,
                (const sofa::helper::vector< double >)0);
    }
    const sofa::helper::vector<Triangle> &triangleArray=container->getTriangleArray();
    for (i=0; i<container->getNumberOfTriangles(); ++i)
    {
        TRBSTriangleCreationFunction(i, (void*) this, triangleInfo[i],
                triangleArray[i],  (const sofa::helper::vector< unsigned int > )0,
                (const sofa::helper::vector< double >)0);
    }

    edgeInfo.setCreateFunction(TRBSEdgeCreationFunction);
    triangleInfo.setCreateFunction(TRBSTriangleCreationFunction);
    triangleInfo.setDestroyFunction(TRBSTriangleDestroyFunction);
    edgeInfo.setCreateParameter( (void *) this );
    edgeInfo.setDestroyParameter( (void *) this );
    triangleInfo.setCreateParameter( (void *) this );
    triangleInfo.setDestroyParameter( (void *) this );

}


template <class DataTypes>
sofa::defaulttype::Vector3::value_type TriangularBiquadraticSpringsForceField<DataTypes>::getPotentialEnergy(const VecCoord& /*x*/)
{
    std::cerr<<"TriangularBiquadraticSpringsForceField::getPotentialEnergy-not-implemented !!!"<<endl;
    return 0;
}
template <class DataTypes>
void TriangularBiquadraticSpringsForceField<DataTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    unsigned int i,j,k,l,v0,v1;
    TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
    unsigned int nbEdges=container->getNumberOfEdges();
    unsigned int nbTriangles=container->getNumberOfTriangles();
    const sofa::helper::vector<Edge> &edgeArray=container->getEdgeArray();
    const sofa::helper::vector< TriangleEdges > &triangleEdgeArray=container->getTriangleEdgeArray() ;
    const sofa::helper::vector< Triangle> &triangleArray=container->getTriangleArray() ;


    Real val,L;
    TriangleRestInformation *tinfo;
    EdgeRestInformation *einfo;

    assert(this->mstate);

    Deriv force;
    Coord dp,dv;
    Real _dampingRatio=f_dampingRatio.getValue();


    for(i=0; i<nbEdges; i++ )
    {
        einfo=&edgeInfo[i];
        v0=edgeArray[i][0];
        v1=edgeArray[i][1];
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
        for(i=0; i<nbTriangles; i++ )
        {
            tinfo=&triangleInfo[i];
            /// describe the jth edge index of triangle no i
            const TriangleEdges &tea= triangleEdgeArray[i];
            /// describe the jth vertex index of triangle no i
            const Triangle &ta= triangleArray[i];

            // store points
            for(j=0; j<3; ++j)
            {
                k=(j+1)%3;
                l=(j+2)%3;
                force=(x[ta[k]] - x[ta[l]])*
                        (edgeInfo[tea[k]].deltaL2 * tinfo->gamma[l] +edgeInfo[tea[l]].deltaL2 * tinfo->gamma[k]);
                f[ta[l]]+=force;
                f[ta[k]]-=force;
            }
        }
        //	std::cerr << "tinfo->gamma[0] "<<tinfo->gamma[0]<<std::endl;

    }

    updateMatrix=true;
}


template <class DataTypes>
void TriangularBiquadraticSpringsForceField<DataTypes>::addDForce(VecDeriv& df, const VecDeriv& dx)
{
    unsigned int i,j,k,l;
    TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
    //unsigned int nbEdges=container->getNumberOfEdges();
    unsigned int nbTriangles=container->getNumberOfTriangles();
    //const sofa::helper::vector<Edge> &edgeArray=container->getEdgeArray();
    const sofa::helper::vector< TriangleEdges > &triangleEdgeArray=container->getTriangleEdgeArray() ;
    const sofa::helper::vector< Triangle> &triangleArray=container->getTriangleArray() ;

    TriangleRestInformation *tinfo;

//	std::cerr << "start addDForce" << std::endl;


    assert(this->mstate);
    VecDeriv& x = *this->mstate->getX();


    Deriv deltax,res;

    if (updateMatrix)
    {
        int u,v;
        Real val1,val2,vali,valj,valk;
        Coord dpj,dpk,dpi;

        //	std::cerr <<"updating matrix"<<std::endl;
        updateMatrix=false;
        for(l=0; l<nbTriangles; l++ )
        {
            tinfo=&triangleInfo[l];
            /// describe the jth edge index of triangle no i
            const TriangleEdges &tea= triangleEdgeArray[l];
            /// describe the jth vertex index of triangle no i
            const Triangle &ta= triangleArray[l];

            // store points
            for(k=0; k<3; ++k)
            {
                i=(k+1)%3;
                j=(k+2)%3;
                Mat3 &m=tinfo->DfDx[k];
                dpk = x[ta[i]]- x[ta[j]];

                if (f_useAngularSprings.getValue()==false)
                {
                    val1 = -tinfo->stiffness[k]*edgeInfo[tea[k]].deltaL2;

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
                    dpj = x[ta[i]]- x[ta[k]];
                    dpi = x[ta[j]]- x[ta[k]];

                    val1 = -(tinfo->stiffness[k]*edgeInfo[tea[k]].deltaL2+
                            tinfo->gamma[i]*edgeInfo[tea[j]].deltaL2+
                            tinfo->gamma[j]*edgeInfo[tea[i]].deltaL2);

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
        }

    }

    for(l=0; l<nbTriangles; l++ )
    {
        tinfo=&triangleInfo[l];
        /// describe the jth vertex index of triangle no l
        const Triangle &ta= triangleArray[l];

        // store points
        for(k=0; k<3; ++k)
        {
            i=(k+1)%3;
            j=(k+2)%3;
            deltax= dx[ta[i]] -dx[ta[j]];
            res=tinfo->DfDx[k]*deltax;
            df[ta[i]]+=res;
            df[ta[j]]-= tinfo->DfDx[k].transposeMultiply(deltax);
        }
    }
}


template<class DataTypes>
void TriangularBiquadraticSpringsForceField<DataTypes>::updateLameCoefficients()
{
    lambda= f_youngModulus.getValue()*f_poissonRatio.getValue()/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
    mu = f_youngModulus.getValue()*(1-f_poissonRatio.getValue())/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
//	std::cerr << "initialized Lame coef : lambda=" <<lambda<< " mu="<<mu<<std::endl;
}


template<class DataTypes>
void TriangularBiquadraticSpringsForceField<DataTypes>::draw()
{
    unsigned int i;
    if (!getContext()->getShowForceFields()) return;
    if (!this->mstate) return;

    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    VecCoord& x = *this->mstate->getX();
    TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
    unsigned int nbTriangles=container->getNumberOfTriangles();
    const sofa::helper::vector< Triangle> &triangleArray=container->getTriangleArray() ;


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
