#include <sofa/component/forcefield/TriangularQuadraticSpringsForceField.h>
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
void TriangularQuadraticSpringsForceField<DataTypes>::TRQSEdgeCreationFunction(int edgeIndex, void* param, EdgeRestInformation &ei,
        const Edge& ,  const std::vector< unsigned int > &,
        const std::vector< double >&)
{
    TriangularQuadraticSpringsForceField<DataTypes> *ff= (TriangularQuadraticSpringsForceField<DataTypes> *)param;
    if (ff)
    {
        TriangleSetTopology<DataTypes> *_mesh=ff->getTriangularTopology();
        assert(_mesh!=0);
        TriangleSetGeometryAlgorithms<DataTypes> *ga=_mesh->getTriangleSetGeometryAlgorithms();

        // store the rest length of the edge created
        ei.restLength=ga->computeRestEdgeLength(edgeIndex);
        ei.stiffness=0;
    }
}



template< class DataTypes>
void TriangularQuadraticSpringsForceField<DataTypes>::TRQSTriangleCreationFunction (int triangleIndex, void* param,
        TriangleRestInformation &tinfo,
        const Triangle& ,
        const std::vector< unsigned int > &,
        const std::vector< double >&)
{
    TriangularQuadraticSpringsForceField<DataTypes> *ff= (TriangularQuadraticSpringsForceField<DataTypes> *)param;
    if (ff)
    {
        TriangleSetTopology<DataTypes> *_mesh=ff->getTriangularTopology();
        assert(_mesh!=0);
        TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
        //const std::vector<Edge> &edgeArray=container->getEdgeArray();
        const std::vector< TriangleEdges > &triangleEdgeArray=container->getTriangleEdgeArray() ;
        unsigned int j,k,l;

        EdgeData<typename TriangularQuadraticSpringsForceField<DataTypes>::EdgeRestInformation> &edgeInfo=ff->getEdgeInfo();
        typename DataTypes::Real area,squareRestLength[3],restLength[3],cotangent[3];
        typename DataTypes::Real lambda=ff->getLambda();
        typename DataTypes::Real mu=ff->getMu();

        /// describe the jth edge index of triangle no i
        const TriangleEdges &te= triangleEdgeArray[triangleIndex];
        // store square rest length
        for(j=0; j<3; ++j)
        {
            restLength[j]=edgeInfo[te[j]].restLength;
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
            std::cerr<<"negative cotangent["<<i<<"]["<<j<<"]"<<std::endl;
            else
            std::cerr<<"cotangent="<<cotangent[j]<<std::endl;*/

        }
        for(j=0; j<3; ++j)
        {
            k=(j+1)%3;
            l=(j+2)%3;
            tinfo.gamma[j]=restLength[k]*restLength[l]*(2*cotangent[k]*cotangent[l]*(lambda+mu)-mu)/(8*area);
            tinfo.stiffness[j]=restLength[j]*restLength[j]*(2*cotangent[j]*cotangent[j]*(lambda+mu)+mu)/(8*area);
            edgeInfo[te[j]].stiffness+=tinfo.stiffness[j];
        }

    }

}


template< class DataTypes>
void TriangularQuadraticSpringsForceField<DataTypes>::TRQSTriangleDestroyFunction(int triangleIndex, void* param, typename TriangularQuadraticSpringsForceField<DataTypes>::TriangleRestInformation &tinfo)
{
    TriangularQuadraticSpringsForceField<DataTypes> *ff= (TriangularQuadraticSpringsForceField<DataTypes> *)param;
    if (ff)
    {
        TriangleSetTopology<DataTypes> *_mesh=ff->getTriangularTopology();
        assert(_mesh!=0);
        TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
        const std::vector< TriangleEdges > &triangleEdgeArray=container->getTriangleEdgeArray() ;
        unsigned int j;

        EdgeData<typename TriangularQuadraticSpringsForceField<DataTypes>::EdgeRestInformation> &edgeInfo=ff->getEdgeInfo();

        /// describe the jth edge index of triangle no i
        const TriangleEdges &te= triangleEdgeArray[triangleIndex];
        // store square rest length
        for(j=0; j<3; ++j)
        {
            edgeInfo[te[j]].stiffness -= tinfo.stiffness[j];
        }

    }
}
template <class DataTypes> TriangularQuadraticSpringsForceField<DataTypes>::TriangularQuadraticSpringsForceField()
    : _mesh(NULL)
    , updateMatrix(true)
    , f_poissonRatio(dataField(&f_poissonRatio,(Real)0.3,"poissonRatio","Poisson ratio in Hooke's law"))
    , f_youngModulus(dataField(&f_youngModulus,(Real)1000.,"youngModulus","Young modulus in Hooke's law"))
    , f_dampingRatio(dataField(&f_dampingRatio,(Real)0.,"dampingRatio","Ratio damping/stiffness"))
    , f_useAngularSprings(dataField(&f_useAngularSprings,true,"useAngularSprings","If Angular Springs should be used or not"))
    , lambda(0)
    , mu(0)
{
}

template <class DataTypes> void TriangularQuadraticSpringsForceField<DataTypes>::handleTopologyChange()
{
    sofa::core::componentmodel::topology::BaseTopology *topology = static_cast<sofa::core::componentmodel::topology::BaseTopology *>(getContext()->getMainTopology());

    std::list<const TopologyChange *>::const_iterator itBegin=topology->firstChange();
    std::list<const TopologyChange *>::const_iterator itEnd=topology->lastChange();

    edgeInfo.handleTopologyEvents(itBegin,itEnd);
    triangleInfo.handleTopologyEvents(itBegin,itEnd);
}

template <class DataTypes> TriangularQuadraticSpringsForceField<DataTypes>::~TriangularQuadraticSpringsForceField()
{

}

template <class DataTypes> void TriangularQuadraticSpringsForceField<DataTypes>::init()
{
    std::cerr << "initializing TriangularQuadraticSpringsForceField" << std::endl;
    this->Inherited::init();
    _mesh =0;
    if (getContext()->getMainTopology()!=0)
        _mesh= dynamic_cast<TriangleSetTopology<DataTypes>*>(getContext()->getMainTopology());

    if ((_mesh==0) || (_mesh->getTriangleSetTopologyContainer()->getNumberOfTriangles()==0))
    {
        std::cerr << "ERROR(TriangularQuadraticSpringsForceField): object must have a Triangular Set Topology.\n";
        return;
    }
    updateLameCoefficients();

    TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();

    /// prepare to store info in the triangle array
    triangleInfo.resize(container->getNumberOfTriangles());
    /// prepare to store info in the edge array
    edgeInfo.resize(container->getNumberOfEdges());

    // get restPosition
    VecCoord& p = *this->mstate->getX();
    _initialPoints = p;

    unsigned int i;
    const std::vector<Edge> &edgeArray=container->getEdgeArray();
    for (i=0; i<container->getNumberOfEdges(); ++i)
    {
        TRQSEdgeCreationFunction(i, (void*) this, edgeInfo[i],
                edgeArray[i],  (const std::vector< unsigned int > )0,
                (const std::vector< double >)0);
    }
    const std::vector<Triangle> &triangleArray=container->getTriangleArray();
    for (i=0; i<container->getNumberOfTriangles(); ++i)
    {
        TRQSTriangleCreationFunction(i, (void*) this, triangleInfo[i],
                triangleArray[i],  (const std::vector< unsigned int > )0,
                (const std::vector< double >)0);
    }

    edgeInfo.setCreateFunction(TRQSEdgeCreationFunction);
    triangleInfo.setCreateFunction(TRQSTriangleCreationFunction);
    triangleInfo.setDestroyFunction(TRQSTriangleDestroyFunction);
    edgeInfo.setCreateParameter( (void *) this );
    edgeInfo.setDestroyParameter( (void *) this );
    triangleInfo.setCreateParameter( (void *) this );
    triangleInfo.setDestroyParameter( (void *) this );

}


template <class DataTypes>
double TriangularQuadraticSpringsForceField<DataTypes>::getPotentialEnergy(const VecCoord& /*x*/)
{
    std::cerr<<"TriangularQuadraticSpringsForceField::getPotentialEnergy-not-implemented !!!"<<endl;
    return 0;
}
template <class DataTypes>
void TriangularQuadraticSpringsForceField<DataTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    unsigned int i,j,k,l,v0,v1;
    TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
    unsigned int nbEdges=container->getNumberOfEdges();
    unsigned int nbTriangles=container->getNumberOfTriangles();
    const std::vector<Edge> &edgeArray=container->getEdgeArray();
    const std::vector< TriangleEdges > &triangleEdgeArray=container->getTriangleEdgeArray() ;
    const std::vector< Triangle> &triangleArray=container->getTriangleArray() ;


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
        v0=edgeArray[i].first;
        v1=edgeArray[i].second;
        dp=x[v0]-x[v1];
        dv=v[v0]-v[v1];
        L=einfo->currentLength=dp.norm();
        einfo->dl=einfo->currentLength-einfo->restLength +_dampingRatio*dot(dv,dp)/L;
        /*if (i==0) {
        	cerr << "dl= " <<  einfo->dl<<std::endl;
        	cerr << "damping= " <<  (_dampingRatio*dot(dv,dp)*einfo->restLength/(L*L))<<std::endl;
        }*/
        val=einfo->stiffness*(einfo->dl)/L;
        f[v1]+=dp*val;
        f[v0]-=dp*val;
        //	std::cerr << "einfo->stiffness= "<<einfo->stiffness<<std::endl;
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
                        (edgeInfo[tea[k]].dl * tinfo->gamma[l] +edgeInfo[tea[l]].dl * tinfo->gamma[k])/edgeInfo[tea[j]].currentLength;
                f[ta[l]]+=force;
                f[ta[k]]-=force;
            }
        }
        //	std::cerr << "tinfo->gamma[0] "<<tinfo->gamma[0]<<std::endl;

    }

    updateMatrix=true;
    //std::cerr << "end addForce" << std::endl;
}


template <class DataTypes>
void TriangularQuadraticSpringsForceField<DataTypes>::addDForce(VecDeriv& df, const VecDeriv& dx)
{
    unsigned int i,j,k,l;
    TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
    //unsigned int nbEdges=container->getNumberOfEdges();
    unsigned int nbTriangles=container->getNumberOfTriangles();
    //const std::vector<Edge> &edgeArray=container->getEdgeArray();
    const std::vector< TriangleEdges > &triangleEdgeArray=container->getTriangleEdgeArray() ;
    const std::vector< Triangle> &triangleArray=container->getTriangleArray() ;

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
                    val1 = -tinfo->stiffness[k]*edgeInfo[tea[k]].dl;
                    val1/=edgeInfo[tea[k]].currentLength;

                    val2= -tinfo->stiffness[k]*edgeInfo[tea[k]].restLength;
                    val2/=edgeInfo[tea[k]].currentLength*edgeInfo[tea[k]].currentLength*edgeInfo[tea[k]].currentLength;

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

                    val1 = -(tinfo->stiffness[k]*edgeInfo[tea[k]].dl+
                            tinfo->gamma[i]*edgeInfo[tea[j]].dl+
                            tinfo->gamma[j]*edgeInfo[tea[i]].dl);

                    val2= -val1 - tinfo->stiffness[k]*edgeInfo[tea[k]].currentLength;
                    val1/=edgeInfo[tea[k]].currentLength;
                    val2/=edgeInfo[tea[k]].currentLength*edgeInfo[tea[k]].currentLength*edgeInfo[tea[k]].currentLength;
                    valk=tinfo->gamma[k]/(edgeInfo[tea[j]].currentLength*
                            edgeInfo[tea[i]].currentLength);
                    vali=tinfo->gamma[i]/(edgeInfo[tea[j]].currentLength*
                            edgeInfo[tea[k]].currentLength);
                    valj=tinfo->gamma[j]/(edgeInfo[tea[k]].currentLength*
                            edgeInfo[tea[i]].currentLength);


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
void TriangularQuadraticSpringsForceField<DataTypes>::updateLameCoefficients()
{
    lambda= f_youngModulus.getValue()*f_poissonRatio.getValue()/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
    mu = f_youngModulus.getValue()*(1-f_poissonRatio.getValue())/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
//	std::cerr << "initialized Lame coef : lambda=" <<lambda<< " mu="<<mu<<std::endl;
}


template<class DataTypes>
void TriangularQuadraticSpringsForceField<DataTypes>::draw()
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
