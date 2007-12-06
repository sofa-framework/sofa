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
#include <sofa/component/forcefield/TriangularFEMForceField.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/helper/gl/template.h>
#include <sofa/component/topology/TriangleData.inl>
//#include <sofa/component/topology/EdgeData.inl>
#if defined (__APPLE__)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <vector>
#include <sofa/defaulttype/Vec3Types.h>
#include <assert.h>

#ifdef _WIN32
#include <windows.h>
#endif


// #define DEBUG_TRIANGLEFEM

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

SOFA_DECL_CLASS(TriangularFEMForceField)

template class TriangularFEMForceField<Vec3dTypes>;
template class TriangularFEMForceField<Vec3fTypes>;


// Register in the Factory
int TriangularFEMForceFieldClass = core::RegisterObject("Triangular finite elements")
        .add< TriangularFEMForceField<Vec3dTypes> >()
        .add< TriangularFEMForceField<Vec3fTypes> >()
        ;


template< class DataTypes>
void TriangularFEMForceField<DataTypes>::TRQSTriangleCreationFunction (int /*triangleIndex*/, void* param,
        TriangleInformation &tinfo,
        const Triangle& t,
        const sofa::helper::vector< unsigned int > &,
        const sofa::helper::vector< double >&)
{
    TriangularFEMForceField<DataTypes> *ff= (TriangularFEMForceField<DataTypes> *)param;
    if (ff)
    {
        TriangleSetTopology<DataTypes> *_mesh=ff->getTriangularTopology();
        assert(_mesh!=0);
        //TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
        //const std::vector< Triangle > &triangleArray=container->getTriangleArray() ;
        //const Triangle &t=triangleArray[triangleIndex];

        const typename DataTypes::VecCoord& vect_c = *_mesh->getDOF()->getX0();

        Index a = t[0];
        Index b = t[1];
        Index c = t[2];

        tinfo.materialMatrix[0][0] = 1;
        tinfo.materialMatrix[0][1] = ff->f_poisson.getValue();
        tinfo.materialMatrix[0][2] = 0;
        tinfo.materialMatrix[1][0] = ff->f_poisson.getValue();
        tinfo.materialMatrix[1][1] = 1;
        tinfo.materialMatrix[1][2] = 0;
        tinfo.materialMatrix[2][0] = 0;
        tinfo.materialMatrix[2][1] = 0;
        tinfo.materialMatrix[2][2] = 0.5f * (1 - ff->f_poisson.getValue());

        tinfo.materialMatrix *= (ff->f_young.getValue() / (12 * (1 - ff->f_poisson.getValue() * ff->f_poisson.getValue())));

        switch(ff->method)
        {
        case SMALL :
            //ff->creation_computeMaterialStiffnesses(); //ff->computeMaterialStiffness(triangleIndex,a,b,c)
            //ff->initSmall(); //ff-> triangleIndex,a,b,c

            break;
        case LARGE :
            //ff->creation_computeMaterialStiffnesses(); //ff->computeMaterialStiffness(triangleIndex,a,b,c)

            //ff->initLarge(); //ff-> triangleIndex,a,b,c

            Transformation R_0_1;
            //cerr<<"TriangularFEMForceField<DataTypes>::initLarge(), x.size() = "<<_object->getX()->size()<<", _initialPoints.getValue().size() = "<<_initialPoints.getValue().size()<<endl;
            ff->computeRotationLarge( R_0_1, vect_c, a, b, c );

            tinfo.rotatedInitialElements[0] = R_0_1 * vect_c[a]; //ff->_initialPoints.getValue()[a];
            tinfo.rotatedInitialElements[1] = R_0_1 * vect_c[b];
            tinfo.rotatedInitialElements[2] = R_0_1 * vect_c[c];

            tinfo.rotatedInitialElements[1] -= tinfo.rotatedInitialElements[0];
            tinfo.rotatedInitialElements[2] -= tinfo.rotatedInitialElements[0];
            tinfo.rotatedInitialElements[0] = Coord(0,0,0);

            break;
        }
    }

}



template <class DataTypes>
TriangularFEMForceField<DataTypes>::
TriangularFEMForceField()
    : _mesh(NULL)
//, _indexedElements(NULL)
    , _initialPoints(initData(&_initialPoints, "initialPoints", "Initial Position"))
    , method(LARGE)
    , f_method(initData(&f_method,std::string("large"),"method","large: large displacements, small: small displacements"))
    , f_poisson(initData(&f_poisson,(Real)0.3,"poissonRatio","Poisson ratio in Hooke's law"))
    , f_young(initData(&f_young,(Real)1000.,"youngModulus","Young modulus in Hooke's law"))
    , f_damping(initData(&f_damping,(Real)0.,"damping","Ratio damping/stiffness"))
{}


template <class DataTypes> void TriangularFEMForceField<DataTypes>::handleTopologyChange()
{
    sofa::core::componentmodel::topology::BaseTopology *topology = static_cast<sofa::core::componentmodel::topology::BaseTopology *>(getContext()->getMainTopology());

    std::list<const TopologyChange *>::const_iterator itBegin=topology->firstChange();
    std::list<const TopologyChange *>::const_iterator itEnd=topology->lastChange();

    triangleInfo.handleTopologyEvents(itBegin,itEnd);
}

template <class DataTypes>
TriangularFEMForceField<DataTypes>::~TriangularFEMForceField()
{
}


template <class DataTypes>
void TriangularFEMForceField<DataTypes>::init()
{
    std::cerr << "initializing TriangularFEMForceField" << std::endl;
    this->Inherited::init();
    if (f_method.getValue() == "small")
        method = SMALL;
    else if (f_method.getValue() == "large")
        method = LARGE;

    //cerr<<"TriangularFEMForceField<DataTypes>::init(), node = "<<this->getContext()->getName()<<endl;


    _mesh =0;
    if (getContext()->getMainTopology()!=0)
        _mesh= dynamic_cast<TriangleSetTopology<DataTypes>*>(getContext()->getMainTopology());

    if ((_mesh==0) || (_mesh->getTriangleSetTopologyContainer()->getNumberOfTriangles()==0))
    {
        std::cerr << "ERROR(TriangularQuadraticSpringsForceField): object must have a Triangular Set Topology.\n";
        return;
    }

    TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();

    /// prepare to store info in the triangle array
    triangleInfo.resize(container->getNumberOfTriangles());
    /// prepare to store info in the edge array
    //edgeInfo.resize(container->getNumberOfEdges());

    if (_initialPoints.getValue().size() == 0)
    {
        VecCoord& p = *this->mstate->getX(); //getX0(); ???
        _initialPoints.setValue(p);
    }

    unsigned int i;

    const sofa::helper::vector<Triangle> &triangleArray=container->getTriangleArray();
    for (i=0; i<container->getNumberOfTriangles(); ++i)
    {
        TRQSTriangleCreationFunction(i, (void*) this, triangleInfo[i],
                triangleArray[i],  (const sofa::helper::vector< unsigned int > )0,
                (const sofa::helper::vector< double >)0);
    }


    triangleInfo.setCreateFunction(TRQSTriangleCreationFunction);
    //triangleInfo.setDestroyFunction(TRQSTriangleDestroyFunction);

    triangleInfo.setCreateParameter( (void *) this );
    triangleInfo.setDestroyParameter( (void *) this );

    //_strainDisplacements.resize(container->getNumberOfTriangles());
    //_rotations.resize(container->getNumberOfTriangles());

    computeMaterialStiffnesses();

    initSmall();
    initLarge();
}



template <class DataTypes>
void TriangularFEMForceField<DataTypes>::reinit()
{
    if (f_method.getValue() == "small")
        method = SMALL;
    else if (f_method.getValue() == "large")
        method = LARGE;

    computeMaterialStiffnesses();

    initSmall();
    initLarge();
}


template <class DataTypes>
void TriangularFEMForceField<DataTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/)
{
    TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
    unsigned int nbTriangles=container->getNumberOfTriangles();
    //const sofa::helper::vector< Triangle> &triangleArray=container->getTriangleArray() ;

    //TriangleInformation *tinfo;

    //EdgeInformation *einfo;

    f.resize(x.size());

    if(f_damping.getValue() != 0)
    {
        if(method == SMALL)
        {
            for( unsigned int i=0; i<nbTriangles; i+=3 ) //unsigned int i=0; i<_indexedElements->size(); i+=3
            {
                accumulateForceSmall( f, x, i/3, true );
                accumulateDampingSmall( f, i/3 );
            }
        }
        else
        {
            for( unsigned int i=0; i<nbTriangles; i+=3 )
            {
                accumulateForceLarge( f, x, i/3, true );
                accumulateDampingLarge( f, i/3 );
            }
        }
    }
    else
    {
        if(method==SMALL)
        {
            //typename VecElement::const_iterator it;
            //unsigned int i(0);

            for(unsigned int i=0; i<nbTriangles; i+=1) //it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i
            {
                accumulateForceSmall( f, x, i, true );
            }
        }
        else
        {
            //typename VecElement::const_iterator it;
            //unsigned int i(0);

            for(unsigned int i=0; i<nbTriangles; i+=1) //it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i
            {
                accumulateForceLarge( f, x, i, true );
            }
        }
    }
}


template <class DataTypes>
void TriangularFEMForceField<DataTypes>::addDForce(VecDeriv& df, const VecDeriv& dx)
{
    Real h=1;
    df.resize(dx.size());

    if (method == SMALL)
    {
        applyStiffnessSmall( df,h,dx );
    }
    else
    {
        applyStiffnessLarge( df,h,dx );
    }
}

template <class DataTypes>
double TriangularFEMForceField<DataTypes>::getPotentialEnergy(const VecCoord& /*x*/)
{
    cerr<<"TriangularFEMForceField::getPotentialEnergy-not-implemented !!!"<<endl;
    return 0;
}

template <class DataTypes>
void TriangularFEMForceField<DataTypes>::applyStiffness( VecCoord& v, Real h, const VecCoord& x )
{
    if (method == SMALL)
    {
        applyStiffnessSmall( v,h,x );
    }
    else
    {
        applyStiffnessLarge( v,h,x );
    }
}


template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeStrainDisplacement( StrainDisplacement &J, Coord /*a*/, Coord b, Coord c )
{
#ifdef DEBUG_TRIANGLEFEM
    std::cout << "TriangularFEMForceField::computeStrainDisplacement\n";
#endif

    //Coord ab_cross_ac = cross(b, c);
    Real determinant = b[0] * c[1]; // Surface

    J[0][0] = J[1][2] = -c[1] / determinant;
    J[0][2] = J[1][1] = (c[0] - b[0]) / determinant;
    J[2][0] = J[3][2] = c[1] / determinant;
    J[2][2] = J[3][1] = -c[0] / determinant;
    J[4][0] = J[5][2] = 0;
    J[4][2] = J[5][1] = b[0] / determinant;
    J[1][0] = J[3][0] = J[5][0] = J[0][1] = J[2][1] = J[4][1] = 0;
}


template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeMaterialStiffnesses()
{
    //_materialsStiffnesses.resize(_indexedElements->size());

    TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
    unsigned int nbTriangles=container->getNumberOfTriangles();
    //const sofa::helper::vector< Triangle> &triangleArray=container->getTriangleArray() ;

    TriangleInformation *tinfo;

    for(unsigned i = 0; i < nbTriangles; ++i) //_indexedElements->size()
    {
        tinfo=&triangleInfo[i];

        tinfo->materialMatrix[0][0] = 1;
        tinfo->materialMatrix[0][1] = f_poisson.getValue();
        tinfo->materialMatrix[0][2] = 0;
        tinfo->materialMatrix[1][0] = f_poisson.getValue();
        tinfo->materialMatrix[1][1] = 1;
        tinfo->materialMatrix[1][2] = 0;
        tinfo->materialMatrix[2][0] = 0;
        tinfo->materialMatrix[2][1] = 0;
        tinfo->materialMatrix[2][2] = 0.5f * (1 - f_poisson.getValue());

        tinfo->materialMatrix *= (f_young.getValue() / (12 * (1 - f_poisson.getValue() * f_poisson.getValue())));
    }
}


template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeForce( Displacement &F, const Displacement &Depl, const MaterialStiffness &K, const StrainDisplacement &J )
{
    Mat<3,6,Real> Jt;
    Jt.transpose( J );

    Vec<3,Real> JtD;

    // Optimisations: The following values are 0 (per computeStrainDisplacement )

    // Jt[0][1]
    // Jt[0][3]
    // Jt[0][4]
    // Jt[0][5]
    // Jt[1][0]
    // Jt[1][2]
    // Jt[1][4]
    // Jt[2][5]


    //	JtD = Jt * Depl;

    JtD[0] = Jt[0][0] * Depl[0] + /* Jt[0][1] * Depl[1] + */ Jt[0][2] * Depl[2]
            /* + Jt[0][3] * Depl[3] + Jt[0][4] * Depl[4] + Jt[0][5] * Depl[5] */ ;

    JtD[1] = /* Jt[1][0] * Depl[0] + */ Jt[1][1] * Depl[1] + /* Jt[1][2] * Depl[2] + */
            Jt[1][3] * Depl[3] + /* Jt[1][4] * Depl[4] + */ Jt[1][5] * Depl[5];

    JtD[2] = Jt[2][0] * Depl[0] + Jt[2][1] * Depl[1] + Jt[2][2] * Depl[2] +
            Jt[2][3] * Depl[3] + Jt[2][4] * Depl[4] /* + Jt[2][5] * Depl[5] */ ;

    Vec<3,Real> KJtD;

    //	KJtD = K * JtD;

    // Optimisations: The following values are 0 (per computeMaterialStiffnesses )

    // K[0][2]
    // K[1][2]
    // K[2][0]
    // K[2][1]

    KJtD[0] = K[0][0] * JtD[0] + K[0][1] * JtD[1] /* + K[0][2] * JtD[2] */;

    KJtD[1] = K[1][0] * JtD[0] + K[1][1] * JtD[1] /* + K[1][2] * JtD[2] */;

    KJtD[2] = /* K[2][0] * JtD[0] + K[2][1] * JtD[1] */ + K[2][2] * JtD[2];

    //	F = J * KJtD;


    // Optimisations: The following values are 0 (per computeStrainDisplacement )

    // J[0][1]
    // J[1][0]
    // J[2][1]
    // J[3][0]
    // J[4][0]
    // J[4][1]
    // J[5][0]
    // J[5][2]

    F[0] = J[0][0] * KJtD[0] + /* J[0][1] * KJtD[1] + */ J[0][2] * KJtD[2];

    F[1] = /* J[1][0] * KJtD[0] + */ J[1][1] * KJtD[1] + J[1][2] * KJtD[2];

    F[2] = J[2][0] * KJtD[0] + /* J[2][1] * KJtD[1] + */ J[2][2] * KJtD[2];

    F[3] = /* J[3][0] * KJtD[0] + */ J[3][1] * KJtD[1] + J[3][2] * KJtD[2];

    F[4] = /* J[4][0] * KJtD[0] + J[4][1] * KJtD[1] + */ J[4][2] * KJtD[2];

    F[5] = /* J[5][0] * KJtD[0] + */ J[5][1] * KJtD[1] /* + J[5][2] * KJtD[2] */ ;
}


/*
** SMALL DEFORMATION METHODS
*/


template <class DataTypes>
void TriangularFEMForceField<DataTypes>::initSmall()
{

#ifdef DEBUG_TRIANGLEFEM
    std::cout << "TriangularFEMForceField::initSmall\n";
#endif

}


template <class DataTypes>
void TriangularFEMForceField<DataTypes>::accumulateForceSmall( VecCoord &f, const VecCoord &p, Index elementIndex, bool implicit )
{

#ifdef DEBUG_TRIANGLEFEM
    std::cout << "TriangularFEMForceField::accumulateForceSmall\n";
#endif

    TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
    //unsigned int nbTriangles=container->getNumberOfTriangles();
    const sofa::helper::vector< Triangle> &triangleArray=container->getTriangleArray() ;

    Index a = triangleArray[elementIndex][0]; //(*_indexedElements)
    Index b = triangleArray[elementIndex][1];
    Index c = triangleArray[elementIndex][2];

    Coord deforme_a, deforme_b, deforme_c;
    deforme_b = p[b]-p[a];
    deforme_c = p[c]-p[a];
    deforme_a = Coord(0,0,0);

    // displacements
    Displacement D;
    D[0] = 0;
    D[1] = 0;
    D[2] = (_initialPoints.getValue()[b][0]-_initialPoints.getValue()[a][0]) - deforme_b[0];
    D[3] = 0;
    D[4] = (_initialPoints.getValue()[c][0]-_initialPoints.getValue()[a][0]) - deforme_c[0];
    D[5] = (_initialPoints.getValue()[c][1]-_initialPoints.getValue()[a][1]) - deforme_c[1];


    StrainDisplacement J;
    computeStrainDisplacement(J,deforme_a,deforme_b,deforme_c);
    if (implicit)
        triangleInfo[elementIndex].strainDisplacementMatrix = J;

    // compute force on element
    Displacement F;
    computeForce( F, D, triangleInfo[elementIndex].materialMatrix, J );

    f[a] += Coord( F[0], F[1], 0);
    f[b] += Coord( F[2], F[3], 0);
    f[c] += Coord( F[4], F[5], 0);
}


template <class DataTypes>
void TriangularFEMForceField<DataTypes>::accumulateDampingSmall(VecCoord&, Index )
{

#ifdef DEBUG_TRIANGLEFEM
    std::cout << "TriangularFEMForceField::accumulateDampingSmall\n";
#endif

}


template <class DataTypes>
void TriangularFEMForceField<DataTypes>::applyStiffnessSmall(VecCoord &v, Real h, const VecCoord &x)
{

#ifdef DEBUG_TRIANGLEFEM
    std::cout << "TriangularFEMForceField::applyStiffnessSmall\n";
#endif

    TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
    unsigned int nbTriangles=container->getNumberOfTriangles();
    const sofa::helper::vector< Triangle> &triangleArray=container->getTriangleArray() ;

    //typename VecElement::const_iterator it;
    //unsigned int i(0);

    for(unsigned int i=0; i<nbTriangles; i+=1) //it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i
    {
        Index a = triangleArray[i][0]; //(*it)[0];
        Index b = triangleArray[i][1];
        Index c = triangleArray[i][2];

        Displacement X;

        X[0] = x[a][0];
        X[1] = x[a][1];

        X[2] = x[b][0];
        X[3] = x[b][1];

        X[4] = x[c][0];
        X[5] = x[c][1];

        Displacement F;
        computeForce( F, X, triangleInfo[i].materialMatrix, triangleInfo[i].strainDisplacementMatrix );

        v[a] += Coord(-h*F[0], -h*F[1], 0);
        v[b] += Coord(-h*F[2], -h*F[3], 0);
        v[c] += Coord(-h*F[4], -h*F[5], 0);
    }
}


/*
** LARGE DEFORMATION METHODS
*/

template <class DataTypes>
void TriangularFEMForceField<DataTypes>::initLarge()
{

#ifdef DEBUG_TRIANGLEFEM
    std::cout << "TriangularFEMForceField::initLarge\n";
#endif

    TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
    unsigned int nbTriangles=container->getNumberOfTriangles();
    const sofa::helper::vector< Triangle> &triangleArray=container->getTriangleArray() ;

    //_rotatedInitialElements.resize(_indexedElements->size());

    //typename VecElement::const_iterator it;
    //unsigned int i(0);

    TriangleInformation *tinfo;


    for(unsigned int i=0; i<nbTriangles; i+=1) //it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i
    {
        tinfo=&triangleInfo[i];

        Index a = triangleArray[i][0]; //(*it)[0];
        Index b = triangleArray[i][1];
        Index c = triangleArray[i][2];

        // Rotation matrix (initial triangle/world)
        // first vector on first edge
        // second vector in the plane of the two first edges
        // third vector orthogonal to first and second
        Transformation R_0_1;
        //cerr<<"TriangularFEMForceField<DataTypes>::initLarge(), x.size() = "<<_object->getX()->size()<<", _initialPoints.getValue().size() = "<<_initialPoints.getValue().size()<<endl;
        computeRotationLarge( R_0_1, _initialPoints.getValue(), a, b, c );

        tinfo->rotatedInitialElements[0] = R_0_1 * _initialPoints.getValue()[a]; //_rotatedInitialElements ... _initialPoints.getValue()
        tinfo->rotatedInitialElements[1] = R_0_1 * _initialPoints.getValue()[b];
        tinfo->rotatedInitialElements[2] = R_0_1 * _initialPoints.getValue()[c];

        tinfo->rotatedInitialElements[1] -= tinfo->rotatedInitialElements[0];
        tinfo->rotatedInitialElements[2] -= tinfo->rotatedInitialElements[0];
        tinfo->rotatedInitialElements[0] = Coord(0,0,0);
    }
}


template <class DataTypes>
void TriangularFEMForceField<DataTypes>::computeRotationLarge( Transformation &r, const VecCoord &p, const Index &a, const Index &b, const Index &c)
{

#ifdef DEBUG_TRIANGLEFEM
    std::cout << "TriangularFEMForceField::computeRotationLarge\n";
#endif

    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second

    Coord edgex = p[b] - p[a];
    edgex.normalize();

    Coord edgey = p[c] - p[a];
    edgey.normalize();

    Coord edgez;
    edgez = cross(edgex, edgey);
    edgez.normalize();

    edgey = cross(edgez, edgex);
    edgey.normalize();

    r[0][0] = edgex[0];
    r[0][1] = edgex[1];
    r[0][2] = edgex[2];
    r[1][0] = edgey[0];
    r[1][1] = edgey[1];
    r[1][2] = edgey[2];
    r[2][0] = edgez[0];
    r[2][1] = edgez[1];
    r[2][2] = edgez[2];
}


template <class DataTypes>
void TriangularFEMForceField<DataTypes>::accumulateForceLarge(VecCoord &f, const VecCoord &p, Index elementIndex, bool implicit )
{

#ifdef DEBUG_TRIANGLEFEM
    std::cout << "TriangularFEMForceField::accumulateForceLarge\n";
#endif

    TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
    //unsigned int nbTriangles=container->getNumberOfTriangles();
    const sofa::helper::vector< Triangle> &triangleArray=container->getTriangleArray() ;

    Index a = triangleArray[elementIndex][0];
    Index b = triangleArray[elementIndex][1];
    Index c = triangleArray[elementIndex][2];

    // Rotation matrix (deformed and displaced Triangle/world)
    Transformation R_2_0, R_0_2;
    computeRotationLarge( R_0_2, p, a, b, c);
    R_2_0.transpose(R_0_2);


    // positions of the deformed and displaced Tetrahedre in its frame
    Coord deforme_a, deforme_b, deforme_c;
    //deforme_a = R_0_2 * p[a];
    //deforme_b = R_0_2 * p[b];
    //deforme_c = R_0_2 * p[c];
    //deforme_b -= deforme_a;
    //deforme_c -= deforme_a;
    //deforme_a = Coord(0,0,0);
    deforme_b = R_0_2 * (p[b]-p[a]);
    deforme_c = R_0_2 * (p[c]-p[a]);

    // displacements
    Displacement D;
    D[0] = 0;
    D[1] = 0;
    D[2] = triangleInfo[elementIndex].rotatedInitialElements[1][0] - deforme_b[0];
    D[3] = 0;
    D[4] = triangleInfo[elementIndex].rotatedInitialElements[2][0] - deforme_c[0];
    D[5] = triangleInfo[elementIndex].rotatedInitialElements[2][1] - deforme_c[1];

    // shape functions matrix
    StrainDisplacement J;
    computeStrainDisplacement(J,deforme_a,deforme_b,deforme_c);

    if(implicit)
    {
        triangleInfo[elementIndex].strainDisplacementMatrix = J;
        triangleInfo[elementIndex].rotation = R_2_0 ;
    }

    // compute force on element
    Displacement F;
    computeForce( F, D, triangleInfo[elementIndex].materialMatrix, J );

    f[a] += R_2_0 * Coord(F[0], F[1], 0);
    f[b] += R_2_0 * Coord(F[2], F[3], 0);
    f[c] += R_2_0 * Coord(F[4], F[5], 0);
}


template <class DataTypes>
void TriangularFEMForceField<DataTypes>::accumulateDampingLarge(VecCoord &, Index )
{

#ifdef DEBUG_TRIANGLEFEM
    std::cout << "TriangularFEMForceField::accumulateDampingLarge\n";
#endif

}


template <class DataTypes>
void TriangularFEMForceField<DataTypes>::applyStiffnessLarge(VecCoord &v, Real h, const VecCoord &x)
{

#ifdef DEBUG_TRIANGLEFEM
    std::cout << "TriangularFEMForceField::applyStiffnessLarge\n";
#endif

    TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
    unsigned int nbTriangles=container->getNumberOfTriangles();
    const sofa::helper::vector< Triangle> &triangleArray=container->getTriangleArray() ;

    //typename VecElement::const_iterator it;
    //unsigned int i(0);

    for(unsigned int i=0; i<nbTriangles; i+=1) //it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i
    {
        Index a = triangleArray[i][0];
        Index b = triangleArray[i][1];
        Index c = triangleArray[i][2];

        Transformation R_0_2;
        R_0_2.transpose(triangleInfo[i].rotation);

        Displacement X;
        Coord x_2;

        x_2 = R_0_2 * x[a];
        X[0] = x_2[0];
        X[1] = x_2[1];

        x_2 = R_0_2 * x[b];
        X[2] = x_2[0];
        X[3] = x_2[1];

        x_2 = R_0_2 * x[c];
        X[4] = x_2[0];
        X[5] = x_2[1];

        Displacement F;
        computeForce( F, X, triangleInfo[i].materialMatrix, triangleInfo[i].strainDisplacementMatrix );

        v[a] += triangleInfo[i].rotation * Coord(-h*F[0], -h*F[1], 0);
        v[b] += triangleInfo[i].rotation * Coord(-h*F[2], -h*F[3], 0);
        v[c] += triangleInfo[i].rotation * Coord(-h*F[4], -h*F[5], 0);
    }
}


template<class DataTypes>
void TriangularFEMForceField<DataTypes>::draw()
{
    if (!getContext()->getShowForceFields())
        return;
//     if (!this->_object)
//         return;

    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    const VecCoord& x = *this->mstate->getX();
    TriangleSetTopologyContainer *container=_mesh->getTriangleSetTopologyContainer();
    unsigned int nbTriangles=container->getNumberOfTriangles();
    const sofa::helper::vector< Triangle> &triangleArray=container->getTriangleArray() ;

    glDisable(GL_LIGHTING);

    glBegin(GL_TRIANGLES);
    //typename VecElement::const_iterator it;
    unsigned int i;
    for(i=0; i<nbTriangles; ++i) //it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it
    {
        Index a = triangleArray[i][0];//(*it)[0];
        Index b = triangleArray[i][1];//(*it)[1];
        Index c = triangleArray[i][2];//(*it)[2];

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

} // namespace component

} // namespace sofa
