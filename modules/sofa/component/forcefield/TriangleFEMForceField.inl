/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGLEFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_TRIANGLEFEMFORCEFIELD_INL

#include <sofa/component/forcefield/TriangleFEMForceField.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/helper/gl/template.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <vector>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/component.h>

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





template <class DataTypes>
TriangleFEMForceField<DataTypes>::
TriangleFEMForceField()
    : _mesh(NULL)
    , _indexedElements(NULL)
    , _initialPoints(initData(&_initialPoints, "initialPoints", "Initial Position"))
    , method(LARGE)
    , f_method(initData(&f_method,std::string("large"),"method","large: large displacements, small: small displacements"))
    , f_poisson(initData(&f_poisson,(Real)0.3,"poissonRatio","Poisson ratio in Hooke's law"))
    , f_young(initData(&f_young,(Real)1000.,"youngModulus","Young modulus in Hooke's law"))
    , f_damping(initData(&f_damping,(Real)0.,"damping","Ratio damping/stiffness"))
{}

template <class DataTypes>
TriangleFEMForceField<DataTypes>::~TriangleFEMForceField()
{
}


template <class DataTypes>
void TriangleFEMForceField<DataTypes>::init()
{
    this->Inherited::init();
    if (f_method.getValue() == "small")
        method = SMALL;
    else if (f_method.getValue() == "large")
        method = LARGE;

    serr<<"TriangleFEMForceField<DataTypes>::init(), node = "<<this->getContext()->getName()<<sendl;
    _mesh = this->getContext()->getMeshTopology();

    if (_mesh==NULL || (_mesh->getTriangles().empty() && _mesh->getNbQuads()<=0))
    {
        serr << "ERROR(TriangleFEMForceField): object must have a triangular MeshTopology."<<sendl;
        return;
    }
    if (!_mesh->getTriangles().empty())
    {
        _indexedElements = & (_mesh->getTriangles());
    }
    else
    {
        sofa::core::topology::BaseMeshTopology::SeqTriangles* trias = new sofa::core::topology::BaseMeshTopology::SeqTriangles;
        int nbcubes = _mesh->getNbQuads();
        trias->reserve(nbcubes*2);
        for (int i=0; i<nbcubes; i++)
        {
            sofa::core::topology::BaseMeshTopology::Quad q = _mesh->getQuad(i);
            trias->push_back(Element(q[0],q[1],q[2]));
            trias->push_back(Element(q[0],q[2],q[3]));
        }
        _indexedElements = trias;
    }

    if (_initialPoints.getValue().size() == 0)
    {
        const VecCoord& p = *this->mstate->getX0();
        _initialPoints.setValue(p);
    }

    _strainDisplacements.resize(_indexedElements->size());
    _rotations.resize(_indexedElements->size());

    computeMaterialStiffnesses();

    initSmall();
    initLarge();
}



template <class DataTypes>
void TriangleFEMForceField<DataTypes>::reinit()
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
void TriangleFEMForceField<DataTypes>::addForce(DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& /* v */, const core::MechanicalParams* /* mparams */)
{
    VecDeriv& f1 = *f.beginEdit();
    const VecCoord& x1 = x.getValue();

    f1.resize(x1.size());

    if(f_damping.getValue() != 0)
    {
        if(method == SMALL)
        {
            for( unsigned int i=0; i<_indexedElements->size(); i+=3 )
            {
                accumulateForceSmall( f1, x1, i/3, true );
                accumulateDampingSmall( f1, i/3 );
            }
        }
        else
        {
            for( unsigned int i=0; i<_indexedElements->size(); i+=3 )
            {
                accumulateForceLarge( f1, x1, i/3, true );
                accumulateDampingLarge( f1, i/3 );
            }
        }
    }
    else
    {
        if(method==SMALL)
        {
            typename VecElement::const_iterator it;
            unsigned int i(0);

            for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
            {
                accumulateForceSmall( f1, x1, i, true );
            }
        }
        else
        {
            typename VecElement::const_iterator it;
            unsigned int i(0);

            for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
            {
                accumulateForceLarge( f1, x1, i, true );
            }
        }
    }

    f.endEdit();
}


template <class DataTypes>
void TriangleFEMForceField<DataTypes>::addDForce(DataVecDeriv& df, const DataVecDeriv& dx, const core::MechanicalParams* /* mparams */)
{
    VecDeriv& df1 = *df.beginEdit();
    const VecDeriv& dx1 = dx.getValue();

    Real h=1;
    df1.resize(dx1.size());

    if (method == SMALL)
    {
        applyStiffnessSmall( df1,h,dx1 );
    }
    else
    {
        applyStiffnessLarge( df1,h,dx1 );
    }
}

template <class DataTypes>
double TriangleFEMForceField<DataTypes>::getPotentialEnergy(const DataVecCoord& /* x */, const core::MechanicalParams* /* mparams */) const
{
    serr<<"TriangleFEMForceField::getPotentialEnergy-not-implemented !!!"<<sendl;
    return 0;
}

template <class DataTypes>
void TriangleFEMForceField<DataTypes>::applyStiffness( VecCoord& v, Real h, const VecCoord& x )
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
void TriangleFEMForceField<DataTypes>::computeStrainDisplacement( StrainDisplacement &J, Coord /*a*/, Coord b, Coord c )
{
#ifdef DEBUG_TRIANGLEFEM
    sout << "TriangleFEMForceField::computeStrainDisplacement"<<sendl;
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
void TriangleFEMForceField<DataTypes>::computeMaterialStiffnesses()
{
    _materialsStiffnesses.resize(_indexedElements->size());

    for(unsigned i = 0; i < _indexedElements->size(); ++i)
    {
        _materialsStiffnesses[i][0][0] = 1;
        _materialsStiffnesses[i][0][1] = f_poisson.getValue();
        _materialsStiffnesses[i][0][2] = 0;
        _materialsStiffnesses[i][1][0] = f_poisson.getValue();
        _materialsStiffnesses[i][1][1] = 1;
        _materialsStiffnesses[i][1][2] = 0;
        _materialsStiffnesses[i][2][0] = 0;
        _materialsStiffnesses[i][2][1] = 0;
        _materialsStiffnesses[i][2][2] = 0.5f * (1 - f_poisson.getValue());

        _materialsStiffnesses[i] *= (f_young.getValue() / (12 * (1 - f_poisson.getValue() * f_poisson.getValue())));
    }
}


template <class DataTypes>
void TriangleFEMForceField<DataTypes>::computeForce( Displacement &F, const Displacement &Depl, const MaterialStiffness &K, const StrainDisplacement &J )
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
void TriangleFEMForceField<DataTypes>::initSmall()
{

#ifdef DEBUG_TRIANGLEFEM
    sout << "TriangleFEMForceField::initSmall"<<sendl;
#endif

}


template <class DataTypes>
void TriangleFEMForceField<DataTypes>::accumulateForceSmall( VecCoord &f, const VecCoord &p, Index elementIndex, bool implicit )
{

#ifdef DEBUG_TRIANGLEFEM
    sout << "TriangleFEMForceField::accumulateForceSmall"<<sendl;
#endif

    Index a = (*_indexedElements)[elementIndex][0];
    Index b = (*_indexedElements)[elementIndex][1];
    Index c = (*_indexedElements)[elementIndex][2];

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
        _strainDisplacements[elementIndex] = J;

    // compute force on element
    Displacement F;
    computeForce( F, D, _materialsStiffnesses[elementIndex], J );

    f[a] += Coord( F[0], F[1], 0);
    f[b] += Coord( F[2], F[3], 0);
    f[c] += Coord( F[4], F[5], 0);
}


template <class DataTypes>
void TriangleFEMForceField<DataTypes>::accumulateDampingSmall(VecCoord&, Index )
{

#ifdef DEBUG_TRIANGLEFEM
    sout << "TriangleFEMForceField::accumulateDampingSmall"<<sendl;
#endif

}


template <class DataTypes>
void TriangleFEMForceField<DataTypes>::applyStiffnessSmall(VecCoord &v, Real h, const VecCoord &x)
{

#ifdef DEBUG_TRIANGLEFEM
    sout << "TriangleFEMForceField::applyStiffnessSmall"<<sendl;
#endif

    typename VecElement::const_iterator it;
    unsigned int i(0);

    for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
    {
        Index a = (*it)[0];
        Index b = (*it)[1];
        Index c = (*it)[2];

        Displacement X;

        X[0] = x[a][0];
        X[1] = x[a][1];

        X[2] = x[b][0];
        X[3] = x[b][1];

        X[4] = x[c][0];
        X[5] = x[c][1];

        Displacement F;
        computeForce( F, X, _materialsStiffnesses[i], _strainDisplacements[i] );

        v[a] += Coord(-h*F[0], -h*F[1], 0);
        v[b] += Coord(-h*F[2], -h*F[3], 0);
        v[c] += Coord(-h*F[4], -h*F[5], 0);
    }
}


/*
** LARGE DEFORMATION METHODS
*/

template <class DataTypes>
void TriangleFEMForceField<DataTypes>::initLarge()
{

#ifdef DEBUG_TRIANGLEFEM
    sout << "TriangleFEMForceField::initLarge"<<sendl;
#endif

    _rotatedInitialElements.resize(_indexedElements->size());

    typename VecElement::const_iterator it;
    unsigned int i(0);

    for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
    {
        Index a = (*it)[0];
        Index b = (*it)[1];
        Index c = (*it)[2];

        // Rotation matrix (initial triangle/world)
        // first vector on first edge
        // second vector in the plane of the two first edges
        // third vector orthogonal to first and second
        Transformation R_0_1;
        //serr<<"TriangleFEMForceField<DataTypes>::initLarge(), x.size() = "<<_object->getX()->size()<<", _initialPoints.getValue().size() = "<<_initialPoints.getValue().size()<<sendl;
        computeRotationLarge( R_0_1, _initialPoints.getValue(), a, b, c );

        _rotatedInitialElements[i][0] = R_0_1 * _initialPoints.getValue()[a];
        _rotatedInitialElements[i][1] = R_0_1 * _initialPoints.getValue()[b];
        _rotatedInitialElements[i][2] = R_0_1 * _initialPoints.getValue()[c];

        _rotatedInitialElements[i][1] -= _rotatedInitialElements[i][0];
        _rotatedInitialElements[i][2] -= _rotatedInitialElements[i][0];
        _rotatedInitialElements[i][0] = Coord(0,0,0);
    }
}


template <class DataTypes>
void TriangleFEMForceField<DataTypes>::computeRotationLarge( Transformation &r, const VecCoord &p, const Index &a, const Index &b, const Index &c)
{

#ifdef DEBUG_TRIANGLEFEM
    sout << "TriangleFEMForceField::computeRotationLarge"<<sendl;
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
void TriangleFEMForceField<DataTypes>::accumulateForceLarge(VecCoord &f, const VecCoord &p, Index elementIndex, bool implicit )
{

#ifdef DEBUG_TRIANGLEFEM
    sout << "TriangleFEMForceField::accumulateForceLarge"<<sendl;
#endif

    Index a = (*_indexedElements)[elementIndex][0];
    Index b = (*_indexedElements)[elementIndex][1];
    Index c = (*_indexedElements)[elementIndex][2];

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
    D[2] = _rotatedInitialElements[elementIndex][1][0] - deforme_b[0];
    D[3] = 0;
    D[4] = _rotatedInitialElements[elementIndex][2][0] - deforme_c[0];
    D[5] = _rotatedInitialElements[elementIndex][2][1] - deforme_c[1];

    // shape functions matrix
    StrainDisplacement J;
    computeStrainDisplacement(J,deforme_a,deforme_b,deforme_c);

    if(implicit)
    {
        _strainDisplacements[elementIndex] = J;
        _rotations[elementIndex] = R_2_0 ;
    }

    // compute force on element
    Displacement F;
    computeForce( F, D, _materialsStiffnesses[elementIndex], J );

    f[a] += R_2_0 * Coord(F[0], F[1], 0);
    f[b] += R_2_0 * Coord(F[2], F[3], 0);
    f[c] += R_2_0 * Coord(F[4], F[5], 0);
}


template <class DataTypes>
void TriangleFEMForceField<DataTypes>::accumulateDampingLarge(VecCoord &, Index )
{

#ifdef DEBUG_TRIANGLEFEM
    sout << "TriangleFEMForceField::accumulateDampingLarge"<<sendl;
#endif

}


template <class DataTypes>
void TriangleFEMForceField<DataTypes>::applyStiffnessLarge(VecCoord &v, Real h, const VecCoord &x)
{

#ifdef DEBUG_TRIANGLEFEM
    sout << "TriangleFEMForceField::applyStiffnessLarge"<<sendl;
#endif

    typename VecElement::const_iterator it;
    unsigned int i(0);

    for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
    {
        Index a = (*it)[0];
        Index b = (*it)[1];
        Index c = (*it)[2];

        Transformation R_0_2;
        R_0_2.transpose(_rotations[i]);

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
        computeForce( F, X, _materialsStiffnesses[i], _strainDisplacements[i] );

        v[a] += _rotations[i] * Coord(-h*F[0], -h*F[1], 0);
        v[b] += _rotations[i] * Coord(-h*F[2], -h*F[3], 0);
        v[c] += _rotations[i] * Coord(-h*F[4], -h*F[5], 0);
    }
}


template<class DataTypes>
void TriangleFEMForceField<DataTypes>::draw()
{
    if (!this->getContext()->getShowForceFields())
        return;
//     if (!this->_object)
//         return;

    if (this->getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    const VecCoord& x = *this->mstate->getX();

    glDisable(GL_LIGHTING);

    glBegin(GL_TRIANGLES);
    typename VecElement::const_iterator it;
    for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it)
    {
        Index a = (*it)[0];
        Index b = (*it)[1];
        Index c = (*it)[2];

        glColor4f(0,1,0,1);
        helper::gl::glVertexT(x[a]);
        glColor4f(0,0.5,0.5,1);
        helper::gl::glVertexT(x[b]);
        glColor4f(0,0,1,1);
        helper::gl::glVertexT(x[c]);
    }
    glEnd();

    if (this->getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // #ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGLEFEMFORCEFIELD_INL
