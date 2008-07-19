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
#include <sofa/component/forcefield/TriangularAnisotropicFEMForceField.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sofa/helper/gl/template.h>
#include <sofa/component/topology/TriangleData.inl>
#include <sofa/component/topology/EdgeData.inl>
#include <sofa/component/topology/PointData.inl>
#include <sofa/helper/system/gl.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <vector>
#include <algorithm>
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

SOFA_DECL_CLASS(TriangularAnisotropicFEMForceField)

template <class DataTypes>
TriangularAnisotropicFEMForceField<DataTypes>::
TriangularAnisotropicFEMForceField()
    : f_young2(initData(&f_young2,(Real)(0.5*Inherited::f_young.getValue()),"transverseYoungModulus","Young modulus along transverse direction"))
    , f_theta(initData(&f_theta,(Real)(0.0),"fiberAngle","Fiber angle in global reference frame (in degrees)"))
{

}

template< class DataTypes>
void TriangularAnisotropicFEMForceField<DataTypes>::TRQSTriangleCreationFunction (int triangleIndex, void* param,
        TriangleInformation &/*tinfo*/,
        const Triangle& /*t*/,
        const sofa::helper::vector< unsigned int > &,
        const sofa::helper::vector< double >&)
{
    TriangularAnisotropicFEMForceField<DataTypes> *ff= (TriangularAnisotropicFEMForceField<DataTypes> *)param;
    if (ff)
    {
        TriangleSetTopology<DataTypes> *_mesh = ff->getTriangularTopology();
        assert(_mesh!=0);
        TriangleSetTopologyContainer *container = _mesh->getTriangleSetTopologyContainer();
        const std::vector< Triangle > &triangleArray=container->getTriangleArray() ;
        const Triangle &t = triangleArray[triangleIndex];

//		const typename DataTypes::VecCoord& vect_c = *_mesh->getDOF()->getX0();

        Index a = t[0];
        Index b = t[1];
        Index c = t[2];

        switch(ff->method)
        {
        case SMALL :
            ff->computeMaterialStiffness(triangleIndex, a, b, c);
            ff->initSmall();
            break;
        case LARGE :
            ff->computeMaterialStiffness(triangleIndex, a, b, c);
            ff->initLarge(triangleIndex,a,b,c);
            break;
        }
    }
}

template< class DataTypes>
void TriangularAnisotropicFEMForceField<DataTypes>::init()
{
    Inherited::init();
    //reinit();
}

template <class DataTypes>void TriangularAnisotropicFEMForceField<DataTypes>::reinit()
{
    f_poisson2.setValue(Inherited::f_poisson.getValue()*(f_young2.getValue()/Inherited::f_young.getValue()));
    Inherited::reinit();
}

template <class DataTypes>
void TriangularAnisotropicFEMForceField<DataTypes>::computeMaterialStiffness(int i, Index& v1, Index& v2, Index& v3)
{
// 	TriangleSetTopologyContainer *container = Inherited::_mesh-> getTriangleSetTopologyContainer();
    TriangleInformation *tinfo = &Inherited::triangleInfo[i];

    Real Q11, Q12, Q22, Q66;
    Q11 = Inherited::f_young.getValue()/(1-Inherited::f_poisson.getValue()*f_poisson2.getValue());
    Q12 = Inherited::f_poisson.getValue()*f_young2.getValue()/(1-Inherited::f_poisson.getValue()*f_poisson2.getValue());
    Q22 = f_young2.getValue()/(1-Inherited::f_poisson.getValue()*f_poisson2.getValue());
    Q66 = Inherited::f_young.getValue() / (2.0*(1 + Inherited::f_poisson.getValue()));

    Real c, s, c2, s2, c3, s3,c4, s4;
    double theta = (double)f_theta.getValue()*M_PI/180.0;
    //double theta_ref;

    Coord fiberDir((Real)cos(theta), (Real)sin(theta), 0);
    Mat<3,3,Real> bary,baryInv;
    bary[0] = Inherited::_initialPoints.getValue()[v2]-Inherited::_initialPoints.getValue()[v1];
    bary[1] = Inherited::_initialPoints.getValue()[v3]-Inherited::_initialPoints.getValue()[v1];
    bary[2] = cross(bary[0],bary[1]);
    bary.transpose();
    baryInv.invert(bary);
    if (i >= (int) fiberDirRefs.size())
        fiberDirRefs.resize(i+1);
    Deriv& fiberDirRef = fiberDirRefs[i];
    fiberDirRef = baryInv * fiberDir;
    fiberDirRef[2] = 0;
    fiberDirRef.normalize();
    c = fiberDirRef[0]; //cos(theta_ref);
    s = fiberDirRef[1]; //sin(theta_ref);
    c2 = c*c;
    s2 = s*s;
    c3 = c2*c;
    s3 = s2*s;
    s4 = s2*s2;
    c4 = c2*c2;

    Real K11= c4 * Q11 + 2 *c2*s2 *(Q12+2*Q66) + s4 * Q22;
    Real K12 = c2*s2 * (Q11+Q22-4*Q66) + (c4+ s4) * Q12;
    Real K22 = s4* Q11 + 2  *c2*s2 * (Q12+2*Q66) + c4 * Q22;
    Real K16 = c3 * s * (Q11 - Q12) + c*s3* (Q12 - Q22) - 2*c*s* (c2 -s2) * Q66;
    Real K26 = c*s3 * (Q11-Q12) + c3*s* (Q12 - Q22) + 2*c*s* (c2 -s2) * Q66;
    Real K66 = c2*s2 * (Q11+Q22-2*Q12 - 2*Q66) + (c4+ s4) * Q66;

    tinfo->materialMatrix[0][0] = K11;
    tinfo->materialMatrix[0][1] = K12;
    tinfo->materialMatrix[0][2] = K16;
    tinfo->materialMatrix[1][0] = K12;
    tinfo->materialMatrix[1][1] = K22;
    tinfo->materialMatrix[1][2] = K26;
    tinfo->materialMatrix[2][0] = K16;
    tinfo->materialMatrix[2][1] = K26;
    tinfo->materialMatrix[2][2] = K66;

    tinfo->materialMatrix *= (Real)(1.0/12.0);

    cout << "Young1=" << Inherited::f_young.getValue() << endl;
    cout << "Young2=" << f_young2.getValue() << endl;
    cout << "Poisson1=" << Inherited::f_poisson.getValue() << endl;
    cout << "Poisson2=" << f_poisson2.getValue() << endl;
}


template <class DataTypes>void TriangularAnisotropicFEMForceField<DataTypes>::draw()
{
    glPolygonOffset(1.0, 2.0);
    glEnable(GL_POLYGON_OFFSET_FILL);
    Inherited::draw();
    glDisable(GL_POLYGON_OFFSET_FILL);
    if (!fiberDirRefs.empty())
    {
        const VecCoord& x = *this->mstate->getX();
        TriangleSetTopologyContainer *container=Inherited::_mesh->getTriangleSetTopologyContainer();
        unsigned int nbTriangles=container->getNumberOfTriangles();
        const sofa::helper::vector< Triangle> &triangleArray=container->getTriangleArray();
        glColor3f(1,1,1);
        glBegin(GL_LINES);
        //typename VecElement::const_iterator it;
        unsigned int i;
        for(i=0; i<nbTriangles; ++i) //it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it
        {
            Index a = triangleArray[i][0];//(*it)[0];
            Index b = triangleArray[i][1];//(*it)[1];
            Index c = triangleArray[i][2];//(*it)[2];
            Coord center = (x[a]+x[b]+x[c])/3;
            Coord d = (x[b]-x[a])*fiberDirRefs[i][0] + (x[c]-x[a])*fiberDirRefs[i][1];
            d*=0.5;
            helper::gl::glVertexT(center-d);
            helper::gl::glVertexT(center+d);
        }
        glEnd();
    }
}


// Register in the Factory
int TriangularAnisotropicFEMForceFieldClass = core::RegisterObject("Triangular finite element model using anisotropic material")
#ifndef SOFA_FLOAT
        .add< TriangularAnisotropicFEMForceField<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< TriangularAnisotropicFEMForceField<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class TriangularAnisotropicFEMForceField<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class TriangularAnisotropicFEMForceField<Vec3fTypes>;
#endif


} // namespace forcefield

} // namespace component

} // namespace sofa
