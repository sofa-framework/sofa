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
#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGULARANISOTROPICFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_TRIANGULARANISOTROPICFEMFORCEFIELD_INL

#include "TriangularAnisotropicFEMForceField.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/helper/gl/template.h>
#include <SofaBaseTopology/TopologyData.inl>
#include <sofa/helper/system/gl.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <vector>
#include <algorithm>
#include <sofa/defaulttype/Vec3Types.h>
#include <assert.h>

// #define DEBUG_TRIANGLEFEM

namespace sofa
{

namespace component
{

namespace forcefield
{

template <class DataTypes>
TriangularAnisotropicFEMForceField<DataTypes>::TriangularAnisotropicFEMForceField()
    : f_young2(initData(&f_young2,helper::vector<Real>(1,1000.0),"transverseYoungModulus","transverseYoungModulus","Young modulus along transverse direction"))
    , f_theta(initData(&f_theta,(Real)(0.0),"fiberAngle","Fiber angle in global reference frame (in degrees)"))
    , f_fiberCenter(initData(&f_fiberCenter,"fiberCenter","Concentric fiber center in global reference frame"))
    , showFiber(initData(&showFiber,true,"showFiber","Flag activating rendering of fiber directions within each triangle"))
    , localFiberDirection(initData(&localFiberDirection,"localFiberDirection", "Computed fibers direction within each triangle"))
{
    this->_anisotropicMaterial = true;
    triangleHandler = new TRQSTriangleHandler(this, &localFiberDirection);

    f_young2.setRequired(true);
}


template <class DataTypes>
TriangularAnisotropicFEMForceField<DataTypes>::~TriangularAnisotropicFEMForceField()
{
    if(triangleHandler) delete triangleHandler;
}

template< class DataTypes>
void TriangularAnisotropicFEMForceField<DataTypes>::TRQSTriangleHandler::applyCreateFunction(unsigned int triangleIndex, helper::vector<triangleInfo> &, const core::topology::BaseMeshTopology::Triangle &t, const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
{
    if (ff)
    {
        //const Triangle &t = ff->_topology->getTriangle(triangleIndex);
        Index a = t[0];
        Index b = t[1];
        Index c = t[2];

        switch(ff->method)
        {
        case TriangularFEMForceField<DataTypes>::SMALL :
            ff->initSmall(triangleIndex,a,b,c);
            ff->computeMaterialStiffness(triangleIndex, a, b, c);
            break;
        case TriangularFEMForceField<DataTypes>::LARGE :
            ff->initLarge(triangleIndex,a,b,c);
            ff->computeMaterialStiffness(triangleIndex, a, b, c);
            break;
        }
    }
}

template< class DataTypes>
void TriangularAnisotropicFEMForceField<DataTypes>::init()
{
    _topology = this->getContext()->getMeshTopology();

    // Create specific handler for TriangleData
    localFiberDirection.createTopologicalEngine(_topology, triangleHandler);
    localFiberDirection.registerTopologicalData();

    Inherited::init();
    reinit();
}

template <class DataTypes>
void TriangularAnisotropicFEMForceField<DataTypes>::reinit()
{
    localFiberDirection.beginEdit();
    //f_poisson2.setValue(Inherited::f_poisson.getValue()*(f_young2.getValue()/Inherited::f_young.getValue()));
    helper::vector<Real> poiss2;
    const helper::vector<Real> & youngArray = Inherited::f_young.getValue();
    const helper::vector<Real> & young2Array = f_young2.getValue();
    const helper::vector<Real> & poissonArray = Inherited::f_poisson.getValue();

    for (unsigned int i = 0; i < poissonArray.size(); i++)
    {
        poiss2.push_back( poissonArray[i]*(young2Array[i]/youngArray[i]));
    }

    f_poisson2.setValue(poiss2);

    helper::vector<Deriv>& lfd = *(localFiberDirection.beginEdit());
    lfd.resize(_topology->getNbTriangles());
    localFiberDirection.endEdit();
    Inherited::reinit();
}


template <class DataTypes>
void TriangularAnisotropicFEMForceField<DataTypes>::getFiberDir(int element, Deriv& dir)
{
    helper::vector<Deriv>& lfd = *(localFiberDirection.beginEdit());

    if ((unsigned)element < lfd.size())
    {
        const Deriv& ref = lfd[element];
        const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
        core::topology::BaseMeshTopology::Triangle t = _topology->getTriangle(element);
        dir = (x[t[1]]-x[t[0]])*ref[0] + (x[t[2]]-x[t[0]])*ref[1];
    }
    else
    {
        dir.clear();
    }
    localFiberDirection.endEdit();
}

template <class DataTypes>
void TriangularAnisotropicFEMForceField<DataTypes>::computeMaterialStiffness(int i, Index& v1, Index& v2, Index& v3)
{
    const  VecCoord& initialPoints = (this->mstate->read(core::ConstVecCoordId::restPosition())->getValue());

    Real Q11, Q12, Q22, Q66;
    Coord fiberDirGlobal;  // orientation of the fiber in the global frame of reference

    Coord fiberDirLocalOrtho; //  // orientation of the fiber in the local orthonormal frame of the element
    defaulttype::Mat<3,3,Real> T, Tinv;

    helper::vector<TriangleInformation>& triangleInf = *(Inherited::triangleInfo.beginEdit());

    TriangleInformation *tinfo = &triangleInf[i];

    //TODO(dmarchal 2017-05-03) I will remove this code soon !!!
    /*Q11 = Inherited::f_young.getValue()/(1-Inherited::f_poisson.getValue()*f_poisson2.getValue());
    Q12 = Inherited::f_poisson.getValue()*f_young2.getValue()/(1-Inherited::f_poisson.getValue()*f_poisson2.getValue());
    Q22 = f_young2.getValue()/(1-Inherited::f_poisson.getValue()*f_poisson2.getValue());
    Q66 = (Real)(Inherited::f_young.getValue() / (2.0*(1 + Inherited::f_poisson.getValue())));*/

    const helper::vector<Real> & youngArray = Inherited::f_young.getValue();
    const helper::vector<Real> & young2Array = f_young2.getValue();
    const helper::vector<Real> & poissonArray = Inherited::f_poisson.getValue();
    const helper::vector<Real> & poisson2Array = f_poisson2.getValue();

    unsigned int index = 0;
    if (i < (int) youngArray.size() )
        index = i;

    Q11 = youngArray[index] /(1-poissonArray[index]*poisson2Array[index]);
    Q12 = poissonArray[index]*young2Array[index]/(1-poissonArray[index]*poisson2Array[index]);
    Q22 = young2Array[index]/(1-poissonArray[index]*poisson2Array[index]);
    Q66 = (Real)(youngArray[index] / (2.0*(1 + poissonArray[index])));

    T[0] = (initialPoints)[v2]-(initialPoints)[v1];
    T[1] = (initialPoints)[v3]-(initialPoints)[v1];
    T[2] = cross(T[0], T[1]);

    if (T[2] == Coord())
    {
        msg_error() << "Cannot compute material stiffness for a flat triangle. Abort computation. ";
        return;
    }

    if (!f_fiberCenter.getValue().empty()) // in case we have concentric fibers
    {
        Coord tcenter = ((initialPoints)[v1]+(initialPoints)[v2]+(initialPoints)[v3])*(Real)(1.0/3.0);
        Coord fcenter = f_fiberCenter.getValue()[0];
        fiberDirGlobal = cross(T[2], fcenter-tcenter);  // was fiberDir
    }
    else // for unidirectional fibers
    {
        double theta = (double)f_theta.getValue()*M_PI/180.0;
        fiberDirGlobal = Coord((Real)cos(theta), (Real)sin(theta), 0); // was fiberDir
    }

    helper::vector<Deriv>& lfd = *(localFiberDirection.beginEdit());

    if ((unsigned int)i >= lfd.size())
    {
        /* ********************************************************************************************
         * this can happen after topology changes
         * apparently, the topological changes are not propagated through localFiberDirection
         * that's why we resize this vector to triangleInf size to hack the crash when we're looking for
         * a element which index is more than the size
         * This hack is probably useless if there would be a good topological propagation
        ***********************************************************************************************/
        dmsg_warning() << "Get an element in localFiberDirection with index more than its size: i=" << i
                       << " and size=" << lfd.size() << ". The size should be "  <<  triangleInf.size() <<" (see comments in TriangularAnisotropicFEMForceField::computeMaterialStiffness)" ;
        lfd.resize(triangleInf.size() );
        dmsg_info() << "LocalFiberDirection resized to " << lfd.size() ;
    }
    else
    {
        Deriv& fiberDirLocal = lfd[i]; // orientation of the fiber in the local frame of the element (orthonormal frame)
        T.transpose();
        Tinv.invert(T);
        fiberDirLocal = Tinv * fiberDirGlobal;
        fiberDirLocal[2] = 0;
        fiberDirLocal.normalize();
    }

    T[0] = (initialPoints)[v2]-(initialPoints)[v1];
    T[1] = (initialPoints)[v3]-(initialPoints)[v1];
    T[2] = cross(T[0], T[1]);
    T[1] = cross(T[2], T[0]);
    T[0].normalize();
    T[1].normalize();
    T[2].normalize();
    T.transpose();
    Tinv.invert(T);
    fiberDirLocalOrtho = Tinv * fiberDirGlobal;
    fiberDirLocalOrtho[2] = 0;
    fiberDirLocalOrtho.normalize();

    Real c, s, c2, s2, c3, s3,c4, s4;
    c = fiberDirLocalOrtho[0];
    s = fiberDirLocalOrtho[1];

    c2 = c*c;
    s2 = s*s;
    c3 = c2*c;
    s3 = s2*s;
    s4 = s2*s2;
    c4 = c2*c2;

    // K(1,1)=Q11*COS(THETA)^4 * + 2.0*(Q12+2*Q66)*SIN(THETA)^2*COS(THETA)^2 + Q22*SIN(THETA)^4 => c4*Q11+2*s2*c2*(Q12+2*Q66)+s4*Q22
    // K(1,2)=(Q11+Q22-4*Q66)*SIN(THETA)^2*COS(THETA)^2 + Q12*(SIN(THETA)^4+COS(THETA)^4) => s2*c2*(Q11+Q22-4*Q66) + (s4+c4)*Q12
    // K(2,1)=K(1,2)
    // K(2,2)=Q11*SIN(THETA)^4 + 2.0*(Q12+2*Q66)*SIN(THETA)^2*COS(THETA)^2 + Q22*COS(THETA)^4 => s4*Q11 + 2.0*s2*c2*(Q12+2*Q66) + c4*Q22
    // K(6,1)=(Q11-Q12-2*Q66)*SIN(THETA)*COS(THETA)^3 + (Q12-Q22+2*Q66)*SIN(THETA)^3*COS(THETA) => s*c3*(Q11-Q12-2*Q66)+s3*c*(Q12-Q22+2*Q66)
    // K(1,6)=K(6,1)
    // K(6,2)=(Q11-Q12-2*Q66)*SIN(THETA)^3*COS(THETA)+(Q12-Q22+2*Q66)*SIN(THETA)*COS(THETA)^3 => (Q11-Q12-2*Q66)*s3*c + (Q12-Q22+2*Q66)*s*c3
    // K(2,6)=K(6,2)
    // K(6,6)=(Q11+Q22-2*Q12-2*Q66)*SIN(THETA)^2 * COS(THETA)^2+ Q66*(SIN(THETA)^4+COS(THETA)^4) => (Q11+Q22-2*Q12-2*Q66)*s2*c2+ Q66*(s4+c4)

    Real K11= c4 * Q11 + 2 * c2 * s2 * (Q12+2*Q66) + s4 * Q22;
    Real K12 = c2 * s2 * (Q11+Q22-4*Q66) + (c4+s4) * Q12;
    Real K22 = s4* Q11 + 2 * c2 * s2 * (Q12+2*Q66) + c4 * Q22;
    Real K16 = s * c3 * (Q11-Q12-2*Q66) + s3 * c * (Q12-Q22+2*Q66);
    Real K26 = s3 * c * (Q11-Q12-2*Q66) + s * c3 * (Q12-Q22+2*Q66);
    Real K66 = c2 * s2 * (Q11+Q22-2*Q12-2*Q66) + (c4+s4) * Q66;

    tinfo->materialMatrix[0][0] = K11;
    tinfo->materialMatrix[0][1] = K12;
    tinfo->materialMatrix[0][2] = K16;
    tinfo->materialMatrix[1][0] = K12;
    tinfo->materialMatrix[1][1] = K22;
    tinfo->materialMatrix[1][2] = K26;
    tinfo->materialMatrix[2][0] = K16;
    tinfo->materialMatrix[2][1] = K26;
    tinfo->materialMatrix[2][2] = K66;

    localFiberDirection.endEdit();
    Inherited::triangleInfo.endEdit();
}

// ----------------------------------------------------------------
// ---	Display
// ----------------------------------------------------------------
template <class DataTypes>
void TriangularAnisotropicFEMForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    glPolygonOffset(1.0, 2.0);
    glEnable(GL_POLYGON_OFFSET_FILL);
    Inherited::draw(vparams);
    glDisable(GL_POLYGON_OFFSET_FILL);
    if (!vparams->displayFlags().getShowForceFields())
        return;

    helper::vector<Deriv>& lfd = *(localFiberDirection.beginEdit());

    if (showFiber.getValue() && lfd.size() >= (unsigned)_topology->getNbTriangles())
    {
        const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
        int nbTriangles=_topology->getNbTriangles();
        glColor3f(0,0,0);
        glBegin(GL_LINES);

        for(int i=0; i<nbTriangles; ++i)
        {

            if ( (unsigned int)i < lfd.size())
            {
                Index a = _topology->getTriangle(i)[0];
                Index b = _topology->getTriangle(i)[1];
                Index c = _topology->getTriangle(i)[2];

                Coord center = (x[a]+x[b]+x[c])/3;
                Coord d = (x[b]-x[a])*lfd[i][0] + (x[c]-x[a])*lfd[i][1];
                d*=0.25;
                helper::gl::glVertexT(center-d);
                helper::gl::glVertexT(center+d);
            }
        }
        glEnd();
    }
    localFiberDirection.endEdit();
#endif /* SOFA_NO_OPENGL */
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_TRIANGULARANISOTROPICFEMFORCEFIELD_INL
