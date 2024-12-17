/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/component/solidmechanics/fem/elastic/TriangularAnisotropicFEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/TriangularFEMForceField.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/topology/TopologyData.inl>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <vector>
#include <algorithm>
#include <sofa/defaulttype/VecTypes.h>
#include <cassert>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes>
TriangularAnisotropicFEMForceField<DataTypes>::TriangularAnisotropicFEMForceField()
    : d_young2(initData(&d_young2, type::vector<Real>(1, 1000.0), "transverseYoungModulus", "transverseYoungModulus", "Young modulus along transverse direction"))
    , d_theta(initData(&d_theta, (Real)(0.0), "fiberAngle", "Fiber angle in global reference frame (in degrees)"))
    , d_fiberCenter(initData(&d_fiberCenter, "fiberCenter", "Concentric fiber center in global reference frame"))
    , d_showFiber(initData(&d_showFiber, true, "showFiber", "Flag activating rendering of fiber directions within each triangle"))
    , d_localFiberDirection(initData(&d_localFiberDirection, "localFiberDirection", "Computed fibers direction within each triangle"))
{
    this->_anisotropicMaterial = true;

    d_young2.setRequired(true);

    f_young2.setOriginalData(&d_young2);
    f_theta.setOriginalData(&d_theta);
    f_fiberCenter.setOriginalData(&d_fiberCenter);
    showFiber.setOriginalData(&d_showFiber);
    localFiberDirection.setOriginalData(&d_localFiberDirection);

}


template <class DataTypes>
TriangularAnisotropicFEMForceField<DataTypes>::~TriangularAnisotropicFEMForceField()
{

}

template< class DataTypes>
void TriangularAnisotropicFEMForceField<DataTypes>::createTriangleInfo(Index triangleIndex, TriangleFiberDirection&, const core::topology::BaseMeshTopology::Triangle &t, const sofa::type::vector<unsigned int> &, const sofa::type::vector<SReal> &)
{
    Index a = t[0];
    Index b = t[1];
    Index c = t[2];

    switch(this->method)
    {
    case TriangularFEMForceField<DataTypes>::SMALL :
        this->initSmall(triangleIndex,a,b,c);
        computeMaterialStiffness(triangleIndex, a, b, c);
        break;
    case TriangularFEMForceField<DataTypes>::LARGE :
        this->initLarge(triangleIndex,a,b,c);
        computeMaterialStiffness(triangleIndex, a, b, c);
        break;
    }
}

template< class DataTypes>
void TriangularAnisotropicFEMForceField<DataTypes>::init()
{
    Inherited::init();

    if (this->d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Invalid)
    {
        return;
    }

    // Create specific handler for TriangleData
    d_localFiberDirection.createTopologyHandler(this->l_topology);
    d_localFiberDirection.setCreationCallback([this](Index triangleIndex, TriangleFiberDirection& triInfo,
                                                     const core::topology::BaseMeshTopology::Triangle& t,
                                                     const sofa::type::vector< Index >& ancestors,
                                                     const sofa::type::vector< SReal >& coefs)
    {
        createTriangleInfo(triangleIndex, triInfo, t, ancestors, coefs);
    });


    reinit();
}

template <class DataTypes>
void TriangularAnisotropicFEMForceField<DataTypes>::reinit()
{
    d_localFiberDirection.beginEdit();
    //f_poisson2.setValue(Inherited::d_poisson.getValue()*(d_young2.getValue()/Inherited::d_young.getValue()));
    type::vector<Real> poiss2;
    const type::vector<Real> & young2Array = d_young2.getValue();

    for (unsigned int i = 0; i < this->l_topology->getNbTriangles(); i++)
    {
        const auto elementYoungModulus = this->getYoungModulusInElement(i);
        const auto elementPoissonRatio = this->getPoissonRatioInElement(i);
        poiss2.push_back( elementPoissonRatio*(young2Array[i]/elementYoungModulus));
    }

    f_poisson2.setValue(poiss2);

    type::vector<Deriv>& lfd = *(d_localFiberDirection.beginEdit());
    lfd.resize(this->l_topology->getNbTriangles());
    d_localFiberDirection.endEdit();
    Inherited::reinit();
}


template <class DataTypes>
void TriangularAnisotropicFEMForceField<DataTypes>::getFiberDir(int element, Deriv& dir)
{
    type::vector<Deriv>& lfd = *(d_localFiberDirection.beginEdit());

    if ((unsigned)element < lfd.size())
    {
        const Deriv& ref = lfd[element];
        const VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();
        core::topology::BaseMeshTopology::Triangle t = this->l_topology->getTriangle(element);
        dir = (x[t[1]]-x[t[0]])*ref[0] + (x[t[2]]-x[t[0]])*ref[1];
    }
    else
    {
        dir.clear();
    }
    d_localFiberDirection.endEdit();
}

template <class DataTypes>
void TriangularAnisotropicFEMForceField<DataTypes>::computeMaterialStiffness(int i, Index& v1, Index& v2, Index& v3)
{
    const  VecCoord& initialPoints = (this->mstate->read(core::vec_id::read_access::restPosition)->getValue());

    Real Q11, Q12, Q22, Q66;
    Coord fiberDirGlobal;  // orientation of the fiber in the global frame of reference

    Coord fiberDirLocalOrtho; //  // orientation of the fiber in the local orthonormal frame of the element
    type::Mat<3,3,Real> T, Tinv;

    type::vector<TriangleInformation>& triangleInf = *(Inherited::d_triangleInfo.beginEdit());

    TriangleInformation *tinfo = &triangleInf[i];

    //TODO(dmarchal 2017-05-03) I will remove this code soon !!!
    /*Q11 = Inherited::d_young.getValue()/(1-Inherited::d_poisson.getValue()*f_poisson2.getValue());
    Q12 = Inherited::d_poisson.getValue()*d_young2.getValue()/(1-Inherited::d_poisson.getValue()*f_poisson2.getValue());
    Q22 = d_young2.getValue()/(1-Inherited::d_poisson.getValue()*f_poisson2.getValue());
    Q66 = (Real)(Inherited::d_young.getValue() / (2.0*(1 + Inherited::d_poisson.getValue())));*/

    const type::vector<Real> & young2Array = d_young2.getValue();
    const type::vector<Real> & poisson2Array = f_poisson2.getValue();

    unsigned int index = 0;
    if (i < (int) young2Array.size() )
        index = i;

    const auto elementYoungModulus = this->getYoungModulusInElement(i);
    const auto elementPoissonRatio = this->getPoissonRatioInElement(i);

    Q11 = elementYoungModulus /(1-elementPoissonRatio*poisson2Array[index]);
    Q12 = elementPoissonRatio*young2Array[index]/(1-elementPoissonRatio*poisson2Array[index]);
    Q22 = young2Array[index]/(1-elementPoissonRatio*poisson2Array[index]);
    Q66 = (Real)(elementYoungModulus / (2.0*(1 + elementPoissonRatio)));

    T[0] = (initialPoints)[v2]-(initialPoints)[v1];
    T[1] = (initialPoints)[v3]-(initialPoints)[v1];
    T[2] = cross(T[0], T[1]);

    if (T[2] == Coord())
    {
        msg_error() << "Cannot compute material stiffness for a flat triangle. Abort computation. ";
        return;
    }

    if (!d_fiberCenter.getValue().empty()) // in case we have concentric fibers
    {
        Coord tcenter = ((initialPoints)[v1]+(initialPoints)[v2]+(initialPoints)[v3])*(Real)(1.0/3.0);
        Coord fcenter = d_fiberCenter.getValue()[0];
        fiberDirGlobal = cross(T[2], fcenter-tcenter);  // was fiberDir
    }
    else // for unidirectional fibers
    {
        const double theta = (double)d_theta.getValue() * M_PI / 180.0;
        fiberDirGlobal = Coord((Real)cos(theta), (Real)sin(theta), 0); // was fiberDir
    }

    type::vector<Deriv>& lfd = *(d_localFiberDirection.beginEdit());

    if ((unsigned int)i >= lfd.size())
    {
        /* ********************************************************************************************
         * this can happen after topology changes
         * apparently, the topological changes are not propagated through d_localFiberDirection
         * that's why we resize this vector to triangleInf size to hack the crash when we're looking for
         * a element which index is more than the size
         * This hack is probably useless if there would be a good topological propagation
        ***********************************************************************************************/
        dmsg_warning() << "Get an element in d_localFiberDirection with index more than its size: i=" << i
                       << " and size=" << lfd.size() << ". The size should be "  <<  triangleInf.size() <<" (see comments in TriangularAnisotropicFEMForceField::computeMaterialStiffness)" ;
        lfd.resize(triangleInf.size() );
        dmsg_info() << "LocalFiberDirection resized to " << lfd.size() ;
    }
    else
    {
        Deriv& fiberDirLocal = lfd[i]; // orientation of the fiber in the local frame of the element (orthonormal frame)
        T.transpose();
        const bool canInvert = Tinv.invert(T);
        assert(canInvert);
        SOFA_UNUSED(canInvert);
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
    const bool canInvert = Tinv.invert(T);
    assert(canInvert);
    SOFA_UNUSED(canInvert);
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

    d_localFiberDirection.endEdit();
    Inherited::d_triangleInfo.endEdit();
}

// ----------------------------------------------------------------
// ---	Display
// ----------------------------------------------------------------
template <class DataTypes>
void TriangularAnisotropicFEMForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    Inherited::draw(vparams);

    if (!vparams->displayFlags().getShowForceFields())
        return;

    type::vector<Deriv>& lfd = *(d_localFiberDirection.beginEdit());

    if (d_showFiber.getValue() && lfd.size() >= (unsigned)this->l_topology->getNbTriangles())
    {
        const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
        constexpr sofa::type::RGBAColor color = sofa::type::RGBAColor::black();
        std::vector<sofa::type::Vec3> vertices;

        const VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();
        const int nbTriangles=this->l_topology->getNbTriangles();

        for(int i=0; i<nbTriangles; ++i)
        {

            if ( (unsigned int)i < lfd.size())
            {
                Index a = this->l_topology->getTriangle(i)[0];
                Index b = this->l_topology->getTriangle(i)[1];
                Index c = this->l_topology->getTriangle(i)[2];

                Coord center = (x[a]+x[b]+x[c])/3;
                Coord d = (x[b]-x[a])*lfd[i][0] + (x[c]-x[a])*lfd[i][1];
                d*=0.25;
                vertices.push_back(sofa::type::Vec3(center-d));
                vertices.push_back(sofa::type::Vec3(center+d));
            }
        }
        vparams->drawTool()->drawLines(vertices,1,color);

    }
    d_localFiberDirection.endEdit();
}

} // namespace sofa::component::solidmechanics::fem::elastic
