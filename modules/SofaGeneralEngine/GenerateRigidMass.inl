/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ENGINE_GENERATERIGIDMASS_INL
#define SOFA_COMPONENT_ENGINE_GENERATERIGIDMASS_INL

#include <SofaGeneralEngine/GenerateRigidMass.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>

#include <iostream>
#include <fstream>

namespace sofa
{

namespace component
{

namespace engine
{


template <class DataTypes, class MassType>
GenerateRigidMass<DataTypes,MassType>::GenerateRigidMass()
    : m_density(  initData(&m_density,static_cast<Real>(1000.0),"density","input: Density of the object") )
    , m_positions(  initData(&m_positions,"position","input: positions of the vertices") )
    , m_triangles(  initData(&m_triangles,"triangles","input: triangles of the mesh") )
    , m_quads(  initData(&m_quads,"quads","input: quads of the mesh") )
    , m_polygons(  initData(&m_polygons,"polygons","input: polygons of the mesh") )
    , rigidMass(  initData(&rigidMass,"rigidMass","output: rigid mass computed") )
    , mass(  initData(&mass,"mass","output: mass of the mesh") )
    , volume(  initData(&volume,"volume","output: volume of the mesh") )
    , inertiaMatrix(  initData(&inertiaMatrix,"inertiaMatrix","output: the inertia matrix of the mesh") )
    , massCenter(  initData(&massCenter,"massCenter","output: the gravity center of the mesh") )
    , centerToOrigin(  initData(&centerToOrigin,"centerToOrigin","output: vector going from the mass center to the space origin") )
{
}

template <class DataTypes, class MassType>
GenerateRigidMass<DataTypes,MassType>::~GenerateRigidMass()
{
}

template <class DataTypes, class MassType>
void  GenerateRigidMass<DataTypes, MassType>::init()
{
    addInput(&m_density);
    addInput(&m_positions);
    addInput(&m_triangles);
    addInput(&m_quads);
    addInput(&m_polygons);

    addOutput(&rigidMass);
    addOutput(&mass);
    addOutput(&volume);
    addOutput(&inertiaMatrix);
    addOutput(&massCenter);
    addOutput(&centerToOrigin);

    setDirtyValue();

    reinit();
}

template <class DataTypes, class MassType>
void GenerateRigidMass<DataTypes, MassType>::reinit()
{
    update();
}

template <class DataTypes, class MassType>
void GenerateRigidMass<DataTypes, MassType>::update()
{
    integrateMesh();
    cleanDirty();
    generateRigid();
}

template <class DataTypes, class MassType>
void GenerateRigidMass<DataTypes, MassType>::integrateMesh()
{
    for (size_t i=0 ; i<10 ; ++i)
        afIntegral[i] = 0.0;

    const helper::vector<Vector3>& positions = m_positions.getValue();
    const helper::vector<MTriangle>& triangles = m_triangles.getValue();
    const helper::vector<MQuad>& quads = m_quads.getValue();
    const helper::vector<MPolygon>& polygons = m_polygons.getValue();

    // Triangles integration
    for (size_t i=0; i<triangles.size() ; ++i)
    {
        MTriangle triangle = triangles[i];
        integrateTriangle(positions[triangle[0]], positions[triangle[1]], positions[triangle[2]]);
    }
    // Quads integration
    for (size_t i=0; i<quads.size() ; ++i)
    {
        const MQuad& quad = quads[i];
        integrateTriangle(positions[quad[0]], positions[quad[1]], positions[quad[2]]);
        integrateTriangle(positions[quad[0]], positions[quad[2]], positions[quad[3]]);
    }
    // Polygons integration
    for (size_t i=0; i<polygons.size() ; ++i)
    {
        const MPolygon& facet = polygons[i];
        for (size_t j = 2; j < facet.size(); j++)
        {
            integrateTriangle(positions[facet[0]], positions[facet[j-1]], positions[facet[j]]);
        }
    }

    afIntegral[0] /= 6.0;
    afIntegral[1] /= 24.0;
    afIntegral[2] /= 24.0;
    afIntegral[3] /= 24.0;
    afIntegral[4] /= 60.0;
    afIntegral[5] /= 60.0;
    afIntegral[6] /= 60.0;
    afIntegral[7] /= 120.0;
    afIntegral[8] /= 120.0;
    afIntegral[9] /= 120.0;
}

template <class DataTypes, class MassType>
void GenerateRigidMass<DataTypes, MassType>::integrateTriangle(Vector3 kV0,Vector3 kV1,Vector3 kV2)
{
    // order:	1, x, y, z, x^2, y^2, z^2, xy, yz, zx

    // get cross product of edges
    Vector3 kV1mV0 = kV1 - kV0;
    Vector3 kV2mV0 = kV2 - kV0;
    Vector3 kN = cross(kV1mV0,kV2mV0);

    // compute integral terms
    SReal fTmp0, fTmp1, fTmp2;
    SReal fF1x, fF2x, fF3x, fG0x, fG1x, fG2x;
    fTmp0 = kV0[0] + kV1[0];
    fF1x = fTmp0 + kV2[0];
    fTmp1 = kV0[0]*kV0[0];
    fTmp2 = fTmp1 + kV1[0]*fTmp0;
    fF2x = fTmp2 + kV2[0]*fF1x;
    fF3x = kV0[0]*fTmp1 + kV1[0]*fTmp2 + kV2[0]*fF2x;
    fG0x = fF2x + kV0[0]*(fF1x + kV0[0]);
    fG1x = fF2x + kV1[0]*(fF1x + kV1[0]);
    fG2x = fF2x + kV2[0]*(fF1x + kV2[0]);

    SReal fF1y, fF2y, fF3y, fG0y, fG1y, fG2y;
    fTmp0 = kV0[1] + kV1[1];
    fF1y = fTmp0 + kV2[1];
    fTmp1 = kV0[1]*kV0[1];
    fTmp2 = fTmp1 + kV1[1]*fTmp0;
    fF2y = fTmp2 + kV2[1]*fF1y;
    fF3y = kV0[1]*fTmp1 + kV1[1]*fTmp2 + kV2[1]*fF2y;
    fG0y = fF2y + kV0[1]*(fF1y + kV0[1]);
    fG1y = fF2y + kV1[1]*(fF1y + kV1[1]);
    fG2y = fF2y + kV2[1]*(fF1y + kV2[1]);

    SReal fF1z, fF2z, fF3z, fG0z, fG1z, fG2z;
    fTmp0 = kV0[2] + kV1[2];
    fF1z = fTmp0 + kV2[2];
    fTmp1 = kV0[2]*kV0[2];
    fTmp2 = fTmp1 + kV1[2]*fTmp0;
    fF2z = fTmp2 + kV2[2]*fF1z;
    fF3z = kV0[2]*fTmp1 + kV1[2]*fTmp2 + kV2[2]*fF2z;
    fG0z = fF2z + kV0[2]*(fF1z + kV0[2]);
    fG1z = fF2z + kV1[2]*(fF1z + kV1[2]);
    fG2z = fF2z + kV2[2]*(fF1z + kV2[2]);

    // update integrals
    afIntegral[0] += kN[0]*fF1x;
    afIntegral[1] += kN[0]*fF2x;
    afIntegral[2] += kN[1]*fF2y;
    afIntegral[3] += kN[2]*fF2z;
    afIntegral[4] += kN[0]*fF3x;
    afIntegral[5] += kN[1]*fF3y;
    afIntegral[6] += kN[2]*fF3z;
    afIntegral[7] += kN[0]*(kV0[1]*fG0x + kV1[1]*fG1x + kV2[1]*fG2x);
    afIntegral[8] += kN[1]*(kV0[2]*fG0y + kV1[2]*fG1y + kV2[2]*fG2y);
    afIntegral[9] += kN[2]*(kV0[0]*fG0z + kV1[0]*fG1z + kV2[0]*fG2z);
}

template <class DataTypes, class MassType>
void GenerateRigidMass<DataTypes, MassType>::generateRigid()
{
    SReal volume = afIntegral[0];
    if (volume == 0.0)
    {
        msg_warning() << "Mesh volume is nul." ;
        return;
    }

    msg_warning_when(volume < 0.0) << "Mesh volume is negative." ;

    MassType *rigidmass = this->rigidMass.beginWriteOnly();

    // volume
    rigidmass->volume = static_cast<Real>( volume );
    // mass
    rigidmass->mass = static_cast<Real>( volume );
    // center of mass
    Vector3 center(afIntegral[1]/afIntegral[0],afIntegral[2]/afIntegral[0],afIntegral[3]/afIntegral[0]);

    // inertia relative to world origin
    rigidmass->inertiaMatrix[0][0] = static_cast<Real>(  afIntegral[5] + afIntegral[6] );
    rigidmass->inertiaMatrix[0][1] = static_cast<Real>( -afIntegral[7] );
    rigidmass->inertiaMatrix[0][2] = static_cast<Real>( -afIntegral[9] );
    rigidmass->inertiaMatrix[1][0] = static_cast<Real>( rigidmass->inertiaMatrix[0][1] );
    rigidmass->inertiaMatrix[1][1] = static_cast<Real>( afIntegral[4] + afIntegral[6] );
    rigidmass->inertiaMatrix[1][2] = static_cast<Real>( -afIntegral[8] );
    rigidmass->inertiaMatrix[2][0] = static_cast<Real>( rigidmass->inertiaMatrix[0][2] );
    rigidmass->inertiaMatrix[2][1] = static_cast<Real>( rigidmass->inertiaMatrix[1][2] );
    rigidmass->inertiaMatrix[2][2] = static_cast<Real>( afIntegral[4] + afIntegral[5] );

    // inertia relative to center of mass
    rigidmass->inertiaMatrix[0][0] -= static_cast<Real>( volume*(center[1]*center[1] + center[2]*center[2]) );
    rigidmass->inertiaMatrix[0][1] += static_cast<Real>( volume*center[0]*center[1] );
    rigidmass->inertiaMatrix[0][2] += static_cast<Real>( volume*center[2]*center[0] );
    rigidmass->inertiaMatrix[1][0] =  static_cast<Real>( rigidmass->inertiaMatrix[0][1] );
    rigidmass->inertiaMatrix[1][1] -= static_cast<Real>( volume*(center[2]*center[2] + center[0]*center[0]) );
    rigidmass->inertiaMatrix[1][2] += static_cast<Real>( volume*center[1]*center[2] );
    rigidmass->inertiaMatrix[2][0] =  static_cast<Real>( rigidmass->inertiaMatrix[0][2] );
    rigidmass->inertiaMatrix[2][1] =  static_cast<Real>( rigidmass->inertiaMatrix[1][2] );
    rigidmass->inertiaMatrix[2][2] -= static_cast<Real>( volume*(center[0]*center[0] + center[1]*center[1]) );

    rigidmass->inertiaMatrix /= static_cast<Real>( volume );
    rigidmass->recalc();

    *rigidmass *= m_density.getValue();

    // Data updating
    this->mass.setValue(rigidmass->mass);
    this->volume.setValue(rigidmass->volume);
    this->inertiaMatrix.setValue(rigidmass->inertiaMatrix);

    this->rigidMass.endEdit();

    this->massCenter.setValue(center);
    this->centerToOrigin.setValue(-center);
}

template <class DataTypes, class MassType>
std::string GenerateRigidMass<DataTypes, MassType>::getTemplateName() const
{
    return templateName(this);
}

template <class DataTypes, class MassType>
std::string GenerateRigidMass<DataTypes, MassType>::templateName(const GenerateRigidMass<DataTypes, MassType>*)
{
    return DataTypes::Name();
}


} // namespace engine

} // namespace component

} // namespace sofa

#endif
