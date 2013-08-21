/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
 *                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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

#include "INPExporter.h"

#include <sstream>
#include <string>
#include <stdlib.h>
#include <stdio.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/simulation/common/ExportINPVisitor.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>


namespace sofa
{
    
namespace component
{

namespace misc
{

SOFA_DECL_CLASS(INPExporter)

int INPExporterClass = core::RegisterObject("Read State vectors from file")
    .add< INPExporter >();

INPExporter::INPExporter(): BaseExporter()
    , m_position( initData(&m_position, "position", "points coordinates"))
    /*
    *    , m_triangle( initData(&m_triangle, "triangle", "triangle of the object"))
    *    , m_quad( initData(&quad, "quad", "quads of the object"))
    */
    , m_density( initData(&m_density, "density", "density of the object (kg.m-3)"))
    , m_youngModulus( initData(&m_youngModulus, "youngModulus", "Young's Modulus"))
    , m_poissonRatio( initData(&m_poissonRatio, "poissonRatio", "Poisson's Ration"))
    , m_tetrahedra( initData(&m_tetrahedra, "tetrahedra", "tetrahedra indices"))
    , m_hexahedra( initData(&m_hexahedra, "hexahedra", "hexahedra indices"))
    , m_fixedPoints( initData(&m_fixedPoints, "fixedPoints", "fixed points indices"))
{
}

INPExporter::~INPExporter()
{
}

void INPExporter::init()
{
    context = this->getContext();
    context->get(topology); // Topology
    context->get(mstate); // Mechanical state
    context->get(bcondition); // Fixed constraints
    context->get(bmass); // Mass
    context->get(hexaForceField); // HexahedronFEMForceField
    context->get(tetraForceField); // TetrahedronFEMForceField
    
    if(!topology)
    {
        serr << "INPExporter : error, no topology." << sendl;
        return;
    }
    
    // Test if the position has not been modified
    if(!m_position.isSet())
    {        
        sofa::core::objectmodel::BaseData* nam = NULL;
        sofa::core::objectmodel::BaseData* pos = NULL;
        sofa::core::objectmodel::BaseData* tri = NULL;
        sofa::core::objectmodel::BaseData* qua = NULL;
        sofa::core::objectmodel::BaseData* bm = NULL;
        sofa::core::objectmodel::BaseData* bden = NULL;
        sofa::core::objectmodel::BaseData* tetra = NULL;
        sofa::core::objectmodel::BaseData* hexa = NULL;
        sofa::core::objectmodel::BaseData* fpoints = NULL;
        sofa::core::objectmodel::BaseData* ymodul = NULL;
        sofa::core::objectmodel::BaseData* pratio = NULL;
        
        nam = context->findField("name");
        
        pos = topology->findField("position");
        if(!pos)
        {
            serr << "INPExporter : error, missing positions in topology" << sendl;
            return;
        }
        tri = topology->findField("triangles");
        qua = topology->findField("quads");
        if(!tri && !qua)
        {
            serr << "INPExporter : error, neither triangles nor quads in topology" << sendl;
            return;
        }
        tetra = topology->findField("tetrahedra");
    hexa = topology->findField("hexahedra");
    if(!tetra && !hexa)
    {
        serr << "INPExporter : error, neither tetrahedra nor hexahedra in topology" << sendl;
        return;
    }
    
    if(bmass)
    {
        bm = bmass->findField("totalmass"); // looking for mass in mass component
        bden = bmass->findField("massDensity"); //looking for density in mass component
        if(!bden)
            bden = bmass->findField("density"); // looking for density in other ForceFieldAndMass component
            if(!bden)
            {
                if(!bm)
                {
                    serr << "INPExporter : error, missing mass" << sendl;
                    return;
                }
            }
    }
    else
    {
        serr << "INPExporter : error, no mass found or not supported yet" << sendl; // TODO GenerateRigidMass
        return;
    }
    
    if(hexaForceField)
    {
        ymodul = hexaForceField->findField("youngModulus");
        pratio = hexaForceField->findField("poissonRatio");
    }
    else if(tetraForceField)
    {
        ymodul = tetraForceField->findField("youngModulus");
        pratio = tetraForceField->findField("poissonRatio");
    }
    
    if(bcondition)
    {
        fpoints = bcondition->findField("indices");
        if(!fpoints)
        {
            serr << "INPExporter : error, missing fixed points" << sendl;
            return;
        }
    }
    
    m_name.setParent(nam);
    m_position.setParent(pos);
    if(hexa)
        m_hexahedra.setParent(hexa);
    if(tetra)
        m_tetrahedra.setParent(tetra);
    if(ymodul)
        m_youngModulus.setParent(ymodul);
    if(pratio)
        m_poissonRatio.setParent(pratio);
    if(fpoints)
        m_fixedPoints.setParent(fpoints);
    m_triangle.setParent(tri);
    m_quad.setParent(qua);
    m_baseMass.setParent(bm);
    m_baseDensity.setParent(bden);
    }
    
}

void INPExporter::bwdInit()
{
    if(!m_baseDensity.isSet())
    {
        m_density.setValue(computeDensity(m_triangle.getValue(), m_quad.getValue(), m_position.getValue(), m_baseMass.getValue()));
    }
    else if(m_baseDensity.isSet())
    {
        m_density.setValue(m_baseDensity.getValue());
    }
}

bool INPExporter::getINP(vector< std::string >* nameT, vector< defaulttype::Vec3Types::VecCoord >* positionT, vector< double >* densiT, vector< vector< sofa::component::topology::Tetrahedron > >* tetrahedraT, vector< vector< sofa::component::topology::Hexahedron > >* hexahedraT, vector< vector< unsigned int > >* fixedPointT, vector< double >* youngModulusT, vector< double >* poissonRatioT)
{
    helper::ReadAccessor<Data< std::string > > nodeName = m_name;
    helper::ReadAccessor<Data< defaulttype::Vec3Types::VecCoord > > pointsIndices = m_position;
    helper::ReadAccessor<Data< double > > densityIndice = m_density;
    helper::ReadAccessor<Data< vector< sofa::component::topology::Tetrahedron > > > tetraIndices = m_tetrahedra;
    helper::ReadAccessor<Data< vector< sofa::component::topology::Hexahedron > > > hexaIndices = m_hexahedra;
    helper::ReadAccessor<Data< vector< unsigned int > > > boundIndices = m_fixedPoints;
    helper::ReadAccessor<Data< double > > ymIndice = m_youngModulus;
    helper::ReadAccessor<Data< double > > prIndice = m_poissonRatio;
    
    if( nodeName.ref().empty() || pointsIndices.ref().empty() || (tetraIndices.ref().empty() && hexaIndices.ref().empty()) )
        return false;
    
    nameT->push_back(nodeName.ref());
    positionT->push_back(pointsIndices.ref());
    densiT->push_back(densityIndice.ref());
    tetrahedraT->push_back(tetraIndices.ref());
    hexahedraT->push_back(hexaIndices.ref());
    fixedPointT->push_back(boundIndices.ref());
    youngModulusT->push_back(ymIndice.ref());
    poissonRatioT->push_back(prIndice.ref());
    
    return true;
}

float INPExporter::computeDensity(vector< core::topology::Triangle > triangles, vector< core::topology::Quad > quads, vector< defaulttype::Vec3d > positions, double mass)
{
    // total volume is computed using Green - Ostrogradsky theorem
    float totalVolume = float(.0f);
    float dens = float(.0f);
    // first iterate on triangle
    for(unsigned int t = 0 ; t < triangles.size() ; ++t)
    {
        core::topology::Triangle tri = triangles[t];
        defaulttype::Vec3Types::Coord v0, v1, v2;
        
        v0.set((float)positions[tri[0]][2], (float)positions[tri[0]][1], (float)positions[tri[0]][0]);
        v1.set((float)positions[tri[1]][2], (float)positions[tri[1]][1], (float)positions[tri[1]][0]);
        v2.set((float)positions[tri[2]][2], (float)positions[tri[2]][1], (float)positions[tri[2]][0]);
        
        totalVolume += ( ((v1[1] - v0[1]) * (v2[2] - v0[2])) - ((v1[2]-v0[2])*(v2[1]-v0[1])) )*(v0[0]+v1[0]+v2[0]);
    }
    
    // second iterate on quad and convert them to triangle (caution : triangle must be clockwise).
    
    for(unsigned int q = 0 ; q < quads.size() ; ++q )
    {
        core::topology::Quad qua = quads[q];
        defaulttype::Vec3Types::Coord v0, v1, v2;
        v0.set((float)positions[qua[0]][2], (float)positions[qua[0]][1], (float)positions[qua[0]][0]);
        v1.set((float)positions[qua[1]][2], (float)positions[qua[1]][1], (float)positions[qua[1]][0]);
        v2.set((float)positions[qua[2]][2], (float)positions[qua[2]][1], (float)positions[qua[2]][0]);
        
        totalVolume += ( ((v1[1] - v0[1]) * (v2[2] - v0[2])) - ((v1[2]-v0[2])*(v2[1]-v0[1])) )*(v0[0]+v1[0]+v2[0]);
        
        v0.set((float)positions[qua[0]][2], (float)positions[qua[0]][1], (float)positions[qua[0]][0]);
        v1.set((float)positions[qua[2]][2], (float)positions[qua[2]][1], (float)positions[qua[2]][0]);
        v2.set((float)positions[qua[3]][2], (float)positions[qua[3]][1], (float)positions[qua[3]][0]);
        
        totalVolume += ( ((v1[1] - v0[1]) * (v2[2] - v0[2])) - ((v1[2]-v0[2])*(v2[1]-v0[1])) )*(v0[0]+v1[0]+v2[0]);
    }
    
    totalVolume = (totalVolume/6.f)*1E-6;
    sout << "volume (cubic meter) : " << totalVolume << sendl;
    sout << "mass (kilogram) : " << mass << sendl;
    dens = mass/totalVolume;
    return dens;
}

} // namespace misc

} // namespace component

} // namespace sofa
