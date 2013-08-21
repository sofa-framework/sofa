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

#ifndef INPEXPORTER_H_
#define INPEXPORTER_H_

#include <sofa/helper/helper.h>
#include <sofa/core/exporter/BaseExporter.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/component.h>
#include <sofa/component/forcefield/HexahedronFEMForceField.h>
#include <sofa/component/forcefield/TetrahedronFEMForceField.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>
#include <sofa/component/visualmodel/VisualModelImpl.h>
#include <sofa/core/topology/Topology.h>

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

class SOFA_EXPORTER_API INPExporter : public virtual core::exporter::BaseExporter
{
public:
    SOFA_CLASS(INPExporter,core::exporter::BaseExporter);

private:
    sofa::core::topology::BaseMeshTopology* topology;
    sofa::core::behavior::BaseMechanicalState* mstate;
    sofa::core::behavior::BaseProjectiveConstraintSet* bcondition;
    sofa::core::behavior::BaseMass* bmass;
    sofa::core::behavior::ForceField<sofa::defaulttype::Vec3Types>* hexaForceField;
    sofa::core::behavior::ForceField<sofa::defaulttype::Vec3Types>* tetraForceField;
    sofa::core::objectmodel::BaseContext* context;

    std::ofstream* outfile;

public:
    Data< std::string > m_name;
    Data< defaulttype::Vec3Types::VecCoord > m_position;
    Data< vector< core::topology::Triangle > > m_triangle;
    Data< vector< core::topology::Quad > > m_quad;
    Data<double> m_baseMass;
    Data<double> m_baseDensity;
    Data<double> m_density;
    Data<double> m_youngModulus;
    Data<double> m_poissonRatio;
    Data< vector< sofa::component::topology::Tetrahedron > > m_tetrahedra;
    Data< vector< sofa::component::topology::Hexahedron > > m_hexahedra;
    Data< vector< unsigned int > > m_fixedPoints;

protected:
    INPExporter();
    virtual ~INPExporter();
public:
    
    virtual bool getINP(vector< std::string >* nameT, vector< defaulttype::Vec3Types::VecCoord >* positionT, vector< double >* densiT, vector< vector< sofa::component::topology::Tetrahedron > >* tetrahedraT, vector< vector< sofa::component::topology::Hexahedron > >* hexahedraT, vector< vector< unsigned int > >* fixedPointT, vector< double >* youngModulusT, vector< double >* poissonRatioT);
    virtual float computeDensity(vector< core::topology::Triangle > triangles, vector< core::topology::Quad > quads, vector< defaulttype::Vec3d > positions, double mass);
    
    void init();
    void bwdInit();
};

}

}

}

#endif /* INPEXPORTER_H_ */
