/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
 *                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
 *                              SOFA :: Framework                              *
 *                                                                             *
 * Authors: The SOFA Team (see Authors.txt)                                    *
 *                                                                             *
 * Contact information: contact@sofa-framework.org                             *
 ******************************************************************************/
#ifndef SOFA_CORE_EXPORTER_BASEEXPORTER_H
#define SOFA_CORE_EXPORTER_BASEEXPORTER_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/helper/helper.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace core
{
    
namespace exporter
{

class SOFA_CORE_API BaseExporter : public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(BaseExporter, objectmodel::BaseObject);
protected:
    ///Constructor
    BaseExporter() {}
    
    /// Destructor
    virtual ~BaseExporter() {}

public:
    virtual bool getINP(vector< std::string >* nameT, vector< defaulttype::Vec3Types::VecCoord >* positionT, vector< double >* densiT, vector< vector< sofa::core::topology::Topology::Tetrahedron > >* tetrahedraT, vector< vector< sofa::core::topology::Topology::Hexahedron > >* hexahedraT, vector< vector< unsigned int > >* fixedPointT, vector< double >* youngModulusT, vector< double >* poissonRatioT) = 0;

};
    
} // namespace exporter

} // namespace core

} // namespace sofa

#endif //SOFA_CORE_EXPORTER_BASEEXPORTER_H
