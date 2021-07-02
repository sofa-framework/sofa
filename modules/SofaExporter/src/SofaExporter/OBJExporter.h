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
#ifndef OBJEXPORTER_H_
#define OBJEXPORTER_H_
#include <SofaExporter/config.h>

#include <sofa/simulation/BaseSimulationExporter.h>

#include <fstream>

namespace sofa::component
{
namespace _objexporter_
{

using sofa::simulation::BaseSimulationExporter;
using sofa::core::objectmodel::Event;
using sofa::core::objectmodel::Base;

class SOFA_SOFAEXPORTER_API OBJExporter : public BaseSimulationExporter
{
public:
    SOFA_CLASS(OBJExporter, BaseSimulationExporter);

    bool write() override;
    bool writeOBJ();

    void handleEvent(Event *event) override;

protected:
    ~OBJExporter() override;
};

} // namespace _objexporter_

// Import the object in the exporter namespace to avoid having all the object straight in component.
namespace exporter {
    using OBJExporter = _objexporter_::OBJExporter;
} // namespace exporter

// Import the object in the "old" namespaces to allow smooth update of code base.
using OBJExporter
    SOFA_ATTRIBUTE_DEPRECATED__SOFAEXPORTER_NAMESPACE_2106()
    = _objexporter_::OBJExporter;
namespace misc {
    using OBJExporter
        SOFA_ATTRIBUTE_DEPRECATED__SOFAEXPORTER_NAMESPACE_1712()
        = _objexporter_::OBJExporter;
} // namespace misc

} // namespace sofa::component

#endif /* OBJEXPORTER_H_ */
