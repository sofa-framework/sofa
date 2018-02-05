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
#ifndef STLEXPORTER_H_
#define STLEXPORTER_H_
#include "config.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/simulation/BaseSimulationExporter.h>

///////////////////////////// FORWARD DECLARATION //////////////////////////////////////////////////
namespace sofa {
    namespace core {
        namespace objectmodel {
            class BaseMechanicalState;
            class Event ;
        }
        namespace visual {
            class VisualModel ;
        }
    }
}



////////////////////////////////// DECLARATION /////////////////////////////////////////////////////
namespace sofa
{

namespace component
{

namespace _stlexporter_
{

using sofa::core::behavior::BaseMechanicalState ;
using sofa::core::topology::BaseMeshTopology ;
using sofa::core::visual::VisualModel ;
using sofa::core::objectmodel::Event ;
using sofa::simulation::BaseSimulationExporter ;

class SOFA_EXPORTER_API STLExporter : public BaseSimulationExporter
{
public:
    SOFA_CLASS(STLExporter, BaseSimulationExporter);

    Data<bool> d_binaryFormat;      //0 for Ascii Formats, 1 for Binary File Format
    Data<defaulttype::Vec3Types::VecCoord>               d_position;
    Data< helper::vector< BaseMeshTopology::Triangle > > d_triangle;
    Data< helper::vector< BaseMeshTopology::Quad > >     d_quad;

    virtual void doInit() override ;
    virtual void doReInit() override ;
    virtual void handleEvent(Event *) override ;

    virtual bool write() override ;

    bool writeSTL(bool autonumbering=true);
    bool writeSTLBinary(bool autonumbering=true);

protected:
    STLExporter();
    virtual ~STLExporter();

private:
    BaseMeshTopology*    m_inputtopology {nullptr};
    BaseMechanicalState* m_inputmstate   {nullptr};
    VisualModel*         m_inputvmodel   {nullptr};
};

} /// _stlexporter_

//todo(18.06): remove the old namespaces...
/// Import the object in the "old" namespace to allow smooth update of code base.
namespace misc {
    using _stlexporter_::STLExporter ;
}

namespace exporter {
    using _stlexporter_::STLExporter ;
}

}

}

#endif /* STLEXPORTER_H_ */
