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
#include "PythonMacros.h"
#include "Binding.h"

#include "Bindings/SofaModule.h"
#include "Bindings/Data.h"
#include "Bindings/DisplayFlagsData.h"
#include "Bindings/OptionsGroupData.h"
#include "Bindings/DataFileName.h"
#include "Bindings/DataFileNameVector.h"
#include "Bindings/VectorLinearSpringData.h"
#include "Bindings/Link.h"
#include "Bindings/Base.h"
#include "Bindings/BaseObject.h"
#include "Bindings/BaseState.h"
#include "Bindings/BaseContext.h"
#include "Bindings/Context.h"
#include "Bindings/Node.h"
#include "Bindings/Vector.h"
#include "Bindings/TopologyChange.h"
#include "Bindings/BaseLoader.h"
#include "Bindings/MeshLoader.h"
#include "Bindings/Topology.h"
#include "Bindings/BaseMeshTopology.h"
#include "Bindings/MeshTopology.h"
#include "Bindings/GridTopology.h"
#include "Bindings/RegularGridTopology.h"
#include "Bindings/BaseMechanicalState.h"
#include "Bindings/MechanicalObject.h"
#include "Bindings/PythonScriptController.h"
#include "Bindings/PythonScriptDataEngine.h"
#include "Bindings/LinearSpring.h"
#include "Bindings/BaseTopologyObject.h"
#include "Bindings/TriangleSetTopologyModifier.h"
#include "Bindings/PointSetTopologyModifier.h"
#include "Bindings/BaseMapping.h"
#include "Bindings/SubsetMultiMapping.h"
#include "Bindings/VisualModel.h"
#include "Bindings/OBJExporter.h"
#include "Bindings/STLExporter.h"
#include "Bindings/DataEngine.h"
#include "PythonFactory.h"

using sofa::PythonFactory;


void bindSofaPythonModule()
{
    PythonFactory::s_sofaPythonModule = SP_INIT_MODULE(Sofa)

    /// non Base-Inherited types
    SP_ADD_CLASS_IN_SOFAMODULE(Data)

    /// special Data cases
    SP_ADD_CLASS_IN_FACTORY(DisplayFlagsData,sofa::core::objectmodel::Data<sofa::core::visual::DisplayFlags>)
    SP_ADD_CLASS_IN_FACTORY(OptionsGroupData,sofa::core::objectmodel::Data<sofa::helper::OptionsGroup>)
    SP_ADD_CLASS_IN_FACTORY(DataFileName,sofa::core::objectmodel::DataFileName)
    SP_ADD_CLASS_IN_FACTORY(DataFileNameVector,sofa::core::objectmodel::DataFileNameVector)
    SP_ADD_CLASS_IN_SOFAMODULE(PointAncestorElem)
    SP_ADD_CLASS_IN_FACTORY(VectorLinearSpringData,sofa::core::objectmodel::Data<sofa::helper::vector<sofa::component::interactionforcefield::LinearSpring<SReal>>>)

    SP_ADD_CLASS_IN_SOFAMODULE(Link)
    SP_ADD_CLASS_IN_SOFAMODULE(Vector3)
    SP_ADD_CLASS_IN_SOFAMODULE(LinearSpring)

    /// special component categories gettable by static_cast
    SP_ADD_CLASS_IN_SOFAMODULE(Base)
    SP_ADD_CLASS_IN_SOFAMODULE(BaseContext)
    SP_ADD_CLASS_IN_SOFAMODULE(BaseObject)
    SP_ADD_CLASS_IN_SOFAMODULE(BaseTopologyObject)
    SP_ADD_CLASS_IN_SOFAMODULE(BaseState)
    SP_ADD_CLASS_IN_SOFAMODULE(BaseMechanicalState)
    SP_ADD_CLASS_IN_SOFAMODULE(BaseMapping)
    SP_ADD_CLASS_IN_SOFAMODULE(DataEngine)
    SP_ADD_CLASS_IN_SOFAMODULE(VisualModel)
    SP_ADD_CLASS_IN_SOFAMODULE(BaseLoader)
    SP_ADD_CLASS_IN_SOFAMODULE(Topology)
    SP_ADD_CLASS_IN_SOFAMODULE(BaseMeshTopology)

    /// regular component bindings
    SP_ADD_CLASS_IN_FACTORY(Context,sofa::core::objectmodel::Context)
    SP_ADD_CLASS_IN_FACTORY(Node,sofa::simulation::Node)
    SP_ADD_CLASS_IN_FACTORY(VisualModelImpl,sofa::component::visualmodel::VisualModelImpl)
    SP_ADD_CLASS_IN_FACTORY(MeshLoader,sofa::core::loader::MeshLoader)
    SP_ADD_CLASS_IN_FACTORY(MeshTopology,sofa::component::topology::MeshTopology)
    SP_ADD_CLASS_IN_FACTORY(GridTopology,sofa::component::topology::GridTopology)
    SP_ADD_CLASS_IN_FACTORY(RegularGridTopology,sofa::component::topology::RegularGridTopology)
    SP_ADD_CLASS_IN_FACTORY(OBJExporter,sofa::component::misc::OBJExporter)
    SP_ADD_CLASS_IN_FACTORY(STLExporter,sofa::component::misc::STLExporter)
    SP_ADD_CLASS_IN_FACTORY(PythonScriptController,sofa::component::controller::PythonScriptController)
    SP_ADD_CLASS_IN_FACTORY(PythonScriptDataEngine,sofa::component::controller::PythonScriptDataEngine)
    SP_ADD_CLASS_IN_FACTORY(PointSetTopologyModifier,sofa::component::topology::PointSetTopologyModifier)
    SP_ADD_CLASS_IN_FACTORY(TriangleSetTopologyModifier,sofa::component::topology::TriangleSetTopologyModifier)

    /// Custom Exception
    PyObject* PyExc_SofaException = PyErr_NewExceptionWithDoc(
        (char*) "Sofa.SofaException",
        (char*) "Base exception class for the SofaPython module.",
        NULL, NULL);

    if ( PyExc_SofaException )
        PyModule_AddObject(PythonFactory::s_sofaPythonModule, "SofaException", PyExc_SofaException);
}




