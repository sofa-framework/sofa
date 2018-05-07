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
#include "Binding_SofaModule.h"

#include "Binding_Data.h"
#include "Binding_DisplayFlagsData.h"
#include "Binding_OptionsGroupData.h"
#include "Binding_DataFileName.h"
#include "Binding_DataFileNameVector.h"
#include "Binding_VectorLinearSpringData.h"
#include "Binding_Link.h"
#include "Binding_Base.h"
#include "Binding_BaseObject.h"
#include "Binding_BaseState.h"
#include "Binding_BaseContext.h"
#include "Binding_Context.h"
#include "Binding_Node.h"
#include "Binding_Vector.h"
#include "Binding_TopologyChange.h"
#include "Binding_BaseLoader.h"
#include "Binding_MeshLoader.h"
#include "Binding_Topology.h"
#include "Binding_BaseMeshTopology.h"
#include "Binding_MeshTopology.h"
#include "Binding_GridTopology.h"
#include "Binding_RegularGridTopology.h"
#include "Binding_BaseMechanicalState.h"
#include "Binding_MechanicalObject.h"
#include "Binding_PythonScriptController.h"
#include "Binding_PythonScriptDataEngine.h"
#include "Binding_LinearSpring.h"
#include "Binding_BaseTopologyObject.h"
#include "Binding_TriangleSetTopologyModifier.h"
#include "Binding_PointSetTopologyModifier.h"
#include "Binding_BaseMapping.h"
#include "Binding_SubsetMultiMapping.h"
#include "Binding_VisualModel.h"
#include "Binding_OBJExporter.h"
#include "Binding_STLExporter.h"
#include "Binding_DataEngine.h"
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




