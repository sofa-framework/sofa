/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
//#include "PythonCommon.h"
#include "PythonMacros.h"
#include "Binding.h"
#include "Binding_SofaModule.h"

#include "Binding_Data.h"
#include "Binding_DisplayFlagsData.h"
#include "Binding_OptionsGroupData.h"
#include "Binding_Link.h"
#include "Binding_Base.h"
#include "Binding_BaseObject.h"
#include "Binding_BaseState.h"
#include "Binding_BaseContext.h"
#include "Binding_Context.h"
#include "Binding_Node.h"
#include "Binding_Vector.h"
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
#include "Binding_LinearSpring.h"
#include "Binding_BaseMapping.h"
//#include "Binding_Mapping.h"
//#include "Binding_RigidMapping.h"
//#include "Binding_MultiMapping.h"
#include "Binding_SubsetMultiMapping.h"
#include "Binding_VisualModel.h"
#include "Binding_OBJExporter.h"
#include "Binding_DataEngine.h"

PyObject *SofaPythonModule = 0;




void bindSofaPythonModule()
{
    //PyImport_AppendInittab( (char*)"Sofa", &initSofa );

	SofaPythonModule = SP_INIT_MODULE(Sofa)
	SP_ADD_CLASS(SofaPythonModule,Data)
	SP_ADD_CLASS(SofaPythonModule,DisplayFlagsData)
    SP_ADD_CLASS(SofaPythonModule,OptionsGroupData)

    SP_ADD_CLASS(SofaPythonModule,Link)

	SP_ADD_CLASS(SofaPythonModule,Vector3)

	SP_ADD_CLASS(SofaPythonModule,LinearSpring)


	SP_ADD_CLASS(SofaPythonModule,Base)
	SP_ADD_CLASS(SofaPythonModule,BaseContext)
	SP_ADD_CLASS(SofaPythonModule,Context)
	SP_ADD_CLASS(SofaPythonModule,Node)
	SP_ADD_CLASS(SofaPythonModule,BaseObject)
	SP_ADD_CLASS(SofaPythonModule,BaseState)
	SP_ADD_CLASS(SofaPythonModule,BaseMechanicalState)
	SP_ADD_CLASS(SofaPythonModule,MechanicalObject)
	SP_ADD_CLASS(SofaPythonModule,VisualModel)
    SP_ADD_CLASS(SofaPythonModule,VisualModelImpl)
	SP_ADD_CLASS(SofaPythonModule,BaseMapping)
    SP_ADD_CLASS(SofaPythonModule,DataEngine)
	//SP_ADD_CLASS(SofaPythonModule,Mapping)
	//SP_ADD_CLASS(SofaPythonModule,RigidMapping)
	//SP_ADD_CLASS(SofaPythonModule,MultiMapping3_to_3)
	SP_ADD_CLASS(SofaPythonModule,SubsetMultiMapping3_to_3)
	SP_ADD_CLASS(SofaPythonModule,BaseLoader)
	SP_ADD_CLASS(SofaPythonModule,MeshLoader)
	SP_ADD_CLASS(SofaPythonModule,Topology)
	SP_ADD_CLASS(SofaPythonModule,BaseMeshTopology)
	SP_ADD_CLASS(SofaPythonModule,MeshTopology)
	SP_ADD_CLASS(SofaPythonModule,GridTopology)
	SP_ADD_CLASS(SofaPythonModule,RegularGridTopology)
	SP_ADD_CLASS(SofaPythonModule,OBJExporter)
	//SP_ADD_CLASS(SofaPythonModule,BaseController)
	//SP_ADD_CLASS(SofaPythonModule,Controller)
	//SP_ADD_CLASS(SofaPythonModule,ScriptController)
	SP_ADD_CLASS(SofaPythonModule,PythonScriptController)
}




