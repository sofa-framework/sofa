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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "Binding.h"
#include "Binding_SofaModule.h"

#include "Binding_Data.h"
#include "Binding_DisplayFlagsData.h"
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
#include "Binding_BaseMechanicalState.h"
#include "Binding_MechanicalObject.h"
#include "Binding_PythonScriptController.h"

#if defined(_MSC_VER)
// undefine _DEBUG since we want to always link to the release version of
// python and pyconfig.h automatically links debug version if _DEBUG is
// defined.
#ifdef _DEBUG
#define _DEBUG_UNDEFED
#undef _DEBUG
#endif
#endif
#if defined(__APPLE__) && defined(__MACH__)
#include <Python/Python.h>
#else
#include <Python.h>
#endif
#if defined(_MSC_VER)
// redefine _DEBUG if it was undefed
#ifdef _DEBUG_UNDEFED
#define _DEBUG
#endif
#endif

PyObject *SofaPythonModule = 0;




void bindSofaPythonModule()
{
    //PyImport_AppendInittab( (char*)"Sofa", &initSofa );

    SofaPythonModule = SP_INIT_MODULE(Sofa)
    SP_ADD_CLASS(SofaPythonModule,Data)
        SP_ADD_CLASS(SofaPythonModule,DisplayFlagsData)
    SP_ADD_CLASS(SofaPythonModule,Vector3)

        SP_ADD_CLASS(SofaPythonModule,Base)
            SP_ADD_CLASS(SofaPythonModule,BaseContext)
                SP_ADD_CLASS(SofaPythonModule,Context)
                    SP_ADD_CLASS(SofaPythonModule,Node)
            SP_ADD_CLASS(SofaPythonModule,BaseObject)
                SP_ADD_CLASS(SofaPythonModule,BaseState)
                    SP_ADD_CLASS(SofaPythonModule,BaseMechanicalState)
                        SP_ADD_CLASS(SofaPythonModule,MechanicalObject)
            SP_ADD_CLASS(SofaPythonModule,BaseLoader)
                SP_ADD_CLASS(SofaPythonModule,MeshLoader)
            SP_ADD_CLASS(SofaPythonModule,Topology)
/*
            SP_ADD_CLASS(SofaPythonModule,BaseController)
                SP_ADD_CLASS(SofaPythonModule,Controller)
                    SP_ADD_CLASS(SofaPythonModule,ScriptController)
        */
                        SP_ADD_CLASS(SofaPythonModule,PythonScriptController)
}




