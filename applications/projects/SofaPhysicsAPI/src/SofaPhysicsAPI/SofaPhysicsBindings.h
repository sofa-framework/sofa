/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#if defined(_MSC_VER)
#define EXPORT_API extern "C" __declspec(dllexport)
#elif defined(UNITY_METRO)
#define EXPORT_API __declspec(dllexport) __stdcall
#elif defined(UNITY_WIN)
#define EXPORT_API __declspec(dllexport)
#elif defined(__GNUC__)
#define EXPORT_API __attribute__((visibility("default")))
#else
#define EXPORT
#pragma warning Unknown dynamic link import/export semantics.
#endif

#include <string>

//////////////////////////////////////////////////////////
///////////////   Global API Bindings   //////////////////
//////////////////////////////////////////////////////////

/// Test API
EXPORT_API int test_getAPI_ID();

// API creation/destruction methods
EXPORT_API void* sofaPhysicsAPI_create();
EXPORT_API int sofaPhysicsAPI_delete(void* ptr);
EXPORT_API const char* sofaPhysicsAPI_APIName(void* ptr);

// API for scene creation/loading
EXPORT_API int sofaPhysicsAPI_createScene(void* ptr);
EXPORT_API int sofaPhysicsAPI_loadScene(void* ptr, const char* filename);
EXPORT_API int sofaPhysicsAPI_unloadScene(void* ptr);

// API for animation loop
EXPORT_API void sofaPhysicsAPI_start(void* ptr);
EXPORT_API void sofaPhysicsAPI_stop(void* ptr);
EXPORT_API void sofaPhysicsAPI_step(void* ptr);
EXPORT_API void sofaPhysicsAPI_reset(void* ptr);

EXPORT_API float sofaPhysicsAPI_time(void* ptr);
EXPORT_API float sofaPhysicsAPI_timeStep(void* ptr);
EXPORT_API void sofaPhysicsAPI_setTimeStep(void* ptr, double value);

EXPORT_API int sofaPhysicsAPI_getGravity(void* ptr, double* values);
EXPORT_API int sofaPhysicsAPI_setGravity(void* ptr, double* values);


//////////////////////////////////////////////////////////
//////////////    VisualModel Bindings    ////////////////
//////////////////////////////////////////////////////////

EXPORT_API int sofaPhysicsAPI_getNbrVisualModel(void* ptr);
EXPORT_API const char* sofaVisualModel_getName(void* ptr, int VModelID);

EXPORT_API int sofaVisualModel_getNbVertices(void* ptr, const char* name);
EXPORT_API int sofaVisualModel_getVertices(void* ptr, const char* name, float* buffer);
EXPORT_API int sofaVisualModel_getNormals(void* ptr, const char* name, float* buffer);
EXPORT_API int sofaVisualModel_getTexCoords(void* ptr, const char* name, float* buffer);

EXPORT_API int sofaVisualModel_getNbEdges(void* ptr, const char* name);
EXPORT_API int sofaVisualModel_getEdges(void* ptr, const char* name, int* buffer);

EXPORT_API int sofaVisualModel_getNbTriangles(void* ptr, const char* name);
EXPORT_API int sofaVisualModel_getTriangles(void* ptr, const char* name, int* buffer);

EXPORT_API int sofaVisualModel_getNbQuads(void* ptr, const char* name);
EXPORT_API int sofaVisualModel_getQuads(void* ptr, const char* name, int* buffer);
