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


////////////////////////////////////////////////////////////
///////////////   Global API C Bindings   //////////////////
////////////////////////////////////////////////////////////

/// Test API method. This is just a very simple method returning number 2022 to check connection with API
EXPORT_API int test_getAPI_ID();

// API creation/destruction methods
EXPORT_API void* sofaPhysicsAPI_create(); ///< Create a SofaPhysicsAPI instance and return pointer to it
EXPORT_API int sofaPhysicsAPI_delete(void* api_ptr); ///< Method to delete the instance of SofaPhysicsAPI given as input @param api_ptr. Return error code.
EXPORT_API const char* sofaPhysicsAPI_APIName(void* api_ptr); ///< Method to get the api name of @param api_ptr

// API for scene creation/loading
EXPORT_API int sofaPhysicsAPI_createScene(void* api_ptr); ///< Method to create a SOFA scene (scene will be empty with valid Root Node). Return error code.
EXPORT_API int sofaPhysicsAPI_loadScene(void* api_ptr, const char* filename); ///< Method to load a SOFA (.scn) file given by @param filename inside the given instance @param api_ptr. Return error code.
EXPORT_API int sofaPhysicsAPI_unloadScene(void* api_ptr); ///< Method to unload the current SOFA scene and create empty Root Node inside the given instance @param api_ptr. Return error code.
EXPORT_API const char* sofaPhysicsAPI_loadSofaIni(void* api_ptr, const char* filePath); ///< Method to load a SOFA .ini config file at given path @filePath to define resource/example paths. Return share path.
EXPORT_API int sofaPhysicsAPI_loadPlugin(void* api_ptr, const char* pluginPath); ///< Method to load a specific SOFA plugin using it's full path @param pluginPath. Return error code. 

// API for animation loop
EXPORT_API void sofaPhysicsAPI_start(void* api_ptr); ///< Method to start simulation
EXPORT_API void sofaPhysicsAPI_stop(void* api_ptr); ///< Method to stop simulation
EXPORT_API void sofaPhysicsAPI_step(void* api_ptr); ///< Method to perform a single simulation step
EXPORT_API void sofaPhysicsAPI_reset(void* api_ptr); ///< Method to reset current simulation

EXPORT_API float sofaPhysicsAPI_time(void* api_ptr); ///< Getter to the current simulation time
EXPORT_API float sofaPhysicsAPI_timeStep(void* api_ptr); ///< Getter to the current simulation time stepping
EXPORT_API void sofaPhysicsAPI_setTimeStep(void* api_ptr, double value); ///< Setter to the current simulation time stepping

EXPORT_API int sofaPhysicsAPI_getGravity(void* api_ptr, double* values); ///< Getter of scene gravity using the ouptut @param values which is a double[3]. Return error code.
EXPORT_API int sofaPhysicsAPI_setGravity(void* api_ptr, double* values); ///< Setter of current scene gravity using the input @param gravity which is a double[3]. Return error code.

// API for message logging
EXPORT_API int sofaPhysicsAPI_activateMessageHandler(void* api_ptr, bool value); ///< Method to activate/deactivate SOFA MessageHandler according to @param value. Return Error code.
EXPORT_API int sofaPhysicsAPI_getNbMessages(void* api_ptr); ///< Method to get the number of messages in queue
EXPORT_API const char* sofaPhysicsAPI_getMessage(void* api_ptr, int messageId, int* msgType); ///< Method to return the queued message of index @param messageId and its type level inside @param msgType
EXPORT_API int sofaPhysicsAPI_clearMessages(void* api_ptr); ///< Method clear the list of queued messages. Return Error code.


//////////////////////////////////////////////////////////
//////////////    VisualModel Bindings    ////////////////
//////////////////////////////////////////////////////////

EXPORT_API int sofaPhysicsAPI_getNbrVisualModel(void* api_api_ptr); ///< Return the number of SofaPhysicsOutputMesh in the current simulation
EXPORT_API const char* sofaVisualModel_getName(void* api_api_ptr, int VModelID); ///< Return the name of the SofaPhysicsOutputMesh with id @param VModelID in the current simulation

/// API to get SofaPhysicsOutputMesh position and topology information. 
/// SofaPhysicsOutputMesh Name is used as identifier as the index of registration could change from one loading to another. Or if scene is modified.
EXPORT_API int sofaVisualModel_getNbVertices(void* api_api_ptr, const char* name); ///< Return the number of vertices of the SofaPhysicsOutputMesh with name: @param name
EXPORT_API int sofaVisualModel_getVertices(void* api_api_ptr, const char* name, float* buffer); ///< Get the positions/vertices using ouput @param values (type float[ 3*nbVertices ]) of the SofaPhysicsOutputMesh with name: @param name. Return error code.
EXPORT_API int sofaVisualModel_getNormals(void* api_ptr, const char* name, float* buffer); ///< Get the normals using ouput @param values (type float[ 3*nbVertices ]) of the SofaPhysicsOutputMesh with name: @param name. Return error code.
EXPORT_API int sofaVisualModel_getTexCoords(void* api_ptr, const char* name, float* buffer); ///< Get the texture coordinates using ouput @param values (type float[ 2*nbVertices ]) of the SofaPhysicsOutputMesh with name: @param name. Return error code.

EXPORT_API int sofaVisualModel_getNbEdges(void* api_ptr, const char* name); ///< Return the number of edges of the SofaPhysicsOutputMesh with name: @param name
EXPORT_API int sofaVisualModel_getEdges(void* api_ptr, const char* name, int* buffer); ///< Get the edges using ouput @param values (type int[ 2*nbEdges ]) of the SofaPhysicsOutputMesh with name: @param name. Return error code.

EXPORT_API int sofaVisualModel_getNbTriangles(void* api_ptr, const char* name); ///< Return the number of triangles of the SofaPhysicsOutputMesh with name: @param name
EXPORT_API int sofaVisualModel_getTriangles(void* api_ptr, const char* name, int* buffer); ///< Get the triangles using ouput @param values (type int[ 3*nbTriangles ]) of the SofaPhysicsOutputMesh with name: @param name. Return error code.

EXPORT_API int sofaVisualModel_getNbQuads(void* api_ptr, const char* name); ///< Return the number of quads of the SofaPhysicsOutputMesh with name: @param name
EXPORT_API int sofaVisualModel_getQuads(void* api_ptr, const char* name, int* buffer); ///< Get the quads using ouput @param values (type int[ 4*nbQuads ]) of the SofaPhysicsOutputMesh with name: @param name. Return error code.
