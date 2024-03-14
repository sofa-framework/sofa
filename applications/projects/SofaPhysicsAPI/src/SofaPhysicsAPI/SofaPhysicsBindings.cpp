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
#include "SofaPhysicsBindings.h"
#include "SofaPhysicsAPI.h"
#include <string.h>

/// Test API
int test_getAPI_ID()
{
    return 2022;
}

// API creation/destruction methods
void* sofaPhysicsAPI_create()
{
    return new SofaPhysicsAPI();
}

int sofaPhysicsAPI_delete(void* api_ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api)
        delete api;
    else
        return API_NULL;
    api_ptr = NULL;

    return API_SUCCESS;
}

const char* sofaPhysicsAPI_APIName(void* api_ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api)
    {
        return api->APIName();
    }
    else
        return "none";
}


// API for scene creation/loading
int sofaPhysicsAPI_createScene(void* api_ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api) {
        api->createScene();
        return API_SUCCESS;
    }
    else
        return API_NULL;
}

int sofaPhysicsAPI_loadScene(void* api_ptr, const char* filename)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api) {
        return api->load(filename);
    }
    else
        return API_NULL;
}

int sofaPhysicsAPI_unloadScene(void* api_ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api) {
        return api->unload();
    }
    else
        return API_NULL;
}

const char* sofaPhysicsAPI_loadSofaIni(void* api_ptr, const char* filePath)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api) {
        return api->loadSofaIni(filePath);
    }
    else
        return "Error: API_NULL";
}

int sofaPhysicsAPI_loadPlugin(void* api_ptr, const char* pluginPath)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api) {
        return api->loadPlugin(pluginPath);
    }
    else
        return API_NULL;
}


// API for animation loop
void sofaPhysicsAPI_start(void* api_ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api) {
        return api->start();
    }
}

void sofaPhysicsAPI_stop(void* api_ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api) {
        return api->stop();
    }
}


void sofaPhysicsAPI_step(void* api_ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api) {
        return api->step();
    }
}


void sofaPhysicsAPI_reset(void* api_ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api) {
        return api->reset();
    }
}



float sofaPhysicsAPI_time(void* api_ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api) {
        return api->getTime();
    }
    else
        return -1.0;
}


float sofaPhysicsAPI_timeStep(void* api_ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api) {
        return api->getTime();
    }
    else
        return -1.0;
}


void sofaPhysicsAPI_setTimeStep(void* api_ptr, double value)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api) {
        return api->setTimeStep(value);
    }
}



int sofaPhysicsAPI_getGravity(void* api_ptr, double* values)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api) {
        return api->getGravity(values);
    }
    else
        return API_NULL;
}


int sofaPhysicsAPI_setGravity(void* api_ptr, double* values)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api) {
        api->setGravity(values);
        return API_SUCCESS;
    }
    else
        return API_NULL;
}



int sofaPhysicsAPI_activateMessageHandler(void* api_ptr, bool value)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api)
        return api->activateMessageHandler(value);
    else
        return API_NULL;
}

int sofaPhysicsAPI_getNbMessages(void* api_ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api)
        return api->getNbMessages();
    else
        return API_NULL;
}

const char* sofaPhysicsAPI_getMessage(void* api_ptr, int messageId, int* msgType)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api)
    {
        return api->getMessage(messageId, msgType[0]);
    }
    else
        return "Error: API_NULL";
}

int sofaPhysicsAPI_clearMessages(void* api_ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api)
        return api->clearMessages();
    else
        return API_NULL;
}


//////////////////////////////////////////////////////////
//////////////    VisualModel Bindings    ////////////////
//////////////////////////////////////////////////////////

int sofaPhysicsAPI_getNbrVisualModel(void* api_ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api) {
        return api->getNbOutputMeshes();
    }
    else
        return API_NULL;
}


const char* sofaVisualModel_getName(void* api_ptr, int VModelID)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api)
    {
        SofaPhysicsOutputMesh* mesh = api->getOutputMeshPtr(VModelID);
        return mesh->getName();
    }
    else
        return "Error: SAPAPI_NULL_API";
}



int sofaVisualModel_getNbVertices(void* api_ptr, const char* name)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api)
    {
        SofaPhysicsOutputMesh* mesh = api->getOutputMeshPtr(name);
        if (mesh == nullptr)
            return API_MESH_NULL;
        else
            return mesh->getNbVertices();
    }

    return API_NULL;
}


int sofaVisualModel_getVertices(void* api_ptr, const char* name, float* buffer)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api)
    {
        SofaPhysicsOutputMesh* mesh = api->getOutputMeshPtr(name);
        if (mesh == nullptr)
            return API_MESH_NULL;
        else
            return mesh->getVPositions(buffer);
    }

    return API_NULL;
}


int sofaVisualModel_getNormals(void* api_ptr, const char* name, float* buffer)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api)
    {
        SofaPhysicsOutputMesh* mesh = api->getOutputMeshPtr(name);
        if (mesh == nullptr)
            return API_MESH_NULL;
        else
            return mesh->getVNormals(buffer);
    }

    return API_NULL;
}


int sofaVisualModel_getTexCoords(void* api_ptr, const char* name, float* buffer)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api)
    {
        SofaPhysicsOutputMesh* mesh = api->getOutputMeshPtr(name);
        if (mesh == nullptr)
            return API_MESH_NULL;
        else
            return mesh->getVTexCoords(buffer);
    }

    return API_NULL;
}



int sofaVisualModel_getNbEdges(void* api_ptr, const char* name)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api)
    {
        SofaPhysicsOutputMesh* mesh = api->getOutputMeshPtr(name);
        if (mesh == nullptr)
            return API_MESH_NULL;
        else
            return mesh->getNbLines();
    }

    return API_NULL;
}


int sofaVisualModel_getEdges(void* /*api_ptr*/, const char* /*name*/, int* /*buffer*/)
{
    //TODO

    return API_NULL;
}



int sofaVisualModel_getNbTriangles(void* api_ptr, const char* name)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api)
    {
        SofaPhysicsOutputMesh* mesh = api->getOutputMeshPtr(name);
        if (mesh == nullptr)
            return API_MESH_NULL;
        else
            return mesh->getNbTriangles();
    }

    return API_NULL;
}


int sofaVisualModel_getTriangles(void* api_ptr, const char* name, int* buffer)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api)
    {
        SofaPhysicsOutputMesh* mesh = api->getOutputMeshPtr(name);
        if (mesh == nullptr)
            return API_MESH_NULL;
        else
            return mesh->getTriangles(buffer);
    }

    return API_NULL;
}



int sofaVisualModel_getNbQuads(void* api_ptr, const char* name)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api)
    {
        SofaPhysicsOutputMesh* mesh = api->getOutputMeshPtr(name);
        if (mesh == nullptr)
            return API_MESH_NULL;
        else
            return mesh->getNbQuads();
    }

    return API_NULL;
}


int sofaVisualModel_getQuads(void* api_ptr, const char* name, int* buffer)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)api_ptr;
    if (api)
    {
        SofaPhysicsOutputMesh* mesh = api->getOutputMeshPtr(name);
        if (mesh == nullptr)
            return API_MESH_NULL;
        else
            return mesh->getQuads(buffer);
    }

    return API_NULL;
}

