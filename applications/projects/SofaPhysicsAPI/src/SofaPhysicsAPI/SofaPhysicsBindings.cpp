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

// Exit code
#define API_SUCCESS EXIT_SUCCESS

#define API_NULL -1


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

int sofaPhysicsAPI_delete(void* ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)ptr;
    if (api)
        delete api;
    else
        return API_NULL;
    ptr = NULL;

    return API_SUCCESS;
}

const char* sofaPhysicsAPI_APIName(void* ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)ptr;
    if (api)
    {
        std::string apiName = api->APIName();
        char* cstr = new char[apiName.length() + 1];
#if defined(_MSC_VER)
        std::strcpy(cstr, apiName.c_str());
#endif
        return cstr;
    }
    else
        return "none";
}


// API for scene creation/loading
int sofaPhysicsAPI_createScene(void* ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)ptr;
    if (api) {
        api->createScene();
        return API_SUCCESS;
    }
    else
        return API_NULL;
}

int sofaPhysicsAPI_loadScene(void* ptr, const char* filename)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)ptr;
    if (api) {
        return api->load(filename);
    }
    else
        return API_NULL;
}

int sofaPhysicsAPI_unloadScene(void* ptr)
{
    //SofaPhysicsAPI* api = (SofaPhysicsAPI*)ptr;
    //if (api) {
    //    return api->unloa
    //}
    //else
    return API_NULL;
}


// API for animation loop
void sofaPhysicsAPI_start(void* ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)ptr;
    if (api) {
        return api->start();
    }
}

void sofaPhysicsAPI_stop(void* ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)ptr;
    if (api) {
        return api->stop();
    }
}


void sofaPhysicsAPI_step(void* ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)ptr;
    if (api) {
        return api->step();
    }
}


void sofaPhysicsAPI_reset(void* ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)ptr;
    if (api) {
        return api->reset();
    }
}



float sofaPhysicsAPI_time(void* ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)ptr;
    if (api) {
        return api->getTime();
    }
    else
        return -1.0;
}


float sofaPhysicsAPI_timeStep(void* ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)ptr;
    if (api) {
        return api->getTime();
    }
    else
        return -1.0;
}


void sofaPhysicsAPI_setTimeStep(void* ptr, double value)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)ptr;
    if (api) {
        return api->setTimeStep(value);
    }
}



int sofaPhysicsAPI_getGravity(void* ptr, double* values)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)ptr;
    if (api) {
        return api->getGravity(values);
    }
    else
        return API_NULL;
}


int sofaPhysicsAPI_setGravity(void* ptr, double* values)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)ptr;
    if (api) {
        api->setGravity(values);
        return API_SUCCESS;
    }
    else
        return API_NULL;
}




//////////////////////////////////////////////////////////
//////////////    VisualModel Bindings    ////////////////
//////////////////////////////////////////////////////////

int sofaPhysicsAPI_getNbrVisualModel(void* ptr)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)ptr;
    if (api) {
        return api->getNbOutputMeshes();
    }
    else
        return API_NULL;
}


const char* sofaVisualModel_getName(void* ptr, int VModelID)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)ptr;
    if (api)
    {
        SofaPhysicsOutputMesh* mesh = api->getOutputMeshPtr(VModelID);
        std::string value = mesh->getName();

        char* cstr = new char[value.length() + 1];
#if defined(_MSC_VER)
        std::strcpy(cstr, value.c_str());
#endif
        return cstr;
    }
    else
        return "Error: SAPAPI_NULL_API";
}



int sofaVisualModel_getNbVertices(void* ptr, const char* name)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)ptr;
    if (api)
    {
        SofaPhysicsOutputMesh* mesh = api->getOutputMeshPtr(name);
        return mesh->getNbVertices();
    }

    return API_NULL;
}


int sofaVisualModel_getVertices(void* ptr, const char* name, float* buffer)
{
    //TODO

    return API_NULL;
}


int sofaVisualModel_getNormals(void* ptr, const char* name, float* buffer)
{
    //TODO

    return API_NULL;
}


int sofaVisualModel_getTexCoords(void* ptr, const char* name, float* buffer)
{
    //TODO

    return API_NULL;
}



int sofaVisualModel_getNbEdges(void* ptr, const char* name)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)ptr;
    if (api)
    {
        SofaPhysicsOutputMesh* mesh = api->getOutputMeshPtr(name);
        return mesh->getNbLines();
    }

    return API_NULL;
}


int sofaVisualModel_getEdges(void* ptr, const char* name, int* buffer)
{
    //TODO

    return API_NULL;
}



int sofaVisualModel_getNbTriangles(void* ptr, const char* name)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)ptr;
    if (api)
    {
        SofaPhysicsOutputMesh* mesh = api->getOutputMeshPtr(name);
        return mesh->getNbTriangles();
    }

    return API_NULL;
}


int sofaVisualModel_getTriangles(void* ptr, const char* name, int* buffer)
{
    //TODO

    return API_NULL;
}



int sofaVisualModel_getNbQuads(void* ptr, const char* name)
{
    SofaPhysicsAPI* api = (SofaPhysicsAPI*)ptr;
    if (api)
    {
        SofaPhysicsOutputMesh* mesh = api->getOutputMeshPtr(name);
        return mesh->getNbQuads();
    }

    return API_NULL;
}


int sofaVisualModel_getQuads(void* ptr, const char* name, int* buffer)
{
    //TODO

    return API_NULL;
}

