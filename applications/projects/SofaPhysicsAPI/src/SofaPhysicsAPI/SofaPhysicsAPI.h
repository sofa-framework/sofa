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

#ifndef WIN32
    #ifdef SOFA_BUILD_SOFAPHYSICSAPI
    #	define SOFA_SOFAPHYSICSAPI_API __attribute__ ((visibility ("default")))
    #else
    #   define SOFA_SOFAPHYSICSAPI_API
    #endif
#else
#ifdef SOFA_BUILD_SOFAPHYSICSAPI
    #	define SOFA_SOFAPHYSICSAPI_API __declspec( dllexport )
    #else
    #   define SOFA_SOFAPHYSICSAPI_API __declspec( dllimport )
    #endif
    #   ifdef _MSC_VER
    #       pragma warning(disable : 4231)
    #       pragma warning(disable : 4910)
    #   endif
#endif

class SofaPhysicsOutputMesh;
class SofaPhysicsDataMonitor;
class SofaPhysicsDataController;

typedef unsigned int Index; ///< Type used for topology indices
typedef float Real;         ///< Type used for coordinates
typedef void* ID;           ///< Type used for IDs

/// List of error code to be used to translate methods return values without logging system
#define API_SUCCESS 0                   ///< success value
#define API_NULL -1                     ///< SofaPhysicsAPI created is null
#define API_MESH_NULL -2                ///< If SofaPhysicsOutputMesh requested/accessed is null
#define API_SCENE_NULL -10              ///< Scene creation failed. I.e Root node is null
#define API_SCENE_FAILED -11            ///< Scene loading failed. I.e root node is null but scene is still empty
#define API_PLUGIN_INVALID_LOADING -20  ///< Error while loading SOFA plugin. Plugin library file is invalid.
#define API_PLUGIN_MISSING_SYMBOL -21   ///< Error while loading SOFA plugin. Plugin library has missing symbol such as: initExternalModule
#define API_PLUGIN_FILE_NOT_FOUND -22   ///< Error while loading SOFA plugin. Plugin library file not found
#define API_PLUGIN_LOADING_FAILED -23   ///< Error while loading SOFA plugin. Plugin library loading fail for another unknown reason.

/// Internal implementation sub-class
class SofaPhysicsSimulation;


/// Main class used to control a Sofa Simulation
class SOFA_SOFAPHYSICSAPI_API SofaPhysicsAPI
{
public:
    SofaPhysicsAPI(bool useGUI = false, int GUIFramerate = 0);
    virtual ~SofaPhysicsAPI();

    /// Load an XML file containing the main scene description. Will return API_SUCCESS or API_SCENE_FAILED if loading failed
    int load(const char* filename);
    /// Call unload of the current scene graph. Will return API_SUCCESS or API_SCENE_NULL if scene is null
    int unload();
    /// Method to load a SOFA .ini config file at given path @param pathIniFile to define resource/example paths. Return share path.
    const char* loadSofaIni(const char* pathIniFile);
    /// Method to load a specific SOFA plugin using it's full path @param pluginPath. Return error code.
    int loadPlugin(const char* pluginPath);

    /// Get the current api Name behind this interface.
    virtual const char* APIName();

    /// Create an empty scene with only a SOFA root Node.
    virtual void createScene();

    /// Start the simulation
    /// Currently this simply sets the animated flag to true, but this might
    /// start a separate computation thread in a future version
    void start();

    /// Stop/pause the simulation
    void stop();

    /// Compute one simulation time-step
    void step();

    /// Reset the simulation to its initial state
    void reset();

    /// Send an event to the simulation for custom controls
    /// (such as switching active instrument)
    void sendValue(const char* name, double value);

    /// Reset the camera to its default position
    void resetView();

    /// Render the scene using OpenGL
    void drawGL();

    /// Return the number of currently active output meshes
    unsigned int           getNbOutputMeshes() const;

    /// return pointer to the @param meshID 'th SofaPhysicsOutputMesh 
    SofaPhysicsOutputMesh* getOutputMeshPtr(unsigned int meshID) const;
    /// return pointer to the @param meshID 'th SofaPhysicsOutputMesh. Return nullptr if out of bounds.
    SofaPhysicsOutputMesh* getOutputMeshPtr(const char* name) const;
    /// returns pointer to the SofaPhysicsOutputMesh with the name equal to @param name. Return nullptr if not found.
    SofaPhysicsOutputMesh** getOutputMesh(unsigned int meshID);

    /// Return the number of currently active output Tetrahedron meshes
    unsigned int getNbOutputMeshTetrahedrons();

    /// Return an array of pointers to active output meshes
    SofaPhysicsOutputMesh** getOutputMeshes();

    /// Return true if the simulation is running
    /// Note that currently you must call the step() method
    /// periodically to actually animate the scene
    bool isAnimated() const;

    /// Set the animated state to a given value (requires a
    /// simulation to be loaded)
    void setAnimated(bool val);

    /// Return the main simulation file name (from the last
    /// call to load())
    const char* getSceneFileName() const;

    /// Return the current time-step (or 0 if no simulation
    /// is loaded)
    double getTimeStep() const;
    /// Control the timestep of the simulation (requires a
    /// simulation to be loaded)
    void   setTimeStep(double dt);

    /// Return the current simulated time
    double getTime() const;

    /// Return the current computation speed (averaged over
    /// the last 100 steps)
    double getCurrentFPS() const;

    double* getGravity() const;
    /// Get the current scene gravity using the ouptut @param values which is a double[3]. Return error code.
    int getGravity(double* values) const;
    /// Set the current scene gravity using the input @param gravity which is a double[3]
    void setGravity(double* gravity);

    /// message API
    /// Method to activate/deactivate SOFA MessageHandler according to @param value. Return Error code.
    int activateMessageHandler(bool value);
    /// Method to get the number of messages in queue
    int getNbMessages();
    /// Method to return the queued message of index @param messageId and its type level inside @param msgType
    const char* getMessage(int messageId, int& msgType);
    /// Method clear the list of queued messages. Return Error code.
    int clearMessages();


    /// Return the number of currently active data monitors
    unsigned int getNbDataMonitors();

    /// Return an array of pointers to active data monitors
    SofaPhysicsDataMonitor** getDataMonitors();

    /// Return the number of currently active data controllers
    unsigned int getNbDataControllers();

    /// Return an array of pointers to active data controllers
    SofaPhysicsDataController** getDataControllers();

    /// Internal implementation sub-class
    SofaPhysicsSimulation* impl;
};

/// Class describing one output mesh (i.e. visual model) in the simulation
class SOFA_SOFAPHYSICSAPI_API SofaPhysicsOutputMesh
{
public:

    SofaPhysicsOutputMesh();
    ~SofaPhysicsOutputMesh();

    const char* getName(); ///< (non-unique) name of this object
    ID          getID();   ///< unique ID of this object

    unsigned int getNbVertices(); ///< number of vertices
    const Real* getVPositions();  ///< vertices positions (Vec3)
    int getVPositions(Real* values); ///< get the positions/vertices of this mesh inside ouput @param values, of type Real[ 3*nbVertices ]. Return error code.
    const Real* getVNormals();    ///< vertices normals   (Vec3)
    int getVNormals(Real* values); ///< get the normals per vertex of this mesh inside ouput @param values, of type Real[ 3*nbVertices ]. Return error code.
    const Real* getVTexCoords();  ///< vertices UVs       (Vec2)
    int getVTexCoords(Real* values); ///< get the texture coordinates (UV) per vertex of this mesh inside ouput @param values, of type Real[ 2*nbVertices ]. Return error code.
    int getTexCoordRevision();    ///< changes each time texture coord data are updated
    int getVerticesRevision();    ///< changes each time vertices data are updated

    unsigned int getNbVAttributes();                    ///< number of vertices attributes
    unsigned int getNbAttributes(int index);            ///< number of the attributes in specified vertex attribute 
    const char*  getVAttributeName(int index);          ///< vertices attribute name
    int          getVAttributeSizePerVertex(int index); ///< vertices attribute #
    const Real*  getVAttributeValue(int index);         ///< vertices attribute (Vec#)
    int          getVAttributeRevision(int index);      ///< changes each time vertices attribute is updated

    unsigned int getNbLines(); ///< number of lines
    const Index* getLines();   ///< lines topology (2 indices / line)
    int getLinesRevision();    ///< changes each time lines data is updated

    unsigned int getNbTriangles(); ///< number of triangles
    const Index* getTriangles();   ///< triangles topology (3 indices / triangle)
    int getTriangles(int* values); ///< get the triangle topology inside ouput @param values, of type int[ 3*nbTriangles ]. Return error code.
    int getTrianglesRevision();    ///< changes each time triangles data is updated

    unsigned int getNbQuads(); ///< number of quads
    const Index* getQuads();   ///< quads topology (4 indices / quad)
    int getQuads(int* values); ///< get the quad topology inside ouput @param values, of type int[ 4*nbQuads ]. Return error code.
    int getQuadsRevision();    ///< changes each time quads data is updated

    /// Internal implementation sub-class
    class Impl;
    /// Internal implementation sub-class
    Impl* impl;
};

/// Class for data monitoring
class SOFA_SOFAPHYSICSAPI_API SofaPhysicsDataMonitor
{
public:

    SofaPhysicsDataMonitor();
    ~SofaPhysicsDataMonitor();

    const char* getName(); ///< (non-unique) name of this object
    ID          getID();   ///< unique ID of this object

    /// Get the value of the associated variable
    const char* getValue();

    /// Internal implementation sub-class
    class Impl;
    /// Internal implementation sub-class
    Impl* impl;
};

/// Class for data control
class SOFA_SOFAPHYSICSAPI_API SofaPhysicsDataController
{
public:

    SofaPhysicsDataController();
    ~SofaPhysicsDataController();

    const char* getName(); ///< (non-unique) name of this object
    ID          getID();   ///< unique ID of this object

    /// Set the value of the associated variable
    void setValue(const char* v);

    /// Internal implementation sub-class
    class Impl;
    /// Internal implementation sub-class
    Impl* impl;
};
