#ifndef SOFAPHYSICSAPI_H
#define SOFAPHYSICSAPI_H

class SofaPhysicsSimulation;
class SofaPhysicsOutputMesh;

typedef unsigned int Index; ///< Type used for topology indices
typedef float Real;         ///< Type used for coordinates
typedef void* ID;           ///< Type used for IDs

class SofaPhysicsSimulation
{
public:
    SofaPhysicsSimulation();
    ~SofaPhysicsSimulation();

    bool load(const char* filename);
    void start();
    void stop();
    void step();

    void reset();
    void resetView();
    void sendValue(const char* name, double value);
    void drawGL();

    unsigned int            getNbOutputMeshes();
    SofaPhysicsOutputMesh** getOutputMeshes();

    bool isAnimated() const;
    void setAnimated(bool val);

    const char* getSceneFileName() const;

    double getTimeStep() const;
    void   setTimeStep(double dt);
    double getCurrentFPS() const;

    class Impl;
    Impl* impl;
};


class SofaPhysicsOutputMesh
{
public:

    SofaPhysicsOutputMesh();
    ~SofaPhysicsOutputMesh();

    const char* getName(); ///< (non-unique) name of this object
    ID          getID();   ///< unique ID of this object

    unsigned int getNbVertices(); ///< number of vertices
    const Real* getVPositions();  ///< vertices positions (Vec3)
    const Real* getVNormals();    ///< vertices normals   (Vec3)
    const Real* getVTexCoords();  ///< vertices UVs       (Vec2)
    int getVerticesRevision();    ///< changes each time vertices data are updated

    unsigned int getNbLines(); ///< number of lines
    const Index* getLines();   ///< lines topology (2 indices / line)
    int getLinesRevision();    ///< changes each time lines data is updated

    unsigned int getNbTriangles(); ///< number of triangles
    const Index* getTriangles();   ///< triangles topology (3 indices / triangle)
    int getTrianglesRevision();    ///< changes each time triangles data is updated

    unsigned int getNbQuads(); ///< number of quads
    const Index* getQuads();   ///< quads topology (4 indices / quad)
    int getQuadsRevision();    ///< changes each time quads data is updated

    class Impl;
    Impl* impl;
};

#endif // SOFAPHYSICSAPI_H
