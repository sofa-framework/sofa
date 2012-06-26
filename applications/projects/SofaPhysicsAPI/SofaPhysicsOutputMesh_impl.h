#ifndef SOFAPHYSICSOUTPUTMESH_IMPL_H
#define SOFAPHYSICSOUTPUTMESH_IMPL_H

#include "SofaPhysicsAPI.h"

#include <sofa/component/visualmodel/VisualModelImpl.h>

class SofaPhysicsOutputMesh::Impl
{
public:

    Impl();
    ~Impl();

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

    typedef sofa::component::visualmodel::VisualModelImpl SofaOutputMesh;
    typedef SofaOutputMesh::DataTypes DataTypes;
    typedef SofaOutputMesh::Coord Coord;
    typedef SofaOutputMesh::Deriv Deriv;
    typedef SofaOutputMesh::TexCoord TexCoord;
    typedef SofaOutputMesh::Triangle Triangle;
    typedef SofaOutputMesh::Quad Quad;

protected:
    SofaOutputMesh::SPtr sObj;

public:
    SofaOutputMesh* getObject() { return sObj.get(); }
    void setObject(SofaOutputMesh* o) { sObj = o; }
};

#endif // SOFAPHYSICSOUTPUTMESH_IMPL_H
