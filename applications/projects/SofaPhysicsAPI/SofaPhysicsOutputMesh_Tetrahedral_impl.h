#ifndef SOFAPHYSICSOUTPUTMESH_TETRAHEDRAL_IMPL_H
#define SOFAPHYSICSOUTPUTMESH_TETRAHEDRAL_IMPL_H

#include "SofaPhysicsAPI.h"

//#include <sofa/component/visualmodel/VisualModelImpl.h>
#include <sofa/component/visualmodel/OglTetrahedralModel.h>
#include <sofa/core/visual/Shader.h>
#include <sofa/defaulttype/Vec.h>

class SofaPhysicsOutputMeshTetrahedral::Impl
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
    int getTexCoordRevision();    ///< changes each time texture coord data are updated
    int getVerticesRevision();    ///< changes each time vertices data are updated
    
    unsigned int getNbVAttributes();                    ///< number of vertices attributes
    const char*  getVAttributeName(int index);          ///< vertices attribute name
    int          getVAttributeSizePerVertex(int index); ///< vertices attribute #
    const Real*  getVAttributeValue(int index);         ///< vertices attribute (Vec#)
    int          getVAttributeRevision(int index);      ///< changes each time vertices attribute is updated

    unsigned int getNbLines(); ///< number of lines
    const Index* getLines();   ///< lines topology (2 indices / line)
    int getLinesRevision();    ///< changes each time lines data is updated

    unsigned int getNbTriangles(); ///< number of triangles
    const Index* getTriangles();   ///< triangles topology (3 indices / triangle)
    int getTrianglesRevision();    ///< changes each time triangles data is updated

    unsigned int getNbQuads(); ///< number of quads
    const Index* getQuads();   ///< quads topology (4 indices / quad)
    int getQuadsRevision();    ///< changes each time quads data is updated

    //typedef sofa::component::visualmodel::VisualModelImpl SofaOutputMesh;
    //typedef SofaOutputMeshTetrahedral::DataTypes DataTypes;
    
    typedef sofa::defaulttype::Vec3fTypes Vec3f;

    typedef sofa::component::visualmodel::OglTetrahedralModel<Vec3f> SofaOutputMeshTetrahedral;

    typedef SofaOutputMeshTetrahedral::Coord Coord;

    



protected:
    SofaOutputMeshTetrahedral::SPtr sObj;
    //SofaOutputMesh::SPtr sObj;
    //sofa::helper::vector<SofaVAttribute::SPtr> sVA;

public:
    SofaOutputMeshTetrahedral* getObject(){return sObj.get();}
    void setObject(SofaOutputMeshTetrahedral* o);
};

#endif // SOFAPHYSICSOUTPUTMESH_TETRAHEDRAL_IMPL_H
