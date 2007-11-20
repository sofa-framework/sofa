/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_VISUALMODEL_VISUALMODELIMPL_H
#define SOFA_COMPONENT_VISUALMODEL_VISUALMODELIMPL_H

#include <string>
#include <sofa/core/VisualModel.h>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/TriangleSetTopology.h>
#include <sofa/component/topology/TetrahedronSetTopology.h>
#include <sofa/helper/io/Mesh.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

using namespace sofa::defaulttype;


class RigidMappedModel : public core::componentmodel::behavior::MappedModel< Rigid3fTypes >
{
public:
    VecCoord xforms;
    bool xformsModified;

    RigidMappedModel()
        : xformsModified(false)
    {
    }

    const VecCoord* getX()  const { return &xforms; }
    const VecDeriv* getV()  const { return NULL; }

    VecCoord* getX()  { xformsModified = true; return &xforms;   }
    VecDeriv* getV()  { return NULL; }

    const VecCoord* getRigidX()  const { return getX(); }
    VecCoord* getRigidX()  { return getX(); }
};

class ExtVec3fMappedModel : public core::componentmodel::behavior::MappedModel< ExtVec3fTypes >
{
public:
    ResizableExtVector<Coord>* inputVertices;
    bool modified; ///< True if input vertices modified since last rendering

    ExtVec3fMappedModel()
        : inputVertices(NULL), modified(false)
    {
    }

    const VecCoord* getX()  const { return inputVertices; }
    const VecDeriv* getV()  const { return NULL; }

    VecCoord* getX()  { modified = true; return inputVertices; }
    VecDeriv* getV()  { return NULL; }

    const VecCoord* getVecX()  const { return getX(); }
    VecCoord* getVecX()  { return getX(); }
};

class VisualModelImpl : public core::VisualModel, public ExtVec3fMappedModel, public RigidMappedModel
{
protected:
    // use types from ExtVec3fTypes

    typedef ExtVec3fTypes::Real Real;
    typedef ExtVec3fTypes::Coord Coord;
    typedef ExtVec3fTypes::VecCoord VecCoord;
    typedef ExtVec3fTypes::Deriv Deriv;
    typedef ExtVec3fTypes::VecDeriv VecDeriv;


    typedef Vec<2, float> TexCoord;
    typedef helper::fixed_array<int, 3> Triangle;
    typedef helper::fixed_array<int, 4> Quad;

    //ResizableExtVector<Coord>* inputVertices;

    bool useTopology; ///< True if list of facets should be taken from the attached topology
    int lastMeshRev; ///< Time stamps from the last time the mesh was updated from the topology
    bool useNormals; ///< True if normals should be read from file
    bool castShadow; ///< True if object cast shadows

    /*     Data< ResizableExtVector<Coord> > vertices; */
    DataPtr< ResizableExtVector<Coord> > field_vertices;
    ResizableExtVector<Coord> vertices;

    DataPtr< ResizableExtVector<Coord> > field_vnormals;
    ResizableExtVector<Coord> vnormals;
    DataPtr< ResizableExtVector<TexCoord> > field_vtexcoords;
    ResizableExtVector<TexCoord> vtexcoords;

    DataPtr< ResizableExtVector<Triangle> > field_triangles;
    ResizableExtVector<Triangle> triangles;
    DataPtr< ResizableExtVector<Quad> > field_quads;
    ResizableExtVector<Quad> quads;

    /// If vertices have multiple normals/texcoords, then we need to separate them
    /// This vector store which input position is used for each vertice
    /// If it is empty then each vertex correspond to one position
    ResizableExtVector<int> vertPosIdx;

    /// Similarly this vector store which input normal is used for each vertice
    /// If it is empty then each vertex correspond to one normal
    ResizableExtVector<int> vertNormIdx;

    Data< std::string > texturename;

    Vec3f bbox[2];

    virtual void internalDraw()
    {}

public:
    Data< sofa::helper::io::Mesh::Material > material;

    VisualModelImpl();

    ~VisualModelImpl();

    void parse(core::objectmodel::BaseObjectDescription* arg);

    bool isTransparent();

    void draw();
    void drawTransparent();
    void drawShadow();

    virtual bool loadTexture(const std::string& /*filename*/) { return false; }

    bool load(const std::string& filename, const std::string& loader, const std::string& textureName);

    void applyTranslation(double dx, double dy, double dz);
    void applyRotation(Quat q);
    void applyScale(double s);
    void applyUVTranslation(double dU, double dV);
    void applyUVScale(double su, double sv);

    void flipFaces();

    void setColor(float r, float g, float b, float a);
    void setColor(std::string color);

    void setUseNormals(bool val) { useNormals = val;  }
    bool getUseNormals() const   { return useNormals; }

    void setCastShadow(bool val) { castShadow = val;  }
    bool getCastShadow() const   { return castShadow; }

    virtual void computePositions();
    virtual void computeMesh(sofa::component::topology::MeshTopology* topology);
    virtual void computeMeshFromTopology(sofa::core::componentmodel::topology::BaseTopology* topology);
    virtual void computeNormals();
    virtual void computeBBox();

    virtual void update();

    // handle topological changes
    virtual void handleTopologyChange();

    void init();

    void initTextures();

    bool addBBox(double* minBBox, double* maxBBox);

    //const VecCoord* getX()  const; // { return &x;   }
    //const VecDeriv* getV()  const { return NULL; }

    //VecCoord* getX(); //  { return &x;   }
    //VecDeriv* getV()  { return NULL; }

    /// Append this mesh to an OBJ format stream.
    /// The number of vertices position, normal, and texture coordinates already written is given as parameters
    /// This method should update them
    virtual void exportOBJ(std::string name, std::ostream* out, std::ostream* mtl, int& vindex, int& nindex, int& tindex);
};

//typedef Vec<3,GLfloat> GLVec3f;
//typedef ExtVectorTypes<GLVec3f,GLVec3f> GLExtVec3fTypes;

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif
