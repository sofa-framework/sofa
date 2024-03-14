/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once
#include <sofa/component/visual/config.h>

#include <sofa/type/Vec.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/visual/VisualState.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <string>

namespace sofa::component::visual
{

SOFA_ATTRIBUTE_DISABLED__VEC3STATE_AS_VISUALSTATE() 
typedef sofa::core::visual::VisualState<defaulttype::Vec3Types> Vec3State;


/**
 *  \brief Abstract class which implements partially VisualModel.
 *
 *  This class implemented all non-hardware (i.e OpenGL or DirectX)
 *  specific functions for rendering. It takes a 3D model (basically a .OBJ model)
 *  and apply transformations on it.
 *  At the moment, it is only implemented by OglModel for OpenGL systems.
 *
 */
class SOFA_COMPONENT_VISUAL_API VisualModelImpl : public core::visual::VisualModel, public sofa::core::visual::VisualState<defaulttype::Vec3Types>
{
public:
    SOFA_CLASS2(VisualModelImpl, core::visual::VisualModel, sofa::core::visual::VisualState<defaulttype::Vec3Types>);

    typedef sofa::type::Vec<2, float> TexCoord;
    typedef type::vector<TexCoord> VecTexCoord;

    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Index, sofa::Index);

    //Indices must be unsigned int for drawing
    using visual_index_type = unsigned int;

    typedef type::fixed_array<visual_index_type, 2> VisualEdge;
    typedef type::fixed_array<visual_index_type, 3> VisualTriangle;
    typedef type::fixed_array<visual_index_type, 4> VisualQuad;
    typedef type::vector<VisualEdge> VecVisualEdge;
    typedef type::vector<VisualTriangle> VecVisualTriangle;
    typedef type::vector<VisualQuad> VecVisualQuad;

    typedef sofa::core::visual::VisualState<defaulttype::Vec3Types>::DataTypes DataTypes;
    typedef DataTypes::Real Real;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::VecCoord VecCoord;
    typedef DataTypes::Deriv Deriv;
    typedef DataTypes::VecDeriv VecDeriv;


    bool useTopology; ///< True if list of facets should be taken from the attached topology
    int lastMeshRev; ///< Time stamps from the last time the mesh was updated from the topology
    bool castShadow; ///< True if object cast shadows

    sofa::core::topology::BaseMeshTopology* m_topology;

    Data<bool> m_initRestPositions; ///< True if rest positions should be initialized with initial positions, False if nothing should be done
    Data<bool> m_useNormals; ///< True if normals should be read from file
    Data<bool> m_updateNormals; ///< True if normals should be updated at each iteration
    Data<bool> m_computeTangents; ///< True if tangents should be computed at startup
    Data<bool> m_updateTangents; ///< True if tangents should be updated at each iteration
    Data<bool> m_handleDynamicTopology; ///< True if topological changes should be handled
    Data<bool> m_fixMergedUVSeams; ///< True if UV seams should be handled even when duplicate UVs are merged
    Data<bool> m_keepLines; ///< keep and draw lines (false by default)

    Data< VecCoord > m_vertices2; ///< vertices of the model (only if vertices have multiple normals/texcoords, otherwise positions are used)
    core::topology::PointData< VecTexCoord > m_vtexcoords; ///< coordinates of the texture
    core::topology::PointData< VecCoord > m_vtangents; ///< tangents for normal mapping
    core::topology::PointData< VecCoord > m_vbitangents; ///< tangents for normal mapping
    core::topology::EdgeData< VecVisualEdge > m_edges; ///< edges of the model
    core::topology::TriangleData< VecVisualTriangle > m_triangles; ///< triangles of the model
    core::topology::QuadData< VecVisualQuad > m_quads; ///< quads of the model

    bool m_textureChanged {false};

    /// If vertices have multiple normals/texcoords, then we need to separate them
    /// This vector store which input position is used for each vertex
    /// If it is empty then each vertex correspond to one position
    Data< type::vector<visual_index_type> > m_vertPosIdx;

    /// Similarly this vector store which input normal is used for each vertex
    /// If it is empty then each vertex correspond to one normal
    Data< type::vector<visual_index_type> > m_vertNormIdx;

    /// Rendering method.
    virtual void internalDraw(const core::visual::VisualParams* /*vparams*/, bool /*transparent*/) {}

public:

    sofa::core::objectmodel::DataFileName fileMesh;
    sofa::core::objectmodel::DataFileName texturename;

    /// @name Initial transformation attributes
    /// @{
    typedef sofa::type::Vec<3,Real> Vec3Real;
    Data< Vec3Real > m_translation; ///< Initial Translation of the object
    Data< Vec3Real > m_rotation; ///< Initial Rotation of the object
    Data< Vec3Real > m_scale; ///< Initial Scale of the object

    Data< TexCoord > m_scaleTex; ///< Scale of the texture
    Data< TexCoord > m_translationTex; ///< Translation of the texture

    void applyTranslation(const SReal dx, const SReal dy, const SReal dz) override;

    /// Apply Rotation from Euler angles (in degree!)
    void applyRotation (const SReal rx, const SReal ry, const SReal rz) override;

    void applyRotation(const sofa::type::Quat<SReal> q) override;

    void applyScale(const SReal sx, const SReal sy, const SReal sz) override;

    virtual void applyUVTransformation();

    void applyUVTranslation(const Real dU, const Real dV);

    void applyUVScale(const Real su, const Real sv);

    void setTranslation(SReal dx, SReal dy, SReal dz)
    {
        m_translation.setValue(Vec3Real((Real)dx,(Real)dy,(Real)dz));
    }

    void setRotation(SReal rx, SReal ry, SReal rz)
    {
        m_rotation.setValue(Vec3Real((Real)rx,(Real)ry,(Real)rz));
    }

    void setScale(SReal sx, SReal sy, SReal sz)
    {
        m_scale.setValue(Vec3Real((Real)sx,(Real)sy,(Real)sz));
    }
    /// @}

    sofa::type::Vec3f bbox[2];
    Data< sofa::type::Material > material;
    Data< bool > putOnlyTexCoords;
    Data< bool > srgbTexturing;

    class FaceGroup
    {
    public:
        /// tri0: first triangle index of a group
        /// nbt: number of triangle elements
        visual_index_type tri0, nbt;

        /// quad0: first quad index of a group
        /// nbq: number of quad elements
        visual_index_type quad0, nbq;

        /// edge0: first edge index of a group
        /// nbe: number of edge elements
        visual_index_type edge0, nbe;

        std::string materialName;
        std::string groupName;
        int materialId;
        FaceGroup() : tri0(0), nbt(0), quad0(0), nbq(0), edge0(0), nbe(0), materialName("defaultMaterial"), groupName("defaultGroup"), materialId(-1) {}
        inline friend std::ostream& operator << (std::ostream& out, const FaceGroup &g)
        {
            out << g.groupName << " " << g.materialName << " " << g.materialId << " " << g.tri0 << " " << g.nbt << " " << g.quad0 << " " << g.nbq << " " << g.edge0 << " " << g.nbe;
            return out;
        }
        inline friend std::istream& operator >> (std::istream& in, FaceGroup &g)
        {

            in >> g.groupName >> g.materialName >> g.materialId >> g.tri0 >> g.nbt >> g.quad0 >> g.nbq >> g.edge0 >> g.nbe;
            return in;
        }
    };

    Data< type::vector<sofa::type::Material> > materials;
    Data< type::vector<FaceGroup> > groups;

    /// Link to be set to the topology container in the component graph.
    SingleLink <VisualModelImpl, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;
protected:
    /// Default constructor.
    VisualModelImpl();

    /// Default destructor.
    ~VisualModelImpl() override;

public:
    void parse(core::objectmodel::BaseObjectDescription* arg) override;

    virtual bool hasTransparent();
    bool hasOpaque();
    
    void doDrawVisual(const core::visual::VisualParams* vparams) override;
    void drawTransparent(const core::visual::VisualParams* vparams) override;
    void drawShadow(const core::visual::VisualParams* vparams) override;

    virtual bool loadTextures() {return false;}
    virtual bool loadTexture(const std::string& /*filename*/) { return false; }

    bool load(const std::string& filename, const std::string& loader, const std::string& textureName);

    void flipFaces();

    void setFilename(std::string s)
    {
        fileMesh.setValue(s);
    }

    std::string getFilename() {return fileMesh.getValue();}

    void setColor(float r, float g, float b, float a);

    void setColor(std::string color);

    void setUseNormals(bool val)
    {
        m_useNormals.setValue(val);
    }

    bool getUseNormals() const
    {
        return m_useNormals.getValue();
    }

    void setCastShadow(bool val)
    {
        castShadow = val;
    }

    bool getCastShadow() const
    {
        return castShadow;
    }

    void setMesh(helper::io::Mesh &m, bool tex=false);

    bool isUsingTopology() const
    {
        return useTopology;
    }

    const VecCoord& getVertices() const
    {
        if (!m_vertPosIdx.getValue().empty())
        {
            // Splitted vertices for multiple texture or normal coordinates per vertex.
            return m_vertices2.getValue();
        }

        return m_positions.getValue();
    }

    const VecDeriv& getVnormals() const
    {
        return m_vnormals.getValue();
    }

    const VecTexCoord& getVtexcoords() const
    {
        return m_vtexcoords.getValue();
    }

    const VecCoord& getVtangents() const
    {
        return m_vtangents.getValue();
    }

    const VecCoord& getVbitangents() const
    {
        return m_vbitangents.getValue();
    }

    const VecVisualTriangle& getTriangles() const
    {
        return m_triangles.getValue();
    }

    const VecVisualQuad& getQuads() const
    {
        return m_quads.getValue();
    }

    const VecVisualEdge& getEdges() const
    {
        return m_edges.getValue();
    }

    void setVertices(VecCoord * x)
    {
        this->m_positions.setValue(*x);
    }

    void setVnormals(VecDeriv * vn)
    {
        m_vnormals.setValue(*vn);
    }

    void setVtexcoords(VecTexCoord * vt)
    {
        m_vtexcoords.setValue(*vt);
    }

    void setVtangents(VecCoord * v)
    {
        m_vtangents.setValue(*v);
    }

    void setVbitangents(VecCoord * v)
    {
        m_vbitangents.setValue(*v);
    }

    void setTriangles(VecVisualTriangle * t)
    {
        m_triangles.setValue(*t);
    }

    void setQuads(VecVisualQuad * q)
    {
        m_quads.setValue(*q);
    }

    void setEdges(VecVisualEdge * e)
    {
        m_edges.setValue(*e);
    }

    virtual void computePositions();
    virtual void computeMesh();
    virtual void computeNormals();
    virtual void computeTangents();
    void computeBBox(const core::ExecParams* params, bool=false) override;
    virtual void computeUVSphereProjection();

    virtual void updateBuffers() {}
    virtual void deleteBuffers() {}
    virtual void deleteTextures() {}

    void updateVisual() override;

    void init() override;
    void initFromTopology();
    void initPositionFromVertices();
    void initFromFileMesh();

    void initVisual() override;

    /// Append this mesh to an OBJ format stream.
    /// The number of vertices position, normal, and texture coordinates already written is given as parameters
    /// This method should update them
    void exportOBJ(std::string name, std::ostream* out, std::ostream* mtl, sofa::Index& vindex, sofa::Index& nindex, sofa::Index& tindex, int& count) override;

    /// Returns the sofa class name. By default the name of the c++ class is exposed...
    /// More details on the name customization infrastructure is in NameDecoder.h
    static std::string GetCustomTemplateName()
    {
        return sofa::defaulttype::Vec3Types::Name();
    }

    /// Utility method to compute tangent from vertices and texture coordinates.
    static Coord computeTangent(const Coord &v1, const Coord &v2, const Coord &v3,
                                const TexCoord &t1, const TexCoord &t2, const TexCoord &t3);

    /// Utility method to compute bitangent from vertices and texture coordinates.
    static Coord computeBitangent(const Coord &v1, const Coord &v2, const Coord &v3,
                                  const TexCoord &t1, const TexCoord &t2, const TexCoord &t3);

    /// Temporary added here from RigidState deprecated inheritance
    sofa::defaulttype::Rigid3fTypes::VecCoord xforms;
    bool xformsModified;


    bool insertInNode( core::objectmodel::BaseNode* node ) override { Inherit1::insertInNode(node); Inherit2::insertInNode(node); return true; }
    bool removeInNode( core::objectmodel::BaseNode* node ) override { Inherit1::removeInNode(node); Inherit2::removeInNode(node); return true; }

protected:
    /// Internal buffer to be filled by topology Data @sa m_triangles callback when points are removed. Those dirty triangles will be updated at next updateVisual
    /// This avoid to update the whole mesh.
    std::set< sofa::core::topology::BaseMeshTopology::TriangleID> m_dirtyTriangles;

    /// Internal buffer similar to @sa m_dirtyTriangles but to be used by topolgy Data @sa m_quads callback when points are removed.
    std::set< sofa::core::topology::BaseMeshTopology::QuadID> m_dirtyQuads;
};


} // namespace sofa::component::visual
