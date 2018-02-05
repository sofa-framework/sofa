/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_VISUALMODEL_VISUALMODELIMPL_H
#define SOFA_COMPONENT_VISUALMODEL_VISUALMODELIMPL_H
#include "config.h"

#include <sofa/core/State.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/io/Mesh.h>
#include <SofaBaseTopology/TopologyData.inl>

#include <map>
#include <string>


namespace sofa
{

namespace component
{

namespace visualmodel
{

//class SOFA_BASE_VISUAL_API RigidState : public core::State< sofa::defaulttype::Rigid3fTypes >
//{
//public:
//    VecCoord xforms;
//    bool xformsModified;

//    RigidState()
//        : xformsModified(false)
//    {
//    }

//    virtual void resize(int vsize) { xformsModified = true; xforms.resize( vsize); }
//    virtual int getSize() const { return 0; }

//    const VecCoord* getX()  const { return &xforms; }
//    const VecDeriv* getV()  const { return NULL; }

//    VecCoord* getX()  { xformsModified = true; return &xforms;   }
//    VecDeriv* getV()  { return NULL; }

//    const VecCoord* getRigidX()  const { return getX(); }
//    VecCoord* getRigidX()  { return getX(); }

//    virtual       Data<VecCoord>* write(     core::VecCoordId /* v */) { return NULL; }
//    virtual const Data<VecCoord>*  read(core::ConstVecCoordId /* v */) const { return NULL; }

//    virtual       Data<VecDeriv>* write(     core::VecDerivId /* v */) { return NULL; }
//    virtual const Data<VecDeriv>*  read(core::ConstVecDerivId /* v */) const { return NULL; }

//    virtual       Data<MatrixDeriv>* write(     core::MatrixDerivId /* v */) { return NULL; }
//    virtual const Data<MatrixDeriv>*  read(core::ConstMatrixDerivId /* v */) const {  return NULL; }
//};

class SOFA_BASE_VISUAL_API ExtVec3fState : public core::State< sofa::defaulttype::ExtVec3fTypes >
{
public:
    topology::PointData< sofa::defaulttype::ResizableExtVector<Coord> > m_positions;
    topology::PointData< sofa::defaulttype::ResizableExtVector<Coord> > m_restPositions;
    topology::PointData< sofa::defaulttype::ResizableExtVector<Deriv> > m_vnormals;
    bool modified; ///< True if input vertices modified since last rendering

    ExtVec3fState()
        : m_positions(initData(&m_positions, "position", "Vertices coordinates"))
        , m_restPositions(initData(&m_restPositions, "restPosition", "Vertices rest coordinates"))
        , m_vnormals (initData (&m_vnormals, "normal", "Normals of the model"))
        , modified(false)
    {
        m_positions.setGroup("Vector");
        m_restPositions.setGroup("Vector");
        m_vnormals.setGroup("Vector");
    }

    virtual void resize(size_t vsize)
    {
        helper::WriteOnlyAccessor< Data<sofa::defaulttype::ResizableExtVector<Coord> > > positions = m_positions;
        if( positions.size() == vsize ) return;
        helper::WriteOnlyAccessor< Data<sofa::defaulttype::ResizableExtVector<Coord> > > restPositions = m_restPositions;
        helper::WriteOnlyAccessor< Data<sofa::defaulttype::ResizableExtVector<Deriv> > > normals = m_vnormals;

        positions.resize(vsize);
        restPositions.resize(vsize); // todo allocate restpos only when it is necessary
        normals.resize(vsize);

        modified = true;
    }

    virtual size_t getSize() const { return m_positions.getValue().size(); }

    //State API
    virtual       Data<VecCoord>* write(     core::VecCoordId  v )
    {
        modified = true;

        if( v == core::VecCoordId::position() )
            return &m_positions;
        if( v == core::VecCoordId::restPosition() )
            return &m_restPositions;

        return NULL;
    }
    virtual const Data<VecCoord>*  read(core::ConstVecCoordId  v )  const
    {
        if( v == core::VecCoordId::position() )
            return &m_positions;
        if( v == core::VecCoordId::restPosition() )
            return &m_restPositions;

        return NULL;
    }

    virtual Data<VecDeriv>*	write(core::VecDerivId v )
    {
        if( v == core::VecDerivId::normal() )
            return &m_vnormals;

        return NULL;
    }

    virtual const Data<VecDeriv>* read(core::ConstVecDerivId v ) const
    {
        if( v == core::VecDerivId::normal() )
            return &m_vnormals;

        return NULL;
    }

    virtual       Data<MatrixDeriv>*	write(     core::MatrixDerivId /* v */) { return NULL; }
    virtual const Data<MatrixDeriv>*	read(core::ConstMatrixDerivId /* v */) const {  return NULL; }
};

/**
 *  \brief Abstract class which implements partially VisualModel.
 *
 *  This class implemented all non-hardware (i.e OpenGL or DirectX)
 *  specific functions for rendering. It takes a 3D model (basically a .OBJ model)
 *  and apply transformations on it.
 *  At the moment, it is only implemented by OglModel for OpenGL systems.
 *
 */

class SOFA_BASE_VISUAL_API VisualModelImpl : public core::visual::VisualModel, public ExtVec3fState //, public RigidState
{
public:
    SOFA_CLASS2(VisualModelImpl, core::visual::VisualModel, ExtVec3fState);
//    SOFA_CLASS3(VisualModelImpl, core::visual::VisualModel, ExtVec3fState , RigidState);

    typedef sofa::defaulttype::Vec<2, float> TexCoord;
    //typedef helper::vector<TexCoord> VecTexCoord;
    typedef sofa::defaulttype::ResizableExtVector<TexCoord> VecTexCoord;
    
    typedef sofa::core::topology::BaseMeshTopology::Edge Edge;
    typedef sofa::core::topology::BaseMeshTopology::Triangle Triangle;
    typedef sofa::core::topology::BaseMeshTopology::Quad Quad;

//protected:

    typedef sofa::defaulttype::ExtVec3fTypes DataTypes;
    typedef DataTypes::Real Real;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::VecCoord VecCoord;
    typedef DataTypes::Deriv Deriv;
    typedef DataTypes::VecDeriv VecDeriv;

    //ResizableExtVector<Coord>* inputVertices;

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

    Data< VecCoord > m_vertices2;
    topology::PointData< VecTexCoord > m_vtexcoords;
    topology::PointData< VecCoord > m_vtangents;
    topology::PointData< VecCoord > m_vbitangents;
    Data< sofa::defaulttype::ResizableExtVector< Edge > > m_edges;
    Data< sofa::defaulttype::ResizableExtVector< Triangle > > m_triangles;
    Data< sofa::defaulttype::ResizableExtVector< Quad > > m_quads;
  

    /// If vertices have multiple normals/texcoords, then we need to separate them
    /// This vector store which input position is used for each vertex
    /// If it is empty then each vertex correspond to one position
    Data< sofa::defaulttype::ResizableExtVector<int> > m_vertPosIdx;

    /// Similarly this vector store which input normal is used for each vertex
    /// If it is empty then each vertex correspond to one normal
    Data< sofa::defaulttype::ResizableExtVector<int> > m_vertNormIdx;

    /// Rendering method.
    virtual void internalDraw(const core::visual::VisualParams* /*vparams*/, bool /*transparent*/) {}

    template<class VecType>
    void addTopoHandler(topology::PointData<VecType>* data, int algo = 0);

public:

    sofa::core::objectmodel::DataFileName fileMesh;
    sofa::core::objectmodel::DataFileName texturename;

    /// @name Initial transformation attributes
    /// @{
    typedef sofa::defaulttype::Vec<3,Real> Vec3Real;
    Data< Vec3Real > m_translation;
    Data< Vec3Real > m_rotation;
    Data< Vec3Real > m_scale;

    Data< TexCoord > m_scaleTex;
    Data< TexCoord > m_translationTex;

    virtual void applyTranslation(const SReal dx, const SReal dy, const SReal dz) override;

    /// Apply Rotation from Euler angles (in degree!)
    virtual void applyRotation (const SReal rx, const SReal ry, const SReal rz) override;

    virtual void applyRotation(const sofa::defaulttype::Quat q) override;

    virtual void applyScale(const SReal sx, const SReal sy, const SReal sz) override;

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

    sofa::defaulttype::Vec3f bbox[2];
    Data< sofa::core::loader::Material > material;
    Data< bool > putOnlyTexCoords;
    Data< bool > srgbTexturing;

    class FaceGroup
    {
    public:
        /// tri0: first triangle index of a group
        /// nbt: number of triangle elements
        int tri0, nbt;

        /// quad0: first quad index of a group
        /// nbq: number of quad elements
        int quad0, nbq;

        /// edge0: first edge index of a group
        /// nbe: number of edge elements
        int edge0, nbe;

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

    Data< helper::vector<sofa::core::loader::Material> > materials;
    Data< helper::vector<FaceGroup> > groups;
protected:
    /// Default constructor.
    VisualModelImpl();

    /// Default destructor.
    ~VisualModelImpl();

public:
    virtual void parse(core::objectmodel::BaseObjectDescription* arg) override;

    virtual bool hasTransparent();
    bool hasOpaque();

    virtual void drawVisual(const core::visual::VisualParams* vparams) override;
    virtual void drawTransparent(const core::visual::VisualParams* vparams) override;
    virtual void drawShadow(const core::visual::VisualParams* vparams) override;

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

    const sofa::defaulttype::ResizableExtVector<Coord>& getVertices() const
    {
        if (!m_vertPosIdx.getValue().empty())
        {
            // Splitted vertices for multiple texture or normal coordinates per vertex.
            return m_vertices2.getValue();
        }

        return m_positions.getValue();
    }

    const sofa::defaulttype::ResizableExtVector<Deriv>& getVnormals() const
    {
        return m_vnormals.getValue();
    }

    const VecTexCoord& getVtexcoords() const
    {
        return m_vtexcoords.getValue();
    }

    const sofa::defaulttype::ResizableExtVector<Coord>& getVtangents() const
    {
        return m_vtangents.getValue();
    }

    const sofa::defaulttype::ResizableExtVector<Coord>& getVbitangents() const
    {
        return m_vbitangents.getValue();
    }

    const sofa::defaulttype::ResizableExtVector<Triangle>& getTriangles() const
    {
        return m_triangles.getValue();
    }

    const sofa::defaulttype::ResizableExtVector<Quad>& getQuads() const
    {
        return m_quads.getValue();
    }
    
    const sofa::defaulttype::ResizableExtVector<Edge>& getEdges() const
    {
        return m_edges.getValue();
    }

    void setVertices(sofa::defaulttype::ResizableExtVector<Coord> * x)
    {
        //m_vertices2.setValue(*x);
        this->m_positions.setValue(*x);
    }

    void setVnormals(sofa::defaulttype::ResizableExtVector<Deriv> * vn)
    {
        m_vnormals.setValue(*vn);
    }

    void setVtexcoords(VecTexCoord * vt)
    {
        m_vtexcoords.setValue(*vt);
    }

    void setVtangents(sofa::defaulttype::ResizableExtVector<Coord> * v)
    {
        m_vtangents.setValue(*v);
    }

    void setVbitangents(sofa::defaulttype::ResizableExtVector<Coord> * v)
    {
        m_vbitangents.setValue(*v);
    }

    void setTriangles(sofa::defaulttype::ResizableExtVector<Triangle> * t)
    {
        m_triangles.setValue(*t);
    }

    void setQuads(sofa::defaulttype::ResizableExtVector<Quad> * q)
    {
        m_quads.setValue(*q);
    }
    
    void setEdges(sofa::defaulttype::ResizableExtVector<Edge> * e)
    {
        m_edges.setValue(*e);
    }

    virtual void computePositions();
    virtual void computeMesh();
    virtual void computeNormals();
    virtual void computeTangents();
    virtual void computeBBox(const core::ExecParams* params, bool=false) override;

    virtual void updateBuffers() {}

    virtual void updateVisual() override;

    // Handle topological changes
    virtual void handleTopologyChange() override;

    virtual void init() override;

    virtual void initVisual() override;

    /// Append this mesh to an OBJ format stream.
    /// The number of vertices position, normal, and texture coordinates already written is given as parameters
    /// This method should update them
    virtual void exportOBJ(std::string name, std::ostream* out, std::ostream* mtl, int& vindex, int& nindex, int& tindex, int& count) override;

    virtual std::string getTemplateName() const override
    {
        return ExtVec3fState::getTemplateName();
    }

    static std::string templateName(const VisualModelImpl* p = NULL)
    {
        return ExtVec3fState::templateName(p);
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


    virtual bool insertInNode( core::objectmodel::BaseNode* node ) override { Inherit1::insertInNode(node); Inherit2::insertInNode(node); return true; }
    virtual bool removeInNode( core::objectmodel::BaseNode* node ) override { Inherit1::removeInNode(node); Inherit2::removeInNode(node); return true; }
};


} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif
