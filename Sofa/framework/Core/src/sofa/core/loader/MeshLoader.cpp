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
#include <sofa/core/loader/MeshLoader.h>
#include <sofa/helper/io/Mesh.h>

namespace sofa::core::loader
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

MeshLoader::MeshLoader() : BaseLoader()
  , d_positions(initData(&d_positions, "position", "Vertices of the mesh loaded"))
  , d_polylines(initData(&d_polylines, "polylines", "Polylines of the mesh loaded"))
  , d_edges(initData(&d_edges, "edges", "Edges of the mesh loaded"))
  , d_triangles(initData(&d_triangles, "triangles", "Triangles of the mesh loaded"))
  , d_quads(initData(&d_quads, "quads", "Quads of the mesh loaded"))
  , d_polygons(initData(&d_polygons, "polygons", "Polygons of the mesh loaded"))
  , d_highOrderEdgePositions(initData(&d_highOrderEdgePositions, "highOrderEdgePositions", "High order edge points of the mesh loaded"))
  , d_highOrderTrianglePositions(initData(&d_highOrderTrianglePositions, "highOrderTrianglePositions", "High order triangle points of the mesh loaded"))
  , d_highOrderQuadPositions(initData(&d_highOrderQuadPositions, "highOrderQuadPositions", "High order quad points of the mesh loaded"))
  , d_tetrahedra(initData(&d_tetrahedra, "tetrahedra", "Tetrahedra of the mesh loaded"))
  , d_hexahedra(initData(&d_hexahedra, "hexahedra", "Hexahedra of the mesh loaded"))
  , d_prisms(initData(&d_prisms, "prisms", "Prisms of the mesh loaded"))
  , d_highOrderTetrahedronPositions(initData(&d_highOrderTetrahedronPositions, "highOrderTetrahedronPositions", "High order tetrahedron points of the mesh loaded"))
  , d_highOrderHexahedronPositions(initData(&d_highOrderHexahedronPositions, "highOrderHexahedronPositions", "High order hexahedron points of the mesh loaded"))
  , d_pyramids(initData(&d_pyramids, "pyramids", "Pyramids of the mesh loaded"))
  , d_normals(initData(&d_normals, "normals", "Normals of the mesh loaded"))
  , d_edgesGroups(initData(&d_edgesGroups, "edgesGroups", "Groups of Edges"))
  , d_trianglesGroups(initData(&d_trianglesGroups, "trianglesGroups", "Groups of Triangles"))
  , d_quadsGroups(initData(&d_quadsGroups, "quadsGroups", "Groups of Quads"))
  , d_polygonsGroups(initData(&d_polygonsGroups, "polygonsGroups", "Groups of Polygons"))
  , d_tetrahedraGroups(initData(&d_tetrahedraGroups, "tetrahedraGroups", "Groups of Tetrahedra"))
  , d_hexahedraGroups(initData(&d_hexahedraGroups, "hexahedraGroups", "Groups of Hexahedra"))
  , d_prismsGroups(initData(&d_prismsGroups, "prismsGroups", "Groups of Prisms"))
  , d_pyramidsGroups(initData(&d_pyramidsGroups, "pyramidsGroups", "Groups of Pyramids"))
  , d_flipNormals(initData(&d_flipNormals, false, "flipNormals", "Flip Normals"))
  , d_triangulate(initData(&d_triangulate, false, "triangulate", "Divide all polygons into triangles"))
  , d_createSubelements(initData(&d_createSubelements, false, "createSubelements", "Divide all n-D elements into their (n-1)-D boundary elements (e.g. tetrahedra to triangles)"))
  , d_onlyAttachedPoints(initData(&d_onlyAttachedPoints, false, "onlyAttachedPoints", "Only keep points attached to elements of the mesh"))
  , d_translation(initData(&d_translation, Vec3(), "translation", "Translation of the DOFs"))
  , d_rotation(initData(&d_rotation, Vec3(), "rotation", "Rotation of the DOFs"))
  , d_scale(initData(&d_scale, Vec3(1.0, 1.0, 1.0), "scale3d", "Scale of the DOFs in 3 dimensions"))
  , d_transformation(initData(&d_transformation, type::Matrix4::Identity(), "transformation", "4x4 Homogeneous matrix to transform the DOFs (when present replace any)"))
  , d_previousTransformation(type::Matrix4::Identity() )
{
    d_pentahedra.setOriginalData(&d_prisms);
    d_pentahedraGroups.setOriginalData(&d_prismsGroups);

    addAlias(&d_tetrahedra, "tetras");
    addAlias(&d_hexahedra, "hexas");
    addAlias(&d_prisms, "pentas");
    addAlias(&d_prisms, "pentahedra");
    addAlias(&d_prismsGroups, "pentahedraGroups");

    d_flipNormals.setAutoLink(false);
    d_triangulate.setAutoLink(false);
    d_createSubelements.setAutoLink(false);
    d_onlyAttachedPoints.setAutoLink(false);
    d_translation.setAutoLink(false);
    d_rotation.setAutoLink(false);
    d_scale.setAutoLink(false);
    d_transformation.setAutoLink(false);
    d_transformation.setDirtyValue();

    d_positions.setGroup("Vectors");
    d_polylines.setGroup("Vectors");
    d_edges.setGroup("Vectors");
    d_triangles.setGroup("Vectors");
    d_quads.setGroup("Vectors");
    d_polygons.setGroup("Vectors");
    d_tetrahedra.setGroup("Vectors");
    d_hexahedra.setGroup("Vectors");
    d_prisms.setGroup("Vectors");
    d_pyramids.setGroup("Vectors");
    d_normals.setGroup("Vectors");
    d_highOrderTetrahedronPositions.setGroup("Vectors");
    d_highOrderEdgePositions.setGroup("Vectors");
    d_highOrderHexahedronPositions.setGroup("Vectors");
    d_highOrderQuadPositions.setGroup("Vectors");
    d_highOrderTrianglePositions.setGroup("Vectors");

    d_edgesGroups.setGroup("Groups");
    d_quadsGroups.setGroup("Groups");
    d_polygonsGroups.setGroup("Groups");
    d_pyramidsGroups.setGroup("Groups");
    d_hexahedraGroups.setGroup("Groups");
    d_trianglesGroups.setGroup("Groups");
    d_prismsGroups.setGroup("Groups");
    d_tetrahedraGroups.setGroup("Groups");

    d_positions.setReadOnly(true);
    d_polylines.setReadOnly(true);
    d_edges.setReadOnly(true);
    d_triangles.setReadOnly(true);
    d_quads.setReadOnly(true);
    d_polygons.setReadOnly(true);
    d_highOrderEdgePositions.setReadOnly(true);
    d_highOrderTrianglePositions.setReadOnly(true);
    d_highOrderQuadPositions.setReadOnly(true);
    d_tetrahedra.setReadOnly(true);
    d_hexahedra.setReadOnly(true);
    d_prisms.setReadOnly(true);
    d_highOrderTetrahedronPositions.setReadOnly(true);
    d_highOrderHexahedronPositions.setReadOnly(true);
    d_pyramids.setReadOnly(true);
    d_normals.setReadOnly(true);

    /// name filename => component state update + change of all data field...but not visible ?
    addUpdateCallback("filename", {&d_filename}, [this](const core::DataTracker& t)
    {
        SOFA_UNUSED(t);
        if(load()){
            clearLoggedMessages();
            return sofa::core::objectmodel::ComponentState::Valid;
        }
        return sofa::core::objectmodel::ComponentState::Invalid;
    }, {&d_positions, &d_normals,
        &d_edges, &d_triangles, &d_quads, &d_tetrahedra, &d_hexahedra, &d_prisms, &d_pyramids,
        &d_polylines, &d_polygons,
        &d_highOrderEdgePositions, &d_highOrderTrianglePositions, &d_highOrderQuadPositions, &d_highOrderHexahedronPositions, &d_highOrderTetrahedronPositions,
        &d_edgesGroups, &d_quadsGroups, &d_polygonsGroups, &d_pyramidsGroups, &d_hexahedraGroups, &d_trianglesGroups, &d_prismsGroups, &d_tetrahedraGroups}
    );

    addUpdateCallback("updateTransformPosition", {&d_translation, &d_rotation, &d_scale, &d_transformation}, [this](const core::DataTracker& )
    {
        reinit();
        return sofa::core::objectmodel::ComponentState::Valid;
    }, {&d_positions});
}

void MeshLoader::clearBuffers()
{
    getWriteOnlyAccessor(d_positions).clear();
    getWriteOnlyAccessor(d_normals).clear();

    getWriteOnlyAccessor(d_edges).clear();
    getWriteOnlyAccessor(d_triangles).clear();
    getWriteOnlyAccessor(d_quads).clear();
    getWriteOnlyAccessor(d_tetrahedra).clear();
    getWriteOnlyAccessor(d_hexahedra).clear();
    getWriteOnlyAccessor(d_prisms).clear();
    getWriteOnlyAccessor(d_pyramids).clear();
    getWriteOnlyAccessor(d_polygons).clear();
    getWriteOnlyAccessor(d_polylines).clear();

    getWriteOnlyAccessor(d_highOrderEdgePositions).clear();
    getWriteOnlyAccessor(d_highOrderTrianglePositions).clear();
    getWriteOnlyAccessor(d_highOrderQuadPositions).clear();
    getWriteOnlyAccessor(d_highOrderTetrahedronPositions).clear();
    getWriteOnlyAccessor(d_highOrderHexahedronPositions).clear();

    getWriteOnlyAccessor(d_edgesGroups).clear();
    getWriteOnlyAccessor(d_trianglesGroups).clear();
    getWriteOnlyAccessor(d_quadsGroups).clear();
    getWriteOnlyAccessor(d_tetrahedraGroups).clear();
    getWriteOnlyAccessor(d_hexahedraGroups).clear();
    getWriteOnlyAccessor(d_prismsGroups).clear();
    getWriteOnlyAccessor(d_pyramidsGroups).clear();
    getWriteOnlyAccessor(d_polygonsGroups).clear();

    doClearBuffers();
}

void MeshLoader::parse(sofa::core::objectmodel::BaseObjectDescription* arg)
{
    objectmodel::BaseObject::parse(arg);

    if (arg->getAttribute("scale"))
    {
        const SReal s = (SReal) arg->getAttributeAsFloat("scale", 1.0);
        d_scale.setValue(d_scale.getValue()*s);
    }

    // File not loaded, component is set to invalid
    if (!canLoad())
        d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
}

void MeshLoader::init()
{
    BaseLoader::init();
    this->reinit();
}

void MeshLoader::reinit()
{
    type::Matrix4 transformation = d_transformation.getValue();
    const Vec3& scale = d_scale.getValue();
    const Vec3& rotation = d_rotation.getValue();
    const Vec3& translation = d_translation.getValue();


    this->applyTransformation(d_previousTransformation);
    d_previousTransformation.identity();


    if (transformation != type::Matrix4::Identity())
    {
        if (d_scale.getValue() != Vec3(1.0, 1.0, 1.0) || d_rotation.getValue() != Vec3(0.0, 0.0, 0.0) || d_translation.getValue() != Vec3(0.0, 0.0, 0.0))
        {
            msg_info() << "Parameters scale, rotation, translation ignored in favor of transformation matrix";
        }
    }
    else
    {
        // Transformation of the local frame: scale along the translated and rotated axes, then rotation around the translated origin, then translation
        // is applied to the points to implement the matrix product TRSx

        transformation = type::Matrix4::transformTranslation(translation) *
                type::Matrix4::transformRotation(type::Quat< SReal >::createQuaterFromEuler(rotation * M_PI / 180.0)) *
                type::Matrix4::transformScale(scale);
    }

    if (transformation != type::Matrix4::Identity())
    {
        this->applyTransformation(transformation);
        d_previousTransformation.transformInvert(transformation);
    }

    updateMesh();
}

bool MeshLoader::load()
{
    // Clear previously loaded buffers
    clearBuffers();

    const bool loaded = doLoad();

    // Clear (potentially) partially filled buffers
    if (!loaded)
        clearBuffers();
    return loaded;
}



bool MeshLoader::canLoad()
{
    return BaseLoader::canLoad();
}

void MeshLoader::updateMesh()
{
    updateElements();
    updatePoints();
    updateNormals();
}

template<class Vec>
static inline Vec uniqueOrder(Vec v)
{
    // simple insertion sort
    for (Size j = 1; j < v.size(); ++j)
    {
        typename Vec::value_type key = v[j];
        Size i = j;
        while (i > 0 && v[i - 1] > key)
        {
            v[i] = v[i - 1];
            --i;
        }
        v[i] = key;
    }
    return v;
}

void MeshLoader::updateElements()
{
    if (d_triangulate.getValue())
    {
        helper::WriteAccessor<Data<type::vector< Quad > > > waQuads = d_quads;
        helper::WriteAccessor<Data<type::vector< Triangle > > > waTriangles = d_triangles;

        for (Size i = 0; i < waQuads.size() ; i++)
        {
            const Quad& q = waQuads[i];
            addTriangle(waTriangles.wref(), q[0], q[1], q[2]);
            addTriangle(waTriangles.wref(), q[0], q[2], q[3]);
        }
        waQuads.clear();
    }
    if (d_hexahedra.getValue().size() > 0 && d_createSubelements.getValue())
    {
        helper::ReadAccessor<Data<type::vector< Hexahedron > > > hexahedra = this->d_hexahedra;
        helper::WriteAccessor<Data<type::vector< Quad > > > quads = this->d_quads;
        std::set<Quad > eSet;
        for (Size i = 0; i < quads.size(); ++i)
        {
            eSet.insert(uniqueOrder(quads[i]));
        }
        int nbnew = 0;
        for (Size i = 0; i < hexahedra.size(); ++i)
        {
            Hexahedron h = hexahedra[i];
            type::fixed_array< Quad, 6 > e;
            e[0] = Quad(h[0], h[3], h[2], h[1]);
            e[1] = Quad(h[4], h[5], h[6], h[7]);
            e[2] = Quad(h[0], h[1], h[5], h[4]);
            e[3] = Quad(h[1], h[2], h[6], h[5]);
            e[4] = Quad(h[2], h[3], h[7], h[6]);
            e[5] = Quad(h[3], h[0], h[4], h[7]);
            for (Size j = 0; j < e.size(); ++j)
            {
                if (eSet.insert(uniqueOrder(e[j])).second) // the element was inserted
                {
                    quads.push_back(e[j]);
                    ++nbnew;
                }
            }
        }
        if (nbnew > 0)
        {
            msg_info() << nbnew << " quads were missing around the hexahedra";
        }
    }
    if (d_prisms.getValue().size() > 0 && d_createSubelements.getValue())
    {
        helper::ReadAccessor<Data<type::vector< Prism > > > prisms = this->d_prisms;
        helper::WriteAccessor<Data<type::vector< Quad > > > quads = this->d_quads;
        helper::WriteAccessor<Data<type::vector< Triangle > > > triangles = this->d_triangles;

        std::set<Quad > eSetQuad;
        for (Size i = 0; i < quads.size(); ++i)
        {
            eSetQuad.insert(uniqueOrder(quads[i]));
        }
        int nbnewQuad = 0;

        std::set<Triangle > eSetTri;
        for (Size i = 0; i < triangles.size(); ++i)
        {
            eSetTri.insert(uniqueOrder(triangles[i]));
        }
        int nbnewTri = 0;

        for (Size i = 0; i < prisms.size(); ++i)
        {
            Prism p = prisms[i];
            //vtk ordering http://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf
            Quad quad1 = Quad(p[0], p[3], p[4], p[1]);
            Quad quad2 = Quad(p[0], p[2], p[5], p[3]);
            Quad quad3 = Quad(p[1], p[4], p[5], p[2]);
            Triangle tri1(p[0], p[1], p[2]);
            Triangle tri2(p[4], p[3], p[5]);

            if (eSetQuad.insert(uniqueOrder(quad1)).second)  // the element was inserted
            {
                quads.push_back(quad1);
                ++nbnewQuad;
            }
            if (eSetQuad.insert(uniqueOrder(quad2)).second)  // the element was inserted
            {
                quads.push_back(quad2);
                ++nbnewQuad;
            }
            if (eSetQuad.insert(uniqueOrder(quad3)).second)  // the element was inserted
            {
                quads.push_back(quad3);
                ++nbnewQuad;
            }
            if (eSetTri.insert(uniqueOrder(tri1)).second)
            {
                triangles.push_back(tri1);
                ++nbnewTri;
            }
            if (eSetTri.insert(uniqueOrder(tri2)).second)
            {
                triangles.push_back(tri2);
                ++nbnewTri;
            }

        }
        if (nbnewQuad > 0 || nbnewTri > 0 )
        {
            msg_info() << nbnewQuad << " quads, " << nbnewTri << " triangles were missing around the prism";
        }
    }
    if (d_pyramids.getValue().size() > 0 && d_createSubelements.getValue())
    {
        helper::ReadAccessor<Data<type::vector< Pyramid > > > pyramids = this->d_pyramids;

        helper::WriteAccessor<Data<type::vector< Quad > > > quads = this->d_quads;
        helper::WriteAccessor<Data<type::vector< Triangle > > > triangles = this->d_triangles;

        std::set<Quad > eSetQuad;
        for (Size i = 0; i < quads.size(); ++i)
        {
            eSetQuad.insert(uniqueOrder(quads[i]));
        }
        int nbnewQuad = 0;

        std::set<Triangle > eSetTri;
        for (Size i = 0; i < triangles.size(); ++i)
        {
            eSetTri.insert(uniqueOrder(triangles[i]));
        }
        int nbnewTri = 0;

        for (Size i = 0; i < pyramids.size(); ++i)
        {
            Pyramid p = pyramids[i];
            Quad quad = Quad(p[0], p[3], p[2], p[1]);
            Triangle tri1(p[0], p[1], p[4]);
            Triangle tri2(p[1], p[2], p[4]);
            Triangle tri3(p[3], p[4], p[2]);
            Triangle tri4(p[0], p[4], p[3]);

            if (eSetQuad.insert(uniqueOrder(quad)).second)  // the element was inserted
            {
                quads.push_back(quad);
                ++nbnewQuad;
            }
            if (eSetTri.insert(uniqueOrder(tri1)).second)
            {
                triangles.push_back(tri1);
                ++nbnewTri;
            }
            if (eSetTri.insert(uniqueOrder(tri2)).second)
            {
                triangles.push_back(tri2);
                ++nbnewTri;
            }
            if (eSetTri.insert(uniqueOrder(tri3)).second)
            {
                triangles.push_back(tri3);
                ++nbnewTri;
            }
            if (eSetTri.insert(uniqueOrder(tri4)).second)
            {
                triangles.push_back(tri4);
                ++nbnewTri;
            }

        }
        if (nbnewTri > 0 || nbnewQuad > 0)
        {
            msg_info() << nbnewTri << " triangles and " << nbnewQuad << " quads were missing around the pyramids";
        }
    }
    if (d_tetrahedra.getValue().size() > 0 && d_createSubelements.getValue())
    {
        helper::ReadAccessor<Data<type::vector< Tetrahedron > > > tetrahedra = this->d_tetrahedra;
        helper::WriteAccessor<Data<type::vector< Triangle > > > triangles = this->d_triangles;
        std::set<Triangle > eSet;
        for (Size i = 0; i < triangles.size(); ++i)
        {
            eSet.insert(uniqueOrder(triangles[i]));
        }
        int nbnew = 0;
        for (Size i = 0; i < tetrahedra.size(); ++i)
        {
            Tetrahedron t = tetrahedra[i];
            Triangle e1(t[0], t[2], t[1]);
            Triangle e2(t[0], t[1], t[3]);
            Triangle e3(t[0], t[3], t[2]);
            Triangle e4(t[1], t[2], t[3]); //vtk ordering http://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf
            if (eSet.insert(uniqueOrder(e1)).second)  // the element was inserted
            {
                triangles.push_back(e1);
                ++nbnew;
            }
            if (eSet.insert(uniqueOrder(e2)).second)
            {
                triangles.push_back(e2);
                ++nbnew;
            }
            if (eSet.insert(uniqueOrder(e3)).second)
            {
                triangles.push_back(e3);
                ++nbnew;
            }
            if (eSet.insert(uniqueOrder(e4)).second)
            {
                triangles.push_back(e4);
                ++nbnew;
            }
        }
        if (nbnew > 0)
        {
            msg_info() << nbnew << " triangles were missing around the tetrahedra";
        }
    }
    if (d_quads.getValue().size() > 0 && d_createSubelements.getValue())
    {
        helper::ReadAccessor<Data<type::vector< Quad > > > quads = this->d_quads;
        helper::WriteAccessor<Data<type::vector< Edge > > > edges = this->d_edges;
        std::set<Edge > eSet;
        for (Size i = 0; i < edges.size(); ++i)
        {
            eSet.insert(uniqueOrder(edges[i]));
        }
        int nbnew = 0;
        for (Size i = 0; i < quads.size(); ++i)
        {
            Quad t = quads[i];
            for (Size j = 0; j < t.size(); ++j)
            {
                Edge e(t[(j + 1) % t.size()], t[(j + 2) % t.size()]);
                if (eSet.insert(uniqueOrder(e)).second) // the element was inserted
                {
                    edges.push_back(e);
                    ++nbnew;
                }
            }
        }
        if (nbnew > 0)
        {
            msg_info() << nbnew << " edges were missing around the quads";
        }
    }
    if (d_triangles.getValue().size() > 0 && d_createSubelements.getValue())
    {
        helper::ReadAccessor<Data<type::vector< Triangle > > > triangles = this->d_triangles;
        helper::WriteAccessor<Data<type::vector< Edge > > > edges = this->d_edges;
        std::set<Edge > eSet;
        for (Size i = 0; i < edges.size(); ++i)
        {
            eSet.insert(uniqueOrder(edges[i]));
        }
        int nbnew = 0;
        for (Size i = 0; i < triangles.size(); ++i)
        {
            Triangle t = triangles[i];
            for (Size j = 0; j < t.size(); ++j)
            {
                Edge e(t[(j + 1) % t.size()], t[(j + 2) % t.size()]);
                if (eSet.insert(uniqueOrder(e)).second) // the element was inserted
                {
                    edges.push_back(e);
                    ++nbnew;
                }
            }
        }
        if (nbnew > 0)
        {
            msg_info() << nbnew << " edges were missing around the triangles";
        }
    }
}

void MeshLoader::updatePoints()
{
    if (d_onlyAttachedPoints.getValue())
    {
        std::set<sofa::Index> attachedPoints;
        {
            const helper::ReadAccessor<Data< type::vector< Edge > > > elems = d_edges;
            for (Size i = 0; i < elems.size(); ++i)
                for (Size j = 0; j < elems[i].size(); ++j)
                {
                    attachedPoints.insert(elems[i][j]);
                }
        }
        {
            const helper::ReadAccessor<Data< type::vector< Triangle > > > elems = d_triangles;
            for (Size i = 0; i < elems.size(); ++i)
                for (Size j = 0; j < elems[i].size(); ++j)
                {
                    attachedPoints.insert(elems[i][j]);
                }
        }
        {
            const helper::ReadAccessor<Data< type::vector< Quad > > > elems = d_quads;
            for (Size i = 0; i < elems.size(); ++i)
                for (Size j = 0; j < elems[i].size(); ++j)
                {
                    attachedPoints.insert(elems[i][j]);
                }
        }
        {
            const helper::ReadAccessor<Data< type::vector< Tetrahedron > > > elems = d_tetrahedra;
            for (Size i = 0; i < elems.size(); ++i)
                for (Size j = 0; j < elems[i].size(); ++j)
                {
                    attachedPoints.insert(elems[i][j]);
                }
        }
        {
            const helper::ReadAccessor<Data< type::vector< Prism > > > elems = d_prisms;
            for (Size i = 0; i < elems.size(); ++i)
                for (Size j = 0; j < elems[i].size(); ++j)
                {
                    attachedPoints.insert(elems[i][j]);
                }
        }
        {
            const helper::ReadAccessor<Data< type::vector< Pyramid > > > elems = d_pyramids;
            for (Size i = 0; i < elems.size(); ++i)
                for (Size j = 0; j < elems[i].size(); ++j)
                {
                    attachedPoints.insert(elems[i][j]);
                }
        }
        {
            const helper::ReadAccessor<Data< type::vector< Hexahedron > > > elems = d_hexahedra;
            for (Size i = 0; i < elems.size(); ++i)
                for (Size j = 0; j < elems[i].size(); ++j)
                {
                    attachedPoints.insert(elems[i][j]);
                }
        }
        const Size newsize = Size(attachedPoints.size());
        if (newsize == d_positions.getValue().size())
        {
            return;    // all points are attached
        }
        helper::WriteAccessor<Data<type::vector<sofa::type::Vec3 > > > waPositions = d_positions;
        type::vector<sofa::Index> old2new;
        old2new.resize(waPositions.size());
        sofa::Index p = 0;
        for (std::set<sofa::Index>::const_iterator it = attachedPoints.begin(), itend = attachedPoints.end(); it != itend; ++it)
        {
            const sofa::Index newp = *it;
            old2new[newp] = p;
            if (p != newp)
            {
                waPositions[p] = waPositions[newp];
            }
            ++p;
        }
        waPositions.resize(newsize);
        {
            helper::WriteAccessor<Data< type::vector< Edge > > > elems = d_edges;
            for (Size i = 0; i < elems.size(); ++i)
                for (Size j = 0; j < elems[i].size(); ++j)
                {
                    elems[i][j] = old2new[elems[i][j]];
                }
        }
        {
            helper::WriteAccessor<Data< type::vector< Triangle > > > elems = d_triangles;
            for (Size i = 0; i < elems.size(); ++i)
                for (Size j = 0; j < elems[i].size(); ++j)
                {
                    elems[i][j] = old2new[elems[i][j]];
                }
        }
        {
            helper::WriteAccessor<Data< type::vector< Quad > > > elems = d_quads;
            for (Size i = 0; i < elems.size(); ++i)
                for (Size j = 0; j < elems[i].size(); ++j)
                {
                    elems[i][j] = old2new[elems[i][j]];
                }
        }
        {
            helper::WriteAccessor<Data< type::vector< Tetrahedron > > > elems = d_tetrahedra;
            for (Size i = 0; i < elems.size(); ++i)
                for (Size j = 0; j < elems[i].size(); ++j)
                {
                    elems[i][j] = old2new[elems[i][j]];
                }
        }
        {
            helper::WriteAccessor<Data< type::vector< Prism > > > elems = d_prisms;
            for (Size i = 0; i < elems.size(); ++i)
                for (Size j = 0; j < elems[i].size(); ++j)
                {
                    elems[i][j] = old2new[elems[i][j]];
                }
        }
        {
            helper::WriteAccessor<Data< type::vector< Pyramid > > > elems = d_pyramids;
            for (Size i = 0; i < elems.size(); ++i)
                for (Size j = 0; j < elems[i].size(); ++j)
                {
                    elems[i][j] = old2new[elems[i][j]];
                }
        }
        {
            helper::WriteAccessor<Data< type::vector< Hexahedron > > > elems = d_hexahedra;
            for (Size i = 0; i < elems.size(); ++i)
                for (Size j = 0; j < elems[i].size(); ++j)
                {
                    elems[i][j] = old2new[elems[i][j]];
                }
        }
    }
}

void MeshLoader::updateNormals()
{
    const helper::ReadAccessor<Data<type::vector<sofa::type::Vec3 > > > raPositions = d_positions;
    const helper::ReadAccessor<Data< type::vector< Triangle > > > raTriangles = d_triangles;
    const helper::ReadAccessor<Data< type::vector< Quad > > > raQuads = d_quads;

    //look if we already have loaded normals
    if (d_normals.getValue().size() == raPositions.size())
    {
        return;
    }

    helper::WriteAccessor<Data<type::vector<sofa::type::Vec3 > > > waNormals = d_normals;

    waNormals.resize(raPositions.size());

    for (Size i = 0; i < raTriangles.size() ; i++)
    {
        const sofa::type::Vec3  v1 = raPositions[raTriangles[i][0]];
        const sofa::type::Vec3  v2 = raPositions[raTriangles[i][1]];
        const sofa::type::Vec3  v3 = raPositions[raTriangles[i][2]];
        sofa::type::Vec3 n = cross(v2 - v1, v3 - v1);

        n.normalize();
        waNormals[raTriangles[i][0]] += n;
        waNormals[raTriangles[i][1]] += n;
        waNormals[raTriangles[i][2]] += n;

    }
    for (Size i = 0; i < raQuads.size() ; i++)
    {
        const sofa::type::Vec3& v1 = raPositions[raQuads[i][0]];
        const sofa::type::Vec3& v2 = raPositions[raQuads[i][1]];
        const sofa::type::Vec3& v3 = raPositions[raQuads[i][2]];
        const sofa::type::Vec3& v4 = raPositions[raQuads[i][3]];
        sofa::type::Vec3 n1 = cross(v2 - v1, v4 - v1);
        sofa::type::Vec3 n2 = cross(v3 - v2, v1 - v2);
        sofa::type::Vec3 n3 = cross(v4 - v3, v2 - v3);
        sofa::type::Vec3 n4 = cross(v1 - v4, v3 - v4);
        n1.normalize();
        n2.normalize();
        n3.normalize();
        n4.normalize();
        waNormals[raQuads[i][0]] += n1;
        waNormals[raQuads[i][1]] += n2;
        waNormals[raQuads[i][2]] += n3;
        waNormals[raQuads[i][3]] += n4;
    }

    for (Size i = 0; i < waNormals.size(); i++)
    {
        waNormals[i].normalize();
    }
}


void MeshLoader::applyTransformation(type::Matrix4 const& T)
{
    if (!T.isTransform())
    {
        msg_info() << "applyTransformation: ignored matrix which is not a transformation T=" << T ;
        return;
    }
    sofa::helper::WriteAccessor <Data< type::vector<sofa::type::Vec3 > > > my_positions = d_positions;
    for (Size i = 0; i < my_positions.size(); i++)
    {
        my_positions[i] = T.transform(my_positions[i]);
    }
}

void MeshLoader::addPosition(type::vector<sofa::type::Vec3 >& pPositions, const sofa::type::Vec3& p)
{
    pPositions.push_back(p);
}

void MeshLoader::addPosition(type::vector<sofa::type::Vec3 >& pPositions,  SReal x, SReal y, SReal z)
{
    addPosition(pPositions, sofa::type::Vec3(x, y, z));
}

void MeshLoader::addPolyline(type::vector<Polyline>& pPolylines, Polyline p)
{
    pPolylines.push_back(p);
}

void MeshLoader::addEdge(type::vector<Edge >& pEdges, const Edge& p)
{
    pEdges.push_back(p);
}

void MeshLoader::addEdge(type::vector<Edge >& pEdges, sofa::Index p0, sofa::Index p1)
{
    addEdge(pEdges, Edge(p0, p1));
}

void MeshLoader::addTriangle(type::vector<Triangle >& pTriangles, const Triangle& p)
{
    if (d_flipNormals.getValue())
    {
        Triangle revertP;
        std::reverse_copy(p.begin(), p.end(), revertP.begin());
        pTriangles.push_back(revertP);
    }
    else
    {
        pTriangles.push_back(p);
    }
}

void MeshLoader::addTriangle(type::vector<Triangle >& pTriangles, sofa::Index p0, sofa::Index p1, sofa::Index p2)
{
    addTriangle(pTriangles, Triangle(p0, p1, p2));
}

void MeshLoader::addQuad(type::vector<Quad >& pQuads, const Quad& p)
{
    if (d_flipNormals.getValue())
    {
        Quad revertP;
        std::reverse_copy(p.begin(), p.end(), revertP.begin());

        pQuads.push_back(revertP);
    }
    else
    {
        pQuads.push_back(p);
    }
}

void MeshLoader::addQuad(type::vector<Quad >& pQuads, sofa::Index p0, sofa::Index p1, sofa::Index p2, sofa::Index p3)
{
    addQuad(pQuads, Quad(p0, p1, p2, p3));
}

void MeshLoader::addPolygon(type::vector< type::vector<sofa::Index> >& pPolygons, const type::vector<sofa::Index>& p)
{
    if (d_flipNormals.getValue())
    {
        type::vector<sofa::Index> revertP(p.size());
        std::reverse_copy(p.begin(), p.end(), revertP.begin());

        pPolygons.push_back(revertP);
    }
    else
    {
        pPolygons.push_back(p);
    }
}


void MeshLoader::addTetrahedron(type::vector< Tetrahedron >& pTetrahedra, const Tetrahedron& p)
{
    pTetrahedra.push_back(p);
}

void MeshLoader::addTetrahedron(type::vector< Tetrahedron >& pTetrahedra, sofa::Index p0, sofa::Index p1, sofa::Index p2, sofa::Index p3)
{
    addTetrahedron(pTetrahedra, Tetrahedron(p0, p1, p2, p3));
}

void MeshLoader::addHexahedron(type::vector< Hexahedron >& pHexahedra,
                               sofa::Index p0, sofa::Index p1, sofa::Index p2, sofa::Index p3,
                               sofa::Index p4, sofa::Index p5, sofa::Index p6, sofa::Index p7)
{
    addHexahedron(pHexahedra, Hexahedron(p0, p1, p2, p3, p4, p5, p6, p7));
}

void MeshLoader::addHexahedron(type::vector< Hexahedron >& pHexahedra, const Hexahedron& p)
{
    pHexahedra.push_back(p);
}

void MeshLoader::addPrism(type::vector< Prism >& pPrism,
                                sofa::Index p0, sofa::Index p1, sofa::Index p2, sofa::Index p3,
                                sofa::Index p4, sofa::Index p5)
{
    addPrism(pPrism, Prism(p0, p1, p2, p3, p4, p5));
}

void MeshLoader::addPrism(type::vector< Prism >& pPrism, const Prism& p)
{
    pPrism.push_back(p);
}

void MeshLoader::addPyramid(type::vector< Pyramid >& pPyramids,
                            sofa::Index p0, sofa::Index p1, sofa::Index p2, sofa::Index p3, sofa::Index p4)
{
    addPyramid(pPyramids, Pyramid(p0, p1, p2, p3, p4));
}

void MeshLoader::addPyramid(type::vector< Pyramid >& pPyramids, const Pyramid& p)
{
    pPyramids.push_back(p);
}

void MeshLoader::copyMeshToData(sofa::helper::io::Mesh& _mesh)
{
    // copy vertices
    d_positions.setValue(_mesh.getVertices());

    // copy 3D elements
    d_edges.setValue(_mesh.getEdges());
    d_triangles.setValue(_mesh.getTriangles());
    d_quads.setValue(_mesh.getQuads());

    // copy 3D elements
    d_tetrahedra.setValue(_mesh.getTetrahedra());
    d_hexahedra.setValue(_mesh.getHexahedra());

    // copy groups
    d_edgesGroups.setValue(_mesh.getEdgesGroups());
    d_trianglesGroups.setValue(_mesh.getTrianglesGroups());
    d_quadsGroups.setValue(_mesh.getQuadsGroups());
    d_polygonsGroups.setValue(_mesh.getPolygonsGroups());
    d_tetrahedraGroups.setValue(_mesh.getTetrahedraGroups());
    d_hexahedraGroups.setValue(_mesh.getHexahedraGroups());
    d_prismsGroups.setValue(_mesh.getPrismsGroups());
    d_pyramidsGroups.setValue(_mesh.getPyramidsGroups());

    // copy high order
    d_highOrderEdgePositions.setValue(_mesh.getHighOrderEdgePositions());
    d_highOrderTrianglePositions.setValue(_mesh.getHighOrderTrianglePositions());
    d_highOrderQuadPositions.setValue(_mesh.getHighOrderQuadPositions());
}

} // namespace sofa::core::loader





