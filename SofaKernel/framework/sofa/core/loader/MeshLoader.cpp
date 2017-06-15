/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <sofa/core/loader/MeshLoader.h>
#include <cstdlib>

namespace sofa
{

namespace core
{

namespace loader
{

using namespace sofa::defaulttype;

MeshLoader::MeshLoader() : BaseLoader()
    , d_positions(initData(&d_positions,"position","Vertices of the mesh loaded"))
    , d_edges(initData(&d_edges,"edges","Edges of the mesh loaded"))
    , d_triangles(initData(&d_triangles,"triangles","Triangles of the mesh loaded"))
    , d_quads(initData(&d_quads,"quads","Quads of the mesh loaded"))
    , d_polygons(initData(&d_polygons,"polygons","Polygons of the mesh loaded"))
    , d_highOrderEdgePositions(initData(&d_highOrderEdgePositions,"highOrderEdgePositions","High order edge points of the mesh loaded"))
    , d_highOrderTrianglePositions(initData(&d_highOrderTrianglePositions,"highOrderTrianglePositions","High order triangle points of the mesh loaded"))
    , d_highOrderQuadPositions(initData(&d_highOrderQuadPositions,"highOrderQuadPositions","High order quad points of the mesh loaded"))
    , d_tetrahedra(initData(&d_tetrahedra,"tetrahedra","Tetrahedra of the mesh loaded"))
    , d_hexahedra(initData(&d_hexahedra,"hexahedra","Hexahedra of the mesh loaded"))
    , d_pentahedra(initData(&d_pentahedra,"pentahedra","Pentahedra of the mesh loaded"))
    , d_pyramids(initData(&d_pyramids,"pyramids","Pyramids of the mesh loaded"))
    , d_highOrderTetrahedronPositions(initData(&d_highOrderTetrahedronPositions,"highOrderTetrahedronPositions","High order tetrahedron points of the mesh loaded"))
    , d_highOrderHexahedronPositions(initData(&d_highOrderHexahedronPositions,"highOrderHexahedronPositions","High order hexahedron points of the mesh loaded"))
    , d_normals(initData(&d_normals,"normals","Normals of the mesh loaded"))
    , d_edgesGroups(initData(&d_edgesGroups,"edgesGroups","Groups of Edges"))
    , d_trianglesGroups(initData(&d_trianglesGroups,"trianglesGroups","Groups of Triangles"))
    , d_quadsGroups(initData(&d_quadsGroups,"quadsGroups","Groups of Quads"))
    , d_polygonsGroups(initData(&d_polygonsGroups,"polygonsGroups","Groups of Polygons"))
    , d_tetrahedraGroups(initData(&d_tetrahedraGroups,"tetrahedraGroups","Groups of Tetrahedra"))
    , d_hexahedraGroups(initData(&d_hexahedraGroups,"hexahedraGroups","Groups of Hexahedra"))
    , d_pentahedraGroups(initData(&d_pentahedraGroups,"pentahedraGroups","Groups of Pentahedra"))
    , d_pyramidsGroups(initData(&d_pyramidsGroups,"pyramidsGroups","Groups of Pyramids"))
    , d_flipNormals(initData(&d_flipNormals, false,"flipNormals","Flip Normals"))
    , d_triangulate(initData(&d_triangulate,false,"triangulate","Divide all polygons into triangles"))
    , d_createSubelements(initData(&d_createSubelements,false,"createSubelements","Divide all n-D elements into their (n-1)-D boundary elements (e.g. tetrahedra to triangles)"))
    , d_onlyAttachedPoints(initData(&d_onlyAttachedPoints, false,"onlyAttachedPoints","Only keep points attached to elements of the mesh"))
    , d_translation(initData(&d_translation, Vector3(), "translation", "Translation of the DOFs"))
    , d_rotation(initData(&d_rotation, Vector3(), "rotation", "Rotation of the DOFs"))
    , d_scale(initData(&d_scale, Vector3(1.0,1.0,1.0), "scale3d", "Scale of the DOFs in 3 dimensions"))
    , d_transformation(initData(&d_transformation, Matrix4::s_identity, "transformation", "4x4 Homogeneous matrix to transform the DOFs (when present replace any)"))
    , d_previousTransformation( Matrix4::s_identity )
{
    addAlias(&d_tetrahedra,"tetras");
    addAlias(&d_hexahedra,"hexas");
    addAlias(&d_pentahedra,"pentas");

    d_flipNormals.setAutoLink(false);
    d_triangulate.setAutoLink(false);
    d_createSubelements.setAutoLink(false);
    d_onlyAttachedPoints.setAutoLink(false);
    d_translation.setAutoLink(false);
    d_rotation.setAutoLink(false);
    d_scale.setAutoLink(false);
    d_transformation.setAutoLink(false);
    d_transformation.setDirtyValue();

    d_positions.setPersistent(false);
    d_edges.setPersistent(false);
    d_triangles.setPersistent(false);
    d_quads.setPersistent(false);
    d_polygons.setPersistent(false);
    d_tetrahedra.setPersistent(false);
    d_hexahedra.setPersistent(false);
    d_pentahedra.setPersistent(false);
    d_pyramids.setPersistent(false);
    d_normals.setPersistent(false);
}


void MeshLoader::parse(sofa::core::objectmodel::BaseObjectDescription* arg)
{
    objectmodel::BaseObject::parse(arg);

    if (arg->getAttribute("scale"))
    {
        SReal s = (SReal) arg->getAttributeAsFloat("scale", 1.0);
        d_scale.setValue(d_scale.getValue()*s);
    }


    if (canLoad())
        load(/*m_filename.getFullPath().c_str()*/);
    else
        sout << "Doing nothing" << sendl;
}

void MeshLoader::init()
{
    BaseLoader::init();
    this->reinit();
}

void MeshLoader::reinit()
{
    Matrix4 transformation = d_transformation.getValue();
    const Vector3& scale = d_scale.getValue();
    const Vector3& rotation = d_rotation.getValue();
    const Vector3& translation = d_translation.getValue();


    this->applyTransformation(d_previousTransformation);
    d_previousTransformation.identity();


    if (transformation != Matrix4::s_identity) {
        if (d_scale != Vector3(1.0,1.0,1.0) || d_rotation != Vector3(0.0,0.0,0.0) || d_translation != Vector3(0.0,0.0,0.0))
            sout<< "Parameters scale, rotation, translation ignored in favor of transformation matrix" << sendl;
    }
    else {
        // Transformation of the local frame: scale along the translated and rotated axes, then rotation around the translated origin, then translation
        // is applied to the points to implement the matrix product TRSx

        transformation = Matrix4::transformTranslation(translation) *
                Matrix4::transformRotation(helper::Quater< SReal >::createQuaterFromEuler(rotation * M_PI / 180.0)) *
                Matrix4::transformScale(scale);
    }

    if (transformation != Matrix4::s_identity) {
        this->applyTransformation(transformation);
        d_previousTransformation.transformInvert(transformation);
    }

    updateMesh();
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
    for (size_t j = 1; j < v.size(); ++j)
    {
        typename Vec::value_type key = v[j];
        size_t i = j;
        while (i>0 && v[i-1] > key)
        {
            v[i] = v[i-1];
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
        helper::WriteAccessor<Data<helper::vector< Quad > > > waQuads = d_quads;
        helper::WriteAccessor<Data<helper::vector< Triangle > > > waTriangles = d_triangles;

        for (size_t i = 0; i < waQuads.size() ; i++)
        {
            const Quad& q = waQuads[i];
            addTriangle(&waTriangles.wref(), q[0], q[1], q[2]);
            addTriangle(&waTriangles.wref(), q[0], q[2], q[3]);
        }
        waQuads.clear();
    }
    if (d_hexahedra.getValue().size() > 0 && d_createSubelements.getValue())
    {
        helper::ReadAccessor<Data<helper::vector< Hexahedron > > > hexahedra = this->d_hexahedra;
        helper::WriteAccessor<Data<helper::vector< Quad > > > quads = this->d_quads;
        std::set<Quad > eSet;
        for (size_t i = 0; i < quads.size(); ++i)
            eSet.insert(uniqueOrder(quads[i]));
        int nbnew = 0;
        for (size_t i = 0; i < hexahedra.size(); ++i)
        {
            Hexahedron h = hexahedra[i];
            helper::fixed_array< Quad, 6 > e;
            e[0] = Quad(h[0],h[3],h[2],h[1]);
            e[1] = Quad(h[4],h[5],h[6],h[7]);
            e[2] = Quad(h[0],h[1],h[5],h[4]);
            e[3] = Quad(h[1],h[2],h[6],h[5]);
            e[4] = Quad(h[2],h[3],h[7],h[6]);
            e[5] = Quad(h[3],h[0],h[4],h[7]);
            for (size_t j = 0; j < e.size(); ++j)
            {
                if (eSet.insert(uniqueOrder(e[j])).second) // the element was inserted
                {
                    quads.push_back(e[j]);
                    ++nbnew;
                }
            }
        }
        if (nbnew > 0)
            sout << nbnew << " quads were missing around the hexahedra" << sendl;
    }
    if (d_pentahedra.getValue().size() > 0 && d_createSubelements.getValue())
    {
        helper::ReadAccessor<Data<helper::vector< Pentahedron > > > pentahedra = this->d_pentahedra;
        helper::WriteAccessor<Data<helper::vector< Quad > > > quads = this->d_quads;
        helper::WriteAccessor<Data<helper::vector< Triangle > > > triangles = this->d_triangles;

        std::set<Quad > eSetQuad;
        for (size_t i = 0; i < quads.size(); ++i)
            eSetQuad.insert(uniqueOrder(quads[i]));
        int nbnewQuad = 0;

        std::set<Triangle > eSetTri;
        for (size_t i = 0; i < triangles.size(); ++i)
            eSetTri.insert(uniqueOrder(triangles[i]));
        int nbnewTri = 0;

        for (size_t i = 0; i < pentahedra.size(); ++i)
        {
            Pentahedron p = pentahedra[i];
            //vtk ordering http://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf
            Quad quad1 = Quad(p[0],p[3],p[4],p[1]);
            Quad quad2 = Quad(p[0],p[2],p[5],p[3]);
            Quad quad3 = Quad(p[1],p[4],p[5],p[2]);
            Triangle tri1(p[0],p[1],p[2]);
            Triangle tri2(p[4],p[3],p[5]);

            if (eSetQuad.insert(uniqueOrder(quad1)).second){ // the element was inserted
                quads.push_back(quad1);
                ++nbnewQuad;
            }
            if (eSetQuad.insert(uniqueOrder(quad2)).second){ // the element was inserted
                quads.push_back(quad2);
                ++nbnewQuad;
            }
            if (eSetQuad.insert(uniqueOrder(quad3)).second){ // the element was inserted
                quads.push_back(quad3);
                ++nbnewQuad;
            }
            if (eSetTri.insert(uniqueOrder(tri1)).second){
                triangles.push_back(tri1);
                ++nbnewTri;
            }
            if (eSetTri.insert(uniqueOrder(tri2)).second){
                triangles.push_back(tri2);
                ++nbnewTri;
            }

        }
        if (nbnewQuad > 0 || nbnewTri>0 )
            sout << nbnewQuad << " quads, "<<nbnewTri<<" triangles were missing around the pentahedra" << sendl;
    }
    if (d_pyramids.getValue().size() > 0 && d_createSubelements.getValue())
    {
        helper::ReadAccessor<Data<helper::vector< Pyramid > > > pyramids = this->d_pyramids;

        helper::WriteAccessor<Data<helper::vector< Quad > > > quads = this->d_quads;
        helper::WriteAccessor<Data<helper::vector< Triangle > > > triangles = this->d_triangles;

        std::set<Quad > eSetQuad;
        for (size_t i = 0; i < quads.size(); ++i)
            eSetQuad.insert(uniqueOrder(quads[i]));
        int nbnewQuad = 0;

        std::set<Triangle > eSetTri;
        for (size_t i = 0; i < triangles.size(); ++i)
            eSetTri.insert(uniqueOrder(triangles[i]));
        int nbnewTri = 0;

        for (size_t i = 0; i < pyramids.size(); ++i)
        {
            Pyramid p = pyramids[i];
            Quad quad = Quad(p[0],p[3],p[2],p[1]);
            Triangle tri1(p[0],p[1],p[4]);
            Triangle tri2(p[1],p[2],p[4]);
            Triangle tri3(p[3],p[4],p[2]);
            Triangle tri4(p[0],p[4],p[3]);

            if (eSetQuad.insert(uniqueOrder(quad)).second){ // the element was inserted
                quads.push_back(quad);
                ++nbnewQuad;
            }
            if (eSetTri.insert(uniqueOrder(tri1)).second){
                triangles.push_back(tri1);
                ++nbnewTri;
            }
            if (eSetTri.insert(uniqueOrder(tri2)).second){
                triangles.push_back(tri2);
                ++nbnewTri;
            }
            if (eSetTri.insert(uniqueOrder(tri3)).second){
                triangles.push_back(tri3);
                ++nbnewTri;
            }
            if (eSetTri.insert(uniqueOrder(tri4)).second){
                triangles.push_back(tri4);
                ++nbnewTri;
            }

        }
        if (nbnewTri > 0 || nbnewQuad > 0)
            sout << nbnewTri << " triangles and "<<nbnewQuad<<" quads were missing around the pyramids" << sendl;
    }
    if (d_tetrahedra.getValue().size() > 0 && d_createSubelements.getValue())
    {
        helper::ReadAccessor<Data<helper::vector< Tetrahedron > > > tetrahedra = this->d_tetrahedra;
        helper::WriteAccessor<Data<helper::vector< Triangle > > > triangles = this->d_triangles;
        std::set<Triangle > eSet;
        for (size_t i = 0; i < triangles.size(); ++i)
            eSet.insert(uniqueOrder(triangles[i]));
        int nbnew = 0;
        for (size_t i = 0; i < tetrahedra.size(); ++i)
        {
            Tetrahedron t = tetrahedra[i];
            Triangle e1(t[0],t[2],t[1]);
            Triangle e2(t[0],t[1],t[3]);
            Triangle e3(t[0],t[3],t[2]);
            Triangle e4(t[1],t[2],t[3]);  //vtk ordering http://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf
            if (eSet.insert(uniqueOrder(e1)).second){ // the element was inserted
                triangles.push_back(e1);
                ++nbnew;
            }
            if (eSet.insert(uniqueOrder(e2)).second){
                triangles.push_back(e2);
                ++nbnew;
            }
            if (eSet.insert(uniqueOrder(e3)).second){
                triangles.push_back(e3);
                ++nbnew;
            }
            if (eSet.insert(uniqueOrder(e4)).second){
                triangles.push_back(e4);
                ++nbnew;
            }
        }
        if (nbnew > 0)
            sout << nbnew << " triangles were missing around the tetrahedra" << sendl;
    }
    if (d_quads.getValue().size() > 0 && d_createSubelements.getValue())
    {
        helper::ReadAccessor<Data<helper::vector< Quad > > > quads = this->d_quads;
        helper::WriteAccessor<Data<helper::vector< Edge > > > edges = this->d_edges;
        std::set<Edge > eSet;
        for (size_t i = 0; i < edges.size(); ++i)
            eSet.insert(uniqueOrder(edges[i]));
        int nbnew = 0;
        for (size_t i = 0; i < quads.size(); ++i)
        {
            Quad t = quads[i];
            for (size_t j = 0; j < t.size(); ++j)
            {
                Edge e(t[(j+1)%t.size()],t[(j+2)%t.size()]);
                if (eSet.insert(uniqueOrder(e)).second) // the element was inserted
                {
                    edges.push_back(e);
                    ++nbnew;
                }
            }
        }
        if (nbnew > 0)
            sout << nbnew << " edges were missing around the quads" << sendl;
    }
    if (d_triangles.getValue().size() > 0 && d_createSubelements.getValue())
    {
        helper::ReadAccessor<Data<helper::vector< Triangle > > > triangles = this->d_triangles;
        helper::WriteAccessor<Data<helper::vector< Edge > > > edges = this->d_edges;
        std::set<Edge > eSet;
        for (size_t i = 0; i < edges.size(); ++i)
            eSet.insert(uniqueOrder(edges[i]));
        int nbnew = 0;
        for (size_t i = 0; i < triangles.size(); ++i)
        {
            Triangle t = triangles[i];
            for (size_t j = 0; j < t.size(); ++j)
            {
                Edge e(t[(j+1)%t.size()],t[(j+2)%t.size()]);
                if (eSet.insert(uniqueOrder(e)).second) // the element was inserted
                {
                    edges.push_back(e);
                    ++nbnew;
                }
            }
        }
        if (nbnew > 0)
            sout << nbnew << " edges were missing around the triangles" << sendl;
    }
}

void MeshLoader::updatePoints()
{
    if (d_onlyAttachedPoints.getValue())
    {
        std::set<unsigned int> attachedPoints;
        {
            helper::ReadAccessor<Data< helper::vector< Edge > > > elems = d_edges;
            for (size_t i=0; i<elems.size(); ++i)
                for (size_t j=0; j<elems[i].size(); ++j)
                    attachedPoints.insert(elems[i][j]);
        }
        {
            helper::ReadAccessor<Data< helper::vector< Triangle > > > elems = d_triangles;
            for (size_t i=0; i<elems.size(); ++i)
                for (size_t j=0; j<elems[i].size(); ++j)
                    attachedPoints.insert(elems[i][j]);
        }
        {
            helper::ReadAccessor<Data< helper::vector< Quad > > > elems = d_quads;
            for (size_t i=0; i<elems.size(); ++i)
                for (size_t j=0; j<elems[i].size(); ++j)
                    attachedPoints.insert(elems[i][j]);
        }
        {
            helper::ReadAccessor<Data< helper::vector< Tetrahedron > > > elems = d_tetrahedra;
            for (size_t i=0; i<elems.size(); ++i)
                for (size_t j=0; j<elems[i].size(); ++j)
                    attachedPoints.insert(elems[i][j]);
        }
        {
            helper::ReadAccessor<Data< helper::vector< Pentahedron > > > elems = d_pentahedra;
            for (size_t i=0; i<elems.size(); ++i)
                for (size_t j=0; j<elems[i].size(); ++j)
                    attachedPoints.insert(elems[i][j]);
        }
        {
            helper::ReadAccessor<Data< helper::vector< Pyramid > > > elems = d_pyramids;
            for (size_t i=0; i<elems.size(); ++i)
                for (size_t j=0; j<elems[i].size(); ++j)
                    attachedPoints.insert(elems[i][j]);
        }
        {
            helper::ReadAccessor<Data< helper::vector< Hexahedron > > > elems = d_hexahedra;
            for (size_t i=0; i<elems.size(); ++i)
                for (size_t j=0; j<elems[i].size(); ++j)
                    attachedPoints.insert(elems[i][j]);
        }
        const size_t newsize = attachedPoints.size();
        if (newsize == d_positions.getValue().size()) return; // all points are attached
        helper::WriteAccessor<Data<helper::vector<sofa::defaulttype::Vec<3,SReal> > > > waPositions = d_positions;
        helper::vector<unsigned int> old2new;
        old2new.resize(waPositions.size());
        unsigned int p = 0;
        for (std::set<unsigned int>::const_iterator it = attachedPoints.begin(), itend = attachedPoints.end(); it != itend; ++it)
        {
            unsigned int newp = *it;
            old2new[newp] = p;
            if (p != newp) waPositions[p] = waPositions[newp];
            ++p;
        }
        waPositions.resize(newsize);
        {
            helper::WriteAccessor<Data< helper::vector< Edge > > > elems = d_edges;
            for (size_t i=0; i<elems.size(); ++i)
                for (size_t j=0; j<elems[i].size(); ++j)
                    elems[i][j] = old2new[elems[i][j]];
        }
        {
            helper::WriteAccessor<Data< helper::vector< Triangle > > > elems = d_triangles;
            for (size_t i=0; i<elems.size(); ++i)
                for (size_t j=0; j<elems[i].size(); ++j)
                    elems[i][j] = old2new[elems[i][j]];
        }
        {
            helper::WriteAccessor<Data< helper::vector< Quad > > > elems = d_quads;
            for (size_t i=0; i<elems.size(); ++i)
                for (size_t j=0; j<elems[i].size(); ++j)
                    elems[i][j] = old2new[elems[i][j]];
        }
        {
            helper::WriteAccessor<Data< helper::vector< Tetrahedron > > > elems = d_tetrahedra;
            for (size_t i=0; i<elems.size(); ++i)
                for (size_t j=0; j<elems[i].size(); ++j)
                    elems[i][j] = old2new[elems[i][j]];
        }
        {
            helper::WriteAccessor<Data< helper::vector< Pentahedron > > > elems = d_pentahedra;
            for (size_t i=0; i<elems.size(); ++i)
                for (size_t j=0; j<elems[i].size(); ++j)
                    elems[i][j] = old2new[elems[i][j]];
        }
        {
            helper::WriteAccessor<Data< helper::vector< Pyramid > > > elems = d_pyramids;
            for (size_t i=0; i<elems.size(); ++i)
                for (size_t j=0; j<elems[i].size(); ++j)
                    elems[i][j] = old2new[elems[i][j]];
        }
        {
            helper::WriteAccessor<Data< helper::vector< Hexahedron > > > elems = d_hexahedra;
            for (size_t i=0; i<elems.size(); ++i)
                for (size_t j=0; j<elems[i].size(); ++j)
                    elems[i][j] = old2new[elems[i][j]];
        }
    }
}

void MeshLoader::updateNormals()
{
    helper::ReadAccessor<Data<helper::vector<sofa::defaulttype::Vec<3,SReal> > > > raPositions = d_positions;
    helper::ReadAccessor<Data< helper::vector< Triangle > > > raTriangles = d_triangles;
    helper::ReadAccessor<Data< helper::vector< Quad > > > raQuads = d_quads;

    //look if we already have loaded normals
    if (d_normals.getValue().size() == raPositions.size())
        return;

    helper::WriteAccessor<Data<helper::vector<sofa::defaulttype::Vec<3,SReal> > > > waNormals = d_normals;

    waNormals.resize(raPositions.size());

    for (size_t i = 0; i < raTriangles.size() ; i++)
    {
        const sofa::defaulttype::Vec<3,SReal>  v1 = raPositions[raTriangles[i][0]];
        const sofa::defaulttype::Vec<3,SReal>  v2 = raPositions[raTriangles[i][1]];
        const sofa::defaulttype::Vec<3,SReal>  v3 = raPositions[raTriangles[i][2]];
        sofa::defaulttype::Vec<3,SReal> n = cross(v2-v1, v3-v1);

        n.normalize();
        waNormals[raTriangles[i][0]] += n;
        waNormals[raTriangles[i][1]] += n;
        waNormals[raTriangles[i][2]] += n;

    }
    for (size_t i = 0; i < raQuads.size() ; i++)
    {
        const sofa::defaulttype::Vec<3,SReal> & v1 = raPositions[raQuads[i][0]];
        const sofa::defaulttype::Vec<3,SReal> & v2 = raPositions[raQuads[i][1]];
        const sofa::defaulttype::Vec<3,SReal> & v3 = raPositions[raQuads[i][2]];
        const sofa::defaulttype::Vec<3,SReal> & v4 = raPositions[raQuads[i][3]];
        sofa::defaulttype::Vec<3,SReal> n1 = cross(v2-v1, v4-v1);
        sofa::defaulttype::Vec<3,SReal> n2 = cross(v3-v2, v1-v2);
        sofa::defaulttype::Vec<3,SReal> n3 = cross(v4-v3, v2-v3);
        sofa::defaulttype::Vec<3,SReal> n4 = cross(v1-v4, v3-v4);
        n1.normalize(); n2.normalize(); n3.normalize(); n4.normalize();
        waNormals[raQuads[i][0]] += n1;
        waNormals[raQuads[i][1]] += n2;
        waNormals[raQuads[i][2]] += n3;
        waNormals[raQuads[i][3]] += n4;
    }

    for (size_t i = 0; i < waNormals.size(); i++)
    {
        waNormals[i].normalize();
    }
}


void MeshLoader::applyTransformation(Matrix4 const& T)
{
    if (!T.isTransform()) {
        serr << "applyTransformation: ignored matrix which is not a transformation T=" << T << sendl;
        return;
    }
    sofa::helper::WriteAccessor <Data< helper::vector<sofa::defaulttype::Vec<3,SReal> > > > my_positions = d_positions;
    for (size_t i = 0; i < my_positions.size(); i++)
        my_positions[i] = T.transform(my_positions[i]);
}

void MeshLoader::addPosition(helper::vector<sofa::defaulttype::Vec<3,SReal> >* pPositions, const sofa::defaulttype::Vec<3,SReal> &p)
{
    pPositions->push_back(p);
}

void MeshLoader::addPosition(helper::vector<sofa::defaulttype::Vec<3,SReal> >* pPositions,  SReal x, SReal y, SReal z)
{
    addPosition(pPositions, sofa::defaulttype::Vec<3,SReal>(x, y, z));
}


void MeshLoader::addEdge(helper::vector<Edge >* pEdges, const Edge &p)
{
    pEdges->push_back(p);
}

void MeshLoader::addEdge(helper::vector<Edge >* pEdges, unsigned int p0, unsigned int p1)
{
    addEdge(pEdges, Edge(p0, p1));
}

void MeshLoader::addTriangle(helper::vector<Triangle >* pTriangles, const Triangle &p)
{
    if (d_flipNormals.getValue())
    {
        Triangle revertP;
        std::reverse_copy(p.begin(), p.end(), revertP.begin());

        pTriangles->push_back(revertP);
    }
    else
        pTriangles->push_back(p);
}

void MeshLoader::addTriangle(helper::vector<Triangle >* pTriangles, unsigned int p0, unsigned int p1, unsigned int p2)
{
    addTriangle(pTriangles, Triangle(p0, p1, p2));
}

void MeshLoader::addQuad(helper::vector<Quad >* pQuads, const Quad &p)
{
    if (d_flipNormals.getValue())
    {
        Quad revertP;
        std::reverse_copy(p.begin(), p.end(), revertP.begin());

        pQuads->push_back(revertP);
    }
    else
        pQuads->push_back(p);
}

void MeshLoader::addQuad(helper::vector<Quad >* pQuads, unsigned int p0, unsigned int p1, unsigned int p2, unsigned int p3)
{
    addQuad(pQuads, Quad(p0, p1, p2, p3));
}

void MeshLoader::addPolygon(helper::vector< helper::vector <unsigned int> >* pPolygons, const helper::vector<unsigned int> &p)
{
    if (d_flipNormals.getValue())
    {
        helper::vector<unsigned int> revertP(p.size());
        std::reverse_copy(p.begin(), p.end(), revertP.begin());

        pPolygons->push_back(revertP);
    }
    else
        pPolygons->push_back(p);
}


void MeshLoader::addTetrahedron(helper::vector< Tetrahedron >* pTetrahedra, const Tetrahedron &p)
{
    pTetrahedra->push_back(p);
}

void MeshLoader::addTetrahedron(helper::vector< Tetrahedron >* pTetrahedra, unsigned int p0, unsigned int p1, unsigned int p2, unsigned int p3)
{
    addTetrahedron(pTetrahedra, Tetrahedron(p0, p1, p2, p3));
}

void MeshLoader::addHexahedron(helper::vector< Hexahedron >* pHexahedra,
        unsigned int p0, unsigned int p1, unsigned int p2, unsigned int p3,
        unsigned int p4, unsigned int p5, unsigned int p6, unsigned int p7)
{
    addHexahedron(pHexahedra, Hexahedron(p0, p1, p2, p3, p4, p5, p6, p7));
}

void MeshLoader::addHexahedron(helper::vector< Hexahedron >* pHexahedra, const Hexahedron &p)
{
    pHexahedra->push_back(p);
}

void MeshLoader::addPentahedron(helper::vector< Pentahedron >* pPentahedra,
        unsigned int p0, unsigned int p1, unsigned int p2, unsigned int p3,
        unsigned int p4, unsigned int p5)
{
    addPentahedron(pPentahedra, Pentahedron(p0, p1, p2, p3, p4, p5));
}

void MeshLoader::addPentahedron(helper::vector< Pentahedron >* pPentahedra, const Pentahedron &p)
{
    pPentahedra->push_back(p);
}

void MeshLoader::addPyramid(helper::vector< Pyramid >* pPyramids,
        unsigned int p0, unsigned int p1, unsigned int p2, unsigned int p3, unsigned int p4)
{
    addPyramid(pPyramids, Pyramid(p0, p1, p2, p3, p4));
}

void MeshLoader::addPyramid(helper::vector< Pyramid >* pPyramids, const Pyramid &p)
{
    pPyramids->push_back(p);
}

} // namespace loader

} // namespace core

} // namespace sofa

