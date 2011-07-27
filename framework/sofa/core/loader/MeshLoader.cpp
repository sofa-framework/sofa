/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/loader/MeshLoader.h>
#include <stdlib.h>

namespace sofa
{

namespace core
{

namespace loader
{

using namespace sofa::defaulttype;

MeshLoader::MeshLoader() : BaseLoader()
    , positions(initData(&positions,"position","Vertices of the mesh loaded"))
    , edges(initData(&edges,"edges","Edges of the mesh loaded"))
    , triangles(initData(&triangles,"triangles","Triangles of the mesh loaded"))
    , quads(initData(&quads,"quads","Quads of the mesh loaded"))
    , polygons(initData(&polygons,"polygons","Polygons of the mesh loaded"))
    , tetrahedra(initData(&tetrahedra,"tetrahedra","Tetrahedra of the mesh loaded"))
    , hexahedra(initData(&hexahedra,"hexahedra","Hexahedra of the mesh loaded"))
    , normals(initData(&normals,"normals","Normals of the mesh loaded"))
    , edgesGroups(initData(&edgesGroups,"edgesGroups","Groups of Edges"))
    , trianglesGroups(initData(&trianglesGroups,"trianglesGroups","Groups of Triangles"))
    , quadsGroups(initData(&quadsGroups,"quadsGroups","Groups of Quads"))
    , polygonsGroups(initData(&polygonsGroups,"polygonsGroups","Groups of Polygons"))
    , tetrahedraGroups(initData(&tetrahedraGroups,"tetrahedraGroups","Groups of Tetrahedra"))
    , hexahedraGroups(initData(&hexahedraGroups,"hexahedraGroups","Groups of Hexahedra"))
    , flipNormals(initData(&flipNormals, false,"flipNormals","Flip Normals"))
    , triangulate(initData(&triangulate,false,"triangulate","Divide all polygons into triangles"))
    , onlyAttachedPoints(initData(&onlyAttachedPoints, false,"onlyAttachedPoints","Only keep points attached to elements of the mesh"))
    , translation(initData(&translation, Vector3(), "translation", "Translation of the DOFs"))
    , rotation(initData(&rotation, Vector3(), "rotation", "Rotation of the DOFs"))
    , scale(initData(&scale, Vector3(1.0,1.0,1.0), "scale3d", "Scale of the DOFs in 3 dimensions"))
{
    addAlias(&tetrahedra,"tetras");
    addAlias(&hexahedra,"hexas");
    //TODO: check if necessary!
    positions.setPersistent(false);
    edges.setPersistent(false);
    triangles.setPersistent(false);
    quads.setPersistent(false);
    polygons.setPersistent(false);
    tetrahedra.setPersistent(false);
    hexahedra.setPersistent(false);
    normals.setPersistent(false);
}


void MeshLoader::parse(sofa::core::objectmodel::BaseObjectDescription* arg)
{
    BaseLoader::parse(arg);

    if (arg->getAttribute("scale"))
    {
        SReal s = (SReal) atof(arg->getAttribute("scale"));
        scale.setValue(scale.getValue()*s);
    }

    if (canLoad())
        load(/*m_filename.getFullPath().c_str()*/);
    else
        sout << "Doing nothing" << sendl;

    updateMesh();

    if (translation.getValue() != Vector3(0.0,0.0,0.0))
        this->applyTranslation(translation.getValue()[0], translation.getValue()[1], translation.getValue()[2]);
    if (rotation.getValue() != Vector3(0.0,0.0,0.0))
        this->applyRotation(rotation.getValue()[0], rotation.getValue()[1], rotation.getValue()[2]);
    if (scale.getValue() != Vector3(1.0,1.0,1.0))
        this->applyScale(scale.getValue()[0],scale.getValue()[1],scale.getValue()[2]);
}


void MeshLoader::reinit()
{
    if (translation.getValue() != Vector3(0.0,0.0,0.0))
        this->applyTranslation(translation.getValue()[0], translation.getValue()[1], translation.getValue()[2]);
    if (rotation.getValue() != Vector3(0.0,0.0,0.0))
        this->applyRotation(rotation.getValue()[0], rotation.getValue()[1], rotation.getValue()[2]);
    if (scale.getValue() != Vector3(1.0,1.0,1.0))
        this->applyScale(scale.getValue()[0],scale.getValue()[1],scale.getValue()[2]);
}



bool MeshLoader::canLoad()
{
    std::string cmd;

    // -- Check filename field:
    if(m_filename.getValue() == "")
    {
        serr << "Error: MeshLoader: No file name given." << sendl;
        return false;
    }


    // -- Check if file exist:
    const char* filename = m_filename.getFullPath().c_str();
    std::string sfilename (filename);

    if (!sofa::helper::system::DataRepository.findFile(sfilename))
    {
        serr << "Error: MeshLoader: File '" << m_filename << "' not found. " << sendl;
        return false;
    }

    std::ifstream file(filename);

    // -- Check if file is readable:
    if (!file.good())
    {
        serr << "Error: MeshLoader: Cannot read file '" << m_filename << "'." << sendl;
        return false;
    }

    // -- Step 2.2: Check first line.
    file >> cmd;
    if (cmd.empty())
    {
        serr << "Error: MeshLoader: Cannot read first line in file '" << m_filename << "'." << sendl;
        file.close();
        return false;
    }

    file.close();
    return true;
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
    for (unsigned int j = 1; j < v.size(); ++j)
    {
        typename Vec::value_type key = v[j];
        unsigned int i = j;
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
    if (triangulate.getValue())
    {
        helper::WriteAccessor<Data<helper::vector< helper::fixed_array <unsigned int,4> > > > waQuads = quads;
        helper::WriteAccessor<Data<helper::vector< helper::fixed_array <unsigned int,3> > > > waTriangles = triangles;

        for (unsigned int i = 0; i < waQuads.size() ; i++)
        {
            const helper::fixed_array <unsigned int,4>& q = waQuads[i];
            addTriangle(&waTriangles.wref(), q[0], q[1], q[2]);
            addTriangle(&waTriangles.wref(), q[0], q[2], q[3]);
        }
        waQuads.clear();
    }
    // If ND topological elements are presents as well as (N-1)D elements, make sure all neighbors are created
    if (hexahedra.getValue().size() > 0 && quads.getValue().size() > 0)
    {
        helper::ReadAccessor<Data<helper::vector< helper::fixed_array <unsigned int,8> > > > hexahedra = this->hexahedra;
        helper::WriteAccessor<Data<helper::vector< helper::fixed_array <unsigned int,4> > > > quads = this->quads;
        std::set<helper::fixed_array <unsigned int,4> > eSet;
        for (unsigned int i = 0; i < quads.size(); ++i)
            eSet.insert(uniqueOrder(quads[i]));
        int nbnew = 0;
        for (unsigned int i = 0; i < hexahedra.size(); ++i)
        {
            helper::fixed_array<unsigned int,8> h = hexahedra[i];
            helper::fixed_array< helper::fixed_array<unsigned int,4>, 6 > e;
            e[0] = helper::make_array(h[0],h[3],h[2],h[1]);
            e[1] = helper::make_array(h[4],h[5],h[6],h[7]);
            e[2] = helper::make_array(h[0],h[1],h[5],h[4]);
            e[3] = helper::make_array(h[1],h[2],h[6],h[5]);
            e[4] = helper::make_array(h[2],h[3],h[7],h[6]);
            e[5] = helper::make_array(h[3],h[0],h[4],h[7]);
            for (unsigned int j = 0; j < e.size(); ++j)
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
    if (tetrahedra.getValue().size() > 0 && triangles.getValue().size() > 0)
    {
        helper::ReadAccessor<Data<helper::vector< helper::fixed_array <unsigned int,4> > > > tetrahedra = this->tetrahedra;
        helper::WriteAccessor<Data<helper::vector< helper::fixed_array <unsigned int,3> > > > triangles = this->triangles;
        std::set<helper::fixed_array <unsigned int,3> > eSet;
        for (unsigned int i = 0; i < triangles.size(); ++i)
            eSet.insert(uniqueOrder(triangles[i]));
        int nbnew = 0;
        for (unsigned int i = 0; i < tetrahedra.size(); ++i)
        {
            helper::fixed_array<unsigned int,4> t = tetrahedra[i];
            for (unsigned int j = 0; j < t.size(); ++j)
            {
                helper::fixed_array<unsigned int,3> e(t[(j+1)%t.size()],t[(j+2)%t.size()],t[(j+3)%t.size()]);
                if (eSet.insert(uniqueOrder(e)).second) // the element was inserted
                {
                    triangles.push_back(e);
                    ++nbnew;
                }
            }
        }
        if (nbnew > 0)
            sout << nbnew << " triangles were missing around the tetrahedra" << sendl;
    }
    if (quads.getValue().size() > 0 && edges.getValue().size() > 0)
    {
        helper::ReadAccessor<Data<helper::vector< helper::fixed_array <unsigned int,4> > > > quads = this->quads;
        helper::WriteAccessor<Data<helper::vector< helper::fixed_array <unsigned int,2> > > > edges = this->edges;
        std::set<helper::fixed_array <unsigned int,2> > eSet;
        for (unsigned int i = 0; i < edges.size(); ++i)
            eSet.insert(uniqueOrder(edges[i]));
        int nbnew = 0;
        for (unsigned int i = 0; i < quads.size(); ++i)
        {
            helper::fixed_array<unsigned int,4> t = quads[i];
            for (unsigned int j = 0; j < t.size(); ++j)
            {
                helper::fixed_array<unsigned int,2> e(t[(j+1)%t.size()],t[(j+2)%t.size()]);
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
    if (triangles.getValue().size() > 0 && edges.getValue().size() > 0)
    {
        helper::ReadAccessor<Data<helper::vector< helper::fixed_array <unsigned int,3> > > > triangles = this->triangles;
        helper::WriteAccessor<Data<helper::vector< helper::fixed_array <unsigned int,2> > > > edges = this->edges;
        std::set<helper::fixed_array <unsigned int,2> > eSet;
        for (unsigned int i = 0; i < edges.size(); ++i)
            eSet.insert(uniqueOrder(edges[i]));
        int nbnew = 0;
        for (unsigned int i = 0; i < triangles.size(); ++i)
        {
            helper::fixed_array<unsigned int,3> t = triangles[i];
            for (unsigned int j = 0; j < t.size(); ++j)
            {
                helper::fixed_array<unsigned int,2> e(t[(j+1)%t.size()],t[(j+2)%t.size()]);
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
    if (onlyAttachedPoints.getValue())
    {
        std::set<unsigned int> attachedPoints;
        {
            helper::ReadAccessor<Data< helper::vector< helper::fixed_array <unsigned int,2> > > > elems = edges;
            for (unsigned int i=0; i<elems.size(); ++i)
                for (unsigned int j=0; j<elems[i].size(); ++j)
                    attachedPoints.insert(elems[i][j]);
        }
        {
            helper::ReadAccessor<Data< helper::vector< helper::fixed_array <unsigned int,3> > > > elems = triangles;
            for (unsigned int i=0; i<elems.size(); ++i)
                for (unsigned int j=0; j<elems[i].size(); ++j)
                    attachedPoints.insert(elems[i][j]);
        }
        {
            helper::ReadAccessor<Data< helper::vector< helper::fixed_array <unsigned int,4> > > > elems = quads;
            for (unsigned int i=0; i<elems.size(); ++i)
                for (unsigned int j=0; j<elems[i].size(); ++j)
                    attachedPoints.insert(elems[i][j]);
        }
        {
            helper::ReadAccessor<Data< helper::vector< helper::fixed_array <unsigned int,4> > > > elems = tetrahedra;
            for (unsigned int i=0; i<elems.size(); ++i)
                for (unsigned int j=0; j<elems[i].size(); ++j)
                    attachedPoints.insert(elems[i][j]);
        }
        {
            helper::ReadAccessor<Data< helper::vector< helper::fixed_array <unsigned int,8> > > > elems = hexahedra;
            for (unsigned int i=0; i<elems.size(); ++i)
                for (unsigned int j=0; j<elems[i].size(); ++j)
                    attachedPoints.insert(elems[i][j]);
        }
        const unsigned int newsize = attachedPoints.size();
        if (newsize == positions.getValue().size()) return; // all points are attached
        helper::WriteAccessor<Data<helper::vector<sofa::defaulttype::Vec<3,SReal> > > > waPositions = positions;
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
            helper::WriteAccessor<Data< helper::vector< helper::fixed_array <unsigned int,2> > > > elems = edges;
            for (unsigned int i=0; i<elems.size(); ++i)
                for (unsigned int j=0; j<elems[i].size(); ++j)
                    elems[i][j] = old2new[elems[i][j]];
        }
        {
            helper::WriteAccessor<Data< helper::vector< helper::fixed_array <unsigned int,3> > > > elems = triangles;
            for (unsigned int i=0; i<elems.size(); ++i)
                for (unsigned int j=0; j<elems[i].size(); ++j)
                    elems[i][j] = old2new[elems[i][j]];
        }
        {
            helper::WriteAccessor<Data< helper::vector< helper::fixed_array <unsigned int,4> > > > elems = quads;
            for (unsigned int i=0; i<elems.size(); ++i)
                for (unsigned int j=0; j<elems[i].size(); ++j)
                    elems[i][j] = old2new[elems[i][j]];
        }
        {
            helper::WriteAccessor<Data< helper::vector< helper::fixed_array <unsigned int,4> > > > elems = tetrahedra;
            for (unsigned int i=0; i<elems.size(); ++i)
                for (unsigned int j=0; j<elems[i].size(); ++j)
                    elems[i][j] = old2new[elems[i][j]];
        }
        {
            helper::WriteAccessor<Data< helper::vector< helper::fixed_array <unsigned int,8> > > > elems = hexahedra;
            for (unsigned int i=0; i<elems.size(); ++i)
                for (unsigned int j=0; j<elems[i].size(); ++j)
                    elems[i][j] = old2new[elems[i][j]];
        }
    }
}

void MeshLoader::updateNormals()
{
    helper::ReadAccessor<Data<helper::vector<sofa::defaulttype::Vec<3,SReal> > > > raPositions = positions;
    helper::ReadAccessor<Data< helper::vector< helper::fixed_array <unsigned int,3> > > > raTriangles = triangles;
    helper::ReadAccessor<Data< helper::vector< helper::fixed_array <unsigned int,4> > > > raQuads = quads;

    //look if we already have loaded normals
    if (normals.getValue().size() == raPositions.size())
        return;

    helper::WriteAccessor<Data<helper::vector<sofa::defaulttype::Vec<3,SReal> > > > waNormals = normals;

    waNormals.resize(raPositions.size());

    for (unsigned int i = 0; i < raTriangles.size() ; i++)
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
    for (unsigned int i = 0; i < raQuads.size() ; i++)
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

    for (unsigned int i = 0; i < waNormals.size(); i++)
    {
        waNormals[i].normalize();
    }
}


void MeshLoader::applyTranslation(const SReal dx, const SReal dy, const SReal dz)
{
    sofa::helper::WriteAccessor <Data< helper::vector<sofa::defaulttype::Vec<3,SReal> > > > my_positions = positions;
    for (unsigned int i = 0; i < my_positions.size(); i++)
        my_positions[i] += Vector3(dx,dy,dz);
}


void MeshLoader::applyRotation(const SReal rx, const SReal ry, const SReal rz)
{
    Quaternion q = helper::Quater< SReal >::createQuaterFromEuler(Vec< 3, SReal >(rx, ry, rz) * M_PI / 180.0);
    applyRotation(q);
}


void MeshLoader::applyRotation(const defaulttype::Quat q)
{
    sofa::helper::WriteAccessor <Data< helper::vector<sofa::defaulttype::Vec<3,SReal> > > > my_positions = positions;
    for (unsigned int i = 0; i < my_positions.size(); i++)
    {
        Vec<3,SReal> newposition = q.rotate(my_positions[i]);
        my_positions[i] = newposition;
    }
}


void MeshLoader::applyScale(const SReal sx, const SReal sy, const SReal sz)
{
    sofa::helper::WriteAccessor <Data< helper::vector<sofa::defaulttype::Vec<3,SReal> > > > my_positions = positions;
    for (unsigned int i = 0; i < my_positions.size(); i++)
    {
        my_positions[i][0] *= sx;
        my_positions[i][1] *= sy;
        my_positions[i][2] *= sz;
    }
}


void MeshLoader::addPosition(helper::vector<sofa::defaulttype::Vec<3,SReal> >* pPositions, const sofa::defaulttype::Vec<3,SReal> &p)
{
    pPositions->push_back(p);
}

void MeshLoader::addPosition(helper::vector<sofa::defaulttype::Vec<3,SReal> >* pPositions,  SReal x, SReal y, SReal z)
{
    addPosition(pPositions, sofa::defaulttype::Vec<3,SReal>(x, y, z));
}


void MeshLoader::addEdge(helper::vector<helper::fixed_array <unsigned int,2> >* pEdges, const helper::fixed_array <unsigned int,2> &p)
{
    pEdges->push_back(p);
}

void MeshLoader::addEdge(helper::vector<helper::fixed_array <unsigned int,2> >* pEdges, unsigned int p0, unsigned int p1)
{
    addEdge(pEdges, helper::fixed_array <unsigned int,2>(p0, p1));
}

void MeshLoader::addTriangle(helper::vector<helper::fixed_array <unsigned int,3> >* pTriangles, const helper::fixed_array <unsigned int,3> &p)
{
    if (flipNormals.getValue())
    {
        helper::fixed_array <unsigned int,3> revertP;
        std::reverse_copy(p.begin(), p.end(), revertP.begin());

        pTriangles->push_back(revertP);
    }
    else
        pTriangles->push_back(p);
}

void MeshLoader::addTriangle(helper::vector<helper::fixed_array <unsigned int,3> >* pTriangles, unsigned int p0, unsigned int p1, unsigned int p2)
{
    addTriangle(pTriangles, helper::fixed_array <unsigned int,3>(p0, p1, p2));
}

void MeshLoader::addQuad(helper::vector<helper::fixed_array <unsigned int,4> >* pQuads, const helper::fixed_array <unsigned int,4> &p)
{
    if (flipNormals.getValue())
    {
        helper::fixed_array <unsigned int,4> revertP;
        std::reverse_copy(p.begin(), p.end(), revertP.begin());

        pQuads->push_back(revertP);
    }
    else
        pQuads->push_back(p);
}

void MeshLoader::addQuad(helper::vector<helper::fixed_array <unsigned int,4> >* pQuads, unsigned int p0, unsigned int p1, unsigned int p2, unsigned int p3)
{
    addQuad(pQuads, helper::fixed_array <unsigned int,4>(p0, p1, p2, p3));
}

void MeshLoader::addPolygon(helper::vector< helper::vector <unsigned int> >* pPolygons, const helper::vector<unsigned int> &p)
{
    if (flipNormals.getValue())
    {
        helper::vector<unsigned int> revertP(p.size());
        std::reverse_copy(p.begin(), p.end(), revertP.begin());

        pPolygons->push_back(revertP);
    }
    else
        pPolygons->push_back(p);
}


void MeshLoader::addTetrahedron(helper::vector< helper::fixed_array<unsigned int,4> >* pTetrahedra, const helper::fixed_array<unsigned int,4> &p)
{
    pTetrahedra->push_back(p);
}

void MeshLoader::addTetrahedron(helper::vector< helper::fixed_array<unsigned int,4> >* pTetrahedra, unsigned int p0, unsigned int p1, unsigned int p2, unsigned int p3)
{
    addTetrahedron(pTetrahedra, helper::fixed_array <unsigned int,4>(p0, p1, p2, p3));
}

void MeshLoader::addHexahedron(helper::vector< helper::fixed_array<unsigned int,8> >* pHexahedra,
        unsigned int p0, unsigned int p1, unsigned int p2, unsigned int p3,
        unsigned int p4, unsigned int p5, unsigned int p6, unsigned int p7)
{
    addHexahedron(pHexahedra, helper::fixed_array <unsigned int,8>(p0, p1, p2, p3, p4, p5, p6, p7));
}

void MeshLoader::addHexahedron(helper::vector< helper::fixed_array<unsigned int,8> >* pHexahedra, const helper::fixed_array<unsigned int,8> &p)
{
    pHexahedra->push_back(p);
}


} // namespace loader

} // namespace core

} // namespace sofa

