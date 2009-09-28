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
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/loader/MeshGmshLoader.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace loader
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(MeshGmshLoader)

int MeshGmshLoaderClass = core::RegisterObject("Specific mesh loader for Gmsh file format.")
        .add< MeshGmshLoader >()
        ;

bool MeshGmshLoader::load()
{
    sout << "Loading Gmsh file: " << m_filename << sendl;

    FILE* file;
    char cmd[1024];
    bool fileRead = false;
    unsigned int gmshFormat = 0;

    // -- Loading file
    const char* filename = m_filename.getFullPath().c_str();

    if ((file = fopen(filename, "r")) == NULL)
    {
        serr << "Error: MeshGmshLoader: Cannot read file '" << m_filename << "'." << sendl;
        return false;
    }

    // -- Looking for Gmsh version of this file.
    if (!strncmp(cmd,"$MeshFormat",11)) // Reading gmsh 2.0 file
    {
        gmshFormat = 2;
        // sout << "Gmsh format 2.0" << sendl;
        readLine(cmd, sizeof(cmd), file); // we don't care about this line
        readLine(cmd, sizeof(cmd), file);
        if (strncmp(cmd,"$EndMeshFormat",14)) // it should end with $EndMeshFormat
        {
            serr << "Error: MeshGmshLoader: File '" << m_filename << "' is empty." << sendl;
            fclose(file);
            return false;
        }
        else
            readLine(cmd, sizeof(cmd), file);
    }
    else
    {
        gmshFormat = 1;
        //sout << "Gmsh format 1.0" << sendl;
    }

    // -- Reading file
    readLine(cmd, sizeof(cmd), file);
    if (!strncmp(cmd,"$NOD",4) || !strncmp(cmd,"$Nodes",6)) // Gmsh format
    {
        fileRead = readGmsh(file, gmshFormat);
    }
    else //if it enter this "else", it means there is a problem before in the factory or in canLoad()
    {
        serr << "Error: MeshGmshLoader: File '" << m_filename << "' finally appears not to be a Gmsh file." << sendl;
        fclose(file);
        return false;
    }

    return fileRead;
}



bool MeshGmshLoader::readGmsh(FILE *file, const unsigned int gmshFormat)
{
    sout << "Reading Gmsh file: " << gmshFormat << sendl;

    char cmd[1024];

    unsigned int npoints = 0;
    unsigned int nelems = 0;

    unsigned int nlines = 0;
    unsigned int ntris = 0;
    unsigned int nquads = 0;
    unsigned int ntetrahedra = 0;
    unsigned int ncubes = 0;

    // --- Loading Vertices ---
    if ( fscanf(file, "%d\n", &npoints) == EOF) //Loading number of Point
        serr << "Error: MeshGmshLoader: fscanf function can't read number of point." << sendl;

    helper::vector<sofa::defaulttype::Vector3>& my_positions = *(positions.beginEdit());

    std::vector<unsigned int> pmap; // map for reordering vertices possibly not well sorted
    for (unsigned int i=0; i<npoints; ++i)
    {
        unsigned int index = i;
        double x,y,z;
        if ( fscanf(file, "%d %lf %lf %lf\n", &index, &x, &y, &z) == EOF )
            serr << "Error: MeshGmshLoader: fscanf function can't read vertices." << sendl;

        my_positions.push_back(Vector3(x, y, z));

        if (pmap.size() <= index)
            pmap.resize(index+1);

        pmap[index] = i; // In case of hole or switch
        //sout << "pmap[" << index << "] = " << pmap[index] << sendl;
    }
    positions.endEdit();

    readLine(cmd, sizeof(cmd), file);
    if (strncmp(cmd,"$ENDNOD",7) && strncmp(cmd,"$EndNodes",9))
    {
        serr << "Error: MeshGmshLoader: '$ENDNOD' or '$EndNodes' expected, found '" << cmd << "'" << sendl;
        fclose(file);
        return false;
    }


    // --- Loading Elements ---
    readLine(cmd, sizeof(cmd), file);
    if (strncmp(cmd,"$ELM",4) && strncmp(cmd,"$Elements",9))
    {
        serr << "Error: MeshGmshLoader: '$ELM' or '$Elements' expected, found '" << cmd << "'" << sendl;
        fclose(file);
        return false;
    }

    if ( fscanf(file, "%d\n", &nelems) == EOF ) //Loading number of Element
        serr << "Error: MeshGmshLoader: fscanf function can't read number of element." << sendl;

    helper::vector<helper::fixed_array <unsigned int,2> >& my_edges = *(edges.beginEdit());
    helper::vector<helper::fixed_array <unsigned int,3> >& my_triangles = *(triangles.beginEdit());
    helper::vector<helper::fixed_array <unsigned int,4> >& my_quads = *(quads.beginEdit());
    helper::vector<helper::fixed_array <unsigned int,4> >& my_tetrahedra = *(tetrahedra.beginEdit());
    helper::vector<helper::fixed_array <unsigned int,8> >& my_hexahedra = *(hexahedra.beginEdit());


    for (unsigned int i=0; i<nelems; ++i) // for each elem
    {
        int index, etype, rphys, relem, nnodes, ntags, tag;

        if (gmshFormat==1)
        {
            // version 1.0 format is
            // elm-number elm-type reg-phys reg-elem number-of-nodes <node-number-list ...>
            if ( fscanf(file, "%d %d %d %d %d", &index, &etype, &rphys, &relem, &nnodes) == EOF )
                serr << "Error: MeshGmshLoader: fscanf function can't read elements in gmshFormat 1." << sendl;
        }
        else if (gmshFormat == 2)
        {
            // version 2.0 format is
            // elm-number elm-type number-of-tags < tag > ... node-number-list
            if ( fscanf(file, "%d %d %d", &index, &etype, &ntags) == EOF )
                serr << "Error: MeshGmshLoader: fscanf function can't read elements in gmshFormat 2." << sendl;

            for (int t=0; t<ntags; t++)
            {
                if ( fscanf(file, "%d", &tag) == EOF ) // read the tag but don't use it
                    serr << "Error: MeshGmshLoader: fscanf function can't read tag." << sendl;
            }


            switch (etype)
            {
            case 1: // Line
                nnodes = 2;
                break;
            case 2: // Triangle
                nnodes = 3;
                break;
            case 3: // Quad
                nnodes = 4;
                break;
            case 4: // Tetra
                nnodes = 4;
                break;
            case 5: // Hexa
                nnodes = 8;
                break;
            default:
                serr << "Error: MeshGmshLoader: Elements of type 1, 2, 3, 4, 5, or 6 expected. Element of type " << etype << " found." << sendl;
                nnodes = 0;
            }
        }
        //store real index of node and not line index

        helper::vector <unsigned int> nodes;
        nodes.resize (nnodes);

        for (int n=0; n<nnodes; ++n)
        {
            int t = 0;
            if ( fscanf(file, "%d",&t) == EOF )
                serr << "Error: MeshGmshLoader: fscanf function can't read element nodes." << sendl;

            nodes[n] = (((unsigned int)t)<pmap.size())?pmap[t]:0;
            //sout << "nodes[" << n << "] = " << nodes[n] << sendl;
        }

        switch (etype)
        {
        case 1: // Line
            addEdge(&my_edges, helper::fixed_array <unsigned int,2>(nodes[0], nodes[1]));
            ++nlines;
            break;
        case 2: // Triangle
            addTriangle(&my_triangles, helper::fixed_array <unsigned int,3>(nodes[0], nodes[1], nodes[2]));
            ++ntris;
            break;
        case 3: // Quad
            addQuad(&my_quads, helper::fixed_array <unsigned int,4>(nodes[0], nodes[1], nodes[2], nodes[3]));
            ++nquads;
            break;
        case 4: // Tetra
            addTetrahedron(&my_tetrahedra, helper::fixed_array <unsigned int,4>(nodes[0], nodes[1], nodes[2], nodes[3]));
            ++ntetrahedra;
            break;
        case 5: // Hexa
            helper::fixed_array <unsigned int,8> hexa;
            for (unsigned int n=0; n<8; n++)
                hexa[n] = nodes[n];
            addHexahedron(&my_hexahedra,hexa);
            ++ncubes;
            break;
        }
        skipToEOL(file);
    }

    edges.endEdit();
    triangles.endEdit();
    quads.endEdit();
    tetrahedra.endEdit();
    hexahedra.endEdit();

    readLine(cmd, sizeof(cmd), file);
    if (strncmp(cmd,"$ENDELM",7) && strncmp(cmd,"$EndElements",12))
    {
        serr << "Error: MeshGmshLoader: '$ENDELM' or '$EndElements' expected, found '" << cmd << "'" << sendl;
        fclose(file);
        return false;
    }

    // 	sout << "Loading topology complete:";
    // 	if (npoints>0) sout << ' ' << npoints << " points";
    // 	if (nlines>0)  sout << ' ' << nlines  << " lines";
    // 	if (ntris>0)   sout << ' ' << ntris   << " triangles";
    // 	if (nquads>0)  sout << ' ' << nquads  << " quads";
    // 	if (ntetrahedra>0) sout << ' ' << ntetrahedra << " tetrahedra";
    // 	if (ncubes>0)  sout << ' ' << ncubes  << " cubes";
    // 	sout << sendl;

    fclose(file);
    return true;
}


} // namespace loader

} // namespace component

} // namespace sofa

