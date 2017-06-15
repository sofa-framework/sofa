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
#include "MeshExporter.h"

#include <sstream>
#include <iomanip>

#include <sofa/core/ObjectFactory.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(MeshExporter)

int MeshExporterClass = core::RegisterObject("Export topology and positions into several mesh file formats")
        .add< MeshExporter >();

MeshExporter::MeshExporter()
    : stepCounter(0), nbFiles(0)
    , meshFilename( initData(&meshFilename, "filename", "output Mesh file name \n Important Info ! \n Export can be done by pressing \n the 'E' key (listening required)"))
    , fileFormat( initData(&fileFormat, sofa::helper::OptionsGroup(6,"ALL","vtkxml","vtk","netgen","tetgen","gmsh"), "format", "File format to use"))
    , position( initData(&position, "position", "points position (will use points from topology or mechanical state if this is empty)"))
    , writeEdges( initData(&writeEdges, (bool) true, "edges", "write edge topology"))
    , writeTriangles( initData(&writeTriangles, (bool) true, "triangles", "write triangle topology"))
    , writeQuads( initData(&writeQuads, (bool) true, "quads", "write quad topology"))
    , writeTetras( initData(&writeTetras, (bool) true, "tetras", "write tetra topology"))
    , writeHexas( initData(&writeHexas, (bool) true, "hexas", "write hexa topology"))
//, dPointsDataFields( initData(&dPointsDataFields, "pointsDataFields", "Data to visualize (on points)"))
//, dCellsDataFields( initData(&dCellsDataFields, "cellsDataFields", "Data to visualize (on cells)"))
    , exportEveryNbSteps( initData(&exportEveryNbSteps, (unsigned int)0, "exportEveryNumberOfSteps", "export file only at specified number of steps (0=disable)"))
    , exportAtBegin( initData(&exportAtBegin, false, "exportAtBegin", "export file at the initialization"))
    , exportAtEnd( initData(&exportAtEnd, false, "exportAtEnd", "export file when the simulation is finished"))
{
}

MeshExporter::~MeshExporter()
{
}

void MeshExporter::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    context->get(topology);
    context->get(mstate);

    if (!topology)
    {
        serr << "MeshExporter : error, no topology." << sendl;
        return;
    }

    if (!position.isSet() && mstate)
    {
        sofa::core::objectmodel::BaseData* parent = NULL;
        if (!parent && mstate) parent = mstate->findData("position");
        if (!parent && topology) parent = mstate->findData("topology");
        if (parent)
        {
            position.setParent(parent);
            position.setReadOnly(true);
        }
    }

    // Activate the listening to the event in order to be able to export file at the nth-step
    if(exportEveryNbSteps.getValue() != 0)
        this->f_listening.setValue(true);

    nbFiles = 0;
    /*
        const helper::vector<std::string>& pointsData = dPointsDataFields.getValue();
        const helper::vector<std::string>& cellsData = dCellsDataFields.getValue();

        if (!pointsData.empty())
        {
            fetchDataFields(pointsData, pointsDataObject, pointsDataField, pointsDataName);
        }
        if (!cellsData.empty())
        {
            fetchDataFields(cellsData, cellsDataObject, cellsDataField, cellsDataName);
        }
    */
}

void MeshExporter::writeMesh()
{
    const unsigned int format = fileFormat.getValue().getSelectedId();

    const bool all = (format == 0);
    const bool vtkxml = all || (format == 1);
    const bool vtk    = all || (format == 2);
    const bool netgen = all || (format == 3);
    const bool tetgen = all || (format == 4);
    const bool gmsh   = all || (format == 5);
    sout << "Exporting mesh " << getMeshFilename("") << sendl;

    if (vtkxml)
        writeMeshVTKXML();
    if (vtk)
        writeMeshVTK();
    if (netgen)
        writeMeshNetgen();
    if (tetgen)
        writeMeshTetgen();
    if (gmsh)
        writeMeshGmsh();
    ++nbFiles;

}

std::string MeshExporter::getMeshFilename(const char* ext)
{
    int nbp = position.getValue().size();
    sout << nbp << " points" << sendl;
    sout << topology->getNbEdges() << " edges" << sendl;
    sout << topology->getNbTriangles() << " triangles" << sendl;
    sout << topology->getNbQuads() << " quads" << sendl;
    sout << topology->getNbTetras() << " tetras" << sendl;
    sout << topology->getNbHexas() << " hexas" << sendl;
    unsigned int nbce;
    nbce = ( (writeEdges.getValue()) ? topology->getNbEdges() : 0 )
            + ( (writeTriangles.getValue()) ? topology->getNbTriangles() : 0 )
            + ( (writeQuads.getValue()) ? topology->getNbQuads() : 0 )
            + ( (writeTetras.getValue()) ? topology->getNbTetras() : 0 )
            + ( (writeHexas.getValue()) ? topology->getNbHexas() : 0 );
    unsigned int nbe = 0;
    nbe = ( (writeTetras.getValue()) ? topology->getNbTetras() : 0 )
            + ( (writeHexas.getValue()) ? topology->getNbHexas() : 0 );
    if (!nbe)
        nbe = ( (writeTriangles.getValue()) ? topology->getNbTriangles() : 0 )
                + ( (writeQuads.getValue()) ? topology->getNbQuads() : 0 );
    if (!nbe)
        nbe = ( (writeEdges.getValue()) ? topology->getNbEdges() : 0 );
    if (!nbe)
        nbe = nbp;

    std::ostringstream oss;
    std::string filename = meshFilename.getFullPath();
    std::size_t pos = 0;
    while (pos != std::string::npos)
    {
        std::size_t newpos = filename.find('%',pos);
        oss << filename.substr(pos, (newpos == std::string::npos) ? std::string::npos : newpos-pos);
        pos = newpos;
        if(pos != std::string::npos)
        {
            ++pos;
            char c = filename[pos];
            ++pos;
            switch (c)
            {
            case 'n' : oss << nbFiles; break;
            case 'p' : oss << nbp; break;
            case 'E' : oss << nbce; break;
            case 'e' : oss << nbe; break;
            case '%' : oss << '%';
            default:
                serr << "Invalid special character %" << c << " in filename" << sendl;
            }
        }
    }
    oss << ext;
    return oss.str();
}

void MeshExporter::writeMeshVTKXML()
{
    std::string filename = getMeshFilename(".vtu");

    std::ofstream outfile(filename.c_str());
    if (!outfile.is_open())
    {
        serr << "Error creating file "<<filename<<sendl;
        return;
    }

    outfile << std::setprecision (9);

    //const helper::vector<std::string>& pointsData = dPointsDataFields.getValue();
    //const helper::vector<std::string>& cellsData = dCellsDataFields.getValue();

    helper::ReadAccessor<Data<defaulttype::Vec3Types::VecCoord> > pointsPos = position;

    const int nbp = pointsPos.size();

    unsigned int numberOfCells;
    numberOfCells = ( (writeEdges.getValue()) ? topology->getNbEdges() : 0 )
            +( (writeTriangles.getValue()) ? topology->getNbTriangles() : 0 )
            +( (writeQuads.getValue()) ? topology->getNbQuads() : 0 )
            +( (writeTetras.getValue()) ? topology->getNbTetras() : 0 )
            +( (writeHexas.getValue()) ? topology->getNbHexas() : 0 );
//	unsigned int totalSize =     ( (writeEdges.getValue()) ? 3 * topology->getNbEdges() : 0 )
// 				   +( (writeTriangles.getValue()) ? 4 *topology->getNbTriangles() : 0 )
// 				   +( (writeQuads.getValue()) ? 5 *topology->getNbQuads() : 0 )
// 				   +( (writeTetras.getValue()) ? 5 *topology->getNbTetras() : 0 )
// 				   +( (writeHexas.getValue()) ? 9 *topology->getNbHexas() : 0 );

    //write header
    outfile << "<?xml version=\"1.0\"?>\n";
    outfile << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n";
    outfile << "  <UnstructuredGrid>\n";
    //write piece
    outfile << "    <Piece NumberOfPoints=\"" << nbp << "\" NumberOfCells=\""<< numberOfCells << "\">\n";

    /*
        //write point data
        if (!pointsData.empty())
        {
            outfile << "      <PointData>\n";
            writeDataArray(pointsDataObject, pointsDataField, pointsDataName);
            outfile << "      </PointData>\n";
        }
            //write cell data
        if (!cellsData.empty())
        {
            outfile << "      <CellData>\n";
            writeDataArray(cellsDataObject, cellsDataField, cellsDataName);
            outfile << "      </CellData>\n";
        }

    */

    //write points
    outfile << "      <Points>\n";
    outfile << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i=0 ; i<nbp; i++)
    {
        outfile << "          " << pointsPos[i] << "\n";
    }
    outfile << "        </DataArray>\n";
    outfile << "      </Points>\n";
    //write cells
    outfile << "      <Cells>\n";
    //write connectivity
    outfile << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    if (writeEdges.getValue())
    {
        for (int i=0 ; i<topology->getNbEdges() ; i++)
            outfile << "          " << topology->getEdge(i) << "\n";
    }

    if (writeTriangles.getValue())
    {
        for (int i=0 ; i<topology->getNbTriangles() ; i++)
            outfile << "          " <<  topology->getTriangle(i) << "\n";
    }
    if (writeQuads.getValue())
    {
        for (int i=0 ; i<topology->getNbQuads() ; i++)
            outfile << "          " << topology->getQuad(i) << "\n";
    }
    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++)
            outfile << "          " <<  topology->getTetra(i) << "\n";
    }
    if (writeHexas.getValue())
    {
        for (int i=0 ; i<topology->getNbHexas() ; i++)
            outfile << "          " <<  topology->getHexa(i) << "\n";
    }
    outfile << "        </DataArray>\n";
    //write offsets
    int num = 0;
    outfile << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    outfile << "          ";
    if (writeEdges.getValue())
    {
        for (int i=0 ; i<topology->getNbEdges() ; i++)
        {
            num += 2;
            outfile << num << ' ';
        }
    }
    if (writeTriangles.getValue())
    {
        for (int i=0 ; i<topology->getNbTriangles() ; i++)
        {
            num += 3;
            outfile << num << ' ';
        }
    }
    if (writeQuads.getValue())
    {
        for (int i=0 ; i<topology->getNbQuads() ; i++)
        {
            num += 4;
            outfile << num << ' ';
        }
    }
    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++)
        {
            num += 4;
            outfile << num << ' ';
        }
    }
    if (writeHexas.getValue())
    {
        for (int i=0 ; i<topology->getNbHexas() ; i++)
        {
            num += 6;
            outfile << num << ' ';
        }
    }
    outfile << "\n";
    outfile << "        </DataArray>\n";
    //write types
    outfile << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    outfile << "          ";
    if (writeEdges.getValue())
    {
        for (int i=0 ; i<topology->getNbEdges() ; i++)
            outfile << 3 << ' ';
    }
    if (writeTriangles.getValue())
    {
        for (int i=0 ; i<topology->getNbTriangles() ; i++)
            outfile << 5 << ' ';
    }
    if (writeQuads.getValue())
    {
        for (int i=0 ; i<topology->getNbQuads() ; i++)
            outfile << 9 << ' ';
    }
    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++)
            outfile << 10 << ' ';
    }
    if (writeHexas.getValue())
    {
        for (int i=0 ; i<topology->getNbHexas() ; i++)
            outfile << 12 << ' ';
    }
    outfile << "\n";
    outfile << "        </DataArray>\n";
    outfile << "      </Cells>\n";

    //write end
    outfile << "    </Piece>\n";
    outfile << "  </UnstructuredGrid>\n";
    outfile << "</VTKFile>\n";
    outfile.close();
    sout << filename << " written" << sendl;
}

void MeshExporter::writeMeshVTK()
{
    std::string filename = getMeshFilename(".vtk");

    std::ofstream outfile(filename.c_str());
    if( !outfile.is_open() )
    {
        serr << "Error creating file "<<filename<<sendl;
        return;
    }

    outfile << std::setprecision (9);

    //const helper::vector<std::string>& pointsData = dPointsDataFields.getValue();
    //const helper::vector<std::string>& cellsData = dCellsDataFields.getValue();

    helper::ReadAccessor<Data<defaulttype::Vec3Types::VecCoord> > pointsPos = position;

    const int nbp = pointsPos.size();

    //Write header
    outfile << "# vtk DataFile Version 2.0\n";

    //write Title
    outfile << "SOFA Exported Mesh file\n";

    //write Data type
    outfile << "ASCII\n";

    //write dataset (geometry, unstructured grid)
    outfile << "DATASET " << "UNSTRUCTURED_GRID\n";

    outfile << "POINTS " << nbp << " float\n";

    //Write Points
    for (int i=0 ; i<nbp; i++)
    {
        outfile << pointsPos[i] << "\n";
    }

    //Write Cells
    unsigned int numberOfCells, totalSize;
    numberOfCells = ( (writeEdges.getValue()) ? topology->getNbEdges() : 0 )
            +( (writeTriangles.getValue()) ? topology->getNbTriangles() : 0 )
            +( (writeQuads.getValue()) ? topology->getNbQuads() : 0 )
            +( (writeTetras.getValue()) ? topology->getNbTetras() : 0 )
            +( (writeHexas.getValue()) ? topology->getNbHexas() : 0 );
    totalSize =     ( (writeEdges.getValue()) ? 3 * topology->getNbEdges() : 0 )
            +( (writeTriangles.getValue()) ? 4 *topology->getNbTriangles() : 0 )
            +( (writeQuads.getValue()) ? 5 *topology->getNbQuads() : 0 )
            +( (writeTetras.getValue()) ? 5 *topology->getNbTetras() : 0 )
            +( (writeHexas.getValue()) ? 9 *topology->getNbHexas() : 0 );


    outfile << "CELLS " << numberOfCells << ' ' << totalSize << "\n";

    if (writeEdges.getValue())
    {
        for (int i=0 ; i<topology->getNbEdges() ; i++)
            outfile << 2 << ' ' << topology->getEdge(i) << "\n";
    }

    if (writeTriangles.getValue())
    {
        for (int i=0 ; i<topology->getNbTriangles() ; i++)
            outfile << 3 << ' ' <<  topology->getTriangle(i) << "\n";
    }
    if (writeQuads.getValue())
    {
        for (int i=0 ; i<topology->getNbQuads() ; i++)
            outfile << 4 << ' ' << topology->getQuad(i) << "\n";
    }

    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++)
            outfile << 4 << ' ' <<  topology->getTetra(i) << "\n";
    }
    if (writeHexas.getValue())
    {
        for (int i=0 ; i<topology->getNbHexas() ; i++)
            outfile << 8 << ' ' <<  topology->getHexa(i) << "\n";
    }

    outfile << "CELL_TYPES " << numberOfCells << "\n";

    if (writeEdges.getValue())
    {
        for (int i=0 ; i<topology->getNbEdges() ; i++)
            outfile << 3 << "\n";
    }

    if (writeTriangles.getValue())
    {
        for (int i=0 ; i<topology->getNbTriangles() ; i++)
            outfile << 5 << "\n";
    }
    if (writeQuads.getValue())
    {
        for (int i=0 ; i<topology->getNbQuads() ; i++)
            outfile << 9 << "\n";
    }

    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++)
            outfile << 10 << "\n";
    }
    if (writeHexas.getValue())
    {
        for (int i=0 ; i<topology->getNbHexas() ; i++)
            outfile << 12 << "\n";
    }
    /*
        //write dataset attributes
        if (!pointsData.empty())
        {
            outfile << "POINT_DATA " << nbp << "\n";
            writeData(pointsDataObject, pointsDataField, pointsDataName);
        }

        if (!cellsData.empty())
        {
            outfile << "CELL_DATA " << numberOfCells << "\n";
            writeData(cellsDataObject, cellsDataField, cellsDataName);
        }
    */
    sout << filename << " written" << sendl;
}

/// http://geuz.org/gmsh/doc/texinfo/gmsh.html#File-formats
void MeshExporter::writeMeshGmsh()
{
    std::string filename = getMeshFilename(".gmsh");

    std::ofstream outfile(filename.c_str());
    if( !outfile.is_open() )
    {
        serr << "Error creating file "<<filename<<sendl;
        return;
    }

    outfile << std::setprecision (9);

    //const helper::vector<std::string>& pointsData = dPointsDataFields.getValue();
    //const helper::vector<std::string>& cellsData = dCellsDataFields.getValue();

    helper::ReadAccessor<Data<defaulttype::Vec3Types::VecCoord> > pointsPos = position;

    const int nbp = pointsPos.size();

    //Write header
    outfile << "$MeshFormat\n";
    outfile << "2.1" << ' ' << 0 << ' ' << 8 << "\n";
    outfile << "$EndMeshFormat\n";

    //Write Points
    outfile << "$Nodes\n";

    outfile << nbp << "\n";
    for (int i=0 ; i<nbp; i++)
    {
        outfile << 1+i << ' ' << pointsPos[i] << "\n";
    }

    outfile << "$EndNodes\n";

    //Write Cells
    outfile << "$Elements\n";
    unsigned int numberOfCells/*, totalSize*/;
    numberOfCells = ( (writeEdges.getValue()) ? topology->getNbEdges() : 0 )
            +( (writeTriangles.getValue()) ? topology->getNbTriangles() : 0 )
            +( (writeQuads.getValue()) ? topology->getNbQuads() : 0 )
            +( (writeTetras.getValue()) ? topology->getNbTetras() : 0 )
            +( (writeHexas.getValue()) ? topology->getNbHexas() : 0 );
    /*totalSize =     ( (writeEdges.getValue()) ? 3 * topology->getNbEdges() : 0 )
            +( (writeTriangles.getValue()) ? 4 *topology->getNbTriangles() : 0 )
            +( (writeQuads.getValue()) ? 5 *topology->getNbQuads() : 0 )
            +( (writeTetras.getValue()) ? 5 *topology->getNbTetras() : 0 )
            +( (writeHexas.getValue()) ? 9 *topology->getNbHexas() : 0 );*/


    outfile << numberOfCells << "\n";
    unsigned int elem = 0;
    if (writeEdges.getValue())
    {
        for (int i=0 ; i<topology->getNbEdges() ; i++)
        {
            outfile << ++elem << ' ' << 1 << ' ' << 0;
            sofa::core::topology::BaseMeshTopology::Edge t = topology->getEdge(i);
            for (unsigned int j=0; j<t.size(); ++j)
                outfile << ' ' << 1+t[j];
            outfile << "\n";
        }
    }

    if (writeTriangles.getValue())
    {
        for (int i=0 ; i<topology->getNbTriangles() ; i++)
        {
            outfile << ++elem << ' ' << 2 << ' ' << 0;
            sofa::core::topology::BaseMeshTopology::Triangle t = topology->getTriangle(i);
            for (unsigned int j=0; j<t.size(); ++j)
                outfile << ' ' << 1+t[j];
            outfile << "\n";
        }
    }
    if (writeQuads.getValue())
    {
        for (int i=0 ; i<topology->getNbQuads() ; i++)
        {
            outfile << ++elem << ' ' << 3 << ' ' << 0;
            sofa::core::topology::BaseMeshTopology::Quad t = topology->getQuad(i);
            for (unsigned int j=0; j<t.size(); ++j)
                outfile << ' ' << 1+t[j];
            outfile << "\n";
        }
    }

    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++)
        {
            outfile << ++elem << ' ' << 4 << ' ' << 0;
            sofa::core::topology::BaseMeshTopology::Tetra t = topology->getTetra(i);
            for (unsigned int j=0; j<t.size(); ++j)
                outfile << ' ' << 1+t[j];
            outfile << "\n";
        }
    }
    if (writeHexas.getValue())
    {
        for (int i=0 ; i<topology->getNbHexas() ; i++)
        {
            outfile << ++elem << ' ' << 5 << ' ' << 0;
            sofa::core::topology::BaseMeshTopology::Hexa t = topology->getHexa(i);
            for (unsigned int j=0; j<t.size(); ++j)
                outfile << ' ' << 1+t[j];
            outfile << "\n";
        }
    }

    outfile << "$EndElements\n";

    sout << filename << " written" << sendl;
}

void MeshExporter::writeMeshNetgen()
{
    std::string filename = getMeshFilename(".mesh");

    std::ofstream outfile(filename.c_str());
    if (!outfile.is_open())
    {
        serr << "Error creating file "<<filename<<sendl;
        return;
    }

    outfile << std::setprecision (9);

    //const helper::vector<std::string>& pointsData = dPointsDataFields.getValue();
    //const helper::vector<std::string>& cellsData = dCellsDataFields.getValue();

    helper::ReadAccessor<Data<defaulttype::Vec3Types::VecCoord> > pointsPos = position;

    const int nbp = pointsPos.size();

    //Write Points
    outfile << nbp << "\n";
    for (int i=0 ; i<nbp; i++)
    {
        outfile << pointsPos[i] << "\n";
    }

    //Write Volume Elements
    outfile << ((writeTetras.getValue()) ? topology->getNbTetras() : 0) << "\n";
    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++)
        {
            sofa::core::topology::BaseMeshTopology::Tetra t = topology->getTetra(i);
            outfile << 0; // subdomain
            for (unsigned int j = 0; j < t.size(); ++j)
                outfile << ' ' << 1+t[j];
            outfile << "\n";
        }
    }

    //Write Surface Elements
    int nbtri = 0;
    if (writeTriangles.getValue())
    {
        if (topology->getNbTetras() == 0)
        {
            nbtri += topology->getNbTriangles();
        }
        else
        {
            for (int i=0; i<topology->getNbTriangles(); ++i)
            {
                if (topology->getTetrahedraAroundTriangle(i).size() < 2)
                    ++nbtri;
            }
        }
    }
    outfile << nbtri << "\n";
    if (writeTriangles.getValue())
    {
        if (topology->getNbTetras() == 0)
        {
            for (int i=0 ; i<topology->getNbTriangles() ; i++)
            {
                sofa::core::topology::BaseMeshTopology::Triangle t = topology->getTriangle(i);
                outfile << 0; // subdomain
                for (unsigned int j = 0; j < t.size(); ++j)
                    outfile << ' ' << 1+t[j];
                outfile << "\n";
            }
        }
        else
        {
            for (int i=0; i<topology->getNbTriangles(); ++i)
            {
                if (topology->getTetrahedraAroundTriangle(i).size() < 2)
                    ++nbtri;
                sofa::core::topology::BaseMeshTopology::Triangle t = topology->getTriangle(i);
                outfile << 0; // subdomain
                for (unsigned int j = 0; j < t.size(); ++j)
                    outfile << ' ' << 1+t[j];
                outfile << "\n";
            }
        }
    }
    sout << filename << " written" << sendl;
}

/// http://tetgen.berlios.de/fformats.html
void MeshExporter::writeMeshTetgen()
{
    std::string filename = getMeshFilename(".node");

    std::ofstream outfile(filename.c_str());
    if(!outfile.is_open())
    {
        serr << "Error creating file "<<filename<<sendl;
        return;
    }

    outfile << std::setprecision (9);

    //const helper::vector<std::string>& pointsData = dPointsDataFields.getValue();
    //const helper::vector<std::string>& cellsData = dCellsDataFields.getValue();

    helper::ReadAccessor<Data<defaulttype::Vec3Types::VecCoord> > pointsPos = position;

    // Write Points

    const int nbp = pointsPos.size();

    // http://tetgen.berlios.de/fformats.node.html
    // <# of points> <dimension (must be 3)> <# of attributes> <# of boundary markers (0 or 1)>
    outfile << nbp << ' ' << 3 << ' ' << 0 << ' ' << 0 << "\n";
    // <point #> <x> <y> <z> [attributes] [boundary marker]
    for (int i=0 ; i<nbp; i++)
    {
        outfile << i+1 << ' ' << pointsPos[i] << "\n";
    }

    outfile.close();
    sout << filename << " written" << sendl;

    //Write Volume Elements

    if (writeTetras.getValue())
    {
        // http://tetgen.berlios.de/fformats.ele.html
        filename = getMeshFilename(".ele");
        std::ofstream outfile(filename.c_str());
        if (!outfile.is_open())
        {
            serr << "Error creating file "<<filename<<sendl;
            return;
        }
        // <# of tetrahedra> <nodes per tetrahedron> <# of attributes>
        outfile << ((writeTetras.getValue()) ? topology->getNbTetras() : 0) << ' ' << 4 << ' ' << 0 << "\n";
        // <tetrahedron #> <node> <node> <node> <node> ... [attributes]
        if (writeTetras.getValue())
        {
            for (int i=0 ; i<topology->getNbTetras() ; i++)
            {
                sofa::core::topology::BaseMeshTopology::Tetra t = topology->getTetra(i);
                // check tetra inversion
                if (dot(pointsPos[t[1]]-pointsPos[t[0]],cross(pointsPos[t[2]]-pointsPos[t[0]],pointsPos[t[3]]-pointsPos[t[0]])) > 0)
                {
                    //sout << "Inverting tetra " << i << sendl;
                    unsigned int tmp = t[3]; t[3] = t[2]; t[2] = tmp;
                }

                outfile << 1+i; // id
                for (unsigned int j = 0; j < t.size(); ++j)
                    outfile << ' ' << 1+t[j];
                outfile << "\n";
            }
        }
        outfile.close();
        sout << filename << " written" << sendl;
    }

    //Write Surface Elements
    if (writeTriangles.getValue())
    {
        // http://tetgen.berlios.de/fformats.face.html
        filename = getMeshFilename(".face");
        std::ofstream outfile(filename.c_str());
        if (!outfile.is_open())
        {
            serr << "Error creating file "<<filename<<sendl;
            return;
        }
        int nbtri = 0;
        if (writeTriangles.getValue())
        {
            if (topology->getNbTetras() == 0)
            {
                nbtri += topology->getNbTriangles();
            }
            else
            {
                for (int i=0; i<topology->getNbTriangles(); ++i)
                {
                    if (topology->getTetrahedraAroundTriangle(i).size() < 2)
                        ++nbtri;
                }
            }
        }
        // <# of faces> <boundary marker (0 or 1)>
        outfile << nbtri << ' ' << 0 << "\n";
        // <face #> <node> <node> <node> [boundary marker]
        if (topology->getNbTetras() == 0)
        {
            for (int i=0 ; i<topology->getNbTriangles() ; i++)
            {
                sofa::core::topology::BaseMeshTopology::Triangle t = topology->getTriangle(i);
                outfile << 1+i; // id
                for (unsigned int j = 0; j < t.size(); ++j)
                    outfile << ' ' << 1+t[j];
                outfile << "\n";
            }
        }
        else
        {
            for (int i=0; i<topology->getNbTriangles(); ++i)
            {
                if (topology->getTetrahedraAroundTriangle(i).size() < 2)
                    ++nbtri;
                sofa::core::topology::BaseMeshTopology::Triangle t = topology->getTriangle(i);
                outfile << 1+i; // id
                for (unsigned int j = 0; j < t.size(); ++j)
                    outfile << ' ' << 1+t[j];
                outfile << "\n";
            }
        }
        outfile.close();
        sout << filename << " written" << sendl;
    }
}

void MeshExporter::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (sofa::core::objectmodel::KeypressedEvent::checkEventType(event))
    {
        sofa::core::objectmodel::KeypressedEvent *ev = static_cast<sofa::core::objectmodel::KeypressedEvent *>(event);
        switch(ev->getKey())
        {

        case 'E':
        case 'e':
            writeMesh();
            break;
        }
    }
    else if ( /*simulation::AnimateEndEvent* ev =*/ simulation::AnimateEndEvent::checkEventType(event))
    {
        unsigned int maxStep = exportEveryNbSteps.getValue();
        if (maxStep == 0) return;

        stepCounter++;
        if(stepCounter >= maxStep)
        {
            stepCounter = 0;
            writeMesh();
        }
    }
}

void MeshExporter::cleanup()
{
    if (exportAtEnd.getValue())
        writeMesh();
}

void MeshExporter::bwdInit()
{
    if (exportAtBegin.getValue())
        writeMesh();
}

} // namespace misc

} // namespace component

} // namespace sofa
