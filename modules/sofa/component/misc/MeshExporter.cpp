#include "MeshExporter.h"

#include <sstream>

#include <sofa/core/ObjectFactory.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
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
    : stepCounter(0), outfile(NULL), nbFiles(0)
    , meshFilename( initData(&meshFilename, "filename", "output Mesh file name"))
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
    if (outfile)
        delete outfile;
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
        if (!parent && mstate) parent = mstate->findField("position");
        if (!parent && topology) parent = mstate->findField("topology");
        if (parent)
        {
            position.setParent(parent);
            position.setReadOnly(true);
        }
    }

    nbFiles = 0;
// 	const std::string filename = meshFilename.getFullPath();
// //	std::cout << filename << std::endl;
//
// 	outfile = new std::ofstream(filename.c_str());
// 	if( !outfile->is_open() )
// 	{
// 		serr << "Error creating file "<<filename<<sendl;
// 		delete outfile;
// 		outfile = NULL;
// 		return;
// 	}
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

void MeshExporter::writeMeshVTKXML()
{
    std::string filename;
    std::ostringstream oss;
    oss << meshFilename.getFullPath() << nbFiles;
    filename = oss.str() + ".vtu";
//	std::cout << filename << std::endl;

    outfile = new std::ofstream(filename.c_str());
    if( !outfile->is_open() )
    {
        serr << "Error creating file "<<filename<<sendl;
        delete outfile;
        outfile = NULL;
        return;
    }
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
    *outfile << "<?xml version=\"1.0\"?>" << std::endl;
    *outfile << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">" << std::endl;
    *outfile << "  <UnstructuredGrid>" << std::endl;
    //write piece
    *outfile << "    <Piece NumberOfPoints=\"" << nbp << "\" NumberOfCells=\""<< numberOfCells << "\">" << std::endl;

    /*
    	//write point data
    	if (!pointsData.empty())
    	{
    		*outfile << "      <PointData>" << std::endl;
    		writeDataArray(pointsDataObject, pointsDataField, pointsDataName);
    		*outfile << "      </PointData>" << std::endl;
    	}
    		//write cell data
    	if (!cellsData.empty())
    	{
    		*outfile << "      <CellData>" << std::endl;
    		writeDataArray(cellsDataObject, cellsDataField, cellsDataName);
    		*outfile << "      </CellData>" << std::endl;
    	}

    */

    //write points
    *outfile << "      <Points>" << std::endl;
    *outfile << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
    if (mstate && mstate->getSize() == nbp)
    {
        for (int i = 0; i < mstate->getSize(); i++)
            *outfile << "          " << mstate->getPX(i) << ' ' << mstate->getPY(i) << ' ' << mstate->getPZ(i) << std::endl;
    }
    else
    {
        for (int i = 0; i < nbp; i++)
            *outfile << "          " << topology->getPX(i) << ' ' << topology->getPY(i) << ' ' << topology->getPZ(i) << std::endl;
    }
    *outfile << "        </DataArray>" << std::endl;
    *outfile << "      </Points>" << std::endl;
    //write cells
    *outfile << "      <Cells>" << std::endl;
    //write connectivity
    *outfile << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
    if (writeEdges.getValue())
    {
        for (int i=0 ; i<topology->getNbEdges() ; i++)
            *outfile << "          " << topology->getEdge(i) << std::endl;
    }

    if (writeTriangles.getValue())
    {
        for (int i=0 ; i<topology->getNbTriangles() ; i++)
            *outfile << "          " <<  topology->getTriangle(i) << std::endl;
    }
    if (writeQuads.getValue())
    {
        for (int i=0 ; i<topology->getNbQuads() ; i++)
            *outfile << "          " << topology->getQuad(i) << std::endl;
    }
    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++)
            *outfile << "          " <<  topology->getTetra(i) << std::endl;
    }
    if (writeHexas.getValue())
    {
        for (int i=0 ; i<topology->getNbHexas() ; i++)
            *outfile << "          " <<  topology->getHexa(i) << std::endl;
    }
    *outfile << "        </DataArray>" << std::endl;
    //write offsets
    int num = 0;
    *outfile << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
    *outfile << "          ";
    if (writeEdges.getValue())
    {
        for (int i=0 ; i<topology->getNbEdges() ; i++)
        {
            num += 2;
            *outfile << num << ' ';
        }
    }
    if (writeTriangles.getValue())
    {
        for (int i=0 ; i<topology->getNbTriangles() ; i++)
        {
            num += 3;
            *outfile << num << ' ';
        }
    }
    if (writeQuads.getValue())
    {
        for (int i=0 ; i<topology->getNbQuads() ; i++)
        {
            num += 4;
            *outfile << num << ' ';
        }
    }
    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++)
        {
            num += 4;
            *outfile << num << ' ';
        }
    }
    if (writeHexas.getValue())
    {
        for (int i=0 ; i<topology->getNbHexas() ; i++)
        {
            num += 6;
            *outfile << num << ' ';
        }
    }
    *outfile << std::endl;
    *outfile << "        </DataArray>" << std::endl;
    //write types
    *outfile << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">" << std::endl;
    *outfile << "          ";
    if (writeEdges.getValue())
    {
        for (int i=0 ; i<topology->getNbEdges() ; i++)
            *outfile << 3 << ' ';
    }
    if (writeTriangles.getValue())
    {
        for (int i=0 ; i<topology->getNbTriangles() ; i++)
            *outfile << 5 << ' ';
    }
    if (writeQuads.getValue())
    {
        for (int i=0 ; i<topology->getNbQuads() ; i++)
            *outfile << 9 << ' ';
    }
    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++)
            *outfile << 10 << ' ';
    }
    if (writeHexas.getValue())
    {
        for (int i=0 ; i<topology->getNbHexas() ; i++)
            *outfile << 12 << ' ';
    }
    *outfile << std::endl;
    *outfile << "        </DataArray>" << std::endl;
    *outfile << "      </Cells>" << std::endl;

    //write end
    *outfile << "    </Piece>" << std::endl;
    *outfile << "  </UnstructuredGrid>" << std::endl;
    *outfile << "</VTKFile>" << std::endl;
    outfile->close();
    sout << filename << " written" << sendl;
}

void MeshExporter::writeMeshVTK()
{
    std::string filename;
    std::ostringstream oss;
    oss << meshFilename.getFullPath() << nbFiles;
    filename = oss.str() + ".vtk";
//	std::cout << filename << std::endl;

    outfile = new std::ofstream(filename.c_str());
    if( !outfile->is_open() )
    {
        serr << "Error creating file "<<filename<<sendl;
        delete outfile;
        outfile = NULL;
        return;
    }

    //const helper::vector<std::string>& pointsData = dPointsDataFields.getValue();
    //const helper::vector<std::string>& cellsData = dCellsDataFields.getValue();

    helper::ReadAccessor<Data<defaulttype::Vec3Types::VecCoord> > pointsPos = position;

    const int nbp = pointsPos.size();

    //Write header
    *outfile << "# vtk DataFile Version 2.0" << std::endl;

    //write Title
    *outfile << "SOFA Exported Mesh file" << std::endl;

    //write Data type
    *outfile << "ASCII" << std::endl;

    //write dataset (geometry, unstructured grid)
    *outfile << "DATASET " << "UNSTRUCTURED_GRID" << std::endl;

    *outfile << "POINTS " << nbp << " float" << std::endl;

    //Write Points
    for (int i=0 ; i<nbp; i++)
    {
        *outfile << pointsPos[i] << std::endl;
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


    *outfile << "CELLS " << numberOfCells << ' ' << totalSize << std::endl;

    if (writeEdges.getValue())
    {
        for (int i=0 ; i<topology->getNbEdges() ; i++)
            *outfile << 2 << ' ' << topology->getEdge(i) << std::endl;
    }

    if (writeTriangles.getValue())
    {
        for (int i=0 ; i<topology->getNbTriangles() ; i++)
            *outfile << 3 << ' ' <<  topology->getTriangle(i) << std::endl;
    }
    if (writeQuads.getValue())
    {
        for (int i=0 ; i<topology->getNbQuads() ; i++)
            *outfile << 4 << ' ' << topology->getQuad(i) << std::endl;
    }

    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++)
            *outfile << 4 << ' ' <<  topology->getTetra(i) << std::endl;
    }
    if (writeHexas.getValue())
    {
        for (int i=0 ; i<topology->getNbHexas() ; i++)
            *outfile << 8 << ' ' <<  topology->getHexa(i) << std::endl;
    }

    *outfile << "CELL_TYPES " << numberOfCells << std::endl;

    if (writeEdges.getValue())
    {
        for (int i=0 ; i<topology->getNbEdges() ; i++)
            *outfile << 3 << std::endl;
    }

    if (writeTriangles.getValue())
    {
        for (int i=0 ; i<topology->getNbTriangles() ; i++)
            *outfile << 5 << std::endl;
    }
    if (writeQuads.getValue())
    {
        for (int i=0 ; i<topology->getNbQuads() ; i++)
            *outfile << 9 << std::endl;
    }

    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++)
            *outfile << 10 << std::endl;
    }
    if (writeHexas.getValue())
    {
        for (int i=0 ; i<topology->getNbHexas() ; i++)
            *outfile << 12 << std::endl;
    }
    /*
    	//write dataset attributes
    	if (!pointsData.empty())
    	{
    		*outfile << "POINT_DATA " << nbp << std::endl;
    		writeData(pointsDataObject, pointsDataField, pointsDataName);
    	}

    	if (!cellsData.empty())
    	{
    		*outfile << "CELL_DATA " << numberOfCells << std::endl;
    		writeData(cellsDataObject, cellsDataField, cellsDataName);
    	}
    */
    outfile->close();
    sout << filename << " written" << sendl;
}

/// http://geuz.org/gmsh/doc/texinfo/gmsh.html#File-formats
void MeshExporter::writeMeshGmsh()
{
    std::string filename;
    std::ostringstream oss;
    oss << meshFilename.getFullPath() << nbFiles;
    filename = oss.str() + ".gmsh";
//	std::cout << filename << std::endl;

    outfile = new std::ofstream(filename.c_str());
    if( !outfile->is_open() )
    {
        serr << "Error creating file "<<filename<<sendl;
        delete outfile;
        outfile = NULL;
        return;
    }

    //const helper::vector<std::string>& pointsData = dPointsDataFields.getValue();
    //const helper::vector<std::string>& cellsData = dCellsDataFields.getValue();

    helper::ReadAccessor<Data<defaulttype::Vec3Types::VecCoord> > pointsPos = position;

    const int nbp = pointsPos.size();

    //Write header
    *outfile << "$MeshFormat" << std::endl;
    *outfile << "2.1" << ' ' << 0 << ' ' << 8 << std::endl;
    *outfile << "$EndMeshFormat" << std::endl;

    //Write Points
    *outfile << "$Nodes" << std::endl;

    *outfile << nbp << std::endl;
    for (int i=0 ; i<nbp; i++)
    {
        *outfile << 1+i << ' ' << pointsPos[i] << std::endl;
    }

    *outfile << "$EndNodes" << std::endl;

    //Write Cells
    *outfile << "$Elements" << std::endl;
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


    *outfile << numberOfCells << std::endl;
    unsigned int elem = 0;
    if (writeEdges.getValue())
    {
        for (int i=0 ; i<topology->getNbEdges() ; i++)
        {
            *outfile << ++elem << ' ' << 1 << ' ' << 0;
            sofa::core::topology::BaseMeshTopology::Edge t = topology->getEdge(i);
            for (unsigned int j=0; j<t.size(); ++j)
                *outfile << ' ' << 1+t[j];
            *outfile << std::endl;
        }
    }

    if (writeTriangles.getValue())
    {
        for (int i=0 ; i<topology->getNbTriangles() ; i++)
        {
            *outfile << ++elem << ' ' << 2 << ' ' << 0;
            sofa::core::topology::BaseMeshTopology::Triangle t = topology->getTriangle(i);
            for (unsigned int j=0; j<t.size(); ++j)
                *outfile << ' ' << 1+t[j];
            *outfile << std::endl;
        }
    }
    if (writeQuads.getValue())
    {
        for (int i=0 ; i<topology->getNbQuads() ; i++)
        {
            *outfile << ++elem << ' ' << 3 << ' ' << 0;
            sofa::core::topology::BaseMeshTopology::Quad t = topology->getQuad(i);
            for (unsigned int j=0; j<t.size(); ++j)
                *outfile << ' ' << 1+t[j];
            *outfile << std::endl;
        }
    }

    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++)
        {
            *outfile << ++elem << ' ' << 4 << ' ' << 0;
            sofa::core::topology::BaseMeshTopology::Tetra t = topology->getTetra(i);
            for (unsigned int j=0; j<t.size(); ++j)
                *outfile << ' ' << 1+t[j];
            *outfile << std::endl;
        }
    }
    if (writeHexas.getValue())
    {
        for (int i=0 ; i<topology->getNbHexas() ; i++)
        {
            *outfile << ++elem << ' ' << 5 << ' ' << 0;
            sofa::core::topology::BaseMeshTopology::Hexa t = topology->getHexa(i);
            for (unsigned int j=0; j<t.size(); ++j)
                *outfile << ' ' << 1+t[j];
            *outfile << std::endl;
        }
    }

    *outfile << "$EndElements" << std::endl;

    outfile->close();
    sout << filename << " written" << sendl;
}

void MeshExporter::writeMeshNetgen()
{
    std::string filename;
    std::ostringstream oss;
    oss << meshFilename.getFullPath() << nbFiles;
    filename = oss.str() + ".mesh";
//	std::cout << filename << std::endl;

    outfile = new std::ofstream(filename.c_str());
    if( !outfile->is_open() )
    {
        serr << "Error creating file "<<filename<<sendl;
        delete outfile;
        outfile = NULL;
        return;
    }

    //const helper::vector<std::string>& pointsData = dPointsDataFields.getValue();
    //const helper::vector<std::string>& cellsData = dCellsDataFields.getValue();

    helper::ReadAccessor<Data<defaulttype::Vec3Types::VecCoord> > pointsPos = position;

    const int nbp = pointsPos.size();

    //Write Points
    *outfile << nbp << std::endl;
    for (int i=0 ; i<nbp; i++)
    {
        *outfile << pointsPos[i] << std::endl;
    }

    //Write Volume Elements
    *outfile << ((writeTetras.getValue()) ? topology->getNbTetras() : 0) << std::endl;
    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++)
        {
            sofa::core::topology::BaseMeshTopology::Tetra t = topology->getTetra(i);
            *outfile << 0; // subdomain
            for (unsigned int j = 0; j < t.size(); ++j)
                *outfile << ' ' << 1+t[j];
            *outfile << std::endl;
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
    *outfile << nbtri << std::endl;
    if (writeTriangles.getValue())
    {
        if (topology->getNbTetras() == 0)
        {
            for (int i=0 ; i<topology->getNbTriangles() ; i++)
            {
                sofa::core::topology::BaseMeshTopology::Triangle t = topology->getTriangle(i);
                *outfile << 0; // subdomain
                for (unsigned int j = 0; j < t.size(); ++j)
                    *outfile << ' ' << 1+t[j];
                *outfile << std::endl;
            }
        }
        else
        {
            for (int i=0; i<topology->getNbTriangles(); ++i)
            {
                if (topology->getTetrahedraAroundTriangle(i).size() < 2)
                    ++nbtri;
                sofa::core::topology::BaseMeshTopology::Triangle t = topology->getTriangle(i);
                *outfile << 0; // subdomain
                for (unsigned int j = 0; j < t.size(); ++j)
                    *outfile << ' ' << 1+t[j];
                *outfile << std::endl;
            }
        }
    }
    outfile->close();
    sout << filename << " written" << sendl;
}

/// http://tetgen.berlios.de/fformats.html
void MeshExporter::writeMeshTetgen()
{
    std::string filename;
    std::ostringstream oss;
    oss << meshFilename.getFullPath() << nbFiles;
    filename = oss.str() + ".node";
//	std::cout << filename << std::endl;

    outfile = new std::ofstream(filename.c_str());
    if( !outfile->is_open() )
    {
        serr << "Error creating file "<<filename<<sendl;
        delete outfile;
        outfile = NULL;
        return;
    }

    //const helper::vector<std::string>& pointsData = dPointsDataFields.getValue();
    //const helper::vector<std::string>& cellsData = dCellsDataFields.getValue();

    helper::ReadAccessor<Data<defaulttype::Vec3Types::VecCoord> > pointsPos = position;

    // Write Points

    const int nbp = pointsPos.size();

    // http://tetgen.berlios.de/fformats.node.html
    // <# of points> <dimension (must be 3)> <# of attributes> <# of boundary markers (0 or 1)>
    *outfile << nbp << ' ' << 3 << ' ' << 0 << ' ' << 0 << std::endl;
    // <point #> <x> <y> <z> [attributes] [boundary marker]
    for (int i=0 ; i<nbp; i++)
    {
        *outfile << i+1 << ' ' << pointsPos[i] << std::endl;
    }

    outfile->close();
    sout << filename << " written" << sendl;

    //Write Volume Elements

    if (writeTetras.getValue())
    {
        delete outfile;
        // http://tetgen.berlios.de/fformats.ele.html
        filename = oss.str() + ".ele";
        outfile = new std::ofstream(filename.c_str());
        if( !outfile->is_open() )
        {
            serr << "Error creating file "<<filename<<sendl;
            delete outfile;
            outfile = NULL;
            return;
        }
        // <# of tetrahedra> <nodes per tetrahedron> <# of attributes>
        *outfile << ((writeTetras.getValue()) ? topology->getNbTetras() : 0) << ' ' << 4 << ' ' << 0 << std::endl;
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

                *outfile << 1+i; // id
                for (unsigned int j = 0; j < t.size(); ++j)
                    *outfile << ' ' << 1+t[j];
                *outfile << std::endl;
            }
        }
        outfile->close();
        sout << filename << " written" << sendl;
    }

    //Write Surface Elements
    if (writeTriangles.getValue())
    {
        delete outfile;
        // http://tetgen.berlios.de/fformats.face.html
        filename = oss.str() + ".face";
        outfile = new std::ofstream(filename.c_str());
        if( !outfile->is_open() )
        {
            serr << "Error creating file "<<filename<<sendl;
            delete outfile;
            outfile = NULL;
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
        *outfile << nbtri << ' ' << 0 << std::endl;
        // <face #> <node> <node> <node> [boundary marker]
        if (topology->getNbTetras() == 0)
        {
            for (int i=0 ; i<topology->getNbTriangles() ; i++)
            {
                sofa::core::topology::BaseMeshTopology::Triangle t = topology->getTriangle(i);
                *outfile << 1+i; // id
                for (unsigned int j = 0; j < t.size(); ++j)
                    *outfile << ' ' << 1+t[j];
                *outfile << std::endl;
            }
        }
        else
        {
            for (int i=0; i<topology->getNbTriangles(); ++i)
            {
                if (topology->getTetrahedraAroundTriangle(i).size() < 2)
                    ++nbtri;
                sofa::core::topology::BaseMeshTopology::Triangle t = topology->getTriangle(i);
                *outfile << 1+i; // id
                for (unsigned int j = 0; j < t.size(); ++j)
                    *outfile << ' ' << 1+t[j];
                *outfile << std::endl;
            }
        }
        outfile->close();
        sout << filename << " written" << sendl;
    }
}

void MeshExporter::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
    {
        std::cout << "key pressed " << std::endl;
        switch(ev->getKey())
        {

        case 'E':
        case 'e':
            writeMesh();
            break;
        }
    }
    else if ( /*simulation::AnimateEndEvent* ev =*/  dynamic_cast<simulation::AnimateEndEvent*>(event))
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
