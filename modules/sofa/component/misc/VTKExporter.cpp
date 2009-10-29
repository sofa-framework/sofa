/*
 * VTKExporter.cpp
 *
 *  Created on: 9 sept. 2009
 *      Author: froy
 */

#include "VTKExporter.h"

#include <sofa/core/ObjectFactory.h>

#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(VTKExporter)

int VTKExporterClass = core::RegisterObject("Read State vectors from file at each timestep")
        .add< VTKExporter >();

VTKExporter::VTKExporter()
    : vtkFilename( initData(&vtkFilename, "filename", "output VTK file name"))
    , fileFormat( initData(&fileFormat, (bool) true, "fileFormat", "file format"))
    , writeEdges( initData(&writeEdges, (bool) true, "edges", "write edge topology"))
    , writeTriangles( initData(&writeTriangles, (bool) false, "triangles", "write triangle topology"))
    , writeQuads( initData(&writeQuads, (bool) false, "quads", "write quad topology"))
    , writeTetras( initData(&writeTetras, (bool) false, "tetras", "write tetra topology"))
    , writeHexas( initData(&writeHexas, (bool) false, "hexas", "write hexa topology"))
    , dPointsDataFields( initData(&dPointsDataFields, "pointsDataFields", "Data to visualize (on points)"))
    , dCellsDataFields( initData(&dCellsDataFields, "cellsDataFields", "Data to visualize (on cells)"))
{
    // TODO Auto-generated constructor stub

}

VTKExporter::~VTKExporter()
{
    if (outfile)
        delete outfile;
}

void VTKExporter::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    context->get(topology);

    if (!topology)
    {
        serr << "VTKExporter : error, no topology ." << sendl;
        return;
    }

    const std::string& filename = vtkFilename.getFullPath();
//	std::cout << filename << std::endl;

    outfile = new std::ofstream(filename.c_str());
    if( !outfile->is_open() )
    {
        serr << "Error creating file "<<filename<<sendl;
        delete outfile;
        outfile = NULL;
        return;
    }

    const helper::vector<std::string>& pointsData = dPointsDataFields.getValue();
    const helper::vector<std::string>& cellsData = dCellsDataFields.getValue();

    if (!pointsData.empty())
    {
        fetchDataFields(pointsData, pointsDataObject, pointsDataField);
    }
    if (!cellsData.empty())
    {
        fetchDataFields(cellsData, cellsDataObject, cellsDataField);
    }

}

void VTKExporter::fetchDataFields(const helper::vector<std::string>& strData, helper::vector<std::string>& objects, helper::vector<std::string>& fields)
{
    for (unsigned int i=0 ; i<strData.size() ; i++)
    {
        std::string objectName, dataFieldName;
        std::string::size_type loc = strData[i].find_last_of('.');
        if ( loc != std::string::npos)
        {
            objectName = strData[i].substr(0, loc);
            dataFieldName = strData[i].substr(loc+1);

            objects.push_back(objectName);
            fields.push_back(dataFieldName);
        }
        else
        {
            serr << "VTKExporter : error while parsing dataField names" << sendl;
        }
    }
}

void VTKExporter::writeData(const helper::vector<std::string>& objects, const helper::vector<std::string>& fields)
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();

    //std::cout << "List o: " << objects << std::endl;
    //std::cout << "List f: " << fields << std::endl;

    for (unsigned int i=0 ; i<objects.size() ; i++)
    {
        core::objectmodel::BaseObject* obj = context->get<core::objectmodel::BaseObject> (objects[i]);
        core::objectmodel::BaseData* field = NULL;
        //std::cout << objects[i] << std::endl;
        if (obj)
        {
            std::vector< std::pair<std::string, core::objectmodel::BaseData*> > f = obj->getFields();

            for (unsigned int j=0 ; j<f.size() && !field; j++)
            {
                if(fields[i].compare(f[j].first) == 0)
                    field = f[j].second;
            }
        }

        if (!obj || !field)
        {
            serr << "VTKExporter : error while fetching data field, check object name or field name " << sendl;
        }
        else
        {
            //std::cout << "Type: " << field->getValueTypeString() << std::endl;

            //retrieve data file type
//			if (dynamic_cast<Data< defaulttype::Vec3f >* >(field))
//				std::cout << "Vec3f" << std::endl;
//			if (dynamic_cast<Data< defaulttype::Vec3d >* >(field))
//				std::cout << "Vec3d" << std::endl;

            //Scalars

            std::string line;
            unsigned int sizeSeg=0;
            if (dynamic_cast<sofa::core::objectmodel::TData< helper::vector<float> >* >(field))
            {
                line = "float 1";
                sizeSeg = 1;
            }
            if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector<double> >* >(field))
            {
                line = "double 1";
                sizeSeg = 1;
            }
            if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector< defaulttype::Vec2f > >* > (field))
            {
                line = "float 2";
                sizeSeg = 2;
            }
            if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector< defaulttype::Vec2d > >* >(field))
            {
                line = "double 2";
                sizeSeg = 2;
            }

            //if this is a scalar
            if (!line.empty())
            {
                *outfile << "SCALARS" << " " << fields[i] << " ";
            }
            else
            {
                //Vectors
                if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector< defaulttype::Vec3f > >* > (field))
                {
                    line = "float";
                    sizeSeg = 3;
                }
                if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector< defaulttype::Vec3d > >* >(field))
                {
                    line = "double";
                    sizeSeg = 3;
                }
                *outfile << "VECTORS" << " " << fields[i] << " ";
            }
            *outfile << line << std::endl;
            *outfile << segmentString(field->getValueString(),sizeSeg) << std::endl;
            *outfile << std::endl;
        }
    }
}

void VTKExporter::writeDataArray(const helper::vector<std::string>& objects, const helper::vector<std::string>& fields)
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();

    //std::cout << "List o: " << objects << std::endl;
    //std::cout << "List f: " << fields << std::endl;

    for (unsigned int i=0 ; i<objects.size() ; i++)
    {
        core::objectmodel::BaseObject* obj = context->get<core::objectmodel::BaseObject> (objects[i]);
        core::objectmodel::BaseData* field = NULL;
        //std::cout << objects[i] << std::endl;
        if (obj)
        {
            std::vector< std::pair<std::string, core::objectmodel::BaseData*> > f = obj->getFields();

            for (unsigned int j=0 ; j<f.size() && !field; j++)
            {
                if(fields[i].compare(f[j].first) == 0)
                    field = f[j].second;
            }
        }

        if (!obj || !field)
        {
            serr << "VTKExporter : error while fetching data field, check object name or field name " << sendl;
        }
        else
        {
            //std::cout << "Type: " << field->getValueTypeString() << std::endl;

            //retrieve data file type
//			if (dynamic_cast<Data< defaulttype::Vec3f >* >(field))
//				std::cout << "Vec3f" << std::endl;
//			if (dynamic_cast<Data< defaulttype::Vec3d >* >(field))
//				std::cout << "Vec3d" << std::endl;

            //Scalars
            std::string type;
            unsigned int sizeSeg=0;
            if (dynamic_cast<sofa::core::objectmodel::TData< helper::vector<int> >* >(field))
            {
                type = "Int32";
                sizeSeg = 1;
            }
            if (dynamic_cast<sofa::core::objectmodel::TData< helper::vector<unsigned int> >* >(field))
            {
                type = "UInt32";
                sizeSeg = 1;
            }
            if (dynamic_cast<sofa::core::objectmodel::TData< helper::vector<float> >* >(field))
            {
                type = "Float32";
                sizeSeg = 1;
            }
            if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector<double> >* >(field))
            {
                type = "Float64";
                sizeSeg = 1;
            }

            //Vectors
            if (type.empty())
            {
                if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector< defaulttype::Vec3f > >* > (field))
                {
                    type = "Float32";
                    sizeSeg = 3;
                }
                if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector< defaulttype::Vec3d > >* >(field))
                {
                    type = "Float64";
                    sizeSeg = 3;
                }
            }
            *outfile << "        <DataArray type=\""<< type << "\" Name=\"" << fields[i] << "\" Format=\"ascii\">" << std::endl;
            *outfile << segmentString(field->getValueString(),sizeSeg) << std::endl;
            *outfile << "        </DataArray>" << std::endl;
        }
    }
}


std::string VTKExporter::segmentString(std::string str, unsigned int n)
{
    std::string::size_type loc = 0;
    unsigned int i=0;

    loc = str.find(' ', 0);

    while(loc != std::string::npos )
    {
        i++;
        if (i == n)
        {
            str[loc] = '\n';
            i=0;
        }
        loc = str.find(' ', loc+1);
    }

    return str;
}


void VTKExporter::writeVTKSimple()
{
    const helper::vector<std::string>& pointsData = dPointsDataFields.getValue();
    const helper::vector<std::string>& cellsData = dCellsDataFields.getValue();

    //Write header
    *outfile << "# vtk DataFile Version 2.0" << std::endl;

    //write Title
    *outfile << "Exported VTK file" << std::endl;

    //write Data type
    *outfile << "ASCII" << std::endl;

    *outfile << std::endl;

    //write dataset (geometry, unstructured grid)
    *outfile << "DATASET " << "UNSTRUCTURED_GRID" << std::endl;
    *outfile << "POINTS " << topology->getNbPoints() << " float" << std::endl;
    //write Points
    for (int i=0 ; i<topology->getNbPoints() ; i++)
    {
        *outfile << topology->getPX(i) << " " << topology->getPY(i) << " " << topology->getPZ(i) << std::endl;
    }

    *outfile << std::endl;

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


    *outfile << "CELLS " << numberOfCells << " " << totalSize << std::endl;

    if (writeEdges.getValue())
    {
        for (int i=0 ; i<topology->getNbEdges() ; i++)
            *outfile << 2 << " " << topology->getEdge(i) << std::endl;
    }

    if (writeTriangles.getValue())
    {
        for (int i=0 ; i<topology->getNbTriangles() ; i++)
            *outfile << 3 << " " <<  topology->getTriangle(i) << std::endl;
    }
    if (writeQuads.getValue())
    {
        for (int i=0 ; i<topology->getNbQuads() ; i++)
            *outfile << 4 << " " << topology->getQuad(i) << std::endl;
    }

    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++)
            *outfile << 4 << " " <<  topology->getTetra(i) << std::endl;
    }
    if (writeHexas.getValue())
    {
        for (int i=0 ; i<topology->getNbHexas() ; i++)
            *outfile << 8 << " " <<  topology->getHexa(i) << std::endl;
    }

    *outfile << std::endl;

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

    *outfile << std::endl;

    //write dataset attributes
    if (!pointsData.empty())
    {
        *outfile << "POINT_DATA " << topology->getNbPoints() << std::endl;
        writeData(pointsDataObject, pointsDataField);
    }

    if (!cellsData.empty())
    {
        *outfile << "CELL_DATA " << numberOfCells << std::endl;
        writeData(cellsDataObject, cellsDataField);
    }
    outfile->close();
    std::cout << "VTK written" << std::endl;
}

void VTKExporter::writeVTKXML()
{
    const helper::vector<std::string>& pointsData = dPointsDataFields.getValue();
    const helper::vector<std::string>& cellsData = dCellsDataFields.getValue();

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
    *outfile << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">" << std::endl;
    *outfile << "  <UnstructuredGrid>" << std::endl;
    //write piece
    *outfile << "    <Piece NumberOfPoints=\"" << topology->getNbPoints() << "\" NumberOfCells=\""<< numberOfCells << "\">" << std::endl;




    //write point data
    if (!pointsData.empty())
    {
        *outfile << "      <PointData>" << std::endl;
        writeDataArray(pointsDataObject, pointsDataField);
        *outfile << "      </PointData>" << std::endl;
    }
    //write cell data
    if (!cellsData.empty())
    {
        *outfile << "      <CellData>" << std::endl;
        writeDataArray(cellsDataObject, cellsDataField);
        *outfile << "      </CellData>" << std::endl;
    }



    //write points
    *outfile << "      <Points>" << std::endl;
    *outfile << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" Format=\"ascii\">" << std::endl;
    for (int i = 0; i < topology->getNbPoints(); i++)
        *outfile << "          " << topology->getPX(i) << " " << topology->getPY(i) << " " << topology->getPZ(i) << std::endl;
    *outfile << "        </DataArray>" << std::endl;
    *outfile << "      </Points>" << std::endl;
    //write cells
    *outfile << "      <Cells>" << std::endl;
    //write connectivity
    *outfile << "        <DataArray type=\"Int32\" Name=\"connectivity\" Format=\"ascii\">" << std::endl;
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
    *outfile << "        <DataArray type=\"Int32\" Name=\"offsets\" Format=\"ascii\">" << std::endl;
    *outfile << "          ";
    if (writeEdges.getValue())
    {
        for (int i=0 ; i<topology->getNbEdges() ; i++)
        {
            num += 2;
            *outfile << num << " ";
        }
    }
    if (writeTriangles.getValue())
    {
        for (int i=0 ; i<topology->getNbTriangles() ; i++, num+=3)
        {
            num += 3;
            *outfile << num << " ";
        }
    }
    if (writeQuads.getValue())
    {
        for (int i=0 ; i<topology->getNbQuads() ; i++, num+=4)
        {
            num += 4;
            *outfile << num << " ";
        }
    }
    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++, num+=4)
        {
            num += 4;
            *outfile << num << " ";
        }
    }
    if (writeHexas.getValue())
    {
        for (int i=0 ; i<topology->getNbHexas() ; i++, num+=6)
        {
            num += 6;
            *outfile << num << " ";
        }
    }
    *outfile << std::endl;
    *outfile << "        </DataArray>" << std::endl;
    //write types
    *outfile << "        <DataArray type=\"UInt8\" Name=\"types\" Format=\"ascii\">" << std::endl;
    *outfile << "          ";
    if (writeEdges.getValue())
    {
        for (int i=0 ; i<topology->getNbEdges() ; i++)
            *outfile << 3 << " ";
    }
    if (writeTriangles.getValue())
    {
        for (int i=0 ; i<topology->getNbTriangles() ; i++)
            *outfile << 5 << " ";
    }
    if (writeQuads.getValue())
    {
        for (int i=0 ; i<topology->getNbQuads() ; i++)
            *outfile << 9 << " ";
    }
    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++)
            *outfile << 10 << " ";
    }
    if (writeHexas.getValue())
    {
        for (int i=0 ; i<topology->getNbHexas() ; i++)
            *outfile << 12 << " ";
    }
    *outfile << std::endl;
    *outfile << "        </DataArray>" << std::endl;
    *outfile << "      </Cells>" << std::endl;

    //write end
    *outfile << "    </Piece>" << std::endl;
    *outfile << "  </UnstructuredGrid>" << std::endl;
    *outfile << "</VTKFile>" << std::endl;
    outfile->close();
    std::cout << "VTK written" << std::endl;
}


void VTKExporter::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
    {
        switch(ev->getKey())
        {

        case 'E':
        case 'e':
            if(fileFormat.getValue())
                writeVTKXML();
            else
                writeVTKSimple();
            break;
        }
    }
}


}

}

}
