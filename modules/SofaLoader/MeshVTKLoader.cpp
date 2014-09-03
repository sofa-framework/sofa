/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <SofaLoader/MeshVTKLoader.h>
#include <sofa/core/visual/VisualParams.h>

#include <iostream>
//#include <fstream> // we can't use iostream because the windows implementation gets confused by the mix of text and binary
#include <stdio.h>
#include <sstream>

namespace sofa
{

namespace component
{

namespace loader
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(MeshVTKLoader)

int MeshVTKLoaderClass = core::RegisterObject("Specific mesh loader for VTK file format.")
        .add< MeshVTKLoader >()
        ;

//Base VTK Loader
MeshVTKLoader::MeshVTKLoader() : MeshLoader()
    , reader(NULL)
{
}

MeshVTKLoader::VTKFileType MeshVTKLoader::detectFileType(const char* filename)
{
    std::ifstream inVTKFile(filename, std::ifstream::in | std::ifstream::binary);

    if( !inVTKFile.is_open() )
        return MeshVTKLoader::NONE;

    std::string line;
    std::getline(inVTKFile, line);

    if (line.find("<?xml") != std::string::npos)
    {
        std::getline(inVTKFile, line);

        if (line.find("<VTKFile") != std::string::npos)
            return MeshVTKLoader::XML;
        else
            return MeshVTKLoader::NONE;
    }
    else if (line.find("<VTKFile") != std::string::npos) //... not xml-compliant
        return MeshVTKLoader::XML;
    else if (line.find("# vtk DataFile") != std::string::npos)
        return MeshVTKLoader::LEGACY;
    else //default behavior if the first line is not correct ?
        return MeshVTKLoader::NONE;
}

bool MeshVTKLoader::load()
{
    sout << "Loading VTK file: " << m_filename << sendl;

    bool fileRead = false;

    // -- Loading file
    const char* filename = m_filename.getFullPath().c_str();

    // Detect file type (legacy or vtk)
    MeshVTKLoader::VTKFileType type = detectFileType(filename);
    switch (type)
    {
    case XML:
        reader = new XMLVTKReader();
        break;
    case LEGACY:
        reader = new LegacyVTKReader();
        break;
    case NONE:
    default:
        serr << "Header not recognized" << sendl;
        reader = NULL;
        break;
    }

    if (!reader)
        return false;

    // -- Reading file
    fileRead = reader->readVTK (filename);
    this->setInputsMesh();
    this->setInputsData();

    delete reader;

    return fileRead;
}

bool MeshVTKLoader::setInputsMesh()
{
    helper::vector<sofa::defaulttype::Vector3>& my_positions = *(positions.beginEdit());
    if (reader->inputPoints)
    {
        BaseVTKReader::VTKDataIO<double>* vtkpd =  dynamic_cast<BaseVTKReader::VTKDataIO<double>* > (reader->inputPoints);
        BaseVTKReader::VTKDataIO<float>* vtkpf =  dynamic_cast<BaseVTKReader::VTKDataIO<float>* > (reader->inputPoints);
        if (vtkpd)
        {
            const double* inPoints = (vtkpd->data);
            if (inPoints)
                for (int i=0; i < vtkpd->dataSize; i+=3)
                    my_positions.push_back(sofa::defaulttype::Vector3 ((double)inPoints[i+0], (double)inPoints[i+1], (double)inPoints[i+2]));
            else return false;
        }
        else if (vtkpf)
        {
            const float* inPoints = (vtkpf->data);
            if (inPoints)
                for (int i=0; i < vtkpf->dataSize; i+=3)
                    my_positions.push_back(sofa::defaulttype::Vector3 ((float)inPoints[i+0], (float)inPoints[i+1], (float)inPoints[i+2]));
            else return false;
        }
        else
        {
            serr << "Type of coordinate (X,Y,Z) not supported" << sendl;
            return false;
        }
    }
    else
        return false;

    positions.endEdit();

    helper::vector<Edge >& my_edges = *(edges.beginEdit());
    helper::vector<Triangle >& my_triangles = *(triangles.beginEdit());
    helper::vector<Quad >& my_quads = *(quads.beginEdit());
    helper::vector<Tetrahedron >& my_tetrahedra = *(tetrahedra.beginEdit());
    helper::vector<Hexahedron >& my_hexahedra = *(hexahedra.beginEdit());

    if (reader->inputPolygons)
    {
        const int* inFP = (const int*) reader->inputPolygons->getData();
        int poly = 0;
        for (int i=0; i < reader->inputPolygons->dataSize;)
        {
            int nv = inFP[i]; ++i;
            bool valid = true;
            if (reader->inputPoints)
            {
                for (int j=0; j<nv; ++j)
                    if ((unsigned)inFP[i+j] >= (unsigned)(reader->inputPoints->dataSize/3))
                    {
                        serr << "ERROR: invalid point " << inFP[i+j] << " in polygon " << poly << sendl;
                        valid = false;
                    }
            }
            if (valid)
            {
                if (nv == 4)
                {
                    addQuad(&my_quads, inFP[i+0],inFP[i+1],inFP[i+2],inFP[i+3]);
                }
                else if (nv >= 3)
                {
                    int f[3];
                    f[0] = inFP[i+0];
                    f[1] = inFP[i+1];
                    for (int j=2; j<nv; j++)
                    {
                        f[2] = inFP[i+j];
                        addTriangle(&my_triangles, f[0], f[1], f[2]);
                        f[1] = f[2];
                    }
                }
                i += nv;
            }
            ++poly;
        }
    }
    /*else if (reader->inputLines) {
        const int* inFP = (const int*) reader->inputLines->getData();

        std::cout << "DS: " << reader->inputLines->dataSize << " NDS: " << reader->inputLines->nestedDataSize << std::endl;

        int nbf = reader->numberOfLines;


    }*/
    else if (reader->inputCells && reader->inputCellTypes)
    {
        const int* inFP = (const int*) reader->inputCells->getData();
        //offsets are not used if we have parsed with the legacy method
        const int* offsets = (reader->inputCellOffsets == NULL) ? NULL : (const int*) reader->inputCellOffsets->getData();

        const int* dataT = (int*)(reader->inputCellTypes->getData());

        helper::vector<int> numSubPolyLines;

        int nbf = reader->numberOfCells;
        int i = 0;
        for (int c = 0; c < nbf; ++c)
        {
            int t = dataT[c];// - 48; //ASCII
            int nv;
            if (offsets)
            {
                i = (c == 0) ? 0 : offsets[c-1];
                nv = inFP[i];
            }
            else
            {
                nv = inFP[i]; ++i;
            }

            switch (t)
            {
            case 0: // EMPTY_CELL
                break;
            case 1: // VERTEX
                break;
            case 2: // POLY_VERTEX
                break;
            case 3: // LINE
                addEdge(&my_edges, inFP[i+0], inFP[i+1]);
                break;
            case 4: // POLY_LINE
                numSubPolyLines.push_back(nv);
                for (int v = 0; v < nv-1; ++v) {
                    addEdge(&my_edges, inFP[i+v+0], inFP[i+v+1]);
                    //std::cout << " c = " << c << " i = " << i <<  " v = " << v << "  edge: " << inFP[i+v+0] << " " << inFP[i+v+1] << std::endl;
                }
                break;
            case 5: // TRIANGLE
                addTriangle(&my_triangles,inFP[i+0], inFP[i+1], inFP[i+2]);
                break;
            case 6: // TRIANGLE_STRIP
                for (int j=0; j<nv-2; j++)
                    if (j&1)
                        addTriangle(&my_triangles, inFP[i+j+0],inFP[i+j+1],inFP[i+j+2]);
                    else
                        addTriangle(&my_triangles, inFP[i+j+0],inFP[i+j+2],inFP[i+j+1]);
                break;
            case 7: // POLYGON
                for (int j=2; j<nv; j++)
                    addTriangle(&my_triangles, inFP[i+0],inFP[i+j-1],inFP[i+j]);
                break;
            case 8: // PIXEL
                addQuad(&my_quads, inFP[i+0], inFP[i+1], inFP[i+3], inFP[i+2]);
                break;
            case 9: // QUAD
                addQuad(&my_quads, inFP[i+0], inFP[i+1], inFP[i+2], inFP[i+3]);
                break;
            case 10: // TETRA
                addTetrahedron(&my_tetrahedra, inFP[i+0], inFP[i+1], inFP[i+2], inFP[i+3]);
                break;
            case 11: // VOXEL
                addHexahedron(&my_hexahedra, inFP[i+0], inFP[i+1], inFP[i+3], inFP[i+2],
                        inFP[i+4], inFP[i+5], inFP[i+7], inFP[i+6]);
                break;
            case 12: // HEXAHEDRON
                addHexahedron(&my_hexahedra, inFP[i+0], inFP[i+1], inFP[i+2], inFP[i+3],
                        inFP[i+4], inFP[i+5], inFP[i+6], inFP[i+7]);
                break;
            // more types are defined in vtkCellType.h in libvtk
            default:
                serr << "ERROR: unsupported cell type " << t << sendl;
            }

            if (!offsets)
                i += nv;
        }

        if (numSubPolyLines.size() > 0) {
            size_t sz = reader->inputCellDataVector.size();
            reader->inputCellDataVector.resize(sz+1);
            reader->inputCellDataVector[sz] = reader->newVTKDataIO("int");

            BaseVTKReader::VTKDataIO<int>* cellData = dynamic_cast<BaseVTKReader::VTKDataIO<int>* > (reader->inputCellDataVector[sz]);

            if (cellData == NULL) return false;

            cellData->resize(numSubPolyLines.size());

            for (size_t ii = 0;  ii < numSubPolyLines.size(); ii++)
                cellData->data[ii] = numSubPolyLines[ii];

            cellData->name = "PolyLineSubEdges";
        }

    }
    if (reader->inputPoints) delete reader->inputPoints;
    if (reader->inputPolygons) delete reader->inputPolygons;
    if (reader->inputCells) delete reader->inputCells;
    if (reader->inputCellTypes) delete reader->inputCellTypes;

    edges.endEdit();
    triangles.endEdit();
    quads.endEdit();
    tetrahedra.endEdit();
    hexahedra.endEdit();

    return true;
}

bool MeshVTKLoader::setInputsData()
{
    //std::vector< std::pair<std::string, core::objectmodel::BaseData*> > f = this->getFields();
    //std::cout << "Number of Fields before :" << f.size() << std::endl;

    ///Point Data
    for (size_t i=0 ; i<reader->inputPointDataVector.size() ; i++)
    {
        const char* dataname = reader->inputPointDataVector[i]->name.c_str();

        core::objectmodel::BaseData* basedata = reader->inputPointDataVector[i]->createSofaData();
        this->addData(basedata, dataname);
    }

    ///Cell Data
    for (size_t i=0 ; i<reader->inputCellDataVector.size() ; i++)
    {
        const char* dataname = reader->inputCellDataVector[i]->name.c_str();

        core::objectmodel::BaseData* basedata = reader->inputCellDataVector[i]->createSofaData();
        this->addData(basedata, dataname);
    }

    //f = this->getFields();
    //std::cout << "Number of Fields after :" << f.size() << std::endl;
    return true;
}


//Legacy VTK Loader
bool MeshVTKLoader::LegacyVTKReader::readFile(const char* filename)
{
    std::ifstream inVTKFile(filename, std::ifstream::in | std::ifstream::binary);
    if( !inVTKFile.is_open() )
    {
        return false;
    }
    std::string line;

    // Part 1
    std::getline(inVTKFile, line);
    if (std::string(line,0,23) != "# vtk DataFile Version ")
    {
        serr << "Error: Unrecognized header in file '" << filename << "'." << sendl;
        inVTKFile.close();
        return false;
    }
    std::string version(line,23);

    // Part 2
    std::string header;
    std::getline(inVTKFile, header);

    // Part 3
    std::getline(inVTKFile, line);

    int binary;
    if (line == "BINARY" || line == "BINARY\r" )
    {
        binary = 1;
    }
    else if ( line == "ASCII" || line == "ASCII\r")
    {
        binary = 0;
    }
    else
    {
        serr << "Error: Unrecognized format in file '" << filename << "'." << sendl;
        inVTKFile.close();
        return false;
    }

    if (binary && strlen(filename)>9 && !strcmp(filename+strlen(filename)-9,".vtk_swap"))
        binary = 2; // bytes will be swapped


    // Part 4
    do
        std::getline(inVTKFile, line);
    while (line == "");
    if (line != "DATASET POLYDATA" && line != "DATASET UNSTRUCTURED_GRID"
        && line != "DATASET POLYDATA\r" && line != "DATASET UNSTRUCTURED_GRID\r" )
    {
        serr << "Error: Unsupported data type in file '" << filename << "'." << sendl;
        inVTKFile.close();
        return false;
    }

    sout << (binary == 0 ? "Text" : (binary == 1) ? "Binary" : "Swapped Binary") << " VTK File (version " << version << "): " << header << sendl;
    //VTKDataIO<double>* inputPointsDouble = NULL;
    VTKDataIO<int>* inputPolygonsInt = NULL;
    VTKDataIO<int>* inputCellsInt = NULL;
    VTKDataIO<int>* inputCellTypesInt = NULL;
    inputCellOffsets = NULL;

    while(!inVTKFile.eof())
    {
        std::getline(inVTKFile, line);
        if (line.empty()) continue;
        std::istringstream ln(line);
        std::string kw;
        ln >> kw;
        if (kw == "POINTS")
        {
            int n;
            std::string typestr;
            ln >> n >> typestr;
            sout << "Found " << n << " " << typestr << " points" << sendl;
            inputPoints = newVTKDataIO(typestr);
            //inputPoints = new VTKDataIO<double>;
            if (inputPoints == NULL) return false;
            if (!inputPoints->read(inVTKFile, 3*n, binary)) return false;
            //nbp = n;
        }
        else if (kw == "POLYGONS")
        {
            int n, ni;
            ln >> n >> ni;
            sout << "Found " << n << " polygons ( " << (ni - 3*n) << " triangles )" << sendl;
            inputPolygons = new VTKDataIO<int>;
            inputPolygonsInt = dynamic_cast<VTKDataIO<int>* > (inputPolygons);
            if (!inputPolygons->read(inVTKFile, ni, binary)) return false;
        }
        else if (kw == "CELLS")
        {
            int n, ni;
            ln >> n >> ni;
            sout << "Found " << n << " cells" << sendl;
            inputCells = new VTKDataIO<int>;
            inputCellsInt = dynamic_cast<VTKDataIO<int>* > (inputCells);
            if (!inputCells->read(inVTKFile, ni, binary)) return false;
            numberOfCells = n;
        }
         else if (kw == "LINES")
        {
            int n, ni;
            ln >> n >> ni;
            sout << "Found " << n << " lines" << sendl;
            inputCells = new VTKDataIO<int>;
            inputCellsInt = dynamic_cast<VTKDataIO<int>* > (inputCellsInt);
            if (!inputCells->read(inVTKFile, ni, binary)) return false;
            numberOfCells = n;

            inputCellTypes = new VTKDataIO<int>;
            inputCellTypesInt = dynamic_cast<VTKDataIO<int>* > (inputCellTypes);
            inputCellTypesInt->resize(n);
            for (int i = 0; i < n; i++)
                inputCellTypesInt->data[i] = 4;
        }
        else if (kw == "CELL_TYPES")
        {
            int n;
            ln >> n;
            inputCellTypes = new VTKDataIO<int>;
            inputCellTypesInt = dynamic_cast<VTKDataIO<int>* > (inputCellTypes);
            if (!inputCellTypes->read(inVTKFile, n, binary)) return false;
        }
        else if (kw == "CELL_DATA") {
            int n;
            ln >> n;

            std::getline(inVTKFile, line);
            if (line.empty()) continue;
            /// line defines the type and name such as SCALAR dataset
            std::istringstream lnData(line);
            std::string dataStructure, dataName, dataType;
            lnData >> dataStructure;

            sout << "Data structure: " << dataStructure << sendl;

            if (dataStructure == "SCALARS") {
                size_t sz = inputCellDataVector.size();

                inputCellDataVector.resize(sz+1);
                lnData >> dataName;
                lnData >> dataType;

                inputCellDataVector[sz] = newVTKDataIO(dataType);
                if (inputCellDataVector[sz] == NULL) return false;
                /// one more getline to read LOOKUP_TABLE line, not used here
                std::getline(inVTKFile, line);

                if (!inputCellDataVector[sz]->read(inVTKFile,n, binary)) return false;
                inputCellDataVector[sz]->name = dataName;
                sout << "Read cell data: " << inputCellDataVector[sz]->dataSize << sendl;
            }
            else if (dataStructure == "FIELD") {
                std::getline(inVTKFile,line);
                if (line.empty()) continue;
                /// line defines the type and name such as SCALAR dataset
                std::istringstream lnData(line);
                std::string dataStructure;
                lnData >> dataStructure;

                if (dataStructure == "Topology") {
                    int perCell, cells;
                    lnData >> perCell >> cells;
                    sout << "Reading topology for lines: "<< perCell << " " << cells << sendl;

                    size_t sz = inputCellDataVector.size();

                    inputCellDataVector.resize(sz+1);
                    inputCellDataVector[sz] = newVTKDataIO("int");

                    if (!inputCellDataVector[sz]->read(inVTKFile,perCell*cells, binary))
                        return false;

                    inputCellDataVector[sz]->name = "Topology";
                }
            }
            else  /// TODO
                std::cerr << "WARNING: reading vector data not implemented" << std::endl;
        }
        else if (!kw.empty())
            std::cerr << "WARNING: Unknown keyword " << kw << std::endl;

        sout << "LNG: " << inputCellDataVector.size() << sendl;

        if (inputPoints && inputPolygons) break; // already found the mesh description, skip the rest
        if (inputPoints && inputCells && inputCellTypes && inputCellDataVector.size() > 0) break; // already found the mesh description, skip the rest
    }

    if (binary)
    {
        // detect swapped data
        bool swapped = false;
        if (inputPolygons)
        {
            if ((unsigned)inputPolygonsInt->data[0] > (unsigned)inputPolygonsInt->swapT(inputPolygonsInt->data[0], 1))
                swapped = true;
        }
        else if (inputCells && inputCellTypes)
        {
            if ((unsigned)inputCellTypesInt->data[0] > (unsigned)inputCellTypesInt->swapT(inputCellTypesInt->data[0],1 ))
                swapped = true;
        }
        if (swapped)
        {
            sout << "Binary data is byte-swapped." << sendl;
            if (inputPoints) inputPoints->swap();
            if (inputPolygons) inputPolygons->swap();
            if (inputCells) inputCells->swap();
            if (inputCellTypes) inputCellTypes->swap();
        }
    }

    return true;
}

//XML VTK Loader
#define checkError(A) if (!A) { return false; }
#define checkErrorPtr(A) if (!A) { return NULL; }
#define checkErrorMsg(A, B) if (!A) { serr << B << sendl ; return false; }

bool MeshVTKLoader::XMLVTKReader::readFile(const char* filename)
{
    TiXmlDocument vtkDoc(filename);
    //quick check
    checkErrorMsg(vtkDoc.LoadFile(), "Unknown error while loading VTK Xml doc");

    TiXmlHandle hVTKDoc(&vtkDoc);
    TiXmlElement* pElem;
    TiXmlHandle hVTKDocRoot(0);

    //block VTKFile
    pElem = hVTKDoc.FirstChildElement().ToElement();
    checkErrorMsg(pElem, "VTKFile Node not found");

    hVTKDocRoot = TiXmlHandle(pElem);

    //Endianness
    const char* endiannessStrTemp = pElem->Attribute("byte_order");
    isLittleEndian = (std::string(endiannessStrTemp).compare("LittleEndian") == 0) ;

    //read VTK data format type
    const char* datasetFormatStrTemp = pElem->Attribute("type");
    checkErrorMsg(datasetFormatStrTemp, "Dataset format not defined")
    std::string datasetFormatStr = std::string(datasetFormatStrTemp);
    VTKDatasetFormat datasetFormat;

    if (datasetFormatStr.compare("UnstructuredGrid") == 0)
        datasetFormat = UNSTRUCTURED_GRID;
    else if (datasetFormatStr.compare("PolyData") == 0)
        datasetFormat = POLYDATA;
    else if (datasetFormatStr.compare("RectilinearGrid") == 0)
        datasetFormat = RECTILINEAR_GRID;
    else if (datasetFormatStr.compare("StructuredGrid") == 0)
        datasetFormat = STRUCTURED_GRID;
    else if (datasetFormatStr.compare("StructuredPoints") == 0)
        datasetFormat = STRUCTURED_POINTS;
    else if (datasetFormatStr.compare("ImageData") == 0)
        datasetFormat = IMAGE_DATA;
    else checkErrorMsg(false, "Dataset format " << datasetFormatStr << " not recognized");

    TiXmlHandle datasetFormatHandle = TiXmlHandle(hVTKDocRoot.FirstChild( datasetFormatStr.c_str() ).ToElement());

    bool stateLoading = false;
    switch (datasetFormat)
    {
    case UNSTRUCTURED_GRID :
        stateLoading = loadUnstructuredGrid(datasetFormatHandle);
        break;
    case POLYDATA :
        stateLoading = loadPolydata(datasetFormatHandle);
        break;
    case RECTILINEAR_GRID :
        stateLoading = loadRectilinearGrid(datasetFormatHandle);
        break;
    case STRUCTURED_GRID :
        stateLoading = loadStructuredGrid(datasetFormatHandle);
        break;
    case STRUCTURED_POINTS :
        stateLoading = loadStructuredPoints(datasetFormatHandle);
        break;
    case IMAGE_DATA :
        stateLoading = loadImageData(datasetFormatHandle);
        break;
    default:
        checkErrorMsg(false, "Dataset format not implemented");
        break;
    }
    checkErrorMsg(stateLoading, "Error while parsing XML");

    return true;
}
MeshVTKLoader::BaseVTKReader::BaseVTKDataIO* MeshVTKLoader::XMLVTKReader::loadDataArray(TiXmlElement* dataArrayElement)
{
    return loadDataArray(dataArrayElement,0);
}

MeshVTKLoader::BaseVTKReader::BaseVTKDataIO* MeshVTKLoader::XMLVTKReader::loadDataArray(TiXmlElement* dataArrayElement, int size)
{
    return loadDataArray(dataArrayElement, size, "");
}

MeshVTKLoader::BaseVTKReader::BaseVTKDataIO* MeshVTKLoader::XMLVTKReader::loadDataArray(TiXmlElement* dataArrayElement, int size, std::string type)
{
    //Type
    const char* typeStrTemp;
    if (type.empty())
    {
        typeStrTemp = dataArrayElement->Attribute("type");
        checkErrorPtr(typeStrTemp);
    }
    else
        typeStrTemp = type.c_str();

    //Format
    const char* formatStrTemp = dataArrayElement->Attribute("format");

    checkErrorPtr(formatStrTemp);

    int binary = 0;
    if (std::string(formatStrTemp).compare("ascii") == 0)
        binary = 0;
    else if (isLittleEndian)
        binary = 1;
    else
        binary = 2;

    //NumberOfComponents
    int numberOfComponents;
    if (dataArrayElement->QueryIntAttribute("NumberOfComponents", &numberOfComponents) != TIXML_SUCCESS)
        numberOfComponents = 1;

    //Values
    const char* listValuesStrTemp = dataArrayElement->GetText();

    bool state = false;

    if (!listValuesStrTemp) return NULL;
    if (std::string(listValuesStrTemp).size() < 1) return NULL;

    BaseVTKDataIO* d = BaseVTKReader::newVTKDataIO(std::string(typeStrTemp));

    if (!d) return NULL;

    if (size > 0)
        state = (d->read(std::string(listValuesStrTemp), numberOfComponents*size, binary));
    else
        state = (d->read(std::string(listValuesStrTemp), binary));
    checkErrorPtr(state);

    return d;
}

bool MeshVTKLoader::XMLVTKReader::loadUnstructuredGrid(TiXmlHandle datasetFormatHandle)
{
    TiXmlElement* pieceElem = datasetFormatHandle.FirstChild( "Piece" ).ToElement();

    checkError(pieceElem);
    //for each "Piece" Node
    for( ; pieceElem; pieceElem=pieceElem->NextSiblingElement())
    {
        pieceElem->QueryIntAttribute("NumberOfPoints", &numberOfPoints);
        pieceElem->QueryIntAttribute("NumberOfCells", &numberOfCells);

        //std::cout << "Number Of Points " << numberOfPoints << std::endl;
        //std::cout << "Number Of Cells " << numberOfCells << std::endl;

        TiXmlNode* dataArrayNode;
        TiXmlElement* dataArrayElement;
        TiXmlNode* node = pieceElem->FirstChild();

        for ( ; node ; node = node->NextSibling())
        {
            std::string currentNodeName = std::string(node->Value());

            //std::cout << currentNodeName << std::endl;

            if (currentNodeName.compare("Points") == 0)
            {
                /* Points */
                dataArrayNode = node->FirstChild("DataArray");
                checkError(dataArrayNode);
                dataArrayElement = dataArrayNode->ToElement();
                checkError(dataArrayElement);
                //Force the points coordinates to be stocked as double
                inputPoints = loadDataArray(dataArrayElement, numberOfPoints, "Float64");
                checkError(inputPoints);
            }

            if (currentNodeName.compare("Cells") == 0)
            {
                /* Cells */
                dataArrayNode = node->FirstChild("DataArray");
                for ( ; dataArrayNode; dataArrayNode = dataArrayNode->NextSibling( "DataArray"))
                {
                    dataArrayElement = dataArrayNode->ToElement();
                    checkError(dataArrayElement);
                    std::string currentDataArrayName = std::string(dataArrayElement->Attribute("Name"));
                    ///DA - connectivity
                    if (currentDataArrayName.compare("connectivity") == 0)
                    {
                        //number of elements in values is not known ; have to guess it
                        inputCells = loadDataArray(dataArrayElement);
                        checkError(inputCells);
                    }
                    ///DA - offsets
                    if (currentDataArrayName.compare("offsets") == 0)
                    {
                        inputCellOffsets = loadDataArray(dataArrayElement, numberOfCells-1);
                        checkError(inputCellOffsets);
                    }
                    ///DA - types
                    if (currentDataArrayName.compare("types") == 0)
                    {
                        inputCellTypes = loadDataArray(dataArrayElement, numberOfCells, "Int32");
                        checkError(inputCellTypes);
                    }
                }
            }

            if (currentNodeName.compare("PointData") == 0)
            {
                dataArrayNode = node->FirstChild("DataArray");
                for ( ; dataArrayNode; dataArrayNode = dataArrayNode->NextSibling( "DataArray"))
                {
                    dataArrayElement = dataArrayNode->ToElement();
                    checkError(dataArrayElement);

                    std::string currentDataArrayName = std::string(dataArrayElement->Attribute("Name"));

                    BaseVTKDataIO* pointdata = loadDataArray(dataArrayElement, numberOfPoints);
                    checkError(pointdata);
                    pointdata->name = currentDataArrayName;
                    inputPointDataVector.push_back(pointdata);
                }
            }
            if (currentNodeName.compare("CellData") == 0)
            {
                dataArrayNode = node->FirstChild("DataArray");
                for ( ; dataArrayNode; dataArrayNode = dataArrayNode->NextSibling( "DataArray"))
                {
                    dataArrayElement = dataArrayNode->ToElement();
                    checkError(dataArrayElement);
                    std::string currentDataArrayName = std::string(dataArrayElement->Attribute("Name"));

                    BaseVTKDataIO* celldata = loadDataArray(dataArrayElement, numberOfCells);
                    checkError(celldata);
                    celldata->name = currentDataArrayName;
                    inputCellDataVector.push_back(celldata);
                }
            }
        }
    }

    return true;
}

bool MeshVTKLoader::XMLVTKReader::loadPolydata(TiXmlHandle /* datasetFormatHandle */)
{
    serr << "Polydata dataset not implemented yet" << sendl;
    return false;
}

bool MeshVTKLoader::XMLVTKReader::loadRectilinearGrid(TiXmlHandle /* datasetFormatHandle */)
{
    serr << "RectilinearGrid dataset not implemented yet" << sendl;
    return false;
}

bool MeshVTKLoader::XMLVTKReader::loadStructuredGrid(TiXmlHandle /* datasetFormatHandle */)
{
    serr << "StructuredGrid dataset not implemented yet" << sendl;
    return false;
}

bool MeshVTKLoader::XMLVTKReader::loadStructuredPoints(TiXmlHandle /*datasetFormatHandle */)
{
    serr << "StructuredPoints dataset not implemented yet" << sendl;
    return false;
}

bool MeshVTKLoader::XMLVTKReader::loadImageData(TiXmlHandle /* datasetFormatHandle */)
{
    serr << "ImageData dataset not implemented yet" << sendl;
    return false;
}



} // namespace loader

} // namespace component

} // namespace sofa

