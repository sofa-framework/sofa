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
#include <sofa/component/loader/MeshVTKLoader.h>

#include <iostream>
//#include <fstream> // we can't use iostream because the windows implementation gets confused by the mix of text and binary
#include <stdio.h>
#include <sstream>


#ifdef WIN32
#define strcasecmp stricmp
#endif

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

MeshVTKLoader::MeshVTKLoader() : MeshLoader()
{
}

static bool fgetline(FILE* f, std::string& line)
{
    line.clear();
    char buf[256];
    while (fgets(buf, sizeof(buf), f))
    {
        int n = strlen(buf);
        if (n < (int)sizeof(buf)-1 || buf[n-1]=='\n')
        {
            // end of line
            while (n > 0 && (buf[n-1]=='\n' || buf[n-1]=='\r'))
                --n;
            buf[n]='\0';
            line+=buf;
            return true;
        }
        else // partial line
            line+=buf;
    }
    if (!line.empty() && feof(f)) return true; // we succeeding in reading the last line of the file
    return false; // fgets returned an error
}

// --- VTK classes ---


class BaseVTKDataIO
{
public:
    int dataSize;
    BaseVTKDataIO() : dataSize(0) {}
    virtual ~BaseVTKDataIO() {}
    virtual void resize(int n) = 0;
    virtual bool read(FILE* f, int n, int binary) = 0;
    virtual bool write(FILE* f, int n, int groups, int binary) = 0;
    virtual void addPoints(helper::vector<sofa::defaulttype::Vec<3,SReal> >& my_positions) = 0;
    virtual void swap() = 0;
};

template<class T>
class VTKDataIO : public BaseVTKDataIO
{
public:
    T* data;
    VTKDataIO() : data(NULL) {}
    ~VTKDataIO() { if (data) delete[] data; }
    virtual void resize(int n)
    {
        if (dataSize != n)
        {
            if (data) delete[] data;
            data = new T[n];
        }
        dataSize = n;
    }
    static T swapT(T t)
    {
        union T_chars
        {
            T t;
            char b[sizeof(T)];
        } tmp,rev;
        tmp.t = t;
        for (unsigned int c=0; c<sizeof(T); ++c)
            rev.b[c] = tmp.b[sizeof(T)-1-c];
        return rev.t;
    }
    void swap()
    {
        for (int i=0; i<dataSize; ++i)
            data[i] = swapT(data[i]);
    }
    virtual bool read(FILE* in, int n, int binary)
    {
        resize(n);
        if (binary)
        {
            if ((int)fread(data, sizeof(T), n, in) < n)
            {
                resize(0);
                return false;
            }
            if (binary == 2) // swap bytes
            {
                swap();
            }
        }
        else
        {
            int i = 0;
            std::string line;
            while(i < dataSize)
            {
                if (!fgetline(in, line)) break;
                std::istringstream ln(line);
                while (i < n && ln >> data[i])
                    ++i;
            }
            if (i < n)
            {
                resize(0);
                return false;
            }
        }
        return true;
    }
    virtual bool write(FILE* out, int n, int groups, int binary)
    {
        if (n > dataSize && !data) return false;
        if (binary)
        {
            fwrite(data, sizeof(T), n, out);
        }
        else
        {
            std::ostringstream os;
            if (groups <= 0 || groups > n) groups = n;
            for (int i = 0; i < n; ++i)
            {
                if ((i % groups) > 0)
                    os << ' ';
                os << data[i];
                if (((i % groups) == groups-1) || i == n-1)
                {
                    os << '\n';
                    std::string s = os.str();
                    fwrite(s.c_str(), 1, s.length(), out);
                }
            }
        }
        if (ferror(out))
            return false;
        return true;
    }
    virtual void addPoints(helper::vector<sofa::defaulttype::Vec<3,SReal> >& my_positions)
    {
        if (!data) return;

        for (int i=0; i < dataSize; i+=3)
            my_positions.push_back (Vector3 ((double)data[i+0], (double)data[i+1], (double)data[i+2]));
    }
};

BaseVTKDataIO* newVTKDataIO(const std::string& typestr)
{
    if      (!strcasecmp(typestr.c_str(), "char"))           return new VTKDataIO<char>;
    else if (!strcasecmp(typestr.c_str(), "unsigned_char"))  return new VTKDataIO<unsigned char>;
    else if (!strcasecmp(typestr.c_str(), "short"))          return new VTKDataIO<short>;
    else if (!strcasecmp(typestr.c_str(), "unsigned_short")) return new VTKDataIO<unsigned short>;
    else if (!strcasecmp(typestr.c_str(), "int"))            return new VTKDataIO<int>;
    else if (!strcasecmp(typestr.c_str(), "unsigned_int"))   return new VTKDataIO<unsigned int>;
    else if (!strcasecmp(typestr.c_str(), "long"))           return new VTKDataIO<long long>;
    else if (!strcasecmp(typestr.c_str(), "unsigned_long"))  return new VTKDataIO<unsigned long long>;
    else if (!strcasecmp(typestr.c_str(), "float"))          return new VTKDataIO<float>;
    else if (!strcasecmp(typestr.c_str(), "double"))         return new VTKDataIO<double>;
    else return NULL;
}

// --- Loading VTK functions ---


bool MeshVTKLoader::load()
{

    sout << "Loading VTK file: " << m_filename << sendl;

    bool fileRead = false;

    // -- Loading file
    const char* filename = m_filename.getFullPath().c_str();

    // -- Reading file
    fileRead = this->readVTK (filename);

    return fileRead;
}





bool MeshVTKLoader::readVTK (const char* filename)
{

    // Format doc: http://www.vtk.org/VTK/img/file-formats.pdf
    // http://www.cacr.caltech.edu/~slombey/asci/vtk/vtk_formats.simple.html
    //std::ifstream inVTKFile(filename, std::ifstream::in & std::ifstream::binary);
    //std::ifstream inVTKFile(filename, std::ifstream::in, std::ifstream::binary);

    FILE* inVTKFile = fopen(filename, "rb");

    if (inVTKFile == NULL)
    {
        serr << "Error: Cannot read file '" << filename << "'." << sendl;
        return false;
    }

    std::string line;

    // Part 1
    fgetline(inVTKFile, line);
    if (std::string(line,0,23) != "# vtk DataFile Version ")
    {
        serr << "Error: Unrecognized header in file '" << filename << "'." << sendl;
        fclose(inVTKFile);
        return false;
    }
    std::string version(line,23);

    // Part 2
    std::string header;
    fgetline(inVTKFile, header);

    // Part 3
    fgetline(inVTKFile, line);

    int binary;
    if (line == "BINARY") binary = 1;
    else if (line == "ASCII") binary = 0;
    else
    {
        serr << "Error: Unrecognized format in file '" << filename << "'." << sendl;
        fclose(inVTKFile);
        return false;
    }

    if (binary && strlen(filename)>9 && !strcmp(filename+strlen(filename)-9,".vtk_swap"))
        binary = 2; // bytes will be swapped


    // Part 4
    do
        fgetline(inVTKFile, line);
    while (line == "");
    if (line != "DATASET POLYDATA" && line != "DATASET UNSTRUCTURED_GRID")
    {
        serr << "Error: Unsupported data type in file '" << filename << "'." << sendl;
        fclose(inVTKFile);
        return false;
    }

    sout << (binary == 0 ? "Text" : (binary == 1) ? "Binary" : "Swapped Binary") << " VTK File (version " << version << "): " << header << sendl;
    BaseVTKDataIO* inputPoints = NULL;
    VTKDataIO<int>* inputPolygons = NULL;
    VTKDataIO<int>* inputCells = NULL;
    VTKDataIO<int>* inputCellTypes = NULL;
    int nbp = 0, nbf = 0;
    while(!feof(inVTKFile))
    {
        fgetline(inVTKFile, line);
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
            if (inputPoints == NULL) return false;
            if (!inputPoints->read(inVTKFile, 3*n, binary)) {return false;}
            nbp = n;
        }
        else if (kw == "POLYGONS")
        {
            int n, ni;
            ln >> n >> ni;
            sout << "Found " << n << " polygons ( " << (ni - 3*n) << " triangles )" << sendl;
            inputPolygons = new VTKDataIO<int>;
            if (!inputPolygons->read(inVTKFile, ni, binary)) return false;
            nbf = ni - 3*n;
        }
        else if (kw == "CELLS")
        {
            int n, ni;
            ln >> n >> ni;
            sout << "Found " << n << " cells" << sendl;
            inputCells = new VTKDataIO<int>;
            if (!inputCells->read(inVTKFile, ni, binary)) return false;
            nbf = n;
        }
        else if (kw == "CELL_TYPES")
        {
            int n;
            ln >> n;
            inputCellTypes = new VTKDataIO<int>;
            if (!inputCellTypes->read(inVTKFile, n, binary)) return false;
        }
        else if (!kw.empty())
            serr << "WARNING: Unknown keyword " << kw << sendl;
        if (inputPoints && inputPolygons) break; // already found the mesh description, skip the rest
        if (inputPoints && inputCells && inputCellTypes) break; // already found the mesh description, skip the rest
    }

    if (binary)
    {
        // detect swapped data
        bool swapped = false;
        if (inputPolygons)
        {
            if ((unsigned)inputPolygons->data[0] > (unsigned)inputPolygons->swapT(inputPolygons->data[0]))
                swapped = true;
        }
        else if (inputCells && inputCellTypes)
        {
            if ((unsigned)inputCellTypes->data[0] > (unsigned)inputCellTypes->swapT(inputCellTypes->data[0]))
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
    helper::vector<sofa::defaulttype::Vector3>& my_positions = *(positions.beginEdit());
    inputPoints->addPoints(my_positions);
    positions.endEdit();


    helper::vector<helper::fixed_array <unsigned int,2> >& my_edges = *(edges.beginEdit());
    helper::vector<helper::fixed_array <unsigned int,3> >& my_triangles = *(triangles.beginEdit());
    helper::vector<helper::fixed_array <unsigned int,4> >& my_quads = *(quads.beginEdit());
    helper::vector<helper::fixed_array <unsigned int,4> >& my_tetrahedra = *(tetrahedra.beginEdit());
    helper::vector<helper::fixed_array <unsigned int,8> >& my_hexahedra = *(hexahedra.beginEdit());

    if (inputPolygons)
    {
        const int* inFP = inputPolygons->data;
        int poly = 0;
        for (int i=0; i < inputPolygons->dataSize;)
        {
            int nv = inFP[i]; ++i;
            bool valid = true;
            if (inputPoints)
            {
                for (int j=0; j<nv; ++j)
                    if ((unsigned)inFP[i+j] >= (unsigned)(inputPoints->dataSize/3))
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
    else if (inputCells && inputCellTypes)
    {
        const int* inFP = inputCells->data;
        int i = 0;
        for (int c = 0; c < nbf; ++c)
        {
            int t = inputCellTypes->data[c];
            int nv = inFP[i]; ++i;
            switch (t)
            {
            case 1: // VERTEX
                break;
            case 2: // POLY_VERTEX
                break;
            case 3: // LINE
                addEdge(&my_edges, inFP[i+0], inFP[i+1]);
                break;
            case 4: // POLY_LINE
                for (int v = 0; v < nv-1; ++v)
                    addEdge(&my_edges, inFP[i+v+0], inFP[i+v+1]);
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
            default:
                serr << "ERROR: unsupported cell type " << t << sendl;
            }
            i += nv;
        }
    }
    if (inputPoints) delete inputPoints;
    if (inputPolygons) delete inputPolygons;
    if (inputCells) delete inputCells;
    if (inputCellTypes) delete inputCellTypes;

    edges.endEdit();
    triangles.endEdit();
    quads.endEdit();
    tetrahedra.endEdit();
    hexahedra.endEdit();

    return true;
}



} // namespace loader

} // namespace component

} // namespace sofa

