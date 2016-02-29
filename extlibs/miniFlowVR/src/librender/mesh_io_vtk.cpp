/******* COPYRIGHT ************************************************
*                                                                 *
*                         FlowVR Render                           *
*                   Parallel Rendering Modules                    *
*                                                                 *
*-----------------------------------------------------------------*
* COPYRIGHT (C) 2005 by                                           *
* Laboratoire Informatique et Distribution (UMR5132) and          *
* INRIA Project MOVI. ALL RIGHTS RESERVED.                        *
*                                                                 *
* This source is covered by the GNU GPL, please refer to the      *
* COPYING file for further information.                           *
*                                                                 *
*-----------------------------------------------------------------*
*                                                                 *
*  Original Contributors:                                         *
*    Jeremie Allard,                                              *
*    Clement Menier.                                              *
*                                                                 * 
*******************************************************************
*                                                                 *
* File: ./src/librender/mesh_io_vtk.cpp                           *
*                                                                 *
* Contacts:                                                       *
*                                                                 *
******************************************************************/
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <set>
#include <fstream>
#include <sstream>

#include <flowvr/render/mesh.h>

#if defined(WIN32) || defined(_XBOX)
#define strcasecmp stricmp
#endif

// Format doc: http://www.vtk.org/VTK/img/file-formats.pdf
// http://www.cacr.caltech.edu/~slombey/asci/vtk/vtk_formats.simple.html

namespace flowvr
{

namespace render
{

class BaseVTKDataIO
{
public:
    int dataSize;
    BaseVTKDataIO() : dataSize(0) {}
    virtual ~BaseVTKDataIO() {}
    virtual void resize(int n) = 0;
    virtual bool read(std::ifstream& f, int n, bool binary) = 0;
    virtual bool write(std::ofstream& f, int n, int groups, bool binary) = 0;
    virtual void copyTo(std::vector<Vec3f>& v) = 0;
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
    virtual bool read(std::ifstream& in, int n, bool binary)
    {
        resize(n);
        if (binary)
        {
            in.read((char*)data, n * sizeof(T));
            if (in.eof() || in.bad())
            {
                resize(0);
                return false;
            }
        }
        else
        {
            int i = 0;
            std::string line;
            while(i < dataSize && !in.eof() && !in.bad())
            {
                std::getline(in, line);
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
    virtual bool write(std::ofstream& out, int n, int groups, bool binary)
    {
        if (n > dataSize && !data) return false;
        if (binary)
        {
            out.write((char*)data, n * sizeof(T));
        }
        else
        {
            if (groups <= 0 || groups > n) groups = n;
            for (int i = 0; i < n; ++i)
            {
                if ((i % groups) > 0)
                    out << ' ';
                out << data[i];
                if ((i % groups) == groups-1)
                    out << '\n';
            }
        }
        if (out.bad())
            return false;
        return true;
    }
    virtual void copyTo(std::vector<Vec3f>& v)
    {
        if (!data) return;
        for (unsigned int i=0; i < v.size(); ++i)
            for (int c=0;c<3;++c)
                v[i][c] = (float)data[3*i+c];
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

bool Mesh::loadVtk(const char* filename)
{
    std::ifstream inVTKFile(filename, std::ifstream::in & std::ifstream::binary);
    if( !inVTKFile.is_open() )
    {
        return false;
    }
    std::string line;

    // Part 1
    std::getline(inVTKFile, line);
    if (std::string(line,0,23) != "# vtk DataFile Version ") return false;
    std::string version(line,23);

    // Part 2
    std::string header;
    std::getline(inVTKFile, header);

    // Part 3
    std::getline(inVTKFile, line);

    bool binary;
    if (line == "BINARY") binary = true;
    else if (line == "ASCII") binary = false;
    else return false;

    // Part 4
    do
	std::getline(inVTKFile, line);
    while (line == "");
    if (line != "DATASET POLYDATA")
    {
        return false;
    }

    std::cout << (binary ? "Binary" : "Text") << " VTK File (version " << version << "): " << header << std::endl;
    BaseVTKDataIO* inputPoints = NULL;
    VTKDataIO<int>* inputPolygons = NULL;
    int flags = MESH_POINTS_POSITION|MESH_FACES;
    int nbp = 0, nbf = 0;
    while(!inVTKFile.eof())
    {
        std::getline(inVTKFile, line);
        std::istringstream ln(line);
        std::string kw;
        ln >> kw;
        if (kw == "POINTS")
        {
            int n;
            std::string typestr;
            ln >> n >> typestr;
            std::cout << "Found " << n << " " << typestr << " points" << std::endl;
            inputPoints = newVTKDataIO(typestr);
            if (inputPoints == NULL) return false;
            if (!inputPoints->read(inVTKFile, 3*n, binary))
            {
                if (inputPoints) delete inputPoints;
                if (inputPolygons) delete inputPolygons;
                return false;
            }
            nbp = n;
        }
        else if (kw == "POLYGONS")
        {
            int n, ni;
            ln >> n >> ni;
            std::cout << "Found " << n << " polygons ( " << (ni - 3*n) << " triangles )" << std::endl;
            inputPolygons = new VTKDataIO<int>;
            if (!inputPolygons->read(inVTKFile, ni, binary))
            {
                if (inputPoints) delete inputPoints;
                if (inputPolygons) delete inputPolygons;
                return false;
            }
//            nbf = ni - 3*n;
        }
        else
            std::cerr << "WARNING: Unknown keyword " << kw << std::endl;
        if (inputPoints && inputPolygons) break; // already found the mesh description, skip the rest
    }
    const int* inFP = inputPolygons->data;
    nbf = 0;
    for (int i=0; i < inputPolygons->dataSize;)
    {
        int nv = inFP[i]; ++i;
        if (nv >= 3)
            nbf += nv - 2;
        i += nv;
    }
    //std::cout << "init " << nbp << " points and " << nbf << " faces ( flags = " << flags << " )" << std::endl;
    init(nbp, nbf, flags);
    inputPoints->copyTo(points_p);
    nbf = 0;
    for (int i=0; i < inputPolygons->dataSize;)
    {
        int nv = inFP[i]; ++i;
        if (nv >= 3)
        {
            Vec3i f;
            f[0] = inFP[i+0];
            f[1] = inFP[i+1];
            for (int j=2;j<nv;j++)
            {
                f[2] = inFP[i+j];
                //std::cout << "face " << nbf << " = " << f << std::endl;
                FP(nbf++)=f;
                f[1] = f[2];
            }
        }
        i += nv;
    }
    if (inputPoints) delete inputPoints;
    if (inputPolygons) delete inputPolygons;
    std::cout<<"Loaded file "<<filename<<std::endl;
    
    // Computing normals
    calcBBox();
    calcNormals();
    return true;
}

bool Mesh::saveVtk(const char* filename, bool binary) const
{
    std::ofstream fp(filename, std::ofstream::out & std::ofstream::binary);
    if( !fp.is_open() )
    {
        return false;
    }
    std::cout<<"Writing " << (binary ? "Binary" : "Text") << " VTK file "<<filename<<std::endl;
    fp << "# vtk DataFile Version 2.0\n";
    fp << "Generated by FlowVR Render\n";
    fp << (binary ? "BINARY\n" : "ASCII\n");
    fp << "DATASET POLYDATA\n";
    int np = nbp(), nf = nbf();
    {
        fp << "POINTS " << np << " float\n";
        VTKDataIO<float> out;
        out.resize(np*3);
        for (int p=0;p<np;++p)
        {
            Vec3f v = getPP(p);
            for (int c=0;c<3;++c)
                out.data[3*p+c] = v[c];
        }
        out.write(fp, 3*np, 3, binary);
    }
    {
        fp << "POLYGONS " << nf << " " << 4*nf << "\n";
        VTKDataIO<int> out;
        out.resize(4*nf);
        for (int f=0;f<nf;++f)
        {
            Vec3i v = getFP(f);
            out.data[4*f  ] = 3;
            for (int c=0;c<3;++c)
                out.data[4*f+1+c] = v[c];
        }
        out.write(fp, 4*nf, 4, binary);
    }
    return true;
}

} // namespace render

} // namespace flowvr
