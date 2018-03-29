/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/io/MeshTopologyLoader.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/defaulttype/Vec.h>
#include <string.h>

#if defined(WIN32) || defined(_XBOX)
#define strcasecmp stricmp
#endif


MSG_REGISTER_CLASS(sofa::helper::io::MeshTopologyLoader, "MeshTopologyLoader")

namespace sofa
{

namespace helper
{

namespace io
{

using namespace sofa::defaulttype;

bool MeshTopologyLoader::loadObj(const char *filename)
{
    mesh = helper::io::Mesh::Create(filename);
    if (mesh==NULL)
        return false;

    setNbPoints((int)mesh->getVertices().size());
    for (size_t i=0; i<mesh->getVertices().size(); i++)
    {
        addPoint((SReal)mesh->getVertices()[i][0],
                (SReal)mesh->getVertices()[i][1],
                (SReal)mesh->getVertices()[i][2]);
    }

    const vector< vector < vector <int> > > & facets = mesh->getFacets();
    std::set< std::pair<int,int> > edges;
    for (size_t i=0; i<facets.size(); i++)
    {
        const vector<int>& facet = facets[i][0];
        if (facet.size()==2)
        {
            // Line
            if (facet[0]<facet[1])
                addLine(facet[0],facet[1]);
            else
                addLine(facet[1],facet[0]);
        }
        else if (facet.size()==4)
        {
            // Quad
            addQuad(facet[0],facet[1],facet[2],facet[3]);
        }
        else
        {
            // Triangularize
            for (size_t j=2; j<facet.size(); j++)
                addTriangle(facet[0],facet[j-1],facet[j]);
        }
#if 0
        // Add edges
        if (facet.size()>2)
        {
            for (size_t j=0; j<facet.size(); j++)
            {
                int i1 = facet[j];
                int i2 = facet[(j+1)%facet.size()];
                if (edges.count(std::make_pair(i1,i2))!=0)
                {
                }
                else if (edges.count(std::make_pair(i2,i1))==0)
                {
                    if (i1>i2)
                        addLine(i1,i2);
                    else
                        addLine(i2,i1);
                    edges.insert(std::make_pair(i1,i2));
                }
            }
        }
#endif
    }

    /// delete mesh;
    return true;
}

bool MeshTopologyLoader::loadStl(const char *filename)
{
    mesh = helper::io::Mesh::Create(filename);
    if (mesh==NULL)
        return false;

    setNbPoints((int)mesh->getVertices().size());
    const vector< vector < vector <int> > > & facets = mesh->getFacets();

    for (size_t i=0; i<mesh->getVertices().size(); i++)
    {
        addPoint((SReal)mesh->getVertices()[i][0],
                (SReal)mesh->getVertices()[i][1],
                (SReal)mesh->getVertices()[i][2]);
    }
    for (size_t i=0; i<facets.size(); i++)
    {
        const vector<int>& facet = facets[i][0];
        for (size_t j=2; j<facet.size(); j++)
            addTriangle(facet[0],facet[j-1],facet[j]);
    }
    return true;

}

bool MeshTopologyLoader::loadGmsh(std::ifstream &file, const int gmshFormat)
{
    int npoints = 0;
    int nlines = 0;
    int ntris = 0;
    int nquads = 0;
    int ntetrahedra = 0;
    int ncubes = 0;

    std::string cmd;

    file >> npoints; //nb points
    setNbPoints(npoints);

    std::vector<int> pmap;
    for (int i=0; i<npoints; ++i)
    {
        int index = i;
        double x,y,z;
        file >> index >> x >> y >> z;
        addPoint(x, y, z);
        if ((int)pmap.size() <= index) pmap.resize(index+1);
        pmap[index] = i;
    }

    file >> cmd;
    if (cmd != "$ENDNOD" && cmd != "$EndNodes")
    {
        msg_error() << "'$ENDNOD' or '$EndNodes' expected, found '" << cmd << "'" ;
        return false;
    }

    file >> cmd;
    if (cmd != "$ELM" && cmd != "$Elements")
    {
        msg_error() << "'$ELM' or '$Elements' expected, found '" << cmd << "'" ;
        return false;
    }



    int nelems = 0;
    file >> nelems;
    for (int i=0; i<nelems; ++i)
    {
        int index=-1, etype=-1, nnodes=-1, ntags=-1, tag=-1;
        if (gmshFormat==1)
        {
            // version 1.0 format is
            // elm-number elm-type reg-phys reg-elem number-of-nodes <node-number-list ...>
            int rphys=-1, relem=-1;
            file >> index >> etype >> rphys >> relem >> nnodes;
        }
        else if (gmshFormat == 2)
        {
            // version 2.0 format is
            // elm-number elm-type number-of-tags < tag > ... node-number-list
            file >> index >> etype >> ntags;

            for (int t=0; t<ntags; t++)
            {
                file >> tag;
                // read the tag but don't use it
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
            case 15: // Point
                nnodes = 1;
                break;
            default:
                msg_error() << "Elements of type 1, 2, 3, 4, 5, or 6 expected. Element of type " << etype << " found." ;
                nnodes = 0;
            }
        }

        helper::vector<int> nodes;
        nodes.resize(nnodes);
        for (int n=0; n<nnodes; ++n)
        {
            int t = 0;
            file >> t;
            nodes[n] = (((unsigned int)t)<pmap.size())?pmap[t]:0;
        }
        switch (etype)
        {
        case 1: // Line
            addLine(nodes[0], nodes[1]);
            ++nlines;
            break;
        case 2: // Triangle
            addTriangle(nodes[0], nodes[1], nodes[2]);
            ++ntris;
            break;
        case 3: // Quad
            addQuad(nodes[0], nodes[1], nodes[2], nodes[3]);
            ++nquads;
            break;
        case 4: // Tetra
            addTetra(nodes[0], nodes[1], nodes[2], nodes[3]);
            ++ntetrahedra;
            break;
        case 5: // Hexa
            addCube(nodes[0], nodes[1], nodes[2], nodes[3],nodes[4], nodes[5], nodes[6], nodes[7]);
            ++ncubes;
            break;
        default:
            //if the type is not handled, skip rest of the line
            std::string tmp;
            std::getline(file, tmp);
        }
    }

    file >> cmd;
    if (cmd != "$ENDELM" && cmd!="$EndElements")
    {
        msg_error() << "'$ENDELM' or '$EndElements' expected, found '" << cmd << "'" ;
        return false;
    }

    return true;
}

bool MeshTopologyLoader::loadXsp(std::ifstream &file, bool vector_spring)
{
    std::string cmd;
    file >> cmd;

    // then find out number of masses and springs
    if (cmd == "numm")
    {
        int totalNumMasses;
        file >> totalNumMasses;
        setNbPoints(totalNumMasses);
    }

    if (cmd=="nums")
    {
        int totalNumSprings;
        file >> totalNumSprings;
        setNbLines(totalNumSprings);
    }


    while (!file.eof())
    {
        file  >> cmd;
        if (cmd=="mass")
        {
            int index;
            char location;
            double px,py,pz,vx,vy,vz,mass=0.0,elastic=0.0;
            file >> index >> location >> px >> py >> pz >> vx >> vy >> vz >> mass >> elastic;
            addPoint(px,py,pz);
        }
        else if (cmd=="lspg")	// linear springs connector
        {
            int	index;
            int m1,m2;
            double ks=0.0,kd=0.0,initpos=-1;

            if (vector_spring)
            {
                double restx=0.0,resty=0.0,restz=0.0;
                file >> index >> m1 >> m2 >> ks >> kd >> initpos >> restx >> resty >> restz;
            }
            else
                file >> index >> m1 >> m2 >> ks >> kd >> initpos;
            --m1;
            --m2;

            addLine(m1,m2);
        }
        else if (cmd == "grav")
        {
            double gx,gy,gz;
            file >> gx >> gy >> gz;
        }
        else if (cmd == "visc")
        {
            double viscosity;
            file >> viscosity;
        }
        else if (cmd == "step")
        {
        }
        else if (cmd == "frce")
        {
        }
        else if (cmd[0] == '#')	// it's a comment
        {
        }
        else		// it's an unknown keyword
        {
            msg_error() << "Unknown MassSpring keyword:" ;
            return false;
        }
    }

    return true;
}

bool MeshTopologyLoader::loadMesh(std::ifstream &file)
{
    std::string cmd;
    int npoints = 0;
    int nlines = 0;
    int ntris = 0;
    int nquads = 0;
    int ntetrahedra = 0;
    int ncubes = 0;

    while (!file.eof())
    {
        file >> cmd;
        if (cmd=="line")
        {
            int p1,p2;
            file >> p1 >> p2;
            addLine(p1, p2);
            ++nlines;
        }
        else if (cmd=="triangle")
        {
            int p1,p2,p3;
            file >> p1 >> p2 >> p3;
            addTriangle(p1, p2, p3);
            ++ntris;
        }
        else if (cmd=="quad")
        {
            int p1,p2,p3,p4;
            file >> p1 >> p2 >> p3 >> p4;
            addQuad(p1, p2, p3, p4);
            ++nquads;
        }
        else if (cmd=="tetra")
        {
            int p1,p2,p3,p4;
            file >> p1 >> p2 >> p3 >> p4;
            addTetra(p1, p2, p3, p4);
            ++ntetrahedra;
        }
        else if (cmd=="cube")
        {
            int p1,p2,p3,p4,p5,p6,p7,p8;
            file >> p1 >> p2 >> p3 >> p4 >> p5 >> p6 >> p7 >> p8;
            addCube(p1, p2, p3, p4, p5, p6, p7, p8);
            ++ncubes;
        }
        else if (cmd=="point")
        {
            double px,py,pz;
            file >> px >> py >> pz;
            addPoint(px, py, pz);
            ++npoints;
        }
        else if (cmd=="v")
        {
            double px,py,pz;
            file >> px >> py >> pz;
            addPoint(px, py, pz);
            ++npoints;
        }
        else if (cmd=="f")
        {
            int p1,p2,p3,p4=0;
            file >> p1 >> p2 >> p3 >> p4;
            if (p4)
            {
                addQuad(p1-1, p2-1, p3-1, p4-1);
                ++nquads;
            }
            else
            {
                addTriangle(p1-1, p2-1, p3-1);
                ++ntris;
            }
        }
        else if (cmd=="mass")
        {
            int index;
            char location;
            double px,py,pz,vx,vy,vz,mass=0.0,elastic=0.0;
            file >> index >> location >> px >> py >> pz >> vx >> vy >> vz >> mass >> elastic;
            addPoint(px, py, pz);
            ++npoints;
        }
        else if (cmd=="lspg")
        {
            int	index;
            int m1,m2;
            double ks=0.0,kd=0.0,initpos=-1;
            file >> index >> m1 >> m2 >> ks >> kd >> initpos;
            --m1;
            --m2;
            addLine(m1,m2);
            ++nlines;
        }
        else if (cmd[0] == '#')	// it's a comment
        {
        }
        else		// it's an unknown keyword
        {
            msg_error() << "Unknown Mesh keyword:" << cmd;
            return false;
        }
    }

    return true;
}

bool MeshTopologyLoader::loadMeshFile(const char *filename)
{
    bool fileLoaded = false;

    std::ifstream file(filename);
    if (!file.good()) return false;

    int gmshFormat = 0;

    std::string cmd;
    file >> cmd;

    if (cmd == "$MeshFormat") // Reading gmsh 2.0 file
    {
        gmshFormat = 2;
        std::string line;
        std::getline(file, line); // we don't care about this line
        if (line=="") std::getline(file, line);
        file >> cmd;
        if (cmd != "$EndMeshFormat") // it should end with $EndMeshFormat
        {
            file.close();
            return false;
        }
        else
        {
            file >> cmd;
        }
    }
    else
    {
        gmshFormat = 1;
    }


    if (cmd == "$NOD" || cmd == "$Nodes") // Gmsh format
    {
        fileLoaded = loadGmsh(file, gmshFormat);
    }
    else if (cmd == "Xsp")
    {
        float version = 0.0f;
        file >> version;

        if (version == 3.0)
            fileLoaded = loadXsp(file, false);
        else if (version == 4.0)
            fileLoaded = loadXsp(file, true);
    }
    else
    {
        //Reset the stream to the beginning.
        file.seekg(0, std::ios::beg);
        fileLoaded = loadMesh(file);
    }
    file.close();
    return 	fileLoaded;
}


bool MeshTopologyLoader::loadCGAL(const char *filename)
{
    bool fileLoaded = false;
    std::string cmd;
    std::string line;

    std::ifstream file(filename);
    if (!file.good())
        return false;

    msg_info() << "Loading CGAL mesh file " << filename << " ... " ;

    std::getline(file, line);

    if (line == "MeshVersionFormatted 1") // Reading CGAL 3.7 or 3.8 file
    {
        int npoints = 0;
        int ntri = 0;
        int ntetra = 0;

        std::getline(file, line); // we don't care about this line
        file >> cmd;
        if (cmd == "Vertices")
        {
            file >> npoints;
            setNbPoints(npoints);
            for (int i=0; i<npoints; ++i)
            {
                double x,y,z,tmp;
                file >> x >> y >> z >> tmp;
                addPoint(x, y, z);
            }
        }

        file >> cmd;
        if (cmd == "Triangles")
        {
            file >> ntri;
            setNbTriangles(ntri);
            for (int i=0; i<ntri; ++i)
            {
                int j,k,l,tmp;
                file >> j >> k >> l >> tmp;
                addTriangle(j-1, k-1, l-1);
            }
        }

        file >> cmd;
        if (cmd == "Tetrahedra")
        {
            file >> ntetra;
            setNbTetrahedra(ntetra);
            for (int i=0; i<ntetra; ++i)
            {
                int j,k,l,m, tmp;
                file >> j >> k >> l >> m >> tmp;
                addTetra(j-1, k-1, l-1, m-1);
            }
        }

        msg_info() << "Loading CGAL topology complete:" << msgendl
                   << "   " << npoints << " points" << msgendl
                   << "   " << ntri   << " triangles" << msgendl
                   << "   " << ntetra << " tetrahedra) ";
        fileLoaded = true;
    }
    else
    {
        msg_error() << "Incorrect file format - not recognized as CGAL 'MeshVersionFormatted 1' - (aborting)" ;
    }

    file.close();

    return fileLoaded;
}


/////////////////////////////////////////////////////////////////////////
class BaseVTKDataIO
{
public:
    int dataSize;
    BaseVTKDataIO() : dataSize(0) {}
    virtual ~BaseVTKDataIO() {}
    virtual void resize(int n) = 0;
    virtual bool read(std::ifstream& f, int n, int binary) = 0;
    virtual bool write(std::ofstream& f, int n, int groups, int binary) = 0;
    virtual void addPoints(MeshTopologyLoader* dest) = 0;
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
    virtual bool read(std::ifstream& in, int n, int binary)
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
            if (binary == 2) // swap bytes
            {
                for (int i=0; i<n; ++i)
                {
                    union T_chars
                    {
                        T t;
                        char b[sizeof(T)];
                    } tmp,rev;
                    tmp.t = data[i];
                    for (size_t c=0; c<sizeof(T); ++c)
                        rev.b[c] = tmp.b[sizeof(T)-1-c];
                    data[i] = rev.t;
                }
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
    virtual bool write(std::ofstream& out, int n, int groups, int binary)
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
    virtual void addPoints(MeshTopologyLoader* dest)
    {
        if (!data) return;
        for (int i=0; i < dataSize; i+=3)
            dest->addPoint((double)data[i+0], (double)data[i+1], (double)data[i+2]);
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

bool MeshTopologyLoader::loadVtk(const char *filename)
{
    // Format doc: http://www.vtk.org/VTK/img/file-formats.pdf
    // http://www.cacr.caltech.edu/~slombey/asci/vtk/vtk_formats.simple.html
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

    int binary;
    if (line == "BINARY") binary = 1;
    else if (line == "ASCII") binary = 0;
    else return false;

    if (binary && strlen(filename)>9 && !strcmp(filename+strlen(filename)-9,".vtk_swap"))
        binary = 2; // bytes will be swapped


    // Part 4
    do
        std::getline(inVTKFile, line);
    while (line == "");
    if (line != "DATASET POLYDATA" && line != "DATASET UNSTRUCTURED_GRID")
    {
        return false;
    }

    msg_info() << (binary == 0 ? "Text" : (binary == 1) ? "Binary" : "Swapped Binary") << " VTK File (version " << version << "): " << header ;

    BaseVTKDataIO* inputPoints = NULL;
    VTKDataIO<int>* inputPolygons = NULL;
    VTKDataIO<int>* inputCells = NULL;
    VTKDataIO<int>* inputCellTypes = NULL;
    int /*nbp = 0,*/ nbf = 0;
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
            msg_info() << "Found " << n << " " << typestr << " points" ;
            inputPoints = newVTKDataIO(typestr);
            if (inputPoints == NULL) return false;
            if (!inputPoints->read(inVTKFile, 3*n, binary)) return false;
            //nbp = n;
        }
        else if (kw == "POLYGONS")
        {
            int n, ni;
            ln >> n >> ni;
            msg_info() << "Found " << n << " polygons ( " << (ni - 3*n) << " triangles )" ;
            inputPolygons = new VTKDataIO<int>;
            if (!inputPolygons->read(inVTKFile, ni, binary)) return false;
            nbf = ni - 3*n;
        }
        else if (kw == "CELLS")
        {
            int n, ni;
            ln >> n >> ni;
            msg_info() << "Found " << n << " cells" ;
            inputCells = new VTKDataIO<int>;
            if (!inputCells->read(inVTKFile, ni, binary)) return false;
            nbf = n;
        }
        else if (kw == "CELL_TYPES")
        {
            int n;
            ln >> n;
            inputCellTypes = new VTKDataIO<int>;
            if (!inputCellTypes->read(inVTKFile, n, binary))
            {
                if (inputCellTypes) delete inputCellTypes;
                return false;
            }
        }
        else if (!kw.empty())
            msg_warning() << "Unknown keyword " << kw ;
        if (inputPoints && inputPolygons) break; // already found the mesh description, skip the rest
        if (inputPoints && inputCells && inputCellTypes) break; // already found the mesh description, skip the rest
    }
    inputPoints->addPoints(this);
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
                        msg_error() << "Invalid point " << inFP[i+j] << " in polygon " << poly ;
                        valid = false;
                    }
            }
            if (valid)
            {
                if (nv == 4)
                {
                    addQuad(inFP[i+0],inFP[i+1],inFP[i+2],inFP[i+3]);
                }
                else if (nv >= 3)
                {
                    int f[3];
                    f[0] = inFP[i+0];
                    f[1] = inFP[i+1];
                    for (int j=2; j<nv; j++)
                    {
                        f[2] = inFP[i+j];
                        addTriangle(f[0],f[1],f[2]);
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
                addLine(inFP[i+0], inFP[i+1]);
                break;
            case 4: // POLY_LINE
                for (int v = 0; v < nv-1; ++v)
                    addLine(inFP[i+v+0], inFP[i+v+1]);
                break;
            case 5: // TRIANGLE
                addTriangle(inFP[i+0], inFP[i+1], inFP[i+2]);
                break;
            case 6: // TRIANGLE_STRIP
                for (int j=0; j<nv-2; j++)
                    if (j&1)
                        addTriangle(inFP[i+j+0],inFP[i+j+1],inFP[i+j+2]);
                    else
                        addTriangle(inFP[i+j+0],inFP[i+j+2],inFP[i+j+1]);
                break;
            case 7: // POLYGON
                for (int j=2; j<nv; j++)
                    addTriangle(inFP[i+0],inFP[i+j-1],inFP[i+j]);
                break;
            case 8: // PIXEL
                addQuad(inFP[i+0], inFP[i+1], inFP[i+3], inFP[i+2]);
                break;
            case 9: // QUAD
                addQuad(inFP[i+0], inFP[i+1], inFP[i+2], inFP[i+3]);
                break;
            case 10: // TETRA
                addTetra(inFP[i+0], inFP[i+1], inFP[i+2], inFP[i+3]);
                break;
            case 11: // VOXEL
                addCube(inFP[i+0], inFP[i+1], inFP[i+3], inFP[i+2], inFP[i+4], inFP[i+5], inFP[i+7], inFP[i+6]);
                break;
            case 12: // HEXAHEDRON
                addCube(inFP[i+0], inFP[i+1], inFP[i+2], inFP[i+3], inFP[i+4], inFP[i+5], inFP[i+6], inFP[i+7]);
                break;
            default:
                msg_error() << "Unsupported cell type " << t ;
            }
            i += nv;
        }
    }
    if (inputPoints) delete inputPoints;
    if (inputPolygons) delete inputPolygons;
    if (inputCells) delete inputCells;
    if (inputCellTypes) delete inputCellTypes;
    return true;
}

bool MeshTopologyLoader::load(const char *filename)
{
    std::string fname(filename);
    if (!sofa::helper::system::DataRepository.findFile(fname))
    {
        msg_error() << "Cannot find file: " << filename ;
        return false;
    }

    bool fileLoaded;

    if ((strlen(filename)>4 && !strcmp(filename+strlen(filename)-4,".obj"))
        || (strlen(filename)>6 && !strcmp(filename+strlen(filename)-6,".trian")))
        fileLoaded = loadObj(fname.c_str());
    else if (strlen(filename)>4 && !strcmp(filename+strlen(filename)-4,".vtk"))
        fileLoaded = loadVtk(fname.c_str());
    else if (strlen(filename)>4 && !strcmp(filename+strlen(filename)-4,".stl"))
        fileLoaded = loadStl(fname.c_str());
    else if (strlen(filename)>9 && !strcmp(filename+strlen(filename)-9,".vtk_swap"))
        fileLoaded = loadVtk(fname.c_str());
    else if (strlen(filename)>5 && !strcmp(filename+strlen(filename)-5,".mesh"))
        fileLoaded = loadCGAL(fname.c_str());
    else
        fileLoaded = loadMeshFile(fname.c_str());

    if(!fileLoaded)
        msg_error() << "Unable to load mesh file '" << fname << "'" ;

    return fileLoaded;
}

} // namespace io

} // namespace helper

} // namespace sofa

