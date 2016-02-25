/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LOADER_MeshVTKLoader_H
#define SOFA_COMPONENT_LOADER_MeshVTKLoader_H
#include "config.h"

#include <sofa/core/loader/MeshLoader.h>
#include <sofa/core/objectmodel/BaseData.h>

#include <tinyxml.h>

namespace sofa
{

namespace component
{

namespace loader
{

#if defined(WIN32) || defined(_XBOX)
#define strcasecmp stricmp
#endif

// Format doc: http://www.vtk.org/VTK/img/file-formats.pdf
// http://www.cacr.caltech.edu/~slombey/asci/vtk/vtk_formats.simple.html
class SOFA_LOADER_API MeshVTKLoader : public sofa::core::loader::MeshLoader
{
protected:

    class BaseVTKReader : public sofa::core::objectmodel::BaseObject
    {
    public:
        class BaseVTKDataIO : public sofa::core::objectmodel::BaseObject
        {
        public:
            std::string name;
            int dataSize;
            int nestedDataSize;
            BaseVTKDataIO() : dataSize(0), nestedDataSize(1) {}
            virtual ~BaseVTKDataIO() {}
            virtual void resize(int n) = 0;
            virtual bool read(std::istream& f, int n, int binary) = 0;
            virtual bool read(const std::string& s, int n, int binary) = 0;
            virtual bool read(const std::string& s, int binary) = 0;
            virtual bool write(std::ofstream& f, int n, int groups, int binary) = 0;
            virtual const void* getData() = 0;
            virtual void swap() = 0;

            virtual core::objectmodel::BaseData* createSofaData() = 0;            
        };

        template<class T>
        class VTKDataIO : public BaseVTKDataIO
        {
        public:
            T* data;
            VTKDataIO() : data(NULL) {}
            ~VTKDataIO() { if (data) delete[] data; }
            virtual const void* getData()
            {
                return data;
            }

            virtual void resize(int n)
            {
                if (dataSize != n)
                {
                    if (data) delete[] data;
                    data = new T[n];
                }

                dataSize = n;
            }
            static T swapT(T t, int nestedDataSize)
            {
                /*
                union T_chars
                {
                    T t;
                    char b[sizeof(T)];
                } tmp,rev;
                tmp.t = t;
                for (unsigned int c=0;c<sizeof(T);++c)
                    rev.b[c] = tmp.b[sizeof(T)-1-c];
                return rev.t;
                 */
                T revT;
                char* revB = (char*) &revT;
                const char* tmpB = (char*) &t;

                if (nestedDataSize < 2)
                {
                    for (unsigned int c=0; c<sizeof(T); ++c)
                        revB[c] = tmpB[sizeof(T)-1-c];
                }
                else
                {
                    int singleSize = sizeof(T)/nestedDataSize;
                    for (int i=0; i<nestedDataSize; ++i)
                    {
                        for (unsigned int c=0; c<sizeof(T); ++c)
                            revB[c+i*singleSize] = tmpB[(sizeof(T)-1-c) + i*singleSize];
                    }

                }
                return revT;

            }
            void swap()
            {
                for (int i=0; i<dataSize; ++i)
                    data[i] = swapT(data[i], nestedDataSize);
            }
            virtual bool read(const std::string& s, int n, int binary)
            {
                std::istringstream iss(s);
                return read(iss, n, binary);
            }

            virtual bool read(const std::string& s, int binary)
            {
                int n=0;
                //compute size itself
                if (binary == 0)
                {

                    std::string::size_type begin = 0;
                    std::string::size_type end = s.find(' ', begin);
                    n=1;

                    while (end != std::string::npos)
                    {
                        n++;
                        begin = end + 1;
                        end = s.find(' ', begin);
                    }
                }
                else
                {
                    n = sizeof(s.c_str())/sizeof(T);
                }
                //std::cout << "Guessed size is " << n << std::endl;
                std::istringstream iss(s);

                return read(iss, n, binary);
            }

            virtual bool read(std::istream& in, int n, int binary)
            {
                resize(n);
                if (binary)
                {
                    in.read((char*)data, n *sizeof(T));
                    if (in.eof() || in.bad())
                    {
                        resize(0);
                        return false;
                    }
                    if (binary == 2) // swap bytes
                    {
                        for (int i=0; i<n; ++i)
                        {
                            /*
                            union T_chars
                            {
                            	T t;
                            	char b[sizeof(T)];
                            } tmp,rev;

                            tmp.t = data[i];
                            for (unsigned int c=0;c<sizeof(T);++c)
                            	rev.b[c] = tmp.b[sizeof(T)-1-c];
                            data[i] = rev.t;
                            */
                            data[i] = swapT(data[i], nestedDataSize);
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

            virtual core::objectmodel::BaseData* createSofaData()
            {
                Data<helper::vector<T> >* sdata = new Data<helper::vector<T> >(name.c_str(), true, false);
                sdata->setName(name);
                helper::vector<T>& sofaData = *sdata->beginEdit();

                for (int i=0 ; i<dataSize ; i++)
                    sofaData.push_back(data[i]);
                sdata->endEdit();

                return sdata;
            }
        };

        BaseVTKDataIO* newVTKDataIO(const std::string& typestr)
        {
            if  (!strcasecmp(typestr.c_str(), "char") || !strcasecmp(typestr.c_str(), "Int8"))
                return new VTKDataIO<char>;
            else if (!strcasecmp(typestr.c_str(), "unsigned_char") || !strcasecmp(typestr.c_str(), "UInt8"))
                return new VTKDataIO<unsigned char>;
            else if (!strcasecmp(typestr.c_str(), "short") || !strcasecmp(typestr.c_str(), "Int16"))
                return new VTKDataIO<short>;
            else if (!strcasecmp(typestr.c_str(), "unsigned_short") || !strcasecmp(typestr.c_str(), "UInt16"))
                return new VTKDataIO<unsigned short>;
            else if (!strcasecmp(typestr.c_str(), "int") || !strcasecmp(typestr.c_str(), "Int32"))
                return new VTKDataIO<int>;
            else if (!strcasecmp(typestr.c_str(), "unsigned_int") || !strcasecmp(typestr.c_str(), "UInt32"))
                return new VTKDataIO<unsigned int>;
            //else if (!strcasecmp(typestr.c_str(), "long") || !strcasecmp(typestr.c_str(), "Int64"))
            //	return new VTKDataIO<long long>;
            //else if (!strcasecmp(typestr.c_str(), "unsigned_long") || !strcasecmp(typestr.c_str(), "UInt64"))
            // 	return new VTKDataIO<unsigned long long>;
            else if (!strcasecmp(typestr.c_str(), "float") || !strcasecmp(typestr.c_str(), "Float32"))
                return new VTKDataIO<float>;
            else if (!strcasecmp(typestr.c_str(), "double") || !strcasecmp(typestr.c_str(), "Float64"))
                return new VTKDataIO<double>;
            else return NULL;
        }

        BaseVTKDataIO* newVTKDataIO(const std::string& typestr, int num)
        {
            BaseVTKDataIO* result = NULL;

            if (num == 1)
                result = newVTKDataIO(typestr);
            else
            {
                if (!strcasecmp(typestr.c_str(), "char") || !strcasecmp(typestr.c_str(), "Int8") ||
                    !strcasecmp(typestr.c_str(), "short") || !strcasecmp(typestr.c_str(), "Int32") ||
                    !strcasecmp(typestr.c_str(), "int") || !strcasecmp(typestr.c_str(), "Int32"))
                {
                    switch (num)
                    {
                    case 2:
                        result = new VTKDataIO<defaulttype::Vec<2, int> >;
                        break;
                    case 3:
                        result = new VTKDataIO<defaulttype::Vec<3, int> >;
                        break;
                    default:
                        return NULL;
                    }
                }

                if (!strcasecmp(typestr.c_str(), "unsigned char") || !strcasecmp(typestr.c_str(), "UInt8") ||
                    !strcasecmp(typestr.c_str(), "unsigned short") || !strcasecmp(typestr.c_str(), "UInt32") ||
                    !strcasecmp(typestr.c_str(), "unsigned int") || !strcasecmp(typestr.c_str(), "UInt32"))
                {
                    switch (num)
                    {
                    case 2:
                        result = new VTKDataIO<defaulttype::Vec<2, unsigned int> >;
                        break;
                    case 3:
                        result = new VTKDataIO<defaulttype::Vec<3, unsigned int> >;
                        break;
                    default:
                        return NULL;
                    }
                }
                if (!strcasecmp(typestr.c_str(), "float") || !strcasecmp(typestr.c_str(), "Float32"))
                {
                    switch (num)
                    {
                    case 2:
                        result = new VTKDataIO<defaulttype::Vec<2, float> >;
                        break;
                    case 3:
                        result = new VTKDataIO<defaulttype::Vec<3, float> >;
                        break;
                    default:
                        return NULL;
                    }
                }
                if (!strcasecmp(typestr.c_str(), "double") || !strcasecmp(typestr.c_str(), "Float64"))
                {
                    switch (num)
                    {
                    case 2:
                        result = new VTKDataIO<defaulttype::Vec<2, double> >;
                        break;
                    case 3:
                        result = new VTKDataIO<defaulttype::Vec<3, double> >;
                        break;
                    default:
                        return NULL;
                    }
                }
            }
            result->nestedDataSize = num;
            return result;
        }

    public:
        BaseVTKDataIO* inputPoints;
        BaseVTKDataIO* inputPolygons;
        BaseVTKDataIO* inputCells;        
        BaseVTKDataIO* inputCellOffsets;
        BaseVTKDataIO* inputCellTypes;
        helper::vector<BaseVTKDataIO*> inputPointDataVector;
        helper::vector<BaseVTKDataIO*> inputCellDataVector;
        bool isLittleEndian;

        int numberOfPoints, numberOfCells, numberOfLines;


        BaseVTKReader():inputPoints (NULL), inputPolygons(NULL), inputCells(NULL), inputCellOffsets(NULL), inputCellTypes(NULL),
            numberOfPoints(0),numberOfCells(0)
        {}

        bool readVTK(const char* filename)
        {
            bool state = false;

            state = readFile(filename);

            return state;
        }

        virtual bool readFile(const char* filename) = 0;
    };

    class LegacyVTKReader : public BaseVTKReader
    {
    public:
        bool readFile(const char* filename);
    };

    class XMLVTKReader : public BaseVTKReader
    {
    public:
        bool readFile(const char* filename);
    protected:
        bool loadUnstructuredGrid(TiXmlHandle datasetFormatHandle);
        bool loadPolydata(TiXmlHandle datasetFormatHandle);
        bool loadRectilinearGrid(TiXmlHandle datasetFormatHandle);
        bool loadStructuredGrid(TiXmlHandle datasetFormatHandle);
        bool loadStructuredPoints(TiXmlHandle datasetFormatHandle);
        bool loadImageData(TiXmlHandle datasetFormatHandle);
        BaseVTKDataIO* loadDataArray(TiXmlElement* dataArrayElement, int size, std::string type);
        BaseVTKDataIO* loadDataArray(TiXmlElement* dataArrayElement, int size);
        BaseVTKDataIO* loadDataArray(TiXmlElement* dataArrayElement);


    };

public:
    SOFA_CLASS(MeshVTKLoader,sofa::core::loader::MeshLoader);

    enum VTKFileType { NONE, LEGACY, XML };
    enum VTKDatasetFormat { IMAGE_DATA, STRUCTURED_POINTS, STRUCTURED_GRID, RECTILINEAR_GRID, POLYDATA, UNSTRUCTURED_GRID };
protected:
    MeshVTKLoader();
public:
    virtual bool load();

    template <class T>
    static bool canCreate ( T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg )
    {
        return BaseLoader::canCreate (obj, context, arg);
    }

protected:
    VTKFileType detectFileType(const char* filename);

    BaseVTKReader* reader;
    bool setInputsMesh();
    bool setInputsData();

public:
    core::objectmodel::BaseData* pointsData;
    core::objectmodel::BaseData* edgesData;
    core::objectmodel::BaseData* trianglesData;
    core::objectmodel::BaseData* quadsData;
    core::objectmodel::BaseData* tetrasData;
    core::objectmodel::BaseData* hexasData;
    //Add Data here

};




} // namespace loader

} // namespace component

} // namespace sofa

#endif
