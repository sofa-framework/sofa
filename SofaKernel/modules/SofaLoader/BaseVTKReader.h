/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_LOADER_BASEVTKREADER_H
#define SOFA_COMPONENT_LOADER_BASEVTKREADER_H
#include "config.h"

#include <string>
#include <istream>
#include <fstream>

#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{

namespace component
{

namespace loader
{

/// Use a per-file namespace. The role of this per-file namespace contain the names to make
/// them private. Outside of this namespace the fully qualified name have to be used.
/// At the end of this namespace only a subset of the names are imported into the parent namespace.
/// So that you can access to BaseVTKReader with
/// sofa::component::loader::BaseVTKReader or sofa::component::loader::basevtkreader::BaseVTKReader
namespace basevtkreader
{
using sofa::core::objectmodel::BaseObject ;
using sofa::core::objectmodel::BaseData ;

using std::ofstream ;
using std::istream ;
using std::string ;


enum class VTKDatasetFormat { IMAGE_DATA, STRUCTURED_POINTS,
                              STRUCTURED_GRID, RECTILINEAR_GRID,
                              POLYDATA, UNSTRUCTURED_GRID
                            };

class BaseVTKReader : public BaseObject
{
public:
    class BaseVTKDataIO : public BaseObject
    {
    public:
        string name;
        int dataSize;
        int nestedDataSize;
        BaseVTKDataIO() : dataSize(0), nestedDataSize(1) {}
        virtual ~BaseVTKDataIO() {}
        virtual void resize(int n) = 0;
        virtual bool read(istream& f, int n, int binary) = 0;
        virtual bool read(const string& s, int n, int binary) = 0;
        virtual bool read(const string& s, int binary) = 0;
        virtual bool write(ofstream& f, int n, int groups, int binary) = 0;
        virtual const void* getData() = 0;
        virtual void swap() = 0;

        virtual BaseData* createSofaData() = 0;
    };

    template<class T>
    class VTKDataIO : public BaseVTKDataIO
    {
    public:
        T* data;
        VTKDataIO() : data(NULL) {}
        ~VTKDataIO()
        {
            if (data)
            {
                delete[] data;
            }
        }
        virtual const void* getData() ;
        virtual void resize(int n) ;
        static T swapT(T t, int nestedDataSize) ;
        void swap() ;
        virtual bool read(const string& s, int n, int binary) ;
        virtual bool read(const string& s, int binary) ;
        virtual bool read(istream& in, int n, int binary) ;
        virtual bool write(ofstream& out, int n, int groups, int binary) ;
        virtual BaseData* createSofaData() ;
    };

    BaseVTKDataIO* newVTKDataIO(const string& typestr) ;
    BaseVTKDataIO* newVTKDataIO(const string& typestr, int num) ;

    BaseVTKDataIO* inputPoints;
    BaseVTKDataIO* inputNormals;
    BaseVTKDataIO* inputPolygons;
    BaseVTKDataIO* inputCells;
    BaseVTKDataIO* inputCellOffsets;
    BaseVTKDataIO* inputCellTypes;
    helper::vector<BaseVTKDataIO*> inputPointDataVector;
    helper::vector<BaseVTKDataIO*> inputCellDataVector;
    bool isLittleEndian;

    int numberOfPoints, numberOfCells, numberOfLines;

    BaseVTKReader() ;

    bool readVTK(const char* filename) ;

    virtual bool readFile(const char* filename) = 0;
};

} // basevtkreader

/// Importing the names defined in the per-file namespace into the classical
/// sofa namespace structre so that the classes are accessible with
/// sofa::component::loader::BaseVTKReader instead of
/// sofa::component::loader::basevtkreader::BaseVTKReader which is a bit longer to read and write.
using basevtkreader::VTKDatasetFormat ;
using basevtkreader::BaseVTKReader ;

} // namespace loader

} // namespace component

} // namespace sofa

#endif
