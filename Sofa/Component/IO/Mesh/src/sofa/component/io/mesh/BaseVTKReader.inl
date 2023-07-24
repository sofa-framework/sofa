/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/component/io/mesh/BaseVTKReader.h>

#include <istream>
#include <fstream>

namespace sofa::component::io::mesh::basevtkreader
{

using std::istringstream ;
using sofa::type::Vec ;

template<class T>
const void* BaseVTKReader::VTKDataIO<T>::getData()
{
    return data;
}

template<class T>
void BaseVTKReader::VTKDataIO<T>::resize(int n)
{
    if (dataSize != n)
    {
        if (data) delete[] data;
        data = new T[n];
    }

    dataSize = n;
}

template<class T>
T BaseVTKReader::VTKDataIO<T>::swapT(T t, int nestedDataSize)
{
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
        const int singleSize = sizeof(T)/nestedDataSize;
        for (int i=0; i<nestedDataSize; ++i)
        {
            for (unsigned int c=0; c<sizeof(T); ++c)
                revB[c+i*singleSize] = tmpB[(sizeof(T)-1-c) + i*singleSize];
        }

    }
    return revT;

}

template<class T>
void BaseVTKReader::VTKDataIO<T>::swap()
{
    for (int i=0; i<dataSize; ++i)
        data[i] = swapT(data[i], nestedDataSize);
}

template<class T>
bool BaseVTKReader::VTKDataIO<T>::read(const string& s, int n, int binary)
{
    istringstream iss(s);
    return read(iss, n, binary);
}

template<class T>
bool BaseVTKReader::VTKDataIO<T>::read(const string& s, int binary)
{
    int n=0;
    //compute size itself
    if (binary == 0)
    {

        string::size_type begin = 0;
        string::size_type end = s.find(' ', begin);
        n=1;

        while (end != string::npos)
        {
            n++;
            begin = end + 1;
            end = s.find(' ', begin);
        }
    }
    else
    {
        n = s.size()/sizeof(T);
    }
    istringstream iss(s);

    return read(iss, n, binary);
}

template<class T>
bool BaseVTKReader::VTKDataIO<T>::read(istream& in, int n, int binary)
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
                data[i] = swapT(data[i], nestedDataSize);
            }
        }
    }
    else
    {
        int i = 0;
        string line;
        while(i < dataSize && !in.eof() && !in.bad())
        {
            std::getline(in, line);
            istringstream ln(line);
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

template<class T>
bool BaseVTKReader::VTKDataIO<T>::write(ofstream& out, int n, int groups, int binary)
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

template<class T>
BaseData* BaseVTKReader::VTKDataIO<T>::createSofaData()
{
    Data<type::vector<T> >* sdata = new Data<type::vector<T> >(name.c_str(), true, false);
    sdata->setName(name);
    type::vector<T>& sofaData = *sdata->beginEdit();

    for (int i=0 ; i<dataSize ; i++)
        sofaData.push_back(data[i]);
    sdata->endEdit();

    return sdata;
}


} // namespace sofa::component::io::mesh::basevtkloader
