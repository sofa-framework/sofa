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

// this file contains CImg extensions for SOFA

#include "CImg.h"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <sstream>
#include <string>

namespace cimg_library
{

template<typename T> inline CImg<unsigned char> convertToUC(const CImg<T> &Image)	{	return CImg<unsigned char>((+Image).normalize(0,255)); 	}
inline CImg<unsigned char> convertToUC(const CImg<bool> &Image)	{	return CImg<unsigned char>(Image)*255; }
inline CImg<unsigned char> convertToUC(const CImg<char> &Image) {	return convertToUC(CImg<int>(Image));		}


template<typename T,typename F>
bool save_metaimage(const CImgList<T>& img,const char *const headerFilename, const F *const scale=0, const F *const translation=0, const F *const affine=0, const F *const offsetT=0, const F *const scaleT=0, const bool *const isPerspective=0)
{
    if(!img.size()) return false;

    std::ofstream fileStream (headerFilename, std::ofstream::out);

    if (!fileStream.is_open())	{	std::cout << "Can not open " << headerFilename << std::endl;	return false; }

    fileStream << "ObjectType = Image" << std::endl;

    unsigned int dim[]={img(0).width(),img(0).height(),img(0).depth(), img.size()};
    unsigned int nbdims=(dim[3]==1)?3:4; //  for 2-d, we still need z scale dimension

    fileStream << "NDims = " << nbdims << std::endl;

    fileStream << "ElementNumberOfChannels = " << img(0).spectrum() << std::endl;

    fileStream << "DimSize = "; for(unsigned int i=0;i<nbdims;i++) fileStream << dim[i] << " "; fileStream << std::endl;

    fileStream << "ElementType = ";
    if(cimg::type<T>::id()==3U) fileStream << "MET_CHAR" << std::endl;
    else if(cimg::type<T>::id()==11U) fileStream << "MET_DOUBLE" << std::endl;
    else if(cimg::type<T>::id()==12U) fileStream << "MET_FLOAT" << std::endl;
    else if(cimg::type<T>::id()==8U) fileStream << "MET_INT" << std::endl;
    else if(cimg::type<T>::id()==10U) fileStream << "MET_LONG" << std::endl;
    else if(cimg::type<T>::id()==6U) fileStream << "MET_SHORT" << std::endl;
    else if(cimg::type<T>::id()==2U) fileStream << "MET_UCHAR" << std::endl;
    else if(cimg::type<T>::id()==7U) fileStream << "MET_UINT" << std::endl;
    else if(cimg::type<T>::id()==9U) fileStream << "MET_ULONG" << std::endl;
    else if(cimg::type<T>::id()==5U) fileStream << "MET_USHORT" << std::endl;
    else if(cimg::type<T>::id()==1U) fileStream << "MET_BOOL" << std::endl;
    else fileStream << "MET_UNKNOWN" << std::endl;

    if(scale) { fileStream << "ElementSpacing = "; for(unsigned int i=0;i<3;i++) if(i<nbdims) fileStream << scale[i] << " "; if(nbdims==4) fileStream << *scaleT; fileStream << std::endl; }

    if(translation) { fileStream << "Position = "; for(unsigned int i=0;i<3;i++) if(i<nbdims) fileStream << translation[i] << " "; if(nbdims==4) fileStream << *offsetT; fileStream << std::endl; }

    if(affine) { fileStream << "Orientation = "; for(unsigned int i=0;i<9;i++) fileStream << affine[i] << " "; fileStream << std::endl; }

    if(isPerspective) { fileStream << "isPerpective = " << *isPerspective << std::endl; }

    std::string imageFilename(headerFilename); imageFilename.replace(imageFilename.find_last_of('.')+1,imageFilename.size(),"raw");
    fileStream << "ElementDataFile = " << imageFilename.c_str() << std::endl;
    fileStream.close();

    std::FILE *const nfile = std::fopen(imageFilename.c_str(),"wb");
    if(!nfile) return false;

    cimglist_for(img,l)     cimg::fwrite(img(l)._data,img(l).size(),nfile);
    cimg::fclose(nfile);
    return true;
}


template<typename T,typename F>
CImgList<T> load_metaimage(const char *const  headerFilename, F *const scale=0, F *const translation=0, F *const affine=0, F *const offsetT=0, F *const scaleT=0, bool *const isPerspective=0)
{
    CImgList<T> ret;

    std::ifstream fileStream(headerFilename, std::ifstream::in);
    if (!fileStream.is_open())	{	std::cout << "Can not open " << headerFilename << std::endl;	return ret; }

    std::string str,str2,imageFilename;
    unsigned int inputType=cimg::type<T>::id(),nbchannels=1,nbdims=4,dim[] = {1,1,1,1}; // 3 spatial dimas + time

    while(!fileStream.eof())
    {
        fileStream >> str;

        if(!str.compare("ObjectType") ||
                !str.compare("objectType"))
        {
            fileStream >> str2; // '='
            fileStream >> str2;
            if(str2.compare("Image")) { std::cout << "MetaImageReader: not an image ObjectType "<<std::endl; return ret;}
        }
        else if(!str.compare("ElementDataFile") ||
                !str.compare("elementDataFile"))
        {
            fileStream >> str2; // '='
            fileStream >> imageFilename;
        }
        else if(!str.compare("NDims") ||
                !str.compare("nDims"))
        {
            fileStream >> str2;  // '='
            fileStream >> nbdims;
            if(nbdims>4) { std::cout << "MetaImageReader: dimensions > 4 not supported  "<<std::endl; return ret;}
        }
        else if(!str.compare("ElementNumberOfChannels") ||
                !str.compare("elementNumberOfChannels"))
        {
            fileStream >> str2;  // '='
            fileStream >> nbchannels;
        }
        else if(!str.compare("DimSize") || !str.compare("Dimensions") || !str.compare("Dim") ||
                !str.compare("dimSize") || !str.compare("dimensions") || !str.compare("dim"))
        {
            fileStream >> str2;  // '='
            for(unsigned int i=0;i<nbdims;i++) fileStream >> dim[i];
        }
        else if(!str.compare("ElementSpacing") || !str.compare("Spacing") || !str.compare("Scale3d") || !str.compare("VoxelSize") ||
                !str.compare("elementSpacing") || !str.compare("spacing") || !str.compare("scale3d") || !str.compare("voxelSize"))
        {
            fileStream >> str2; // '='
            double val[4];
            for(unsigned int i=0;i<nbdims;i++) fileStream >> val[i];
            if(scale) for(unsigned int i=0;i<3;i++) if(i<nbdims) scale[i] = (F)val[i];
            if(scaleT) if(nbdims>3) *scaleT = (F)val[3];
        }
        else if(!str.compare("Position") || !str.compare("Offset") || !str.compare("Translation") || !str.compare("Origin") ||
                !str.compare("position") || !str.compare("offset") || !str.compare("translation") || !str.compare("origin"))
        {
            fileStream >> str2; // '='
            double val[4];
            for(unsigned int i=0;i<nbdims;i++) fileStream >> val[i];
            if(translation) for(unsigned int i=0;i<3;i++) if(i<nbdims) translation[i] = (F)val[i];
            if(offsetT) if(nbdims>3) *offsetT = (F)val[3];
        }
        else if(!str.compare("Orientation") || !str.compare("TransformMatrix") || !str.compare("Rotation") ||
                !str.compare("orientation") || !str.compare("transformMatrix") || !str.compare("rotation"))
        {
            fileStream >> str2; // '='
            double val[9];
            for(unsigned int i=0;i<9;i++) fileStream >> val[i];
            if(affine) { for(unsigned int i=0;i<9;i++) affine[i] = (F)val[i]; }
            // to do: handle "CenterOfRotation" Tag
        }
        else if(!str.compare("isPerpective")) { fileStream >> str2; bool val; fileStream >> val; if(isPerspective) *isPerspective=val; }
        else if(!str.compare("ElementType") || !str.compare("VoxelType") ||
                !str.compare("elementType") || !str.compare("voxelType"))  // not used (should be known in advance for template)
        {
            fileStream >> str2; // '='
            fileStream >> str2;

            if(!str2.compare("MET_CHAR")) inputType=3U;
            else if(!str2.compare("MET_DOUBLE")) inputType=11U;
            else if(!str2.compare("MET_FLOAT")) inputType=12U;
            else if(!str2.compare("MET_INT")) inputType=8U;
            else if(!str2.compare("MET_LONG")) inputType=10U;
            else if(!str2.compare("MET_SHORT")) inputType=6U;
            else if(!str2.compare("MET_UCHAR")) inputType=2U;
            else if(!str2.compare("MET_UINT")) inputType=7U;
            else if(!str2.compare("MET_ULONG")) inputType=9U;
            else if(!str2.compare("MET_USHORT")) inputType=5U;
            else if(!str2.compare("MET_BOOL")) inputType=1U;

            if(inputType!=cimg::type<T>::id())  std::cout<<"MetaImageReader: Image type ( "<< str2 <<" ) is converted to Sofa Image type ( "<< cimg::type<T>::string() <<" )"<<std::endl;
        }
    }
    fileStream.close();

    if(!imageFilename.size()) // no specified file name -> replace .mhd by .raw
    {
        imageFilename = std::string(headerFilename);
        imageFilename .replace(imageFilename.find_last_of('.')+1,imageFilename.size(),"raw");
    }
    else // add path to the specified file name
        if(imageFilename.find_last_of('/')==std::string::npos && imageFilename.find_last_of('\\')==std::string::npos) // test if file path is relative
        {
            std::string tmp(headerFilename);
            std::size_t pos=tmp.find_last_of('/');
            if(pos==std::string::npos) pos=tmp.find_last_of('\\');
            if(pos!=std::string::npos) {tmp.erase(pos+1); imageFilename.insert(0,tmp);}
        }

    ret.assign(dim[3],dim[0],dim[1],dim[2],nbchannels);
    unsigned int nb = dim[0]*dim[1]*dim[2]*nbchannels;

    std::FILE *const nfile = std::fopen(imageFilename.c_str(),"rb");
    if(!nfile)
    {
        std::cout<<"Can not open "<<imageFilename.c_str()<<std::endl;
        return ret;
    }

    if(inputType==cimg::type<T>::id())
    {
        cimglist_for(ret,l) cimg::fread(ret(l)._data,nb,nfile);
    }
    else
    {
        if(inputType==3U)
        {
            char *const buffer = new char[dim[3]*nb];
            cimg::fread(buffer,dim[3]*nb,nfile);
            //if (endian) cimg::invert_endianness(buffer,dim[3]*nb);
            cimglist_for(ret,l) cimg_foroff(ret(l),off) ret(l)._data[off] = (T)(buffer[off+l*nb]);
            delete[] buffer;
        }
        else if(inputType==11U)
        {
            double *const buffer = new double[dim[3]*nb];
            cimg::fread(buffer,dim[3]*nb,nfile);
            //if (endian) cimg::invert_endianness(buffer,dim[3]*nb);
            cimglist_for(ret,l) cimg_foroff(ret(l),off) ret(l)._data[off] = (T)(buffer[off+l*nb]);
            delete[] buffer;
        }
        else if(inputType==12U)
        {
            float *const buffer = new float[dim[3]*nb];
            cimg::fread(buffer,dim[3]*nb,nfile);
            //if (endian) cimg::invert_endianness(buffer,dim[3]*nb);
            cimglist_for(ret,l) cimg_foroff(ret(l),off) ret(l)._data[off] = (T)(buffer[off+l*nb]);
            delete[] buffer;
        }
        else if(inputType==8U)
        {
            int *const buffer = new int[dim[3]*nb];
            cimg::fread(buffer,dim[3]*nb,nfile);
            //if (endian) cimg::invert_endianness(buffer,dim[3]*nb);
            cimglist_for(ret,l) cimg_foroff(ret(l),off) ret(l)._data[off] = (T)(buffer[off+l*nb]);
            delete[] buffer;
        }
        else if(inputType==10U)
        {
            long *const buffer = new long[dim[3]*nb];
            cimg::fread(buffer,dim[3]*nb,nfile);
            //if (endian) cimg::invert_endianness(buffer,dim[3]*nb);
            cimglist_for(ret,l) cimg_foroff(ret(l),off) ret(l)._data[off] = (T)(buffer[off+l*nb]);
            delete[] buffer;
        }
        else if(inputType==6U)
        {
            short *const buffer = new short[dim[3]*nb];
            cimg::fread(buffer,dim[3]*nb,nfile);
            //if (endian) cimg::invert_endianness(buffer,dim[3]*nb);
            cimglist_for(ret,l) cimg_foroff(ret(l),off) ret(l)._data[off] = (T)(buffer[off+l*nb]);
            delete[] buffer;
        }
        else if(inputType==2U)
        {
            unsigned char *const buffer = new unsigned char[dim[3]*nb];
            cimg::fread(buffer,dim[3]*nb,nfile);
            //if (endian) cimg::invert_endianness(buffer,dim[3]*nb);
            cimglist_for(ret,l) cimg_foroff(ret(l),off) ret(l)._data[off] = (T)(buffer[off+l*nb]);
            delete[] buffer;
        }
        else if(inputType==7U)
        {
            unsigned int *const buffer = new unsigned int[dim[3]*nb];
            cimg::fread(buffer,dim[3]*nb,nfile);
            //if (endian) cimg::invert_endianness(buffer,dim[3]*nb);
            cimglist_for(ret,l) cimg_foroff(ret(l),off) ret(l)._data[off] = (T)(buffer[off+l*nb]);
            delete[] buffer;
        }
        else if(inputType==9U)
        {
            unsigned long *const buffer = new unsigned long[dim[3]*nb];
            cimg::fread(buffer,dim[3]*nb,nfile);
            //if (endian) cimg::invert_endianness(buffer,dim[3]*nb);
            cimglist_for(ret,l) cimg_foroff(ret(l),off) ret(l)._data[off] = (T)(buffer[off+l*nb]);
            delete[] buffer;
        }
        else if(inputType==5U)
        {
            unsigned short *const buffer = new unsigned short[dim[3]*nb];
            cimg::fread(buffer,dim[3]*nb,nfile);
            //if (endian) cimg::invert_endianness(buffer,dim[3]*nb);
            cimglist_for(ret,l) cimg_foroff(ret(l),off) ret(l)._data[off] = (T)(buffer[off+l*nb]);
            delete[] buffer;
        }
        else if(inputType==1U)
        {
            bool *const buffer = new bool[dim[3]*nb];
            cimg::fread(buffer,dim[3]*nb,nfile);
            //if (endian) cimg::invert_endianness(buffer,dim[3]*nb);
            cimglist_for(ret,l) cimg_foroff(ret(l),off) ret(l)._data[off] = (T)(buffer[off+l*nb]);
            delete[] buffer;
        }
    }
    cimg::fclose(nfile);

    return ret;
}





}
