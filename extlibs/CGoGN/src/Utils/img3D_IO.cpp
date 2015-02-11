/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* aint with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

#include <iostream>

#include "Utils/img3D_IO.h"
#include "Utils/cgognStream.h"

#ifdef CGOGN_WITH_QT
#include <QImage>
#include <QString>
#endif

namespace CGoGN
{

namespace Utils
{

#ifdef CGOGN_WITH_QT
namespace Img3D_IO
{

template<typename DataType>
DataType* compressZ8(DataType* src, int w, int h, int d, int& new_d)
{
	new_d = (d/8);
	int wh=w*h;
	if (d%8) new_d++;

	DataType* newImg = new DataType[(w*h*new_d)+w]; // +w pour stocker entete
	DataType *dest = newImg;

	int z = 0;
	while (z < d)
	{
		DataType* ptrs[8];
		ptrs[0] = src;
		z++;
		for (int i =1; i<8; ++i)
		{
			if (z<d)
				ptrs[i] = ptrs[i-1] + wh;
			else ptrs[i] = NULL;
			z++;
		}

		for (int i=0;i<wh;++i)
		{
			DataType val=0;
			for(int j=7; j>=0; --j)
			{
				val *= 2;
				if (ptrs[j] != NULL)
				{
					if (*((ptrs[j])++) != 0)
					{
						val++;
					}
				}
			}
			*dest++ = val;
		}
		src += 8*wh;
	}

	return newImg;
}


template<typename DataType>
DataType* uncompressZ8(DataType* src, int w, int h, int d)
{
	int wh = w*h;
	
	DataType* newImg = new DataType[wh*d];
	DataType *dest = newImg;

	int z = 0;
	while (z < d)
	{
		DataType* ptrs[8];
		ptrs[0] = dest;
		z++;
		for (int i =1; i<8; ++i)
		{
			if (z<d)
				ptrs[i] = ptrs[i-1] + wh;
			else ptrs[i] = NULL;
			z++;
		}

		for (int i=0;i<wh;++i)
		{
			DataType val=*src++;

			for(int j=0; j<8; ++j)
			{
				if (ptrs[j] != NULL)
				{
					if (val%2)
					{
						*((ptrs[j])++) =  255;
					}	
					else
					{
						*((ptrs[j])++) =  0;
					}
				}
				val /= 2;
			}
		}
		dest += 8*wh;
	}

	return newImg;

}



void saveBool(const std::string& filename, unsigned char* data, int w, int h, int d, float vx, float vy, float vz, int tag)
{
	// compress image from bool to 8bit pp
	int nd;
	unsigned char *dat_comp = compressZ8<unsigned char>(data,w,h,d,nd);
	// add 3D info
	int *entete1=reinterpret_cast<int*>(dat_comp+w*h*nd);
	*entete1++ = BOOL8; // type 1: bool 255 compressee
	*entete1++ = tag;
	*entete1++ = w;
	*entete1++ = h;
	*entete1++ = d;
	float *entete2=reinterpret_cast<float*>(entete1);
	*entete2++ = vx;
	*entete2++ = vy;
	*entete2++ = vz;

	// save image2D

	int hh = (h*nd)+1;
	QImage img(dat_comp, w, hh, QImage::Format_Indexed8);
	img.save(QString(filename.c_str()));
}


unsigned char* loadVal_8(const std::string& filename, int& w, int& h, int &d, float& vx, float& vy, float& vz, int& tag)
{
	QImage* ptrImg = new QImage(filename.c_str());
	if (ptrImg==NULL)
		return NULL;
	
//	get the info of 2D image
	w = ptrImg->width();
	h = ptrImg->height();
	unsigned int bpp = ptrImg->depth();

	// image OK ?
	if ((bpp!=8))
		return NULL;

	unsigned char* ptr=ptrImg->bits();
	int* ptr_int = reinterpret_cast<int*>(ptr+(w*(h-1)));
	int t = *ptr_int++;
	
	if ((t!= BOOL8) && (t!= VAL8))
		return NULL;
		
	tag = *ptr_int++;
	w = *ptr_int++;
	h = *ptr_int++;
	d = *ptr_int++;
	float* ptr_float = reinterpret_cast<float*>(ptr_int);
	vx = *ptr_float++;
	vy = *ptr_float++;
	vz = *ptr_float++;

	if (t== BOOL8)
	{
		// uncompress data
		unsigned char* data = uncompressZ8<unsigned char>(ptr,w,h,d);
	
		delete ptrImg;
		// return 3D image
		return data;
	}
	
	unsigned char* ptr2 = new unsigned char[w*h*d];
	memcpy (ptr2, ptr, w*h*d); 
	delete ptrImg;
	return ptr2;
}






void saveVal(const std::string& filename, unsigned char* data, int w, int h, int d, float vx, float vy, float vz, int tag)
{
	// init image2D
	QImage img(w,(h*d)+1,QImage::Format_Indexed8);
	memcpy(img.bits(), data, w*h*d);
	
	// add 3D info
	int *entete1=reinterpret_cast<int*>(img.bits()+w*h*d); // get the end of 3D data
	*entete1++ = VAL8; 
	*entete1++ = tag;
	*entete1++ = w;
	*entete1++ = h;
	*entete1++ = d;
	float *entete2=reinterpret_cast<float*>(entete1);
	*entete2++ = vx;
	*entete2++ = vy;
	*entete2++ = vz;

	// save image
	img.save(QString(filename.c_str()));
}



unsigned char* loadRGB(const std::string& filename, int& w, int& h, int &d, float& vx, float& vy, float& vz, int& tag)
{
	QImage* ptrImg = new QImage(filename.c_str());
	if (ptrImg==NULL)
		return NULL;
	
//	get the info of 2D image
	w = ptrImg->width();
	h = ptrImg->height();
	unsigned int bpp = ptrImg->depth();

	// image OK ?
	if ((bpp!=24))
		return NULL;

	unsigned char* ptr=ptrImg->bits();
	int* ptr_int = reinterpret_cast<int*>(ptr+3*(w*(h-1)));
	int t = *ptr_int++;
	if (t!= RGB8)
		return NULL;
	tag = *ptr_int++;
	w = *ptr_int++;
	h = *ptr_int++;
	d = *ptr_int++;
	float* ptr_float = reinterpret_cast<float*>(ptr_int);
	vx = *ptr_float++;
	vy = *ptr_float++;
	vz = *ptr_float++;

	unsigned char* ptr2 = new unsigned char[3*w*h*d];
	memcpy (ptr2, ptr, 3*w*h*d); 
	delete ptrImg;
	return ptr2;
}


void saveRGB(const std::string& filename, unsigned char* data, int w, int h, int d, float vx, float vy, float vz, int tag)
{

	// init image2D
	QImage img(w,(h*d)+1,QImage::Format_RGB888);
	memcpy(img.bits(), data, 3*w*h*d);

	// add 3D info
	
	int *entete1=reinterpret_cast<int*>(img.bits()+3*w*h*d); // get the end of 3D data
	*entete1++ = RGB8;
	*entete1++ = tag;
	*entete1++ = w;
	*entete1++ = h;
	*entete1++ = d;
	float *entete2=reinterpret_cast<float*>(entete1);
	*entete2++ = vx;
	*entete2++ = vy;
	*entete2++ = vz;


	// save image
	img.save(QString(filename.c_str()));
}

/*
TODO continue porting to Qt ?
unsigned short* loadVal_16(const std::string& filename, int& w, int& h, int &d, float& vx, float& vy, float& vz, int& tag)
{
	ilOriginFunc(IL_ORIGIN_LOWER_LEFT);
	ilEnable(IL_ORIGIN_SET);
	ILuint imgName;
	ilGenImages(1,&imgName);
	ilBindImage(imgName);
	ilLoadImage(filename);
	
//	get the info of 2D image
	w = ilGetInteger(IL_IMAGE_WIDTH);
	h = ilGetInteger(IL_IMAGE_HEIGHT);
	ILuint bpp = ilGetInteger(IL_IMAGE_BYTES_PER_PIXEL);

	CGoGNout << "BPP:"<<bpp<<CGoGNendl;
	// image OK ?
	if (bpp != 2)
		return NULL;

	unsigned short* ptr= reinterpret_cast<unsigned short*>(ilGetData());
//	CGoGNout << "PTR:"<<long(ptr)<<CGoGNendl;
	int* ptr_int = reinterpret_cast<int*>(ptr+(w*(h-1)));
	int t = *ptr_int++;
	if (t!= VAL16)
		return NULL;
	tag = *ptr_int++;
	w = *ptr_int++;
	h = *ptr_int++;
	d = *ptr_int++;
	float* ptr_float = reinterpret_cast<float*>(ptr_int);
	vx = *ptr_float++;
	vy = *ptr_float++;
	vz = *ptr_float++;

//	CGoGNout << "PTR:"<<long(ptr)<<CGoGNendl;
	return ptr;

}



void saveVal_16(const std::string& filename, unsigned short* data, int w, int h, int d, float vx, float vy, float vz, int tag)
{
	// init image2D
	ILuint imgName;
	ilGenImages(1,&imgName);
	ilBindImage(imgName);
	ilTexImage(w,(h*d)+1,1,1,IL_LUMINANCE, IL_UNSIGNED_SHORT, NULL);
	// copy 3D image in 2D image
	ilSetPixels(0,0,0,w,h*d,1,IL_LUMINANCE, IL_UNSIGNED_SHORT, data);

	// add 3D info
	int *entete1=reinterpret_cast<int*>(ilGetData()+2*w*h*d); // get the end of 3D data
	*entete1++ = VAL16; 
	*entete1++ = tag;
	*entete1++ = w;
	*entete1++ = h;
	*entete1++ = d;
	float *entete2=reinterpret_cast<float*>(entete1);
	*entete2++ = vx;
	*entete2++ = vy;
	*entete2++ = vz;

	// save image
	ilEnable(IL_FILE_OVERWRITE);
	ilSaveImage(filename);
	ilDeleteImages(1,&imgName);
}


float* loadVal_float(const std::string& filename, int& w, int& h, int &d, float& vx, float& vy, float& vz, int& tag)
{
	ilOriginFunc(IL_ORIGIN_LOWER_LEFT);
	ilEnable(IL_ORIGIN_SET);
	ILuint imgName;
	ilGenImages(1,&imgName);
	ilBindImage(imgName);
	ilLoadImage(filename);
	
//	get the info of 2D image
	w = ilGetInteger(IL_IMAGE_WIDTH);
	h = ilGetInteger(IL_IMAGE_HEIGHT);
	ILuint bpp = ilGetInteger(IL_IMAGE_BYTES_PER_PIXEL);

	// image OK ?
	if (bpp != 4)
		return NULL;

	float* ptr = reinterpret_cast<float*>(ilGetData());
	int* ptr_int = reinterpret_cast<int*>(ptr+(w*(h-1)));
	int t = *ptr_int++;
	if (t!= VALFLOAT)
		return NULL;
	tag = *ptr_int++;
	w = *ptr_int++;
	h = *ptr_int++;
	d = *ptr_int++;

	float* ptr_float = reinterpret_cast<float*>(ptr_int);
	vx = *ptr_float++;
	vy = *ptr_float++;
	vz = *ptr_float++;

	return ptr;

}


void saveVal_float(const std::string& filename, float* data, int w, int h, int d, float vx, float vy, float vz, int tag)
{
	// init image2D
	ILuint imgName;
	ilGenImages(1,&imgName);
	ilBindImage(imgName);
	ilTexImage(w,(h*d)+1,1,4,IL_RGBA, IL_UNSIGNED_BYTE, NULL);
	// copy 3D image in 2D image
	ilSetPixels(0,0,0,w,h*d,1,IL_RGBA, IL_UNSIGNED_BYTE, reinterpret_cast<float*>(data));

	// add 3D info
	int *entete1=reinterpret_cast<int*>(ilGetData()+4*w*h*d); // get the end of 3D data
	*entete1++ = VALFLOAT; 
	*entete1++ = tag;
	*entete1++ = w;
	*entete1++ = h;
	*entete1++ = d;
	float *entete2=reinterpret_cast<float*>(entete1);
	*entete2++ = vx;
	*entete2++ = vy;
	*entete2++ = vz;

	// save image
	ilEnable(IL_FILE_OVERWRITE);
	ilSaveImage(filename);
	ilDeleteImages(1,&imgName);
}
*/


} //namespace img3D_IO
#endif

} //namespace Utils

} //namespace CGoGN

