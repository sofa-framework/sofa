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

#ifndef __IMG3D_IO__
#define __IMG3D_IO__

#include "Utils/os_spec.h"

#include "Utils/dll.h"

namespace CGoGN
{

namespace Utils
{
#ifdef CGOGN_WITH_QT
namespace Img3D_IO
{
	/**
	* Image type
	*/
	enum TYPEIMG {BOOL8=1, VAL8, RGB8, VAL16, VALFLOAT};



	/**
	* Load bool image (0/255) 
	* Warning: the allocated data image contain w supplemntary bytes which store information
	* @param filename evident
	* @param w width of image (reference out)
	* @param h height of image (reference out)
	* @param d depth of image (reference out)
	* @param vx voxel size x stored in image (reference out)
	* @param vy voxel size y stored in image (reference out)
	* @param vz voxel size z stored in image (reference out)
	* @param tag image tag (reference out)
	* @return a pointer on the image data (that have been allocated by function)
	*/
	CGoGN_UTILS_API unsigned char* loadBool(char* filename, int& w, int& h, int &d, float& vx, float& vy, float& vz, int& tag);

	/**
	* Save bool image (0/255) 
	* generated image is grey level, each set of 8 plane is compressed to one 8bpp plane
	* Warning: the saved image contain w supplemntary bytes which store informations
	* Warning: origin of image is lower left
	* @param filename evident
	* @param data a pointer on the image data 
	* @param w width of image 
	* @param h height of image 
	* @param d depth of image 
	* @param vx voxel size x 
	* @param vy voxel size y 
	* @param vz voxel size z 
	* @param tag image tag  
	*/
	CGoGN_UTILS_API void saveBool(const std::string& filename, unsigned char* data, int w, int h, int d, float vx, float vy, float vz, int tag);

	/**
	* Load 8 bits image, if image is boolean compressed, it uncompress it !
	* Warning: the allocated data image contain w supplemntary bytes which store information
	* @param filename evident
	* @param w width of image (reference out)
	* @param h height of image (reference out)
	* @param d depth of image (reference out)
	* @param vx voxel size x stored in image (reference out)
	* @param vy voxel size y stored in image (reference out)
	* @param vz voxel size z stored in image (reference out)
	* @param tag image tag (reference out)
	* @return a pointer on the image data (that have been allocated by function)
	*/
	CGoGN_UTILS_API unsigned char* loadVal_8(const std::string& filename, int& w, int& h, int &d, float& vx, float& vy, float& vz, int& tag);

	/**
	* Save 8bits val image 
	* Warning: the saved image contain w supplemntary bytes which store informations
	* Warning: origin of image is lower left
	* @param filename evident
	* @param data a pointer on the image data 
	* @param w width of image 
	* @param h height of image 
	* @param d depth of image 
	* @param vx voxel size x 
	* @param vy voxel size y 
	* @param vz voxel size z 
	* @param tag image tag  
	*/
	CGoGN_UTILS_API void saveVal(const std::string& filename, unsigned char* data, int w, int h, int d, float vx, float vy, float vz, int tag);


	/**
	* Load RGB 8 bits image 
	* Warning: the allocated data image contain w supplemntary bytes which store information
	* @param filename evident
	* @param w width of image (reference out)
	* @param h height of image (reference out)
	* @param d depth of image (reference out)
	* @param vx voxel size x stored in image (reference out)
	* @param vy voxel size y stored in image (reference out)
	* @param vz voxel size z stored in image (reference out)
	* @param tag image tag (reference out)
	* @return a pointer on the image data (that have been allocated by function)
	*/
	CGoGN_UTILS_API unsigned char* loadRGB(const std::string& filename, int& w, int& h, int &d, float& vx, float& vy, float& vz, int& id);

	/**
	* Save RGB 8 bits image 
	* Warning: the saved image contain 3w supplemntary bytes which store informations
	* Warning: origin of image is lower left
	* @param filename evident
	* @param data a pointer on the image data 
	* @param w width of image 
	* @param h height of image 
	* @param d depth of image 
	* @param vx voxel size x 
	* @param vy voxel size y 
	* @param vz voxel size z 
	* @param tag image tag  
	*/
	CGoGN_UTILS_API void saveRGB(const std::string& filename, unsigned char* data, int w, int h, int d, float vx, float vy, float vz, int tag);
	/**
	* Load 16 bits value image 
	* Warning: the allocated data image contain w supplemntary bytes which store information
	* @param filename evident
	* @param w width of image (reference out)
	* @param h height of image (reference out)
	* @param d depth of image (reference out)
	* @param vx voxel size x stored in image (reference out)
	* @param vy voxel size y stored in image (reference out)
	* @param vz voxel size z stored in image (reference out)
	* @param tag image tag (reference out)
	* @return a pointer on the image data (that have been allocated by function)
	*/
//	unsigned short* loadVal_16(const std::string& filename, int& w, int& h, int &d, float& vx, float& vy, float& vz, int& tag);

	/**
	* Save 16bits val image 
	* Warning: the saved image contain w supplemntary bytes which store informations
	* Warning: origin of image is lower left
	* @param filename evident
	* @param data a pointer on the image data 
	* @param w width of image 
	* @param h height of image 
	* @param d depth of image 
	* @param vx voxel size x 
	* @param vy voxel size y 
	* @param vz voxel size z 
	* @param tag image tag  
	*/
//	void saveVal_16(const std::string& filename, unsigned short* data, int w, int h, int d, float vx, float vy, float vz, int tag);

	/**
	* Load float value image 
	* Warning: the allocated data image contain w supplemntary bytes which store information
	* @param filename evident
	* @param w width of image (reference out)
	* @param h height of image (reference out)
	* @param d depth of image (reference out)
	* @param vx voxel size x stored in image (reference out)
	* @param vy voxel size y stored in image (reference out)
	* @param vz voxel size z stored in image (reference out)
	* @param tag image tag (reference out)
	* @return a pointer on the image data (that have been allocated by function)
	*/
//	float* loadVal_float(const std::string& filename, int& w, int& h, int &d, float& vx, float& vy, float& vz, int& id);

	/**
	* Save float val image 
	* Warning: the saved image contain w supplemntary bytes which store informations
	* Warning: origin of image is lower left
	* @param filename evident
	* @param data a pointer on the image data 
	* @param w width of image 
	* @param h height of image 
	* @param d depth of image 
	* @param vx voxel size x 
	* @param vy voxel size y 
	* @param vz voxel size z 
	* @param tag image tag  
	*/
//	void saveVal_float(const std::string& filename, float* data, int w, int h, int d, float vx, float vy, float vz, int tag);


	template<typename DataType>
	DataType* compressZ8(DataType* src, int w, int h, int d, int& new_d)
	{
		new_d = (d / 8);
		int wh = w*h;
		if (d % 8) new_d++;

		DataType* newImg = new DataType[(w*h*new_d) + w]; // +w pour stocker entete
		DataType *dest = newImg;

		int z = 0;
		while (z < d)
		{
			DataType* ptrs[8];
			ptrs[0] = src;
			z++;
			for (int i = 1; i<8; ++i)
			{
				if (z<d)
					ptrs[i] = ptrs[i - 1] + wh;
				else ptrs[i] = NULL;
				z++;
			}

			for (int i = 0; i<wh; ++i)
			{
				DataType val = 0;
				for (int j = 7; j >= 0; --j)
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
			src += 8 * wh;
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
			for (int i = 1; i<8; ++i)
			{
				if (z<d)
					ptrs[i] = ptrs[i - 1] + wh;
				else ptrs[i] = NULL;
				z++;
			}

			for (int i = 0; i<wh; ++i)
			{
				DataType val = *src++;

				for (int j = 0; j<8; ++j)
				{
					if (ptrs[j] != NULL)
					{
						if (val % 2)
						{
							*((ptrs[j])++) = 255;
						}
						else
						{
							*((ptrs[j])++) = 0;
						}
					}
					val /= 2;
				}
			}
			dest += 8 * wh;
		}

		return newImg;

	}



} //namespace
#endif

} //namespace
} //namespace

#endif


