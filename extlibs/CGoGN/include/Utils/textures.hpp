/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009, IGG Team, LSIIT, University of Strasbourg                *
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
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                  *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

#include "Utils/cgognStream.h"


namespace CGoGN
{
namespace Utils
{

// implementation class ImageData

template < unsigned int DIM, typename TYPE >
ImageData<DIM,TYPE>::ImageData():
m_data_ptr(NULL),
m_size(0),
m_sizeSub(0),
m_localAlloc(false)
{
}

template < unsigned int DIM, typename TYPE >
ImageData<DIM,TYPE>::ImageData(const ImageData<DIM,TYPE>& img):
m_size(img.m_size),
m_sizeSub(img.m_sizeSub),
m_localAlloc(true)
{
	if (img.m_qimg == NULL)
	{
		if (img.m_data_ptr != NULL)
		{
			m_data_ptr = new TYPE[m_sizeSub[DIM-1]];
			memcpy(m_data_ptr, img.m_data_ptr, sizeof(TYPE)*m_sizeSub[DIM-1]);
			this->m_qimg = NULL;
		}
	}
}


template < unsigned int DIM, typename TYPE >
ImageData<DIM,TYPE>::ImageData(const COORD& size):
m_size(size),
m_localAlloc(true)
{
	computeSub();
	m_data_ptr = new TYPE[m_sizeSub[DIM-1]];
}



template < unsigned int DIM, typename TYPE >
ImageData<DIM,TYPE>::~ImageData()
{
	if (m_data_ptr!=NULL)
	{
		delete[] m_data_ptr;
	}
}


template < unsigned int DIM, typename TYPE >
void ImageData<DIM,TYPE>::swap(ImageData<DIM,TYPE>& img)
{
	COORD tmp = m_size;
	m_size = img.m_size;
	img.m_size = tmp;

	tmp = m_sizeSub;
	m_sizeSub = img.m_sizeSub;
	img.m_sizeSub = tmp;

	TYPE* prt_temp= m_data_ptr;
	m_data_ptr = img.m_data_ptr;
	img.m_data_ptr = prt_temp;
}



template < unsigned int DIM, typename TYPE >
void ImageData<DIM,TYPE>::computeSub()
{
	unsigned int sub=1;
	for (unsigned int i=0; i<DIM; ++i)
	{
		sub *= m_size[i];
		m_sizeSub[i] = sub;
	}
}


template < unsigned int DIM, typename TYPE >
void ImageData<DIM,TYPE>::create(TYPE* data, const COORD& size)
{
	// first free old image if exist
	if (this->m_qimg != NULL)
	{
		delete this->m_qmg;
	}
	else if (m_data_ptr!=NULL)
	{
		delete[] m_data_ptr;
	}

	m_size = size;
	computeSub();
	m_data_ptr = data;
	this->m_qimg = NULL;
}

template < unsigned int DIM, typename TYPE >
void ImageData<DIM,TYPE>::create(const COORD& size)
{
	// first free old image if exist
	if (m_data_ptr!=NULL)
	{
		delete[] m_data_ptr;
	}

	m_size = size;
	computeSub();
	m_data_ptr = new TYPE[m_sizeSub[DIM-1]];
}

	
template < unsigned int DIM, typename TYPE >
TYPE* ImageData<DIM,TYPE>::getPtrData()
{
	return m_data_ptr;
}

template < unsigned int DIM, typename TYPE >
TYPE& ImageData<DIM,TYPE>::operator()(unsigned int i)
{
	CGoGN_STATIC_ASSERT(DIM==1, incompatible_image_dimension) ;
	assert(i<m_size[0]);
	return m_data_ptr[i];
}

template < unsigned int DIM, typename TYPE >
TYPE& ImageData<DIM,TYPE>::operator()(unsigned int i, unsigned int j)
{
	CGoGN_STATIC_ASSERT(DIM==2, incompatible_image_dimension) ;
	assert(i<m_size[0]);
	assert(j<m_size[1]);
	return m_data_ptr[ i + j*m_sizeSub[0] ];
}


template < unsigned int DIM, typename TYPE >
TYPE& ImageData<DIM,TYPE>::operator()(unsigned int i, unsigned int j, unsigned int k)
{
	CGoGN_STATIC_ASSERT(DIM==3, incompatible_image_dimension) ;
	assert(i<m_size[0]);
	assert(j<m_size[1]);
	assert(k<m_size[2]);
	return m_data_ptr[ i + j*m_sizeSub[0] + k*m_sizeSub[1] ];
}


template < unsigned int DIM, typename TYPE >
TYPE& ImageData<DIM,TYPE>::texel(unsigned int i)
{
	return m_data_ptr[i];
}

template < unsigned int DIM, typename TYPE >
TYPE& ImageData<DIM,TYPE>::texel(unsigned int i, unsigned int j)
{
	return m_data_ptr[ i + j*m_sizeSub[0] ];
}


template < unsigned int DIM, typename TYPE >
TYPE& ImageData<DIM,TYPE>::texel(unsigned int i, unsigned int j, unsigned int k)
{
	return m_data_ptr[ i + j*m_sizeSub[0] + k*m_sizeSub[1] ];
}

template < unsigned int DIM, typename TYPE >
const TYPE& ImageData<DIM,TYPE>::texel(unsigned int i) const
{
	return m_data_ptr[i];
}

template < unsigned int DIM, typename TYPE >
const TYPE& ImageData<DIM,TYPE>::texel(unsigned int i, unsigned int j) const
{
	return m_data_ptr[ i + j*m_sizeSub[0] ];
}


template < unsigned int DIM, typename TYPE >
const TYPE& ImageData<DIM,TYPE>::texel(unsigned int i, unsigned int j, unsigned int k) const
{
	return m_data_ptr[ i + j*m_sizeSub[0] + k*m_sizeSub[1] ];
}



template < unsigned int DIM, typename TYPE >
TYPE& ImageData<DIM,TYPE>::texel(const COORD& coord)
{
	TYPE* ptr = m_data_ptr;
	for (unsigned int i=0; i<DIM; ++i)
		ptr += coord[i]*m_sizeSub[i];
	return *ptr;
}


template < unsigned int DIM, typename TYPE >
const TYPE& ImageData<DIM,TYPE>::texel(const COORD& coord) const
{
	TYPE* ptr = m_data_ptr;
	for (unsigned int i=0; i<DIM; ++i)
		ptr += coord[i]*m_sizeSub[i];
	return *ptr;
}

template < unsigned int DIM, typename TYPE >
template < typename TYPE2 >
void ImageData<DIM,TYPE>::convert(ImageData<DIM,TYPE2>& output, TYPE2 (*convertFunc)(const TYPE&))
{
	output.create(m_size);

	TYPE*  ptrSrc = m_data_ptr;
	TYPE2* ptrDst = output.getDataPtr();

	if (convertFunc)
		for (unsigned int i = 0; i<m_sizeSub[DIM-1]; ++i)
			*ptrDst++ = (*convertFunc)(*ptrSrc++);
}


// implementation class Image

template < unsigned int DIM, typename TYPE >
Image<DIM,TYPE>::Image()
{
}

template < unsigned int DIM, typename TYPE >
Image<DIM,TYPE>::Image(const COORD& size):
ImageData<DIM,TYPE>(size)
{
}


template < unsigned int DIM, typename TYPE >
Image<DIM,TYPE>::~Image()
{
//	this->m_data_ptr = NULL;
}

/*
template < unsigned int DIM, typename TYPE >
void Image<DIM,TYPE>::swap(Image<DIM,TYPE>& img)
{
	if ((this->m_qimg != NULL) && (img.m_qimg != NULL))
	{
		QImage* ptr = this->m_qimg;
		this->m_qimg = img.m_qimg;
		img.m_qimg = ptr;
	}
	
	ImageData<DIM,TYPE>::swap(img);
}
*/

template < unsigned int DIM, typename TYPE >
bool Image<DIM,TYPE>::load(const unsigned char *ptr, unsigned int w, unsigned int h, unsigned int bpp)
{
	CGoGN_STATIC_ASSERT(DIM==2, incompatible_Vector_constructor_dimension);
	
	// compatible TYPE
	if (bpp != sizeof(TYPE))
	{
		CGoGNout << "Image::load incompatible type: bbp=" << bpp << CGoGNendl;
		CGoGNout << "sizeof(TYPE)="<<sizeof(TYPE)<< CGoGNendl;
		delete ptr;
		return false;
	}

	// destroy old data	
	 if (this->m_data_ptr != NULL)
		delete[] this->m_data_ptr;

	this->m_data_ptr = new TYPE[sizeof(TYPE)*w*h];
	memcpy(this->m_data_ptr, ptr, sizeof(TYPE)*w*h);
	this->m_size[0] = w;
	this->m_size[1] = h;

	this->computeSub();

	return true;
}



#ifndef CGOGN_WITH_QT

template < unsigned int DIM, typename TYPE >
bool Image<DIM,TYPE>::load(const std::string& filename)
{
	CGoGN_STATIC_ASSERT(DIM==2, incompatible_image_dimension);
	CGoGN_STATIC_ASSERT(sizeof(TYPE)==3, incompatible_image_type_loading);

	ilInit();
	ilEnable(IL_ORIGIN_SET);
	ilOriginFunc(IL_ORIGIN_LOWER_LEFT);
	ILuint imgName;
	ilGenImages(1,&imgName);
	ilBindImage(imgName);
	ilLoadImage(filename.c_str());

	ILuint bpp = ilGetInteger(IL_IMAGE_BYTES_PER_PIXEL);

	if (bpp != sizeof(TYPE))
	{
		CGoGNerr << "wrong image (" << filename << ") bpp"<< CGoGNendl;
		ilDeleteImage(imgName);
		return false;
	}

	this->m_size[0] = ilGetInteger(IL_IMAGE_WIDTH);
	this->m_size[1] = ilGetInteger(IL_IMAGE_HEIGHT);
	this->computeSub();

	this->m_data_ptr = new TYPE[this->m_size[0]*this->m_size[1]];
	unsigned char* ptr = reinterpret_cast<unsigned char*>(this->m_data_ptr);

	for (int i=this->m_size[1]-1; i>=0; --i)
	{
		ilCopyPixels(0, i, 0, this->m_size[0],1, 1, IL_RGB, IL_UNSIGNED_BYTE, ptr);
		ptr += 3*this->m_size[0];
	}
	ilDeleteImage(imgName);

	return true;
}

template < unsigned int DIM, typename TYPE >
void Image<DIM,TYPE>::save(const std::string& /*filename*/)
{
	CGoGN_STATIC_ASSERT(DIM==2, incompatible_Vector_constructor_dimension);

	CGoGNerr << "Save image not implemented" << CGoGNendl;
}


#else
template < unsigned int DIM, typename TYPE >
bool Image<DIM,TYPE>::load(const std::string& filename)
{
	CGoGN_STATIC_ASSERT(DIM==2, incompatible_Vector_constructor_dimension);

	QImage* ptr = new QImage(filename.c_str());

	if (ptr == NULL)
	{
		CGoGNout << "Impossible to load "<< filename << CGoGNendl;
		return false;
	}
	
	//	get the info of images
	unsigned int bpp = ptr->depth() / 8;

	// compatible TYPE
	if (bpp < sizeof(TYPE))
	{
		CGoGNerr << "Image::load incompatible type: bbp=" << bpp << CGoGNendl;

		CGoGNerr << "sizeof(TYPE)="<<sizeof(TYPE)<< CGoGNendl;
		delete ptr;
		return false;
	}

	// destroy old data	
	 if (this->m_data_ptr != NULL)
		delete[] this->m_data_ptr;

	if (bpp > sizeof(TYPE))
	{
		if (sizeof(TYPE) == 3)
		{
			QImage img = ptr->convertToFormat(QImage::Format_RGB888);
			this->m_data_ptr = new TYPE[3*img.width()*img.height()];
			//copy per line for alignment
			TYPE* ptrOut = this->m_data_ptr;
			for(int i=img.height()-1; i>=0; --i)
			{
				memcpy(ptrOut, img.scanLine(i), 3*img.width());
				ptrOut += img.width(); // not *3 because type is size of 3
			}
		}
		else if (sizeof(TYPE) == 1)
		{
			QImage img = ptr->convertToFormat(QImage::Format_Indexed8);
			this->m_data_ptr = new TYPE[img.width()*img.height()];
			//copy per line for alignment
			TYPE* ptrOut = this->m_data_ptr;
			for(int i=img.height()-1; i>=0; --i)
			{
				memcpy(ptrOut, img.scanLine(i), img.width());
				ptrOut += img.width();
			}
		}
	}
	else
	{
		this->m_data_ptr = new TYPE[ptr->width()*ptr->height()];
		TYPE* ptrOut = this->m_data_ptr;
		//copy per line for alignment
		for(int i=ptr->height()-1; i>=0; --i)
		{
			memcpy(ptrOut, ptr->scanLine(i), ptr->width()*bpp);
			ptrOut += ptr->width();
		}
	}

	this->m_size[0] = ptr->width();
	this->m_size[1] = ptr->height();
	this->computeSub();

	delete ptr;
	return true;
}


template < unsigned int DIM, typename TYPE >
void Image<DIM,TYPE>::save(const std::string& filename)
{
	CGoGN_STATIC_ASSERT(DIM==2, incompatible_Vector_constructor_dimension);

	QImage* ptrIm = NULL;
	if (this->m_data_ptr != NULL)
	{
		switch(sizeof(TYPE))
		{
		case 1:
			ptrIm = new QImage((uchar *)(this->m_data_ptr), this->m_size[0],this->m_size[1], QImage::Format_Indexed8);
			break;
		case 3:
			ptrIm = new QImage((uchar *)(this->m_data_ptr), this->m_size[0],this->m_size[1], QImage::Format_RGB888);
			break;
		case 4:
			ptrIm = new QImage((uchar *)(this->m_data_ptr), this->m_size[0],this->m_size[1], QImage::Format_ARGB32);
			break;
		default:
			break;
		}
	}
	if (ptrIm != NULL)
	{
		ptrIm->save(QString(filename.c_str()));
		delete ptrIm;
	}
}
#endif


template < unsigned int DIM, typename TYPE >
Image<DIM,TYPE>* Image<DIM,TYPE>::subImage(const COORD& origin, const COORD& sz)
{
	COORD szz;
	unsigned int nbt=1;

	for (unsigned int i=0; i<DIM; ++i)
	{
		if ((origin[i] + sz[i]) > this->m_size[i])
			szz[i] = this->m_size[i] - origin[i];
		else
			szz[i] = sz[i];
		nbt *= szz[i];
	}

	Image<DIM,TYPE>* newImg = new Image<DIM,TYPE>;
	newImg->create(szz);

	switch(DIM)
	{
	case 1:
		for (unsigned int i=0; i< szz[0]; ++i)
			newImg->texel(i) = this->texel(origin[0]+i);
		break;
	case 2:
			for (unsigned int j=0; j< szz[1]; ++j)
				for (unsigned int i=0; i< szz[0]; ++i)
					newImg->texel(i,j) = this->texel(origin[0]+i,origin[1]+j);
			break;
	case 3:
			for (unsigned int k=0; k< szz[2]; ++k)
				for (unsigned int j=0; j< szz[1]; ++j)
					for (unsigned int i=0; i< szz[0]; ++i)
						newImg->texel(i,j,k) = this->texel(origin[0]+i,origin[1]+j,origin[2]+k);
		break;
	}

	return newImg;
}

template < unsigned int DIM, typename TYPE >
void Image<DIM,TYPE>::crop(const COORD& origin, const COORD& sz)
{
	Image<DIM,TYPE>* newImg = subImage(origin,sz);
	swap(*newImg);
	delete newImg;
}




template < unsigned int DIM, typename TYPE >
Image<DIM,TYPE>* Image<DIM,TYPE>::scaleNearestToNewImage(const COORD& newSize)
{
	Image<DIM,TYPE>* newImg = new Image<DIM,TYPE>;
	newImg->create(newSize);

	double inc0 = double(this->m_size[0])/double(newSize[0]);

	switch(DIM)
	{
	case 1:
		{
			double p0 = inc0/2.0 - 0.5;
			for (unsigned int i=0; i< newSize[0]; ++i)
			{
				newImg->texel(i) = this->texel( (unsigned int)(p0+0.5) );
				p0 += inc0;
			}
		}
		break;
	case 2:
		{
			double inc1 = double(this->m_size[1])/double(newSize[1]);
			double p1 = inc1/2.0 - 0.5;

			for (unsigned int j=0; j< newSize[1]; ++j)
			{
				double p0 = inc0/2.0 - 0.5;
				for (unsigned int i=0; i< newSize[0]; ++i)
				{
					newImg->texel(i,j) = this->texel( (unsigned int)(p0+0.5), (unsigned int)(p1+0.5) );
					p0 += inc0;
				}
				p1 += inc1;
			}
		}
		break;
	case 3:
		{
			double inc1 = double(this->m_size[1])/double(newSize[1]);
			double inc2 = double(this->m_size[2])/double(newSize[2]);
			double p2 = inc2/2.0 - 0.5;

			for (unsigned int k=0; k< newSize[2]; ++k)
			{
				double p1 = inc1/2.0 - 0.5;
				for (unsigned int j=0; j< newSize[1]; ++j)
				{
					double p0 = inc0/2.0 - 0.5;
					for (unsigned int i=0; i< newSize[0]; ++i)
					{
						newImg->texel(i,j,k) = this->texel( (unsigned int)(p0+0.5), (unsigned int)(p1+0.5), (unsigned int)(p2+0.5) );
						p0 += inc0;
					}
					p1 += inc1;
				}
				p2 += inc2;
			}
		}
		break;
	}

	return newImg;
}



template < unsigned int DIM, typename TYPE >
void Image<DIM,TYPE>::scaleNearest(const COORD& newSize)
{
	if (newSize != this->m_size)
	{
		Image<DIM,TYPE>* newImg = scaleNearestToNewImage(newSize);
		this->swap(*newImg);
		delete newImg;
	}
}


template < unsigned int DIM, typename TYPE >
typename Image<DIM,TYPE>::COORD Image<DIM,TYPE>::newMaxSize(unsigned int maxSize)
{
	unsigned int currentMax=0;
	double sf = 1.0;

	for (unsigned int d=0; d<DIM; ++d)
	{
		if (this->m_size[d] > maxSize)
		{
			currentMax = this->m_size[d];
			sf = double(maxSize) / double(currentMax);
		}
	}

	COORD nc;
	// ensure width of image (texture) is numultiple of 4
	nc[0] = (unsigned int)(this->m_size[0]*sf);
	if (nc[0]%4)
	{
		nc[0] = (nc[0]/4)*4;
		sf = double(nc[0])/double(this->m_size[0]);
	}

	for (unsigned int d=1; d<DIM; ++d)
		nc[d] = (unsigned int)(this->m_size[d]*sf);

	return nc;
}

template < unsigned int DIM, typename TYPE >
template <typename TYPEDOUBLE>
Image<DIM,TYPE>* Image<DIM,TYPE>::subSampleToNewImage2()
{
	COORD newSize = this->m_size/2;
	Image<DIM,TYPE>* newImg = new Image<DIM,TYPE>(newSize);
//	newImg->create(newSize);

	switch(DIM)
	{
	case 1:
			for (unsigned int i=0; i< newSize[0]; ++i)
				newImg->texel(i) =  TYPE(TYPEDOUBLE(this->texel(2*i)) +   TYPEDOUBLE(this->texel(2*i+1)) /2.0);
		break;
	case 2:
			for (unsigned int j=0; j< newSize[1]; ++j)
				for (unsigned int i=0; i< newSize[0]; ++i)
				{
					TYPEDOUBLE sum =  TYPEDOUBLE(this->texel(2*i,2*j));
					sum +=  TYPEDOUBLE(this->texel(2*i+1,2*j));
					sum +=  TYPEDOUBLE(this->texel(2*i+1,2*j+1));
					sum +=  TYPEDOUBLE(this->texel(2*i  ,2*j+1));
					newImg->texel(i,j) = TYPE (sum/4.0);
				}
		break;
	case 3:
			for (unsigned int k=0; k< newSize[2]; ++k)
				for (unsigned int j=0; j< newSize[1]; ++j)
					for (unsigned int i=0; i< newSize[0]; ++i)
					{
						TYPEDOUBLE sum =  TYPEDOUBLE(this->texel(2*i,2*j,2*k));
						sum +=  TYPEDOUBLE(this->texel(2*i+1,2*j  ,2*k));
						sum +=  TYPEDOUBLE(this->texel(2*i+1,2*j+1,2*k));
						sum +=  TYPEDOUBLE(this->texel(2*i  ,2*j+1,2*k));
						sum +=  TYPEDOUBLE(this->texel(2*i,  2*j,  2*k+1));
						sum +=  TYPEDOUBLE(this->texel(2*i+1,2*j  ,2*k+1));
						sum +=  TYPEDOUBLE(this->texel(2*i+1,2*j+1,2*k+1));
						sum +=  TYPEDOUBLE(this->texel(2*i  ,2*j+1,2*k+1));
						newImg->texel(i,j,k) = TYPE (sum/8.0);
					}
		break;
	}
	return newImg;
}

template < unsigned int DIM, typename TYPE >
template <typename TYPEDOUBLE>
void Image<DIM,TYPE>::subSample2()
{
	Image<DIM,TYPE>* newImg = subSampleToNewImage2<TYPEDOUBLE>();
	this->swap(*newImg);
	delete newImg;
}


template < unsigned int DIM, typename TYPE >
void Image<DIM,TYPE>::swapTexels(unsigned int x0, unsigned int x1)
{
	TYPE tempo = this->texel(x0);
	this->texel(x0) = this->texel(x1);
	this->texel(x1) = tempo;
}

template < unsigned int DIM, typename TYPE >
void Image<DIM,TYPE>::swapTexels(unsigned int x0, unsigned int y0, unsigned int x1, unsigned int y1)
{

	TYPE tempo = this->texel(x0,y0);
	this->texel(x0,y0) = this->texel(x1,y1);
	this->texel(x1,y1) = tempo;
}

template < unsigned int DIM, typename TYPE >
void Image<DIM,TYPE>::swapTexels(unsigned int x0, unsigned int y0, unsigned int z0, unsigned int x1, unsigned int y1, unsigned int z1)
{
	TYPE tempo = this->texel(x0,y0,z0);
	this->texel(x0,y0,z0) = this->texel(x1,y1,z1);
	this->texel(x1,y1,z1) = tempo;
}


template < unsigned int DIM, typename TYPE >
void Image<DIM,TYPE>::flip(unsigned int axis)
{
	assert(axis<=DIM);

	switch(DIM)
	{
	case 1:
		{
			unsigned int sz0 = this->m_size[0]/2;
			for (unsigned int i=0; i< sz0; ++i)
				swapTexels(i, this->m_size[0]-1);
		}
		break;
	case 2:
		if (axis==1)
		{
			unsigned int sz0 = this->m_size[0]/2;
			for (unsigned int j=0; j<  this->m_size[1]; ++j)
				for (unsigned int i=0; i< sz0; ++i)
					swapTexels(i,j, this->m_size[0]-1-i,j );
		}
		else if (axis==2)
		{
			unsigned int sz1 = this->m_size[1]/2;
			for (unsigned int j=0; j< sz1; ++j)
				for (unsigned int i=0; i< this->m_size[0]; ++i)
					swapTexels(i,j, i,this->m_size[1]-1-j);
		}
		break;
	case 3:
		if (axis==1)
		{
			unsigned int sz0 = this->m_size[0]/2;
			for (unsigned int k=0; k< this->m_size[2]; ++k)
				for (unsigned int j=0; j< this->m_size[1]; ++j)
					for (unsigned int i=0; i< sz0; ++i)
						swapTexels(i,j,k, this->m_size[0]-1-i,j,k);
		}
		else if (axis==2)
		{
			unsigned int sz1 = this->m_size[1]/2;
			for (unsigned int k=0; k< this->m_size[2]; ++k)
				for (unsigned int j=0; j< sz1; ++j)
					for (unsigned int i=0; i< this->m_size[0]; ++i)
						swapTexels(i,j,k, i,this->m_size[1]-1-j,k);
		}
		else if (axis==3)
		{
			unsigned int sz2 = this->m_size[2]/2;
			for (unsigned int k=0; k< sz2; ++k)
				for (unsigned int j=0; j< this->m_size[1]; ++j)
					for (unsigned int i=0; i< this->m_size[0]; ++i)
						swapTexels(i,j,k, i,j,this->m_size[2]-1-k);
		}
		break;
	}

}


template < unsigned int DIM, typename TYPE >
Image<DIM,TYPE>* Image<DIM,TYPE>::rotate90ToNewImage(int axis)
{
	COORD newSize;
	Image<DIM,TYPE>* newImg=NULL;

	switch(DIM)
	{
	case 1:
		CGoGNerr << " no rotation with image of dimension 1"<<CGoGNendl;
		break;
	case 2:
		newSize[0] = this->m_size[1];
		newSize[1] = this->m_size[0];
		newImg = new Image<DIM,TYPE>(newSize);

		if (axis==3)
		{
			for (unsigned int j=0; j<  newSize[1]; ++j)
				for (unsigned int i=0; i< newSize[0]; ++i)
					newImg->texel(i,j) = this->texel(j, this->m_size[1]-1-i);
		}
		else if (axis==-3)
		{
			for (unsigned int j=0; j<  newSize[1]; ++j)
				for (unsigned int i=0; i< newSize[0]; ++i)
					newImg->texel(i,j) = this->texel(this->m_size[0]-1-j, i);
		}
		break;
	case 3:
		CGoGNerr << " no yet implermented !"<<CGoGNendl;
		break;

	}
	return newImg;
}

template < unsigned int DIM, typename TYPE >
void Image<DIM,TYPE>::rotate90(int axis)
{
	Image<DIM,TYPE>* newImg = rotate90ToNewImage(axis);
	this->swap(*newImg);
	delete newImg;
}


template < unsigned int DIM, typename TYPE >
template <typename TYPEDOUBLE>
TYPE Image<DIM,TYPE>::applyFilterOneTexel(const Filter<DIM>& filter, const COORD& t) const
{
	TYPEDOUBLE val(0);

	COORD tmin = t - filter.size()/2;
	switch(DIM)
	{
	case 1:
		{
			unsigned int sz0 = filter.size()[0];
			for (unsigned int i=0; i<sz0; ++i)
				val += TYPEDOUBLE(texel(tmin[0]+i)) * filter.texel(i);
		}
		break;
	case 2:
		{
			unsigned int sz0 = filter.size()[0];
			unsigned int sz1 = filter.size()[1];
			for (unsigned int j=0; j<sz1; ++j)
				for (unsigned int i=0; i<sz0; ++i)
					val += TYPEDOUBLE(texel(tmin[0]+i,tmin[1]+j)) * filter.texel(i,j);
		}
		break;
	case 3:
		{
			unsigned int sz0 = filter.size()[0];
			unsigned int sz1 = filter.size()[1];
			unsigned int sz2 = filter.size()[2];
			for (unsigned int k=0; k<sz2; ++k)
				for (unsigned int j=0; j<sz1; ++j)
					for (unsigned int i=0; i<sz0; ++i)
						val += TYPEDOUBLE(texel(tmin[0]+i,tmin[1]+j,tmin[2]+k)) * filter.texel(i,j,k);
		}
		break;
	}

	return TYPE(val);
}

template < unsigned int DIM, typename TYPE >
template <typename TYPEDOUBLE>
Image<DIM,TYPE>* Image<DIM,TYPE>::applyFilter(const Filter<DIM>& filter)
{
	Image<DIM,TYPE>* newImg = new Image<DIM,TYPE>;
	newImg->create(this->m_size);

	// on squizz les bords !!
	switch(DIM)
	{
	case 1:

		break;
	case 2:
		{
			unsigned int xb = filter.size()[0]/2;
			unsigned int yb = filter.size()[1]/2;
			unsigned int xe = this->m_size[0] - xb;
			unsigned int ye = this->m_size[1] - yb;
			for (unsigned int j=0; j< this->m_size[1]; ++j)
				for (unsigned int i=0; i< this->m_size[0]; ++i)
					if ((i>xb)&&(i<xe) && (j>yb)&&(j<ye))
						newImg->texel(i,j) = applyFilterOneTexel<TYPEDOUBLE>(filter,COORD(i,j));
					else
						newImg->texel(i,j) = this->texel(i,j);
		}
		break;
	case 3:
		break;
	}

	return newImg;
}



template <unsigned int DIM>
Filter<DIM>* Filter<DIM>::createAverage(int radius)
{
	unsigned int sz=2*radius+1;

	Filter<DIM>* filter = new Filter<DIM>;
	filter->create(COORD(sz));

	switch(DIM)
	{
	case 1:
		{
			double val = 1.0/double(sz);
			for (unsigned int i=0; i< sz; ++i)
				filter->texel(i) = val;
		}
		break;
	case 2:
		{
			double val = 1.0/double(sz*sz);
			for (unsigned int j=0; j< sz; ++j)
				for (unsigned int i=0; i< sz; ++i)
					filter->texel(i,j) = val;
		}
		break;
	case 3:
		{
			double val = 1.0/double(sz*sz*sz);
			for (unsigned int k=0; k< sz; ++k)
				for (unsigned int j=0; j< sz; ++j)
					for (unsigned int i=0; i< sz; ++i)
						filter->texel(i,j) = val;
			}
		break;
	}

	return filter;
}


template <unsigned int DIM>
Filter<DIM>* Filter<DIM>::createGaussian(int radius, double sigma)
{
	int sz=2*radius+1;

	Filter<DIM>* filter = new Filter<DIM>;
	filter->create(COORD(sz));

	double sig2 = 2.0*sigma*sigma;

	switch(DIM)
	{
	case 1:
		{
			double coef = 1.0/(sigma*sqrt(2.0*M_PI));
			for (int i=0; i< sz; ++i)
				filter->texel(i) = coef * exp(-double(i-radius)*(i-radius)/sig2);
		}
		break;
	case 2:
		{
//			double tot=0.0;
			double coef = 1.0/(sig2*M_PI);
			for (int j=0; j< sz; ++j)
				for (int i=0; i< sz; ++i)
					filter->texel(i,j) = coef * exp(-double((i-radius)*(i-radius)+(j-radius)*(j-radius))/sig2);
		}
		break;
	case 3:
		CGoGNout<< "Not implemented"<< CGoGNendl;
		break;
	}

	return filter;

}



template < unsigned int DIM, typename TYPE >
Texture<DIM,TYPE>::Texture(GLenum type):
m_type(type)
{
	switch(DIM)
	{
	case 1:
		m_target = GL_TEXTURE_1D;
		break;
	case 2:
		m_target = GL_TEXTURE_2D;
		break;
	case 3:
		m_target = GL_TEXTURE_3D;
		break;
	}

	switch(type)
	{
	case GL_UNSIGNED_BYTE:
	case GL_BYTE:
		m_compo = sizeof(TYPE);
		break;
	case GL_UNSIGNED_SHORT:
	case GL_SHORT:
		m_compo = sizeof(TYPE)/2;
		break;
	case GL_UNSIGNED_INT:
	case GL_INT:
	case GL_FLOAT:
		m_compo = sizeof(TYPE)/4;
		break;
	}

	glGenTextures(1, &(*m_id));

	glBindTexture(m_target, *m_id);
	glTexParameteri(m_target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(m_target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameteri(m_target, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(m_target, GL_TEXTURE_WRAP_T, GL_CLAMP);

}

template < unsigned int DIM, typename TYPE >
GLenum Texture<DIM,TYPE>::format()
{
	if (m_compo == 1)
		return GL_LUMINANCE;
	if (m_compo == 2)
		return GL_LUMINANCE_ALPHA;
	if (m_compo == 3)
		return GL_RGB;
	return GL_RGBA;
}

template < unsigned int DIM, typename TYPE >
GLenum Texture<DIM,TYPE>::internalFormat()
{
	if (m_type == GL_FLOAT)
	{
		if (m_compo == 1)
			return GL_LUMINANCE32F_ARB;
		if (m_compo == 2)
			return GL_LUMINANCE_ALPHA32F_ARB;
		if (m_compo == 3)
			return GL_RGB32F_ARB;
		return GL_RGBA32F_ARB;
	}
	if (m_compo == 1)
		return GL_LUMINANCE;
	if (m_compo == 2)
		return GL_LUMINANCE_ALPHA;
	if (m_compo == 3)
		return GL_RGB;
	return GL_RGBA;
}

template < unsigned int DIM, typename TYPE >
void Texture<DIM,TYPE>::bind()
{
	glBindTexture(m_target, *m_id);
}


template < unsigned int DIM, typename TYPE >
void Texture<DIM,TYPE>::checkAlignment()
{
	unsigned int sz = sizeof(TYPE)*this->m_size[0];
	if (sz%4==0)
		glPixelStorei(GL_UNPACK_ALIGNMENT,4);
	else if (sz%2==0)
		glPixelStorei(GL_UNPACK_ALIGNMENT,2);
	else
		glPixelStorei(GL_UNPACK_ALIGNMENT,1);

}


template < unsigned int DIM, typename TYPE >
void Texture<DIM,TYPE>::update()
{
	glBindTexture(m_target, *m_id);
//	checkAlignment();
	switch(DIM)
	{
	case 1:
		glTexImage1D(m_target, 0, internalFormat(), this->m_size[0], 0, format(), m_type, this->m_data_ptr);
		break;
	case 2:
		glTexImage2D(m_target, 0, internalFormat(), this->m_size[0], this->m_size[1], 0, format(), m_type, this->m_data_ptr);
		break;
	case 3:
		glTexImage3D(m_target, 0, internalFormat(), this->m_size[0], this->m_size[1], this->m_size[2], 0, format(), m_type, this->m_data_ptr);
		break;
	default:
		CGoGNerr << "Texture of dimension >3 not supported by OpenGL !"<<CGoGNendl;
		break;
	}
}

template < unsigned int DIM, typename TYPE >
void Texture<DIM,TYPE>::update(const COORD& origin, const COORD& sz)
{
	glBindTexture(m_target, *m_id);
	checkAlignment();
	switch(DIM)
	{
	case 1:
		glTexSubImage1D(m_target, 0, origin[0], sz[0], format(), m_type, this->m_data_ptr);
		break;
	case 2:
		glTexSubImage2D(m_target, 0, origin[0], origin[1], sz[0], sz[1], format(), m_type, this->m_data_ptr);
		break;
	case 3:
		glTexSubImage3D(m_target, 0, origin[0], origin[1], origin[2], sz[0], sz[1], sz[2], format(), m_type, this->m_data_ptr);
		break;
	default:
		CGoGNerr << "Texture of dimension >3 not supported by OpenGL !"<<CGoGNendl;
		break;
	}
}


template < unsigned int DIM, typename TYPE >
void Texture<DIM,TYPE>::setFiltering(GLint param)
{
	glBindTexture(m_target, *m_id);
	glTexParameteri(m_target, GL_TEXTURE_MIN_FILTER, param);
	glTexParameteri(m_target, GL_TEXTURE_MAG_FILTER, param);
}



template < unsigned int DIM, typename TYPE >
void Texture<DIM,TYPE>::setWrapping(GLint param)
{
	glBindTexture(m_target, *m_id);
	glTexParameteri(m_target, GL_TEXTURE_WRAP_S, param);
	glTexParameteri(m_target, GL_TEXTURE_WRAP_T, param);
	glTexParameteri(m_target, GL_TEXTURE_WRAP_R, param);
}





}
}
