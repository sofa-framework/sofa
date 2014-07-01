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
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

#include <iostream>
#include <fstream>
#include <string.h>
#include <vector>
#include <math.h>
#include <typeinfo>
#include "Utils/cgognStream.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace MC
{

template< typename  DataType >
Image<DataType>::Image():
	m_Data	(NULL),
	m_OX	(0),
	m_OY	(0),
	m_OZ	(0)
{
}



template< typename  DataType >
Image<DataType>::Image(DataType *data, int wx, int wy, int wz, float sx, float sy, float sz, bool copy ):
	m_WX   (wx),
	m_WY   (wy),
	m_WZ   (wz),
	m_OX   (0),
	m_OY   (0),
	m_OZ   (0),
	m_SX   (sx),
	m_SY   (sy),
	m_SZ   (sz)
{
	if ( copy )
	{
		m_Data = new DataType[wx*wy*wz];
		memcpy( m_Data, data, wx*wy*wz*sizeof(DataType) );
		m_Alloc=true;
	}
	else
	{
		m_Data = data;
		m_Alloc=false;
	}

	m_WXY = m_WX * m_WY;
}



template< typename  DataType >
template< typename  DataTypeIn >
void Image<DataType>::createMask(const Image<DataTypeIn>& img )
{
	m_WX = img.m_WX;
	m_WY = img.m_WY;
	m_WZ = img.m_WZ;
	m_OX = 0;
	m_OY = 0;
	m_OZ = 0;
	m_SX = img.m_SX;
	m_SY = img.m_SY;
	m_SZ = img.m_SZ;
	m_WXY = m_WX * m_WY;
	m_Data = new DataType[m_WXY*m_WZ];

	DataType* ptrO = m_Data;
	DataTypeIn* ptrI = img.m_Data;

	unsigned int nb = m_WXY*m_WZ;
	for (unsigned int i=0; i<nb; ++i)
	{
		if (*ptrI++ != 0)
			*ptrO++ = (DataType)255;
		else
			*ptrO++ = 0;
	}
	m_Alloc=true;
}




template< typename  DataType >
void Image<DataType>::loadRaw(char *filename)
{
	std::ifstream fp( filename, std::ios::in|std::ios::binary);
	if (!fp.good())
	{
		CGoGNerr << "Mesh_Base::loadRaw: Unable to open file " << CGoGNendl;
		exit(0);
	}

	// read size
	fp.read(reinterpret_cast<char*>(&m_WX),sizeof(int));
	fp.read(reinterpret_cast<char*>(&m_WY),sizeof(int));
	fp.read(reinterpret_cast<char*>(&m_WZ),sizeof(int));

	m_WXY = m_WX * m_WY;

	int total = m_WXY * m_WZ;

	m_Data = new DataType[total];
	// read data
	fp.read( reinterpret_cast<char*>(m_Data),total*sizeof(DataType) );

	m_SX = 1.0;
	m_SY = 1.0;
	m_SZ = 1.0;

	m_Alloc=true;
}



template< typename  DataType >
void Image<DataType>::loadVox(char *filename)
{
	std::ifstream in(filename);
	if (!in)
	{
		CGoGNerr << "Mesh_Base::loadVox: Unable to open file " << CGoGNendl;
		exit(0);
	}

	m_SX = 1.0 ;
	m_SY = 1.0 ;
	m_SZ = 1.0 ;

	// read encoding
	while(in.good())
	{
		char line[1024] ;
		in.getline(line, 1024) ;
		std::istringstream line_input(line) ;
		std::string keyword ;
		line_input >> keyword ;

		if(keyword == "encoding") {
			std::string encoding_str ;
			line_input >> encoding_str ;
			if(encoding_str != "GRAY") {
				CGoGNerr << "loadVox : invalid encoding: \'" << encoding_str << "\'" << CGoGNendl ;
				exit(0) ;
			}
		} else if(keyword == "nu") {
			line_input >> m_WX ;
		} else if(keyword == "nv") {
			line_input >> m_WY ;
		} else if(keyword == "nw") {
			line_input >> m_WZ ;
		} else if(keyword == "scale_u") {
			line_input >> m_SX ;
		} else if(keyword == "scale_v") {
			line_input >> m_SY ;
		} else if(keyword == "scale_w") {
			line_input >> m_SZ ;
		}
	}

	m_WXY = m_WX * m_WY;
	int total = m_WXY * m_WZ;

	m_Data = new DataType[total];

	int filename_s = strlen(filename) ;
	char datafile[filename_s] ;
	strcpy(datafile, filename) ;
	datafile[filename_s-3] = 'r' ;
	datafile[filename_s-2] = 'a' ;
	datafile[filename_s-1] = 'w' ;

	std::ifstream fp(datafile, std::ios::in|std::ios::binary);
	fp.read(reinterpret_cast<char*>(m_Data), total*sizeof(DataType));

	m_Alloc=true;
}

#ifdef WITH_QT
template< typename  DataType >
bool Image<DataType>::loadPNG3D(const char* filename)
{

	int tag;
	//en fonction de DataType utiliser la bonne fonction de chargement,
	if ( (typeid(DataType)==typeid(char)) ||  (typeid(DataType)==typeid(unsigned char)) )
		m_Data = reinterpret_cast<DataType*>(CGoGN::Utils::Img3D_IO::loadVal_8(const_cast<char*>(filename), m_WX,m_WY,m_WZ, m_SX,m_SY,m_SZ,tag));

	m_WXY = m_WX * m_WY;


	if (m_Data==NULL)
	{
		CGoGNerr << " ERROR : try to load an image of type " << typeid(DataType).name() << " although file ";
		return false;
	}

	m_Alloc=true;

	return true;
}
#endif

#ifdef WITH_ZINRI
template< typename  DataType >
bool Image<DataType>::loadInrgz(const char* filename)
{
		mImage = readZInrimage(filename);

		// A modifier en verifiant le type de donne
		// contenu dans l'image.

		m_Data = (DataType*)mImage->data;
		m_Alloc = false;
		m_WX = mImage->ncols;
		m_WY = mImage->nrows;
		m_WZ = mImage->nplanes;


		m_WXY = m_WX*m_WY;

		if (m_Data==NULL)
		{
			CGoGNerr << "problem loading image" << CGoGNendl;
			return false;
		}

		m_SX = mImage->vx;
		m_SY = mImage->vy;
		m_SZ = mImage->vz;

		return true;
}

template< typename  DataType >
void Image<DataType>::saveInrMask(const char* filename)
{
	mImage = createInrimage(this->getWidthX(), this->getWidthY(), this->getWidthZ(), 1, WT_UNSIGNED_CHAR);
	mImage->vx = this->getVoxSizeX();
	mImage->vy = this->getVoxSizeY();
	mImage->vz = this->getVoxSizeZ();
	memcpy(mImage->data,this->getData(),this->getWidthX()*this->getWidthY()*this->getWidthZ());
//	setInrimageData(mImage, this->getData());
	writeZInrimage(mImage, filename);
	delete mImage;
	mImage = NULL;
}

#endif

template< typename  DataType >
template< typename T>
void Image<DataType>::readVTKBuffer(std::ifstream& in)
{

	int total = m_WXY * m_WZ;

	if (m_Data != NULL)
			delete[] m_Data;

	m_Data = new DataType[total];

	DataType* ptr = m_Data;
	T* buffer = new T[total];
	in.read(reinterpret_cast<char*>(buffer), total*sizeof(T));
	T* buf = buffer;
	for (int i=0; i<total; ++i)
		if (*buf++ !=0)
			*ptr++ = 255;
		else
			*ptr++ = 0;
	delete[] buffer;
}


template< typename  DataType >
bool Image<DataType>::loadVTKBinaryMask(const char* filename)
{
	std::ifstream in(filename, std::ios::binary);
	if (!in)
	{
		CGoGNerr << "Mesh_Base::loadVox: Unable to open file " << CGoGNendl;
		return false;
	}

	m_SX = 1.0 ;
	m_SY = 1.0 ;
	m_SZ = 1.0 ;

	// read encoding
	std::string keyword ;
	int dtType = 0;

	do
	{
		char line[1024] ;
		in.getline(line, 1024) ;

		std::cout << "LINE = "<< line << std::endl;


		std::istringstream line_input(line) ;

		line_input >> keyword ;

		std::cout << "keyword = "<< keyword << std::endl;

		if(keyword == "VTK") 
		{
				std::cout << "reading vtk file" << std::endl;
		} else if(keyword == "DIMENSIONS") 
		{
			line_input >> m_WX ;
			line_input >> m_WY ;
			line_input >> m_WZ ;

			std::cout << "DIM= "<< m_WX << "/"<< m_WY << "/"<< m_WZ << std::endl;

		} else if(keyword == "SPACING") 
		{
			line_input >> m_SX ;
			line_input >> m_SY ;
			line_input >> m_SZ ;

			std::cout << "SPACING= "<< m_SX << "/"<< m_SY << "/"<< m_SZ << std::endl;
		} 
		else if(keyword == "SCALARS")
		{
			std::string str;
			line_input >> str;
			line_input >> str;

			std::cout << "SCALAR= "<< str << std::endl;

			if (str == "unsigned_char")
				dtType = 1 ;
			else if (str == "unsigned_short")
				dtType = 2 ;
			else if (str == "unsigned_int")
				dtType = 3 ;
			else if (str == "char")
				dtType = 4 ;
			else if (str == "short")
				dtType = 5 ;
			else if (str == "int")
				dtType = 6 ;
		}
		/*else if(keyword == "POINT_DATA")
		{
			
		} */
	}while (keyword != "LOOKUP_TABLE");

	m_WXY = m_WX * m_WY;
//	int total = m_WXY * m_WZ;

//	m_Data = new DataType[total];
//	DataType* ptr = m_Data;

	switch(dtType)
	{
		case 1:
			readVTKBuffer<unsigned char>(in);
//			{
//			unsigned char* buffer = new unsigned char[total];
//			in.read(reinterpret_cast<char*>(buffer), total*sizeof(unsigned char));
//			unsigned char* buf = buffer;
//			for (int i=0; i<total; ++i)
//				if (*buf++ !=0)
//					*ptr++ = 255;
//				else
//					*ptr++ = 0;
//			delete[] buffer;
//			}
			break;

		case 2:
			readVTKBuffer<unsigned short>(in);
//			{
//			unsigned short* buffer = new unsigned short[total];
//			in.read(reinterpret_cast<char*>(buffer), total*sizeof(unsigned short));
//			unsigned short* buf = buffer;
//			for (int i=0; i<total; ++i)
//				if (*buf++ !=0)
//					*ptr++ = 255;
//				else
//					*ptr++ = 0;
//			delete[] buffer;
//			}
			break;

		case 3:
			readVTKBuffer<unsigned int>(in);
//			{
//			unsigned int* buffer = new unsigned int[total];
//			in.read(reinterpret_cast<char*>(buffer), total*sizeof(unsigned int));
//			unsigned int* buf = buffer;
//			for (int i=0; i<total; ++i)
//				if (*buf++ !=0)
//					*ptr++ = 255;
//				else
//					*ptr++ = 0;
//			delete[] buffer;
//			}
			break;
		case 4:
			readVTKBuffer<char>(in);
			break;
		case 5:
			readVTKBuffer<short>(in);
			break;
		case 6:
			readVTKBuffer<int>(in);
			break;
		default:
			CGoGNerr << "unknown format" << CGoGNendl;
			break;
	}
	m_Alloc=true;
	return true;
}


template< typename  DataType >
bool Image<DataType>::loadIPB(const char* filename)
{
	// chargement fichier

	// taille de l'image en X
	m_WX= 256;// A MODIFIER
	// taille de l'image en Y
	m_WY= 256;// A MODIFIER
	// taille de l'image en Z
	m_WZ= 256;// A MODIFIER
// taille d'une coupe XY de l'image
	m_WXY = m_WX * m_WY;

	// taille des voxels en
	m_SX= 1.0f;// A MODIFIER
	m_SY= 1.0f;// A MODIFIER
	m_SZ= 1.0f;// A MODIFIER

	// pointeur sur les donnees
	m_Data =  NULL;// A MODIFIER
	return true;
}




template< typename  DataType >
Image<DataType>::~Image()
{
	// ATTENTION A MODIFIER PEUT-ETRE SI m_data EST ALLOUE AILLEURS !!

	if ( m_Alloc && (m_Data != NULL))
	{
		delete[] m_Data;
	}
}


template< typename  DataType >
DataType Image<DataType>::getVoxel(int _lX, int _lY, int _lZ)
{
  return m_Data[_lX + m_WX*_lY + m_WXY*_lZ];
}


template< typename  DataType >
const DataType* Image<DataType>::getVoxelPtr(int lX, int lY, int lZ) const
{
	return m_Data + lX + m_WX*lY + m_WXY*lZ;
}


template< typename  DataType >
DataType* Image<DataType>::getVoxelPtr(int lX, int lY, int  lZ)
{
	return m_Data + lX + m_WX*lY + m_WXY*lZ;
}



/*
*  add a frame of Zero to the image
*/
template< typename  DataType >
Image<DataType>* Image<DataType>::addFrame(int frameWidth) const
{
	int lTx = m_WX+2*frameWidth;
	int lTy = m_WY+2*frameWidth;
	int lTz = m_WZ+2*frameWidth;
	int lTxy = lTx*lTy;

	// define Zero
	DataType Zero = DataType();

	// allocate new data
	DataType *newData = new DataType[lTxy*lTz];

	// get pointer to original data
	DataType *original = m_Data;


	DataType *data = newData;
	int sizeFrameZ = lTxy * frameWidth;

	// frame Z upper
	for(int i=0; i<sizeFrameZ; i++)
	{
		*data++ = Zero;
	}

	int nbsl = lTz - 2*frameWidth;
	for(int i=0; i<nbsl; i++)
	{
		int sizeFrameY = lTx*frameWidth;
		// frame Y upper
		for(int j=0; j<sizeFrameY; j++)
		{
			*data++ = Zero;
		}

		int nbrow = lTy - 2*frameWidth;
		for(int k=0; k<nbrow; k++)
		{
			// frame X upper
			for(int l=0; l< frameWidth; l++)
			{
				*data++ = Zero;
			}

			// copy original Data
			int nbcol = lTx - 2*frameWidth;
			for(int l=0; l<nbcol; l++)
			{
				*data++ = *original++;
			}

			// frame X lower
			for(int l=0; l< frameWidth; l++)
			{
				*data++ = Zero;
			}

		}

		// frame Y upper
		for(int j=0; j<sizeFrameY; j++)
		{
			*data++ = Zero;
		}
	}

	// frame Z lower
	for(int i=0; i<sizeFrameZ; i++)
	{
		*data++ = Zero;
	}

	Image<DataType>* newImg = new Image<DataType>(newData,lTx,lTy,lTz,getVoxSizeX(),getVoxSizeY(),getVoxSizeZ());

	// set origin of real data in image
	newImg->setOrigin(m_OX+frameWidth, m_OY+frameWidth, m_OZ+frameWidth);

	return newImg;

}


template< typename  DataType >
template< typename Windowing >
float Image<DataType>::computeVolume(const Windowing& wind) const
{
	int nbv = getWidthXY()*getWidthZ();

	const DataType *data = getData();

	// volume in number of voxel
	int vol=0;

	for(int i=0; i<nbv; i++)
	{
		if (wind.inside(*data))
		{
			vol++;
		}
		data++;
	}


	return float(vol);
}

template< typename  DataType >
Image<DataType>* Image<DataType>::Blur3()
{

	int txm = m_WX-1;
	int tym = m_WY-1;
	int tzm = m_WZ-1;

	DataType* data2 = new DataType[m_WX*m_WY*m_WZ];
	Image<DataType>* newImg = new Image<DataType>(data2,m_WX,m_WY,m_WZ,getVoxSizeX(),getVoxSizeY(),getVoxSizeZ());
	newImg->m_Alloc=true;
	// set origin of real data in image ??

	// for frame
	for(int y=0; y<=tym; ++y)
		for(int x=0; x<=txm; ++x)
			*(newImg->getVoxelPtr(x,y,0)) = *(getVoxelPtr(x,y,0));

	for(int z=1; z<tzm; ++z)
	{
		for(int x=0; x<=txm; ++x)
			*(newImg->getVoxelPtr(x,0,z)) = *(getVoxelPtr(x,0,z));
		for(int y=1; y<tym; ++y)
		{
			*(newImg->getVoxelPtr(0,y,z)) = *(getVoxelPtr(0,y,z));

//			#pragma omp parallel for // OpenMP
			for(int x=1; x<txm; ++x)
			{
				DataType* ori = getVoxelPtr(x,y,z);
				DataType* dest = newImg->getVoxelPtr(x,y,z);
				DataType* ptr = ori - m_WXY - m_WX -1;
				double val=0.0;
				for (int i=0; i<3;++i)
				{
					val += (*ptr++);
					val += (*ptr++);
					val += (*ptr);
					ptr += m_WX;
					val += (*ptr--);
					val += (*ptr--);
					val += (*ptr);
					ptr += m_WX;
					val += (*ptr++);
					val += (*ptr++);
					val += (*ptr);
					ptr += m_WXY -( 2+m_WX*2);
				}
				val += 3.0 * (*ori);
				val /= (27.0 + 3.0);
				DataType res(val);
				*dest= res;
			}
			*(newImg->getVoxelPtr(txm,y,z)) = *(getVoxelPtr(txm,y,z));
		}
		for(int x=0; x<=txm; ++x)
			*(newImg->getVoxelPtr(x,tym,z)) = *(getVoxelPtr(x,tym,z));

	}
	for(int y=0; y<=tym; ++y)
		for(int x=1; x<txm; ++x)
			*(newImg->getVoxelPtr(x,y,tzm)) = *(getVoxelPtr(x,y,tzm));


	return newImg;
}

//template<typename DataType>
//void Image<DataType>::createMaskOffsetSphere(std::vector<int>& table, int _i32radius)
//{
//	// compute the width of the sphere for memory allocation
//	int i32Width = 2*_i32radius + 1;
//	// squared radius
//    float fRad2 = (float)(_i32radius*_i32radius);
//
//	// memory allocation
//	// difficult to know how many voxels before computing,
//	// so the reserve for the BB
//	table.reserve(i32Width*i32Width*i32Width);
//	table.clear();
//
//	// scan all the BB of the sphere
//	for (int z = -_i32radius;  z<=_i32radius; z++)
//	{
//		for (int y = -_i32radius;  y<=_i32radius; y++)
//		{
//			for (int x = -_i32radius;  x<=_i32radius; x++)
//			{
//				Geom::Vec3f v((float)x,(float)y,(float)z);
//				float fLength =  v.norm2();
//				// if inside the sphere
//				if (fLength<=fRad2)
//				{
//					// the the index of the voxel
//					int index = z * m_WXY + y * m_WX + x;
//					table.push_back(index);
//				}
//			}
//		}
//	}
//}


template<typename DataType>
void Image<DataType>::createMaskOffsetSphere(std::vector<int>& table, int _i32radius)
{
	float smin = std::min(m_SX, std::min(m_SY,m_SZ));

	float xs = m_SX/smin;
	float ys = m_SY/smin;
	float zs = m_SZ/smin;

	int radX = ceil(float(_i32radius)/xs);
	int radY = ceil(float(_i32radius)/ys);
	int radZ = ceil(float(_i32radius)/zs);

	float sRadius = sqrt( double(_i32radius)/xs*double(_i32radius)/xs +  double(_i32radius)/ys*double(_i32radius)/ys +  double(_i32radius)/zs*double(_i32radius)/zs);

	// memory allocation
	// difficult to know how many voxels before computing,
	// so the reserve for the BB (not really a pb)
	table.reserve(radX*radY*radZ*8);
	table.clear();

	// scan all the BB of the sphere
	for (int z = -radZ; z<radZ; z++)
	{
		for (int y = -radY; y<radY; y++)
		{
			for (int x = -radX; x<radX; x++)
			{
				Geom::Vec3f v(float(x)*xs,float(y)*ys,float(z)*zs);
				float fLength =  v.norm();
				// if inside the sphere
				if (fLength<=sRadius)
				{
					// the the index of the voxel
					int index = z * m_WXY + y * m_WX + x;
					table.push_back(index);
				}
			}
		}
	}
}


//template<typename DataType>
//float Image<DataType>::computeCurvatureCount(const DataType *ptrVox, const std::vector<int>& sphere, DataType val)

template<typename DataType>
float Image<DataType>::computeCurvatureCount(float x, float y, float z, const std::vector<int>& sphere, DataType val)
{
//	std::cout << "XYZ"<< x << " , "<<y<<" , "<<z <<"   => "<< int(round(x/m_SX)) << " , "<<int(round(y/m_SY))<<" , "<<int(round(z/m_SZ)) << std::endl;

	int noir = 0;
	int blanc = 0;
	const DataType *ptrVox = this->getVoxelPtr(int(ceil(x)), int(ceil(y)), int(ceil(z)));

	for (std::vector<int>::const_iterator it=sphere.begin(); it!=sphere.end();it++)
	{
		const DataType *data = ptrVox + *it;
		if (*data != val)
		{
			blanc++;
		}
		else
		{
			noir++;
		}
	}

	if (blanc >= noir)
	{
		return 1.0f - ((float)noir) / ((float)blanc);
	}
	else
	{
		return 1.0f - ((float)blanc)/((float)noir);
	}
}


// TESTING NEW CURVATURE METHODS
template<typename DataType>
void Image<DataType>::createMaskOffsetCylinders(std::vector<int>& tableX, std::vector<int>& tableY, std::vector<int>& tableZ, int _i32radius)
{
	// compute the width of the sphere for memory allocation
	int i32Width = 2*_i32radius + 1;
	// squared radius
    float fRad2 = (float)(_i32radius*_i32radius);

	// memory allocation
	// difficult to know how many voxels before computing,
	// so the reserve for the BB
	tableX.reserve(i32Width*i32Width*7);
	tableX.clear();
	tableY.reserve(i32Width*i32Width*7);
	tableY.clear();
	tableZ.reserve(i32Width*i32Width*7);
	tableZ.clear();


	// scan all the BB of the sphere
	for (int z = -_i32radius;  z<=_i32radius; z++)
	{
		for (int y = -_i32radius;  y<=_i32radius; y++)
		{
			for (int x = -_i32radius;  x<=_i32radius; x++)
			{
				Geom::Vec3f v((float)x,(float)y,(float)z);
				float fLength =  v.norm2();
				// if inside the sphere
				if (fLength<=fRad2)
				{
					// the the index of the voxel
					int index = z * m_WXY + y * m_WX + x;

					if ((x<=3) && (x>=-3))
						tableX.push_back(index);
					if ((y<=3) && (y>=-3))
						tableY.push_back(index);
					if ((z<=3) && (z>=-3))
						tableZ.push_back(index);

				}
			}
		}
	}
}

template<typename DataType>
float Image<DataType>::computeCurvatureCount3(const DataType *ptrVox, const std::vector<int>& cylX, const std::vector<int>& cylY, const std::vector<int>& cylZ, DataType val)
{
	int noir = 0;
	int blanc = 0;

	float vals[3];

	for (std::vector<int>::const_iterator it=cylX.begin(); it!=cylX.end();it++)
	{
		const DataType *data = ptrVox + *it;
		if (*data != val)
		{
			blanc++;
		}
		else
		{
			noir++;
		}
	}


	if (blanc >= noir)
	{
		vals[0] = 1.0f - ((float)noir) / ((float)blanc);
	}
	else
	{
		vals[0] =  -1.0f + ((float)blanc)/((float)noir);
	}

	noir = 0;
	blanc = 0;

	for (std::vector<int>::const_iterator it=cylY.begin(); it!=cylY.end();it++)
	{
		const DataType *data = ptrVox + *it;
		if (*data != val)
		{
			blanc++;
		}
		else
		{
			noir++;
		}
	}

	if (blanc >= noir)
	{
		vals[1] = 1.0f - ((float)noir) / ((float)blanc);
	}
	else
	{
		vals[1] =  -1.0f + ((float)blanc)/((float)noir);
	}


	noir = 0;
	blanc = 0;

	for (std::vector<int>::const_iterator it=cylZ.begin(); it!=cylZ.end();it++)
	{
		const DataType *data = ptrVox + *it;
		if (*data != val)
		{
			blanc++;
		}
		else
		{
			noir++;
		}
	}

	if (blanc >= noir)
	{
		vals[2] = 1.0f - ((float)noir) / ((float)blanc);
	}
	else
	{
		vals[2] =  -1.0f + ((float)blanc)/((float)noir);
	}

//	if ((valZ>valX) && (valZ>valY))
//		return (valX + valY)/2.0f;
//
//	if ((valY>valX) && (valY>valZ))
//		return (valX + valZ)/2.0f;
//
//	return (valY + valZ)/2.0f;

	unsigned short m1,m2;
	if ((fabs(vals[0]) < fabs(vals[1])) && (fabs(vals[0]) < fabs(vals[2])))
	{
		m1 =1;
		m2 =2;
	}
	else
	{
		m1=0;
		if (fabs(vals[1]) < fabs(vals[2]))
			m2 = 2;
		else
			m2 = 1;
	}

	if (vals[m1]>0.0f)
		if (vals[m2]<0.0f)
			if ((vals[m1] - vals[m2])>0.8f)
				return 1.0f;

	if (vals[m1]<0.0f)
		if (vals[m2]>0.0f)
			if ((vals[m2] - vals[m1])>0.8f)
			return 1.0f;

	return std::max(std::max(fabs(vals[0]),fabs(vals[1])),fabs(vals[2]));

}



template< typename  DataType >
void Image<DataType>::addCross()
{
	int zm = m_WZ/2 - 10;
	int ym = m_WY/2 - 10;
	int xm = m_WX/2 - 10;

	for (int z = zm; z < zm+20; z++)
	{
		for (int x = xm; x < xm+20; x++)
		{
			for (int y = 0 ; y < m_WY; y++)
			{
				m_Data[x + m_WX*y + m_WXY*z]=DataType(255);
			}
		}
	}

	for (int z = zm; z < zm+20; z++)
	{
		for (int y = ym; y < ym+20; y++)
		{
			for (int x = 0 ; x < m_WX; x++)
			{
				m_Data[x + m_WX*y + m_WXY*z]=DataType(255);
			}
		}
	}

	for (int y = ym; y < ym+20; y++)
	{
		for (int x = xm; x < xm+20; x++)
		{
			for (int z = 0 ; z < m_WZ; z++)
			{
				m_Data[x + m_WX*y + m_WXY*z]=DataType(255);
			}
		}
	}
}




template<typename DataType>
void Image<DataType>::createNormalSphere(std::vector<Geom::Vec3f>& table, int _i32radius)
{
	// compute the width of the sphere for memory allocation
	int i32Width = 2*_i32radius + 1;

	table.reserve(i32Width*i32Width*i32Width);
	table.clear();

	// scan all the BB of the sphere
	for (int z = -_i32radius;  z<=_i32radius; z++)
	{
		for (int y = -_i32radius;  y<=_i32radius; y++)
		{
			for (int x = -_i32radius;  x<=_i32radius; x++)
			{
				Geom::Vec3f v((float)x,(float)y,(float)z);
                float fLength =  v.norm();
                v/=fLength;
				// if inside the sphere
				if (fLength<=_i32radius)
					table.push_back(v);
				else
					table.push_back(Geom::Vec3f(0.0f,0.0f,0.0f));
			}
		}
	}
}


template<typename DataType>
Geom::Vec3f Image<DataType>::computeNormal(DataType *ptrVox, const std::vector<Geom::Vec3f>& sphere, DataType val, unsigned int radius)
{

	DataType *ucDat1 = ptrVox - radius*(m_WX + m_WX*m_WY);
	Geom::Vec3f normal(0.0f,0.0f,0.0f);
	unsigned int width = 2*radius+1;

	unsigned int ind = 0;
	for (unsigned int i=0; i<width;++i)
	{
		DataType *ucDat2 = ucDat1;
		for (unsigned int j=0; j<width;++j)
		{
			DataType *ucDat3 = ucDat2;
			for (unsigned int k=0; k<width;++k)
			{
				if (*ucDat3 == val)
				{
					normal += sphere[ind];
				}
				++ucDat3;
				++ind;
			}

			ucDat2 += m_WX;
		}
		ucDat1 += m_WXY;
	}

	normal.normalize();
	return normal;
}

template<typename DataType>
bool Image<DataType>::checkSaddlecomputeNormal(const Geom::Vec3f& P, const Geom::Vec3f& normal, unsigned int radius)
{
	Geom::Vec3f V1;
	if ( (fabs(normal[0]) <fabs(normal[1])) && (fabs(normal[0]) <fabs(normal[2])) )
	{
        V1 = normal .cross( Geom::Vec3f(1.0f,0.0f,0.0));
	}
	else if (fabs(normal[1]) <fabs(normal[2]))
	{
        V1 = normal .cross( Geom::Vec3f(0.0f,1.0f,0.0));
	}
	else
	{
        V1 = normal .cross( Geom::Vec3f(0.0f,0.0f,1.0));
	}

    Geom::Vec3f V2 = normal .cross( V1 );

	Geom::Vec3f Q = P + (0.5f * radius)*normal;

	float le = 0.866f * radius; // (cos30)
	Geom::Vec3f Q1 = Q + le*V1;
	Geom::Vec3f Q2 = Q - le*V1;
	DataType X1 = getVoxel(int(floor(Q1[0])), int(floor(Q1[1])), int(floor(Q1[2])));
	DataType X2 = getVoxel(int(floor(Q2[0])), int(floor(Q2[1])), int(floor(Q2[2])));
	Q1 = Q + le*V2;
	Q2 = Q - le*V2;
	DataType X3 = getVoxel(int(floor(Q1[0])), int(floor(Q1[1])), int(floor(Q1[2])));
	DataType X4 = getVoxel(int(floor(Q2[0])), int(floor(Q2[1])), int(floor(Q2[2])));

	le *= 0.707f; // (sqrt(2)/2)
	Q1 = Q + le*(V1+V2);
	Q2 = Q - le*(V1+V2);
	DataType X5 = getVoxel(int(floor(Q1[0])), int(floor(Q1[1])), int(floor(Q1[2])));
	DataType X6 = getVoxel(int(floor(Q2[0])), int(floor(Q2[1])), int(floor(Q2[2])));

	Q1 = Q + le*(V1-V2);
	Q2 = Q - le*(V1-V2);
	DataType X7 = getVoxel(int(floor(Q1[0])), int(floor(Q1[1])), int(floor(Q1[2])));
	DataType X8 = getVoxel(int(floor(Q2[0])), int(floor(Q2[1])), int(floor(Q2[2])));

	if  ((X1 == X2) && (X3==X4) && (X1!=X3))
		return true;

	if  ((X5 == X6) && (X7==X8) && (X5!=X7))
		return true;

	return false;
}



} // end namespace
} // end namespace
} // end namespace
}
