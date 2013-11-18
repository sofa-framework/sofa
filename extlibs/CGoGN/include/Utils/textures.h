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

#ifndef CGOGN_TEXTURES_H_
#define CGOGN_TEXTURES_H_

#include "Utils/gl_def.h"
#include "Geometry/vector_gen.h"
#include <string>

#ifdef WITH_QT
#include <QImage>
//#else
//#error "Texture class need Qt for loading images, please recompile CGoGN with Qt support if necessary"
#else
#include <IL/il.h>
#endif

// TODO externaliser le chargement d'image pour enlever la dependance Qt ??


namespace CGoGN
{
namespace Utils
{

class GTexture
{
public:
	virtual ~GTexture() {}
	virtual void bind() {}
};

/**
* Basic class for image management
*/
template < unsigned int DIM, typename TYPE >
class ImageData: public GTexture
{
public:
	typedef Geom::Vector<DIM,unsigned int> COORD;

protected:
	/// pointer on data
	TYPE* m_data_ptr;

	/// size of image
//	unsigned int m_size[DIM];
	COORD m_size;

	/// size of sub dimension (line/slice/...)
//	unsigned int m_sizeSub[DIM];
	COORD m_sizeSub;

	/// local memory allocation ?
	bool m_localAlloc;

	void computeSub();

public:
	/// constructor 
	ImageData();

	/// copy constructor
	ImageData(const ImageData<DIM,TYPE>& img);

	/**
	* create with memory reservation for given size
	* @param size vector with size values
	*/
	ImageData(const COORD& size);

	/**
	 * swap two images
	 */
	virtual void swap(ImageData<DIM,TYPE>& img);


	/**
	 * destructor
	 */
	~ImageData();


	/**
	* create from existing data (no copy)
	* @param data pointer on image data
	* @param size vector with size values
	*/
	void create(TYPE* data, const COORD& size);

	/**
	* create from nothing (mem allocation)
	* @param size vector with size values
	*/
	void create(const COORD& size);

	/**
	* get the size
	* @return vector with size values
	*/
	const COORD& size() const { return m_size;}

	/**
	 * get direct acces to data
	 */
	TYPE* getDataPtr() const  { return m_data_ptr;}
	
	/**
	* get a pointer on data
	*/
	TYPE* getPtrData();

	/**
	* get pixel value
	*/
	TYPE& operator()(unsigned int i);

	/**
	* get pixel value
	*/
	TYPE& operator()(unsigned int i, unsigned int j);

	/**
	* get pixel value
	*/
	TYPE& operator()(unsigned int i, unsigned int j, unsigned int k);

	/**
	* get pixel value
	* @param coord coordinates stored in vector
	*/
	TYPE& texel(const COORD& coord);
	const TYPE& texel(const COORD& coord) const;
	
	template < typename TYPE2 >
	void convert(ImageData<DIM,TYPE2>& output, TYPE2 (*convertFunc)(const TYPE&));

	///texel access without assertion test (internal use, but not protected because of template)
	TYPE& texel(unsigned int i);

	///texel access without assertion test (internal use)
	TYPE& texel(unsigned int i, unsigned int j);

	///texel access without assertion test (internal use)
	TYPE& texel(unsigned int i, unsigned int j, unsigned int k);

	///texel access without assertion test (internal use, but not protected because of template)
	const TYPE& texel(unsigned int i) const;

	///texel access without assertion test (internal use)
	const TYPE& texel(unsigned int i, unsigned int j) const;

	///texel access without assertion test (internal use)
	const TYPE& texel(unsigned int i, unsigned int j, unsigned int k) const;


};

/**
* class for convolution filter creation
*/
template <unsigned int DIM>
class Filter: public ImageData<DIM,double>
{
public:
	typedef Geom::Vector<DIM,unsigned int> COORD;
	static Filter<DIM>* createGaussian(int radius, double sigma);
	static Filter<DIM>* createAverage(int radius);
};

/**
* class image with some high level function
*/
template < unsigned int DIM, typename TYPE >
class Image: public ImageData<DIM,TYPE>
{
public:
	typedef Geom::Vector<DIM,unsigned int> COORD;

protected:

	template <typename TYPEDOUBLE>
	TYPE applyFilterOneTexel(const Filter<DIM>& filter, const COORD& t) const;

	/// swap two texel in dim 1 image
	void swapTexels(unsigned int x0, unsigned int x1);

	/// swap two texel in dim 2 image
	void swapTexels(unsigned int x0, unsigned int y0, unsigned int x1, unsigned int y1);

	/// swap two texel in dim 3 image
	void swapTexels(unsigned int x0, unsigned int y0, unsigned int z0, unsigned int x1, unsigned int y1, unsigned int z1);


public:
	/// constructor
	Image();


	/// constructor with given size
	Image(const COORD& size);

	/// destructor
	~Image();


	/**
	* Load from memory ptr,
	* a copy if performed (ptr could be destructed just after calling
	* @param ptr a pointer on memory source image (allocation must be coherent with other parameters)
	* @param w width of image
	* @param h heighy of image
	* @param bpp byte per pixel of image 
	*/
	bool load(const unsigned char *ptr, unsigned int w, unsigned int h, unsigned int bpp);
	
	/// load from file
	bool load(const std::string& filename);

	/// load from file
	void save(const std::string& filename);

	/**
	* crop image
	*/
	void crop(const COORD& origin, const COORD& sz);

	/**
	* create a subimage
	*/
	Image<DIM,TYPE>* subImage(const COORD& origin, const COORD& sz);


	template <typename TYPEDOUBLE>
	Image<DIM,TYPE>* subSampleToNewImage2();

	/**
	 * scale image by 1/2
	 */
	template <typename TYPEDOUBLE>
	void subSample2();


	/**
	 * @brief Compute new size by setting the min in all dim to maxSize (sz[0] is kept multiple of 4 for texture)
	 * @param maxSize
	 * @return
	 */
	COORD newMaxSize(unsigned int maxSize);

	/**
	* scale image
	*/
	void scaleNearest(const COORD& newSize);

	/**
	 * scale image to new one
	 */
	Image<DIM,TYPE>* scaleNearestToNewImage(const COORD& newSize);

	/**
	* apply convultion filter
	* Q? what about the borders
	*/
	template <typename TYPEDOUBLE>
	Image<DIM,TYPE>* applyFilter(const Filter<DIM>& filter);

	/**
	* flip image along one axis
	* @param axis 1=X, 2=Y, 3=Z
	*/
	void flip(unsigned int axis);
	

	Image<DIM,TYPE>* rotate90ToNewImage(int axis);

	/**
	* rotation of 90 degrees in counterclockwise direction around axis 
	* @param axis 1=X, 2=Y, 3=Z (negative values : opposite directions)
	* @warning sizes are permutted
	*/
	void rotate90(int axis);
	
};




/**
* Texture class
*/
template < unsigned int DIM, typename TYPE >
class Texture: public Image<DIM,TYPE>
{
protected:
	/**
	* texture id
	*/
	CGoGNGLuint m_id;
	
	/**
	 * dimension of texture
	 */
	GLenum m_target;

	/**
	* DIM_TEXEL
	*	1			LUMINANCE
	*	2
	*	3			RGB
	*	4			RGBA
	*/		
	int m_compo;

	/**
	* GL_UNSIGNED_BYTE,
	* GL_BYTE, 
	* GL_BITMAP,
	* GL_UNSIGNED_SHORT,
	* GL_SHORT,
	* GL_UNSIGNED_INT,
	* GL_INT,
	* GL_FLOAT
	* Extract type from TYPE with dynamic_cast 
	*/
	GLenum m_type;

	/**
	 * give GL format from m_compo
	 */
	GLenum format();

	GLenum internalFormat();

	/// check alignment of texel in memory for optimized transfer
	void checkAlignment();

public:
	typedef Geom::Vector<DIM,unsigned int> COORD;

	/**
	* constructor (gen id)
	* @param type of data in texel (GL_UNSIGNED_BYE../GL_INT.../GL_FLOAT..)
	*/
	Texture(GLenum type = GL_UNSIGNED_BYTE);


	/**
	* bind the texture	
	*/
	void bind();

	/**
	* update texture on graphic memory
	*/
	void update();

	/**
	* update a part of the texture
	* @param origin (lower left corner)
	* @param sz size of sub image to send to GPU
	*/
	void update(const COORD& origin, const COORD& sz);
	
	/**
	* set filtering texture param
	* @param GL_NEAREST orGL_LINEAR
	*/
	void setFiltering(GLint param);

	/**
	* set filtering texture param
	* @param param GL_CLAMP, GL_CLAMP_TO_BORDER, GL_CLAMP_TO_EDGE, GL_MIRRORED_REPEAT, or GL_REPEAT
	*/
	void setWrapping(GLint param);
};


} //endnamespace
} //endnamespace

#include "Utils/textures.hpp"

#endif
