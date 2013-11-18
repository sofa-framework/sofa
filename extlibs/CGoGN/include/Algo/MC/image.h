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

#ifndef IMAGE_H
#define IMAGE_H


#include "Geometry/vector_gen.h"

#include "Utils/img3D_IO.h"

#ifdef WITH_ZINRI
#include "Zinri/Zinrimage.h"
#endif

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace MC
{

/**
 * voxel image class
 *
 * The class Image manage 3D voxel image
 * of any type
 * @param DataType the type of voxel
 */
template< typename  DataType >
class Image
{
protected:
	/**
	 * pointer to data
	 */
	DataType *m_Data;

	/**
	 * X size
	 */
	int m_WX;

	/**
	 * Y size
	 */
	int m_WY;

	/**
	 * Z size
	 */
	int m_WZ;

	/**
	 * slice size
	 */
	int m_WXY;


	/**
	 * X origin
	 */
	int m_OX;

	/**
	 * Y origin
	 */
	int m_OY;

	/**
	 * Z origin
	 */
	int m_OZ;


	/**
	 * Voxel X size
	 */
	float m_SX;

	/**
	 * Voxel Y size
	 */
	float m_SY;

	/**
	 * Voxel Z size
	 */
	float m_SZ;

	/**
	 * memory allocated ?
	 */
	bool m_Alloc;

	/**
	* Test if a point is in the image
	*
	* @param  _V the coordinates to test
	*/
	//bool correct(const gmtl::Vec3i& _V);

#ifdef WITH_ZINRI
	/**
	 * internal inrimage prt
	 */
	PTRINRIMAGE mImage;
#endif

	template< typename T>
	void readVTKBuffer(std::ifstream& in);

public:

	/**
	* default constructor
	*/
	Image();

	/**
	* destructor
	*/
	~Image();


	template< typename  DataTypeIn >
	void createMask(const Image<DataTypeIn>& img );

	/**
	* Load a file (png) in an empty image
	* @param _cName file to open
	*/
	bool loadPNG3D(const char *_cName);


	/**
	* Load an ipb-format file in an empty image
	* @param _cName file to open
	*/
	bool loadIPB(const char* _cName);

	/**
	* Load an inr.gz format file in an empty image
	* @param filname file to open
	*/
	bool loadInrgz(const char* filename);

	/**
	 * @brief load VTK binary mask image
	 * @param filename
	 * @return
	 */
	bool loadVTKBinaryMask(const char* filename);

	/**
	* Constructor
	* @param data pointer on voxel
	* @param wx width in X
	* @param wy width in Y
	* @param wz width in Z
	* @param sx voxel size in X
	* @param sy voxel size in Y
	* @param sz voxel size in Z
	* @param copy	sets to true if data must be copying, false if data are used directly (and release by the destructor).
	*/
	Image(DataType *data, int wx, int wy, int wz, float sx, float sy, float sz, bool copy = false );


	/**
	* Load a raw image
	*/
	void loadRaw(char *filename);

	/**
	 * Load a vox file
	 */
	void loadVox(char *filename);

	/**
	* save current image into file
	* @param _cName file to save
	*/
    void saveInrMask(const char* _cName);

    /**
	* get the width along X axis
	*/
	int getWidthX() const { return m_WX;}

	/**
	* get the width along Y axis
	*/
	int getWidthY() const { return m_WY;}

	/**
	* get the width along Z axis
	*/
	int getWidthZ() const { return m_WZ;}

	/**
	* get widthX*widthY (size of a slice)
	*/
	int getWidthXY() const { return m_WXY;}

	/** set the real size of voxel of image
	 * @param vx x size
	 * @param vy y size
	 * @param vz z size
	 */
	void setVoxelSize(float vx, float vy, float vz) { m_SX = vx; m_SY = vy; m_SZ = vz;}

	/** set origin (equivalent of frame size)
	 * @param ox x size
	 * @param oy y size
	 * @param oz z size
	 */
	void setOrigin(int ox, int oy, int oz) {m_OX = ox; m_OY = oy; m_OZ = oz;}

	/**
	 * get the origin
	 * @return a vector with origin
	 */
	Geom::Vec3i getOrigin() const { return Geom::Vec3i(m_OX, m_OY, m_OZ);}

	/**
	* get the data const version
	*/
	const DataType* getData() const {return m_Data;}

	/**
	* get the data non const version
	*/
	DataType* getData() {return m_Data;}

	/**
	* get the subsampling width in X of current slot
	*/
	float getVoxSizeX() const { return m_SX;}

	/**
	* get the subsampling width in Y of current slot
	*/
	float getVoxSizeY() const { return m_SY;}

	/**
	* get the subsampling width in Z of current slot
	*/
	float getVoxSizeZ() const { return m_SZ;}


	/**
	* get the voxel value
	*
	* @param  _lX,_lY,_lZ position
	* @return the value of the voxel
	*/
	DataType getVoxel(int _lX,int _lY, int _lZ);

	/**
	* get the voxel address (const ptr)
	*
	* @param _lX,_lY,_lZ position
	* @return the address of the voxel
	*/
	const DataType* getVoxelPtr(int _lX,int _lY, int _lZ) const;

	/**
	* get the voxel address
	*
	* @param  _lX,_lY,_lZ position
	* @return the address of the voxel
	*/
	DataType* getVoxelPtr(int _lX,int _lY, int _lZ);


	/**
	* get the voxel value
	*
	* @param  _Vec vector of voxel position
	* @return the value of the voxel
	*/
	DataType getVoxel( const Geom::Vec3i &_Vec);


	/**
	*  Blur the image with pseudo gaussian filter
	* @param   _lWidth width of filtering
	* @return the new image
	*/
	Image<DataType>* filtering(int _lWidth);

	/**
	*  add Frame of zero around the image
	* @param  _lWidth the width of frame to add
	* @return the new image
	*/
	Image<DataType>* addFrame(int _lWidth) const;

	/**
	 * Get the lower corner of bounding AABB
	 */
//	gmtl::Vec3f boundMin() const { return gmtl::Vec3f(0.0f, 0.0f, 0.0f);}
	Geom::Vec3f boundMin() const { return Geom::Vec3f(m_SX*m_OX, m_SY*m_OY, m_SZ*m_OZ);}

	/**
	 * Get the upper corner of bounding AABB
	 */
//	gmtl::Vec3f boundMax() const { return gmtl::Vec3f(m_SX*(m_WX-2*m_OX), m_SY*(m_WY-2*m_OY), m_SZ*(m_WZ-2*m_OZ));}
	Geom::Vec3f boundMax() const { return Geom::Vec3f(m_SX*(m_WX-m_OX), m_SY*(m_WY-m_OY), m_SZ*(m_WZ-m_OZ));}
	/**
	 * Compute the volume in cm3
	 * @param wind the windowing function
	 */
	template< typename Windowing >
	float computeVolume(const Windowing& wind) const;

	/**
	 * local (3x3) blur of image
	 */
	Image<DataType>* Blur3();

	/**
	 * create a virtual sphere for computing curvature
	 * @param table the vector a ptr offset of voxels of sphere
	 * @param _i32radius radius of sphere
	 */
	void createMaskOffsetSphere(std::vector<int>& table, int _i32radius);

	/**
	 * compute the curvature with method that count voxels in a virtual sphere
	 * @param x position in x
	 * @param y position in y
	 * @param z position in z
	 * @param sphere the precomputed sphere
	 * @param outsideVal value of outside object
	 */
	float computeCurvatureCount(float x, float y, float z, const std::vector<int>& sphere, DataType outsideVal);

	float computeCurvatureCount3(const DataType *ptrVox, const std::vector<int>& cylX, const std::vector<int>& cylY, const std::vector<int>& cyl2, DataType val);

	void createMaskOffsetCylinders(std::vector<int>& tableX, std::vector<int>& tableY, std::vector<int>& tableZ, int _i32radius);

	void addCross();

	void createNormalSphere(std::vector<Geom::Vec3f>& table, int _i32radius);

	Geom::Vec3f computeNormal(DataType *ptrVox, const std::vector<Geom::Vec3f>& sphere, DataType val, unsigned int radius);

	bool checkSaddlecomputeNormal(const Geom::Vec3f& P, const Geom::Vec3f& normal, unsigned int radius);
};


} // end namespace
} // end namespace
} // end namespace
}

#include "Algo/MC/image.hpp"



#endif
