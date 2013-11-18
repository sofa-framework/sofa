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

#ifndef _IMPORT_PLY_DATA_H
#define _IMPORT_PLY_DATA_H

#include <stdio.h>
#include <math.h>
#include <string>

#include "Algo/Import/ply.h"

namespace CGoGN
{

class PlyImportData
{
public:

	template <typename VEC>
	void vertexPosition(int i, VEC& P) { P[0] = vlist[i]->x; P[1] = vlist[i]->y; P[2] = vlist[i]->z;}
	
//	gmtl::Vec3f vertexPosition(int i) {return gmtl::Vec3f(vlist[i]->x, vlist[i]->y, vlist[i]->z);}

	template <typename VEC>
	void vertexNormal(int i, VEC& N) { N[0] = vlist[i]->nx; N[1] = vlist[i]->ny; N[2] = vlist[i]->nz;}
	
	template <typename VEC>
	void vertexColorUint8(int i, VEC& C) { C[0] = vlist[i]->red; C[1] = vlist[i]->green; C[2] = vlist[i]->blue;}

	template <typename VEC>
	void vertexColorFloat32(int i, VEC& C) { C[0] = vlist[i]->r; C[1] = vlist[i]->g; C[2] = vlist[i]->b;}


	int nbVertices() { return nverts;}

	int nbFaces() { return nfaces;}

	/**
	* each vertex has a normal vector
	*/
	bool hasNormals() { return has_normals!=0;}

	/**
	* each vertex has a color vector
	*/
	bool hasColors() { return hasColorsUint8() || hasColorsFloat32() ;}
	bool hasColorsUint8() { return per_vertex_color_uint8 != 0 ;}
	bool hasColorsFloat32() { return per_vertex_color_float32 != 0 ;}

	/**
	* get the number of edges of a face
	*/
	int getFaceValence(int i) { return flist[i]->nverts;}

	/**
	* get a table (pointer) of int of vertex indices of 
	*/
	int* getFaceIndices(int i) { return flist[i]->verts;}

	PlyImportData();

	~PlyImportData();

	bool read_file(const std::string& filename);


protected:

	/* vertex and face definitions for a polygonal object */
	
	typedef struct Vertex {
	float x,y,z;
	float r,g,b;
	unsigned char red,green,blue;
	float nx,ny,nz;
	void *other_props;       /* other properties */
	} Vertex;
	
	typedef struct Face {
	unsigned char nverts;    /* number of vertex indices in list */
	int *verts;              /* vertex index list */
	void *other_props;       /* other properties */
	} Face;
	
	static char *elem_names[]; 
	
	static PlyProperty vert_props[];
	
	static PlyProperty face_props[];
	
	
	/*** the PLY object ***/
	
	int nverts,nfaces;
	Vertex **vlist;
	Face **flist;
	
	PlyOtherProp *vert_other,*face_other;
	
	int per_vertex_color_float32, per_vertex_color_uint8 ;
	int has_normals;
	
	char *old_locale;
};

} // namespace CGoGN


#endif

