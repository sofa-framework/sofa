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

#include "Utils/os_spec.h"
#include "Algo/Import/importPlyData.h"
#include <stdlib.h>
#include <locale.h>

namespace CGoGN
{

char* PlyImportData::elem_names[] = { /* list of the elements in the object */
	(char*) "vertex", (char*) "face"
	};
	
PlyProperty PlyImportData::vert_props[] = { /* list of property information for a vertex */
	{(char*) "x", PLY_Float32, PLY_Float32, offsetof(Vertex,x), 0, 0, 0, 0},
	{(char*) "y", PLY_Float32, PLY_Float32, offsetof(Vertex,y), 0, 0, 0, 0},
	{(char*) "z", PLY_Float32, PLY_Float32, offsetof(Vertex,z), 0, 0, 0, 0},
	{(char*) "red", PLY_Uint8, PLY_Uint8, offsetof(Vertex,red), 0, 0, 0, 0},
	{(char*) "green", PLY_Uint8, PLY_Uint8, offsetof(Vertex,green), 0, 0, 0, 0},
	{(char*) "blue", PLY_Uint8, PLY_Uint8, offsetof(Vertex,blue), 0, 0, 0, 0},
	{(char*) "r", PLY_Float32, PLY_Float32, offsetof(Vertex,r), 0, 0, 0, 0},
	{(char*) "g", PLY_Float32, PLY_Float32, offsetof(Vertex,g), 0, 0, 0, 0},
	{(char*) "b", PLY_Float32, PLY_Float32, offsetof(Vertex,b), 0, 0, 0, 0},
	{(char*) "nx", PLY_Float32, PLY_Float32, offsetof(Vertex,nx), 0, 0, 0, 0},
	{(char*) "ny", PLY_Float32, PLY_Float32, offsetof(Vertex,ny), 0, 0, 0, 0},
	{(char*) "nz", PLY_Float32, PLY_Float32, offsetof(Vertex,nz), 0, 0, 0, 0},
	};
	
PlyProperty PlyImportData::face_props[] = { /* list of property information for a face */
	{(char*) "vertex_indices", PLY_Int32, PLY_Int32, offsetof(Face,verts),
	1, PLY_Uint8, PLY_Uint8, offsetof(Face,nverts)},
	};


PlyImportData::PlyImportData(): 
	nverts(0),nfaces(0),
	vlist(NULL),
	flist(NULL),
	vert_other(NULL),
	face_other(NULL),
	per_vertex_color_float32(0),
	per_vertex_color_uint8(0),
	has_normals(0)	
{
	old_locale = setlocale(LC_NUMERIC, NULL);
	setlocale(LC_NUMERIC, "C");
}

PlyImportData::~PlyImportData()
{
// 	if (vlist!= NULL)
// 	{
// 		for (int i=0; i<nverts; ++i)
// 		{
// 			if (vlist[i]!=NULL)
// 				free(vlist[i]);
// 		}
// 		free(vlist);
// 	}
// 	
// 	if (flist!= NULL)
// 	{
// 		for (int i=0; i<nfaces; ++i)
// 		{
// 			if (flist[i]!=NULL)
// 				free(flist[i]);
// 		}
// 		free(flist);
// 	}
		
// need to free *vert_other,*face_other ????
	setlocale(LC_NUMERIC, old_locale);
}
	
	
	
bool PlyImportData::read_file(const std::string& filename)
{

  /*** Read in the original PLY object ***/

  FILE* fp = fopen(filename.c_str(),"r");

  if (fp==NULL) return false;
  
  PlyFile *in_ply  = read_ply (fp);


  if (in_ply==NULL) return false;


  for (int i = 0; i < in_ply->num_elem_types; i++) 
  {
	  int elem_count;
    /* prepare to read the i'th list of elements */
    char *elem_name = setup_element_read_ply (in_ply, i, &elem_count);

    if (equal_strings ((char*) "vertex", elem_name)) {

      /* create a vertex list to hold all the vertices */
      vlist = (Vertex **) malloc (sizeof (Vertex *) * elem_count);
      nverts = elem_count;

      /* set up for getting vertex elements */

      setup_property_ply (in_ply, &vert_props[0]);
      setup_property_ply (in_ply, &vert_props[1]);
      setup_property_ply (in_ply, &vert_props[2]);

      for (int j = 0; j < in_ply->elems[i]->nprops; j++) 
	  {
		PlyProperty *prop;
		prop = in_ply->elems[i]->props[j];
		if (equal_strings ((char*) "red", prop->name)) {
		setup_property_ply (in_ply, &vert_props[3]);
		per_vertex_color_uint8 = 1;
		}
		if (equal_strings ((char*) "green", prop->name)) {
		setup_property_ply (in_ply, &vert_props[4]);
		per_vertex_color_uint8 = 1;
		}
		if (equal_strings ((char*) "blue", prop->name)) {
		setup_property_ply (in_ply, &vert_props[5]);
		per_vertex_color_uint8 = 1;
		}
		if (equal_strings ((char*) "r", prop->name)) {
		setup_property_ply (in_ply, &vert_props[6]);
		per_vertex_color_float32 = 1;
		}
		if (equal_strings ((char*) "g", prop->name)) {
		setup_property_ply (in_ply, &vert_props[7]);
		per_vertex_color_float32 = 1;
		}
		if (equal_strings ((char*) "b", prop->name)) {
		setup_property_ply (in_ply, &vert_props[8]);
		per_vertex_color_float32 = 1;
		}
		if (equal_strings ((char*) "nx", prop->name)) {
		setup_property_ply (in_ply, &vert_props[9]);
		has_normals = 1;
		}
		if (equal_strings ((char*) "ny", prop->name)) {
		setup_property_ply (in_ply, &vert_props[10]);
		has_normals = 1;
		}
		if (equal_strings ((char*) "nz", prop->name)) {
		setup_property_ply (in_ply, &vert_props[11]);
		has_normals = 1;
		}
      }

      vert_other = get_other_properties_ply (in_ply, 
					     offsetof(Vertex,other_props));

      /* grab all the vertex elements */
      for (int j = 0; j < elem_count; j++) {
        vlist[j] = (Vertex *) malloc (sizeof (Vertex));
		vlist[j]->r = 1;
		vlist[j]->g = 1;
		vlist[j]->b = 1;
        get_element_ply (in_ply, (void *) vlist[j]);
      }
    }
    else if (equal_strings ((char*) "face", elem_name)) {

      /* create a list to hold all the face elements */
      flist = (Face **) malloc (sizeof (Face *) * elem_count);
      nfaces = elem_count;

      /* set up for getting face elements */

      setup_property_ply (in_ply, &face_props[0]);
      face_other = get_other_properties_ply (in_ply, 
					     offsetof(Face,other_props));

      /* grab all the face elements */
      for (int j = 0; j < elem_count; j++) {
        flist[j] = (Face *) malloc (sizeof (Face));
        get_element_ply (in_ply, (void *) flist[j]);
      }
    }
    else
      get_other_element_ply (in_ply);
  }
  
  close_ply (in_ply);
  
  free_ply (in_ply);

  return true;
}

} // namespace CGoGN
