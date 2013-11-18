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

#ifndef _IMPORT_FILE_TYPES_H
#define _IMPORT_FILE_TYPES_H

namespace CGoGN
{

namespace Algo
{

namespace Surface
{
namespace Import
{
	enum ImportType { UNKNOWNSURFACE, TRIAN, TRIANBGZ, MESHBIN, PLY, /*PLYPTM, */PLYSLFgeneric, PLYSLFgenericBin, OFF, OBJ, VRML, AHEM, STL, STLB };

	inline ImportType getFileType(const std::string& filename)
	{
		if ((filename.rfind(".trianbgz")!=std::string::npos) || (filename.rfind(".TRIANBGZ")!=std::string::npos))
			return TRIANBGZ;

		if ((filename.rfind(".trian")!=std::string::npos) || (filename.rfind(".TRIAN")!=std::string::npos))
			return TRIAN;

		if ((filename.rfind(".meshbin")!=std::string::npos) || (filename.rfind(".MESHBIN")!=std::string::npos))
			return MESHBIN;

	/*	if ((filename.rfind(".plyptm")!=std::string::npos) || (filename.rfind(".PLYGEN")!=std::string::npos))
			return PLYPTM;
	*/
		if ((filename.rfind(".plyPTMextBin")!=std::string::npos) || (filename.rfind(".plySHrealBin")!=std::string::npos))
			return PLYSLFgenericBin;

		if ((filename.rfind(".plyPTMext")!=std::string::npos) || (filename.rfind(".plySHreal")!=std::string::npos))
			return PLYSLFgeneric;

		if ((filename.rfind(".ply")!=std::string::npos) || (filename.rfind(".PLY")!=std::string::npos))
			return PLY;

		if ((filename.rfind(".off")!=std::string::npos) || (filename.rfind(".OFF")!=std::string::npos))
			return OFF;

		if ((filename.rfind(".obj")!=std::string::npos) || (filename.rfind(".OBJ")!=std::string::npos))
			return OBJ;

		if ((filename.rfind(".ahem")!=std::string::npos) || (filename.rfind(".AHEM")!=std::string::npos))
			return AHEM;

		if ((filename.rfind(".stlb")!=std::string::npos) || (filename.rfind(".STLB")!=std::string::npos))
			return STLB;

		if ((filename.rfind(".stl")!=std::string::npos) || (filename.rfind(".STL")!=std::string::npos))
			return STL;

		return UNKNOWNSURFACE;
	}
}
}


namespace Volume
{
namespace Import
{
	enum ImportType { UNKNOWNVOLUME , TET, OFF, TS, MOKA, NODE, MSH, VTU, NAS, VBGZ, TETMESH/*, OVM*/};

	inline ImportType getFileType(const std::string& filename)
	{
		if ((filename.rfind(".tetmesh")!=std::string::npos) || (filename.rfind(".TETMESH")!=std::string::npos))
			return TETMESH;

		if ((filename.rfind(".tet")!=std::string::npos) || (filename.rfind(".TET")!=std::string::npos))
			return TET;

		if ((filename.rfind(".node")!=std::string::npos) || (filename.rfind(".NODE")!=std::string::npos))
			return NODE;

		if ((filename.rfind(".off")!=std::string::npos) || (filename.rfind(".OFF")!=std::string::npos))
			return OFF;

		if ((filename.rfind(".ts")!=std::string::npos) || (filename.rfind(".TS")!=std::string::npos))
			return TS;

		if ((filename.rfind(".moka")!=std::string::npos) || (filename.rfind(".MOKA")!=std::string::npos))
			return MOKA;

		if ((filename.rfind(".node")!=std::string::npos) || (filename.rfind(".NODE")!=std::string::npos))
			return NODE;

		if ((filename.rfind(".msh")!=std::string::npos) || (filename.rfind(".MSH")!=std::string::npos))
			return MSH;

		if ((filename.rfind(".vtu")!=std::string::npos) || (filename.rfind(".VTU")!=std::string::npos))
			return VTU;

		if ((filename.rfind(".nas")!=std::string::npos) || (filename.rfind(".NAS")!=std::string::npos))
			return NAS;

		if ((filename.rfind(".vbgz")!=std::string::npos) || (filename.rfind(".VBGZ")!=std::string::npos))
			return VBGZ;

//		if ((filename.rfind(".ovm")!=std::string::npos) || (filename.rfind(".OVM")!=std::string::npos))
//			return OVM;

		return UNKNOWNVOLUME;
	}

}
}


}
}

#endif 
