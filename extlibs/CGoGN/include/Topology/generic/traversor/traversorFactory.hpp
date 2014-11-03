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
#include "Topology/generic/cellmarker.h"
#include "Topology/generic/traversor/traversor1.h"
#include "Topology/generic/traversor/traversor2Virt.h"
#include "Topology/generic/traversor/traversor3Virt.h"
#include "Topology/generic/traversor/traversorCellVirt.h"
#include "Topology/generic/traversor/traversorDoO.h"

namespace CGoGN
{

template<typename MAP>
Traversor* TraversorFactory<MAP>::createIncident(MAP& map, Dart dart, unsigned int dim, unsigned int orbX, unsigned int orbY)
{
	int code = 0x100*dim + 0x10*(orbX-VERTEX) + orbY-VERTEX;

	switch(code)
	{
		case 0x301:
			return new VTraversor3XY<MAP, VERTEX, EDGE>(map,dart);
			break;
		case 0x302:
			return new VTraversor3XY<MAP, VERTEX, FACE>(map,dart);
			break;
		case 0x303:
			return new VTraversor3XY<MAP, VERTEX, VOLUME>(map,dart);
			break;

		case 0x310:
			return new VTraversor3XY<MAP, EDGE, VERTEX>(map,dart);
			break;
		case 0x312:
			return new VTraversor3XY<MAP, EDGE, FACE>(map,dart);
			break;
		case 0x313:
			return new VTraversor3XY<MAP, EDGE, VOLUME>(map,dart);
			break;

		case 0x320:
			return new VTraversor3XY<MAP, FACE, VERTEX>(map,dart);
			break;
		case 0x321:
			return new VTraversor3XY<MAP, FACE, EDGE>(map,dart);
			break;
		case 0x323:
			return new VTraversor3XY<MAP, FACE, VOLUME>(map,dart);
			break;

		case 0x330:
			return new VTraversor3XY<MAP, VOLUME, VERTEX>(map,dart);
			break;
		case 0x331:
			return new VTraversor3XY<MAP, VOLUME, EDGE>(map,dart);
			break;
		case 0x332:
			return new VTraversor3XY<MAP, VOLUME, FACE>(map,dart);
			break;

		case 0x201:
			return new VTraversor2VE<MAP>(map,dart);
			break;
		case 0x202:
			return new VTraversor2VF<MAP>(map,dart);
			break;
		case 0x210:
			return new VTraversor2EV<MAP>(map,dart);
			break;
		case 0x212:
			return new VTraversor2EF<MAP>(map,dart);
			break;
		case 0x220:
			return new VTraversor2FV<MAP>(map,dart);
			break;
		case 0x221:
			return new VTraversor2FE<MAP>(map,dart);
			break;


		case 0x101:
			return new Traversor1VE<MAP>(map,dart);
			break;
		case 0x110:
			return new Traversor1EV<MAP>(map,dart);
			break;
		default:
			return NULL;
			break;
	}
	return NULL;
}

template<typename MAP>
Traversor* TraversorFactory<MAP>::createAdjacent(MAP& map, Dart dart, unsigned int dim, unsigned int orbX, unsigned int orbY)
{
	int code = 0x100*dim + 0x10*(orbX-VERTEX) + orbY-VERTEX;

	switch(code)
	{
		case 0x301:
			return new VTraversor3XXaY<MAP, VERTEX, EDGE>(map,dart);
			break;
		case 0x302:
			return new VTraversor3XXaY<MAP, VERTEX, FACE>(map,dart);
			break;
		case 0x303:
			return new VTraversor3XXaY<MAP, VERTEX, VOLUME>(map,dart);
			break;

		case 0x310:
			return new VTraversor3XXaY<MAP, EDGE, VERTEX>(map,dart);
			break;
		case 0x312:
			return new VTraversor3XXaY<MAP, EDGE, FACE>(map,dart);
			break;
		case 0x313:
			return new VTraversor3XXaY<MAP, EDGE, VOLUME>(map,dart);
			break;

		case 0x320:
			return new VTraversor3XXaY<MAP, FACE, VERTEX>(map,dart);
			break;
		case 0x321:
			return new VTraversor3XXaY<MAP, FACE, EDGE>(map,dart);
			break;
		case 0x323:
			return new VTraversor3XXaY<MAP, FACE, VOLUME>(map,dart);
			break;

		case 0x330:
			return new VTraversor3XXaY<MAP, VOLUME, VERTEX>(map,dart);
			break;
		case 0x331:
			return new VTraversor3XXaY<MAP, VOLUME, EDGE>(map,dart);
			break;
		case 0x332:
			return new VTraversor3XXaY<MAP, VOLUME, FACE>(map,dart);
			break;

		case 0x201:
			return new VTraversor2VVaE<MAP>(map,dart);
			break;
		case 0x202:
			return new VTraversor2VVaF<MAP>(map,dart);
			break;
		case 0x210:
			return new VTraversor2EEaV<MAP>(map,dart);
			break;
		case 0x212:
			return new VTraversor2EEaF<MAP>(map,dart);
			break;
		case 0x220:
			return new VTraversor2FFaV<MAP>(map,dart);
			break;
		case 0x221:
			return new VTraversor2FFaE<MAP>(map,dart);
			break;

		case 0x101:
			return new Traversor1VVaE<MAP>(map,dart);
			break;
		case 0x110:
			return new Traversor1EEaV<MAP>(map,dart);
			break;
		default:
			return NULL;
			break;
	}

	return NULL;
}

template<typename MAP>
Traversor* TraversorFactory<MAP>::createCell(MAP& map, unsigned int orb, bool forceDartMarker, unsigned int thread)
{
	switch(orb)
	{
		case VERTEX:
			return new VTraversorCell<MAP,VERTEX>(map,forceDartMarker,thread);
			break;
		case EDGE:
			return new VTraversorCell<MAP,EDGE>(map,forceDartMarker,thread);
			break;
		case FACE:
			return new VTraversorCell<MAP,FACE>(map,forceDartMarker,thread);
			break;
		case VOLUME:
			return new VTraversorCell<MAP,VOLUME>(map,forceDartMarker,thread);
			break;
		case CC:
			return new VTraversorCell<MAP,CC>(map,forceDartMarker,thread);
			break;
		case VERTEX1:
			return new VTraversorCell<MAP,VERTEX1>(map,forceDartMarker,thread);
			break;
		case EDGE1:
			return new VTraversorCell<MAP,EDGE1>(map,forceDartMarker,thread);
			break;
		case VERTEX2:
			return new VTraversorCell<MAP,VERTEX2>(map,forceDartMarker,thread);
			break;
		case EDGE2:
			return new VTraversorCell<MAP,EDGE2>(map,forceDartMarker,thread);
			break;
		case FACE2:
			return new VTraversorCell<MAP,FACE2>(map,forceDartMarker,thread);
			break;
		default:
			return NULL;
			break;
	}
}

template<typename MAP>
Traversor* TraversorFactory<MAP>::createDartsOfOrbits(MAP& map, Dart dart, unsigned int orb)
{
	switch(orb)
	{
		case VERTEX:
			return new VTraversorDartsOfOrbit<MAP,VERTEX>(map,dart);
			break;
		case EDGE:
			return new VTraversorDartsOfOrbit<MAP,EDGE>(map,dart);
			break;
		case FACE:
			return new VTraversorDartsOfOrbit<MAP,FACE>(map,dart);
			break;
		case VOLUME:
			return new VTraversorDartsOfOrbit<MAP,VOLUME>(map,dart);
			break;
		case CC:
			return new VTraversorDartsOfOrbit<MAP,CC>(map,dart);
			break;
		case VERTEX1:
			return new VTraversorDartsOfOrbit<MAP,VERTEX1>(map,dart);
			break;
		case EDGE1:
			return new VTraversorDartsOfOrbit<MAP,EDGE1>(map,dart);
			break;
		case VERTEX2:
			return new VTraversorDartsOfOrbit<MAP,VERTEX2>(map,dart);
			break;
		case EDGE2:
			return new VTraversorDartsOfOrbit<MAP,EDGE2>(map,dart);
			break;
		case FACE2:
			return new VTraversorDartsOfOrbit<MAP,FACE2>(map,dart);
			break;
		default:
			return NULL;
			break;
	}
}

}
