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

#ifndef __QUANTIZATION_H__
#define __QUANTIZATION_H__

#define epsSplitVector 0.00001f

#define epsilonDistortion 0.0001f

#include <vector>
#include <math.h>

namespace CGoGN
{

namespace Algo
{

namespace PMesh
{

template <typename VEC>
struct CodeVector
{
	VEC v ;
	unsigned int regionNbVectors ;
	VEC regionVectorsSum ;
	float regionDistortion ;

	bool operator<(CodeVector<VEC>& c)
	{
		return regionDistortion > c.regionDistortion ;
	}
} ;

template <typename VEC>
class Quantization
{
	typedef typename std::list<CodeVector<VEC> >::iterator CodeVectorID ;

private:
	const std::vector<VEC>& sourceVectors ; // source vectors
	std::vector<CodeVectorID> associatedCodeVectors ; // for each source vector, id of its associated codeVector
	std::list<CodeVector<VEC> > codeVectors ; // codebook
	unsigned int nbCodeVectors ; // size of codebook

	VEC meanSourceVector ;
	float distortion ;
	float discreteEntropy, differentialEntropy ;
	float determinantSigma, traceSigma ;

	void computeMeanSourceVector() ;
	CodeVectorID nearestNeighbour(int v) ; // for the sourceVector of the given index, search the id of the nearest codeVector
	void algoLloydMax() ; // Lloyd Iteration

public:
	Quantization(const std::vector<VEC>& source) ;

//	void scalarQuantization(unsigned int nbCodeVectors, std::vector<VEC>& result) ;

	void vectorQuantizationInit() ;
	void vectorQuantizationNbRegions(unsigned int nbCodeVectors, std::vector<VEC>& result) ;
	void vectorQuantizationDistortion(float distortionGoal, std::vector<VEC>& result) ;

	unsigned int getNbCodeVectors() { return nbCodeVectors ; }

	// only available after a quantization
	float getDiscreteEntropy() { return discreteEntropy ; }
	// available immediately after object construction
	float getDifferentialEntropy() { return differentialEntropy ; }
	float getDeterminantSigma() { return determinantSigma ; }
	float getTraceSigma() { return traceSigma ; }

private:
	void computeDiscreteEntropy() ;
	void computeDifferentialEntropy() ;
	float covariance(int x, int y) ;
	float determinant(float* matrice) ;
} ;


} //namespace PMesh

} //namespace Algo

} //namespace CGoGN

#include "quantization.hpp"

#endif
