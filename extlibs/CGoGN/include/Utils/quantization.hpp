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

#include <limits>

namespace CGoGN
{

namespace Algo
{

namespace PMesh
{

template <typename VEC>
void zero(VEC& v)
{
	for(unsigned int i = 0; i < v.dimension(); ++i)
		v[i] = typename VEC::DATA_TYPE(0) ;
}

template <typename VEC>
void set(VEC& v, typename VEC::DATA_TYPE val)
{
	for(unsigned int i = 0; i < v.dimension(); ++i)
		v[i] = val ;
}


template <typename VEC>
Quantization<VEC>::Quantization(const std::vector<VEC>& source) : sourceVectors(source)
{
	associatedCodeVectors.resize(sourceVectors.size()) ;
	nbCodeVectors = 0 ;
	computeMeanSourceVector() ;
	computeDifferentialEntropy() ;
}

template <typename VEC>
void Quantization<VEC>::computeMeanSourceVector()
{
	zero<VEC>(meanSourceVector) ;
	for(unsigned int i = 0; i < sourceVectors.size(); i++)
		meanSourceVector += sourceVectors[i] ;
	meanSourceVector /= sourceVectors.size() ;
}

// Nearest neighbour search : naive algorithm
template <typename VEC>
typename Quantization<VEC>::CodeVectorID Quantization<VEC>::nearestNeighbour(int v)
{
	VEC x = sourceVectors[v] ;
	float dist_min = std::numeric_limits<float>::max() ;
	CodeVectorID nearest = codeVectors.begin() ;
	/* search minimum of squared length between v and each vector of codebook */
	for(CodeVectorID cv = codeVectors.begin(); cv != codeVectors.end() ; ++cv)
	{
		VEC vec = x - cv->v ;
		float l = vec.norm2() ;
		if(l < dist_min)
		{
			dist_min = l ;
			nearest = cv ;
		}
	}
	return nearest ;
}

template <typename VEC>
void Quantization<VEC>::algoLloydMax()
{
	unsigned int nbLloydIt = 0 ;
	bool finished = false ;
	do
	{
		++nbLloydIt ;

		VEC z ;
		zero<VEC>(z) ;

		// initialize the codeVectors region properties
		for(CodeVectorID cv = codeVectors.begin(); cv != codeVectors.end(); ++cv)
		{
			cv->regionNbVectors = 0 ;
			cv->regionVectorsSum = z ;
			cv->regionDistortion = 0.0f ;
		}

		// For each sourceVector, find its nearest neighbour among the current codeVectors
		// and update nbVectors and distortion associated to each codeVector.
		// In the same time, compute for each codeVector the sum of the sourceVectors of its region
		// (needed if one has to update the positions of the codeVectors)
		for(unsigned int i = 0; i < sourceVectors.size(); ++i)
		{
			CodeVectorID cv = nearestNeighbour(i) ;
			associatedCodeVectors[i] = cv ;
			cv->regionNbVectors += 1 ;
			VEC vec = sourceVectors[i] - cv->v ;
			cv->regionDistortion += vec.norm2() ;
			cv->regionVectorsSum += sourceVectors[i] ;
		}

		float oldDistortion = distortion ;
		distortion = 0.0f ;

		// update the distortion associated to each codeVector
		// and compute the total distortion
		for(CodeVectorID cv = codeVectors.begin(); cv != codeVectors.end(); ++cv)
		{
			if(cv->regionNbVectors > 0)
				distortion += cv->regionDistortion ;
			else
				codeVectors.erase(cv) ;
		}
		distortion /= sourceVectors.size() ;

		if((oldDistortion - distortion) / oldDistortion < epsilonDistortion)
			finished = true ;

		if(!finished)
		{
			// update the codeVectors as the average of the sourceVectors of its region
			for(CodeVectorID cv = codeVectors.begin(); cv != codeVectors.end(); ++cv)
				cv->v = cv->regionVectorsSum / typename VEC::DATA_TYPE(cv->regionNbVectors) ;
		}
	}
	while(!finished) ;

	// sort the codeVectors by ascending distortion
	CGoGNout << "nbLloydIt -> " << nbLloydIt << CGoGNendl ;
	codeVectors.sort() ;
}

// Scalar quantization
//template <typename VEC>
//void Quantization<VEC>::scalarQuantization(unsigned int nbCodeVectors, std::vector<VEC>& result)
//{
//	// Initial codebook
//	for(unsigned int i = 0; i < nbCodeVectors; ++i)
//		codeVectors.push_back(sourceVectors[i]) ;
//
//	// compute the average distortion
//	for(unsigned int i = 0; i < sourceVectors.size(); ++i)
//	{
//		VEC vec = sourceVectors[i] - nearestNeighbour(i) ;
//		distortion += gmtl::lengthSquared(vec) ;
//	}
//	distortion /= (sourceVectors.size() * VEC::Size);
//
//	// Lloyd Iteration
//	algoLloydMax() ;
//
//	for(unsigned int i = 0; i < sourceVectors.size() ; ++i)
//		result.push_back(codeVectors[associatedCodeVectors[i]]) ;
//
//	computeDiscreteEntropy() ;
//}

template <typename VEC>
void Quantization<VEC>::vectorQuantizationInit()
{
	codeVectors.clear() ;
	nbCodeVectors = 0 ;
	distortion = 0.0f ;
	// compute the average distortion
	for(unsigned int i = 0; i < sourceVectors.size(); ++i)
	{
		VEC vec = sourceVectors[i] - meanSourceVector ;
		distortion += vec.norm2() ;
	}
	distortion /= sourceVectors.size() ;

	CodeVector<VEC> mcv ;
	mcv.v = meanSourceVector ;
	mcv.regionNbVectors = sourceVectors.size() ;
	mcv.regionDistortion = distortion ;
	codeVectors.push_back(mcv) ;
	++nbCodeVectors ;
}

// Vectorial quantization with size of codebook as ending case
template <typename VEC>
void Quantization<VEC>::vectorQuantizationNbRegions(unsigned int nbRegions, std::vector<VEC>& result)
{
	vectorQuantizationInit() ;

	// do not want to have more codeVectors than sourceVectors
	nbRegions = nbRegions > sourceVectors.size() ? sourceVectors.size() : nbRegions ;

	VEC eps ;
	set<VEC>(eps, epsSplitVector) ;
	while(nbCodeVectors < nbRegions)
	{
		unsigned int nbNewCV = nbCodeVectors / 3 + 1 ;
		if(nbCodeVectors + nbNewCV > nbRegions)
			nbNewCV = nbRegions - nbCodeVectors ;
		CodeVectorID lastOldCV = --codeVectors.end() ;
		CodeVectorID cv = codeVectors.begin() ;
		unsigned int nbAddedCV = 0 ;
		do
		{
			if(cv->regionNbVectors > 1)
			{
				CodeVector<VEC> newCV = (*cv) ;
				newCV.v -= eps ;
				cv->v += eps ;
				codeVectors.push_back(newCV) ;
				++nbCodeVectors ;
				++nbAddedCV ;
			}
			++cv ;
		} while(nbAddedCV < nbNewCV && cv != lastOldCV) ;

		// Lloyd Iteration
		algoLloydMax() ;
	}

	result.resize(sourceVectors.size()) ;
	for(unsigned int i = 0; i < sourceVectors.size() ; ++i)
		result[i] = associatedCodeVectors[i]->v ;

	computeDiscreteEntropy() ;
}

// Vectorial quantization with distortion as ending case
template <typename VEC>
void Quantization<VEC>::vectorQuantizationDistortion(float distortionGoal, std::vector<VEC>& result)
{
	vectorQuantizationInit() ;

	VEC eps ;
	set<VEC>(eps, epsSplitVector) ;
	while(distortion > distortionGoal)
	{
		unsigned int nbNewCV = nbCodeVectors / 3 + 1 ;
		if(nbCodeVectors + nbNewCV > sourceVectors.size())
			nbNewCV = sourceVectors.size() - nbCodeVectors ;
		CodeVectorID cv = codeVectors.begin() ;
		for(unsigned int i = 0; i < nbNewCV; ++i)
		{
			if(cv->regionNbVectors > 1)
			{
				CodeVector<VEC> newCV = (*cv) ;
				newCV.v -= eps ;
				cv->v += eps ;
				codeVectors.push_back(newCV) ;
				++nbCodeVectors ;
			}
			++cv ;
		}
		// Lloyd Iteration
		algoLloydMax() ;
	}

	result.resize(sourceVectors.size()) ;
	for(unsigned int i = 0; i < sourceVectors.size() ; ++i)
		result[i] = associatedCodeVectors[i]->v ;

	computeDiscreteEntropy() ;
}

inline float log2(float x)
{
    return log(x) / log(2.0f) ;
}

template <typename VEC>
void Quantization<VEC>::computeDiscreteEntropy()
{
	discreteEntropy = 0.0f ;
	for(CodeVectorID cv = codeVectors.begin(); cv != codeVectors.end(); ++cv)
	{
		float p = float(cv->regionNbVectors) / float(sourceVectors.size()) ;
		discreteEntropy += -1.0f * p * log2(p) ;
	}
}

template <typename VEC>
void Quantization<VEC>::computeDifferentialEntropy()
{
	if(VEC::DIMENSION == 1) // unidimensional case
	{
		float variance = 0 ;
		// variance computation
		for(unsigned int i = 0; i < sourceVectors.size(); i++)
		{
			VEC vec = sourceVectors[i] - meanSourceVector ;
			variance += vec.norm2() ;
		}
		variance /= sourceVectors.size() ;

		determinantSigma = variance ;
		traceSigma = variance ;
		differentialEntropy = log(variance * 2.0 * M_PI * exp(1.0)) / 2.0f ;
	}
	else // 2D or 3D case
	{
		float* matrice = new float[VEC::DIMENSION * VEC::DIMENSION] ; // covariance matrix
		float trace = 0.0f ;

		// covariance matrix computation
		for(unsigned int i = 0; i < VEC::DIMENSION; i++)
		{
			for(unsigned int j = i; j < VEC::DIMENSION; j++)
			{
				float cov = covariance(i,j) ;
				matrice[i*VEC::DIMENSION + j] = cov ;
				if(i!=j)
					matrice[j*VEC::DIMENSION + i] = cov ;
				else
					trace += cov ;
			}
		}

		determinantSigma = determinant(matrice) ;
		traceSigma = trace ;
		differentialEntropy = log(pow(2.0*M_PI*exp(1.0), VEC::DIMENSION)* determinantSigma) / 2.0f ;

		delete[] matrice ;
	}
}

template <typename VEC>
float Quantization<VEC>::covariance(int x, int y)
{
	float cov = 0.0f ;
	for(unsigned int i = 0; i < sourceVectors.size(); i++)
	{
		VEC v = sourceVectors[i] ;
		float vx = v[x] - meanSourceVector[x] ;
		float vy = v[y] - meanSourceVector[y] ;
		cov += vx * vy ;
	}
	cov /= sourceVectors.size() ;
	return cov ;
}

template <typename VEC>
float Quantization<VEC>::determinant(float* matrice)
{
	float a = matrice[0] ;
	float b = matrice[1] ;
	float c = matrice[2] ;
	float d = matrice[3] ;

	if(VEC::DIMENSION == 2)
		return a*d - b*c ;
	else // if in tridimensional case
	{
		float e = matrice[4] ;
		float f = matrice[5] ;
		float g = matrice[6] ;
		float h = matrice[7] ;
		float i = matrice[8] ;

		return (a*e*i + d*h*c + g*b*f) - (g*e*c + a*h*f + d*b*i) ;
	}
}

} //namespace PMesh

} //namespace Algo

} //namespace CGoGN
