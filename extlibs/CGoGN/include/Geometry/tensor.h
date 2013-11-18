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

#ifndef TENSOR_H_
#define TENSOR_H_

/*!
 * \file tensor.h
 */

#define CONST_VAL -5212368.54127 // random value


namespace CGoGN
{

namespace Geom
{

/*! \class Tensor tensor.h
 *
 * \brief Class for representing cubic tensors of static size and dynamic order.
 *
 * A cubic Tensor of size SIZE and order ORDER has
 * SIZE x SIZE x ... x SIZE (ORDER times) = SIZE^ORDER elements.
 *
 * \tparam REAL the floating point arithmetics to use
 * \tparam SIZE the base size of the Tensor
 */
template <unsigned int SIZE, typename REAL>
class Tensor
{
	private:
		unsigned int m_order ;
		REAL* m_data ;

	public:
		/**********************************************/
		/*                CONSTRUCTORS                */
		/**********************************************/

		/**
		 * \brief Default constructor
		 *
		 * Tensor class default constructor
		 */
		Tensor() ;

		/**
		 * \brief Copy constructor
		 *
		 * Tensor class copy constructor
		 *
		 * \param T the tensor to copy
		 */
		Tensor(const Tensor& T) ;

		/**
		 * \brief Constructor
		 *
		 * Tensor class order constructor
		 *
		 * \param order the desired order of the Tensor
		 */
		Tensor(unsigned int order) ;

		/**
		 * \brief Destructor
		 *
		 * Tensor class destructor
		 */
		~Tensor() ;

		/**********************************************/
		/*                MODIFIERS                   */
		/**********************************************/

		/**
		 * \brief Modifier: set all to zero
		 *
		 * sets all elements to zero
		 */
		void zero() ;

		/**
		 * \brief Modifier: set identity
		 *
		 * Sets diagonal elements to one
		 */
		void identity() ;

		/**
		 * \brief Modifier: set constant values
		 *
		 * Sets all values to r
		 *
		 * \param r the constant value
		 */
		void setConst(const REAL& r) ;

		/**
		 * \brief Modifier: copy Tensor
		 *
		 * copies argument into current instance
		 *
		 * \param T the tensor to copy
		 */
		void operator=(const Tensor& T) ;

		/**********************************************/
		/*                ACCESSORS                   */
		/**********************************************/

		/**
		 * \brief Accessor: get element
		 *
		 * \param p=(p0, ..., pn) the nD index to access
		 *
		 * \return const reference to the value stored at index p
		 */
		const REAL& operator()(std::vector<unsigned int> p) const ;

		/**
		 * \brief Accessor: get element
		 *
		 * \param p=(p0, ..., pn) the nD index to access
		 *
		 * \return reference to the value stored at index p
		 */
		REAL& operator()(std::vector<unsigned int> p) ;

		/**
		 * \brief Accessor: get element
		 *
		 * \param k the 1D array index to access
		 *
		 * \return const reference to the value stored at index k in the array
		 */
		const REAL& operator[](unsigned int k) const ;

		/**
		 * \brief Accessor: get element
		 *
		 * \param k the 1D array index to access
		 *
		 * \return reference to the value stored at index k in the 1D array representing the tensor
		 */
		REAL& operator[](unsigned int k) ;

		/**
		 * \brief Accessor: get amount of elements in tensor
		 *
		 * \return the amount of elements in the tensor
		 */
		unsigned int nbElem() const ;

		/**
		 * \brief Accessor: get order of tensor
		 *
		 * \return const ref to the order of the tensor
		 */
		const unsigned int& order() const ;

		/**********************************************/
		/*             UTILITY FUNCTIONS              */
		/**********************************************/

		/**
		 * \brief Utility: increment nD index
		 *
		 * Tool for incrementing the nD index (p0,...,pn) depending on the size
		 *
		 * \return true if final index not reached
		 */
		static bool incremIndex(std::vector<unsigned int>& p) ;

		/*!
		 * \brief method to complete a symmetric tensor that was
		 * only filled in its first half (defined by an index that
		 * is order ascendantly)
		 *
		 * \param T the tensor to fill
		 */
		void completeSymmetricTensor() ;

		/**********************************************/
		/*             STREAM OPERATORS               */
		/**********************************************/

		/**
		 * \brief Stream operator: write
		 *
		 * Writes the tensor in an output text stream
		 */
		friend std::ostream& operator<<(std::ostream& out, const Tensor<SIZE, REAL>& T)
		{
			out << "Tensor of order " << T.order() << " and size " << SIZE << std::endl ;
			for (unsigned int i = 0 ; i < T.nbElem() ; ++i)
			{
				out << T[i] << " " ;
				if ((i % SIZE) == (SIZE - 1))
					out << std::endl ;
			}
			return out ;
		}

	private:

		/**********************************************/
		/*             UTILITY FUNCTIONS              */
		/**********************************************/

		/**
		 * \brief Utility: convert nD to 1D index
		 *
		 * Tool for converting an nD index (p0,...,pn) to a 1d array index
		 *
		 * \return 1d index
		 */
		unsigned int getIndex(std::vector<unsigned int> p) const ;
} ;

/**********************************************/
/*           SOME USEFUL TYPEDEFS             */
/**********************************************/
typedef Tensor<2, float> Tensor2f ;
typedef Tensor<2, double> Tensor2d ;
typedef Tensor<3, float> Tensor3f ;
typedef Tensor<3, double> Tensor3d ;

} /* namespace Geom */
} /* namespace CGoGN */

#include "tensor.hpp"

#endif /* TENSOR_H_ */
