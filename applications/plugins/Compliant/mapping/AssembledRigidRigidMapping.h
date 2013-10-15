/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_AssembledRigidRigidMapping_H
#define SOFA_COMPONENT_MAPPING_AssembledRigidRigidMapping_H

#include "AssembledMapping.h"
#include "initCompliant.h"

#include "utils/se3.h" 
#include "utils/pair.h" 

#include <sofa/core/ObjectFactory.h>

namespace sofa
{


namespace component
{

namespace mapping
{


/**

   Adaptation of RigidRigidMapping for sparse jacobian matrices.

   Computes a right-translated rigid-frame:    g |-> g.h

   where h is a fixed local joint frame. multiple joint frames may be
   given for one dof using member data @source

   obseletes Flexible/deformationMapping/JointRigidMapping.h

   TODO .inl

   @author maxime.tournier@inria.fr
   
*/
 


template <class TIn, class TOut>
class SOFA_Compliant_API AssembledRigidRigidMapping : public AssembledMapping<TIn, TOut> {
  public:
	SOFA_CLASS(SOFA_TEMPLATE2(AssembledRigidRigidMapping,TIn,TOut), 
               SOFA_TEMPLATE2(AssembledMapping,TIn,TOut));

	
	AssembledRigidRigidMapping() 
		: source(initData(&source, "source", "input dof and rigid offset for each output dof" )) {
		
	}

  protected:
	typedef SE3< typename TIn::Real > se3;
	
	typedef std::pair<unsigned, typename TIn::Coord> source_type;
	Data< vector< source_type > > source;
 
	typedef AssembledRigidRigidMapping self;
	
	virtual void assemble( const typename self::in_pos_type& in_pos ) {

		typename self::jacobian_type::CompressedMatrix& J = this->jacobian.compressedMatrix;

		assert( in_pos.size() );
		assert( source.getValue().size() );
		
		J.resize(6 * source.getValue().size(),
		         6 * in_pos.size() );
		J.setZero();
		
		for(unsigned i = 0, n = source.getValue().size(); i < n; ++i) {
			const source_type& s = source.getValue()[i];
			
			typename se3::mat66 block = se3::dR(s.second, in_pos[ s.first ] );
			
			for(unsigned j = 0; j < 6; ++j) {
				unsigned row = 6 * i + j;
				
				J.startVec( row );
				
				for(unsigned k = 0; k < 6; ++k) {
					unsigned col = 6 * s.first + k;
					J.insertBack(row, col) = block(j, k);
				}
			}			
 		}

		J.finalize();
	}


	
	virtual void apply(typename self::out_pos_type& out,
	                   const typename self::in_pos_type& in ) {
		assert( out.size() == source.getValue().size() );
		
		for(unsigned i = 0, n = source.getValue().size(); i < n; ++i) {
			const source_type& s = source.getValue()[i];
			out[ i ] = se3::prod( in[ s.first ], s.second );
		}
		
	}
};


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
