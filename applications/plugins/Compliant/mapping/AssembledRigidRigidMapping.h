/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_AssembledRigidRigidMapping_H
#define SOFA_COMPONENT_MAPPING_AssembledRigidRigidMapping_H

#include "AssembledMapping.h"
#include <Compliant/config.h>

#include "../utils/se3.h"
#include <sofa/helper/pair.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/helper/decompose.h>

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
		: source(initData(&source, "source", "input dof and rigid offset for each output dof" )),
        geometricStiffness(initData(&geometricStiffness,
                               0,
                               "geometricStiffness",
                               "assemble (and use) geometric stiffness (0=no GS, 1=non symmetric, 2=symmetrized)")) {
                
    }



    typedef std::pair<unsigned, typename TIn::Coord> source_type;
    typedef helper::vector< source_type > source_vectype;
    Data< helper::vector< source_type > > source;

    Data<int> geometricStiffness;

    typedef typename TIn::Real Real;


 public:

	void init() {
	  const unsigned n = source.getValue().size();
	  if(this->getToModel()->getSize() != n) {
		serr << "init: output size does not match 'source' data, auto-resizing " << n << sendl;
		this->getToModel()->resize( n );
	  }

	  // must resize first, otherwise segfault lol (!??!?!)
	  this->AssembledMapping<TIn, TOut>::init();
	}

	
  protected:
	typedef SE3< typename TIn::Real > se3;
  
	typedef AssembledRigidRigidMapping self;

	
    virtual void assemble_geometric(const typename self::in_pos_type& in_pos,
                                    const typename self::out_force_type& out_force) {

        unsigned geomStiff = geometricStiffness.getValue();

        // we're done
        if( !geomStiff ) return;


        const source_vectype& src = source.getValue();

        // sorted in-out
        typedef std::map<unsigned, helper::vector<unsigned> > in_out_type;
        in_out_type in_out;

        // wahoo it is heavy, can't we find lighter?
		// max: probably we can cache it and rebuild only when needed,
		// which is most likely never
        for(unsigned i = 0, n = src.size(); i < n; ++i) {
            const source_type& s = src[i];
            in_out[ s.first ].push_back(i);
        }

        typedef typename self::geometric_type::CompressedMatrix matrix_type;
        matrix_type& dJ = this->geometric.compressedMatrix;

        dJ.resize( 6 * in_pos.size(),
                   6 * in_pos.size() );          
        dJ.reserve( 9 * src.size() );

        for(in_out_type::const_iterator it = in_out.begin(), end = in_out.end();
            it != end; ++it) {

            const unsigned parentIdx = it->first;

            defaulttype::Mat<3,3,Real> block;

            for( unsigned int w=0 ; w<it->second.size() ; ++w )
            {
                const unsigned i = it->second[w];

                const source_type& s = src[i];
                assert( it->first == s.first );

                const typename TOut::Deriv& lambda = out_force[i];
                const typename TOut::Deriv::Vec3& f = lambda.getLinear();

                const typename TOut::Deriv::Quat& R = in_pos[ parentIdx ].getOrientation();
                const typename TOut::Deriv::Vec3& t = s.second.getCenter();
                const typename TOut::Deriv::Vec3& Rt = R.rotate( t );

                block += defaulttype::crossProductMatrix<Real>( f ) * defaulttype::crossProductMatrix<Real>( Rt );
            }

            if( geomStiff == 2 )
            {
                block.symmetrize(); // symmetrization
                helper::Decompose<Real>::NSDProjection( block ); // negative, semi-definite projection
            }

			for(unsigned j = 0; j < 3; ++j) {
                
                const unsigned row = 6 * parentIdx + 3 + j;
                
				dJ.startVec( row );
				
				for(unsigned k = 0; k < 3; ++k) {
                    const unsigned col = 6 * parentIdx + 3 + k;
                    
                    if( block(j, k) ) dJ.insertBack(row, col) = block[j][k];
				}
			}			
 		}

        dJ.finalize();

    }

    
    // virtual void applyDJT(const core::MechanicalParams* mparams,
    //                       core::MultiVecDerivId inForce,
    //                       core::ConstMultiVecDerivId /* inDx */ ) {
    //     std::cout << "PARANOID TEST YO" << std::endl;
        
    //     const Data<typename self::InVecDeriv>& inDx =
    //         *mparams->readDx(this->fromModel);
            
    //     const core::State<TIn>* from_read = this->getFromModel();
    //     core::State<TIn>* from_write = this->getFromModel();

    //     typename self::in_vel_type lvalue( *inForce[from_write].write() );

    //     typename self::in_pos_type in_pos = this->in_pos();
    //     typename self::out_force_type out_force = this->out_force();

    //     for(unsigned i = 0, n = source.getValue().size(); i < n; ++i) {
    //         const source_type& s = source.getValue()[i];

    //         const typename TOut::Deriv& lambda = out_force[i];
    //         const typename TOut::Deriv::Vec3& f = lambda.getLinear();

    //         const typename TOut::Deriv::Quat& R = in_pos[ s.first ].getOrientation();
    //         const typename TOut::Deriv::Vec3& t = s.second.getCenter();

    //         const typename TOut::Deriv::Vec3& Rt = R.rotate( t );
    //         const typename TIn::Deriv::Vec3& omega = inDx.getValue()[ s.first ].getAngular();
            
    //         lvalue[s.first].getAngular() -= TIn::crosscross(f, omega, Rt) * mparams->kFactor();
    //     }
      
    // }



	virtual void assemble( const typename self::in_pos_type& in_pos ) {

		typename self::jacobian_type::CompressedMatrix& J = this->jacobian.compressedMatrix;

        const source_vectype& src = source.getValue();

		assert( in_pos.size() );
        assert( src.size() );
		
        J.resize(6 * src.size(),
                 6 * in_pos.size() );
        J.reserve( 36 * src.size() );
		
        for(unsigned i = 0, n = src.size(); i < n; ++i) {
            const source_type& s = src[i];
			
            typename se3::mat66 block = se3::dR(s.second, in_pos[ s.first ] );
			
			for(unsigned j = 0; j < 6; ++j) {
				unsigned row = 6 * i + j;
				
				J.startVec( row );
				
				for(unsigned k = 0; k < 6; ++k) {
                    unsigned col = 6 * s.first + k;
					if( block(j, k) ) {
                        J.insertBack(row, col) = block(j, k);
                    }
				}
			}			
 		}

		J.finalize();

	}


	
	virtual void apply(typename self::out_pos_type& out,
	                   const typename self::in_pos_type& in ) {

        const source_vectype& src = source.getValue();

        assert( out.size() == src.size() );
		
        for(unsigned i = 0, n = src.size(); i < n; ++i) {
            const source_type& s = src[i];
            out[ i ] = se3::prod( in[ s.first ], s.second );
		}
		
	}


    virtual void updateForceMask()
    {
        const source_vectype& src = source.getValue();

        for(unsigned i = 0, n = src.size(); i < n; ++i)
        {
            if( this->maskTo->getEntry(i) )
            {
                const source_type& s = src[i];
                this->maskFrom->insertEntry(s.first);
            }
        }
    }

};


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
