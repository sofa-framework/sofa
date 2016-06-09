
#ifndef COMPLIANT_ConstantAssembledMapping_H
#define COMPLIANT_ConstantAssembledMapping_H

#include "AssembledMapping.h"

namespace sofa {
	namespace component {
		namespace mapping {

            /// When its parameters won't change, a Mapping can be *constant*
            /// i.e. its Jacobian and Hessian (for geometric stiffness)
            /// can be precomputed once for all.
			template<class In, class Out>
            class ConstantAssembledMapping : public AssembledMapping<In, Out>
            {

			public:

                SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE2(ConstantAssembledMapping,In,Out), SOFA_TEMPLATE2(AssembledMapping,In,Out));

                Data<bool> d_constant; ///< If constant, the Jacobian and Hessian are build once for all during 'init' (false by default)

                virtual void init() {
                    if( d_constant.getValue() )
                    {
                        in_pos_type in_pos = this->in_pos();
                        this->assemble( in_pos );
                        assemble_hessian( in_pos );
                    }

                    Inherit1::init();
				}

				virtual void apply(const core::MechanicalParams*,
                                   Data<typename Inherit1::OutVecCoord>& out,
                                   const Data<typename Inherit1::InVecCoord>& in) {
					out_pos_type out_pos(out);
					in_pos_type in_pos(in);
	  
                    this->apply(out_pos, in_pos);
                    if( !d_constant.getValue() ) this->assemble( in_pos );
				}

                virtual void updateK( const core::MechanicalParams* /*mparams*/, core::ConstMultiVecDerivId childForce ) {
                    in_pos_type in_pos = this->in_pos();
                    if( !d_constant.getValue() ) assemble_hessian( in_pos );
                    this->assemble_geometric( in_pos, this->out_force( childForce ) );
                }
	

			protected:


                ConstantAssembledMapping()
                    : d_constant( initData(&d_constant, false, "constant", "Can the Jacobian and Hessian be precomputed?") )
                {}


//				enum {Nin = In::deriv_total_size,
//                      Nout = Out::deriv_total_size };

                typedef typename Inherit1::in_pos_type in_pos_type;
                typedef typename Inherit1::out_pos_type out_pos_type;
//                typedef typename Inherit1::out_force_type out_force_type;
//                typedef typename Inherit1::in_vel_type in_vel_type;
                
                // to remove some warnings
                virtual void apply(out_pos_type& out, const in_pos_type& in ) = 0;

                /// The Hessian can be constant, while geometric stiffness is not
                /// K = Hessian * out_force
                /// In a constant mapping, the Hessian can be precomputed during 'init'
                /// and then be used in 'assemble_geometric' to perform Hessian * out_force.
                /// Note the Hessian type is not trivial and the user can choose what is best in his case.
                virtual void assemble_hessian( const in_pos_type& /*in*/ ) {}

			};

		}
	}
}



#endif
