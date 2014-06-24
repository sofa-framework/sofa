
#ifndef COMPLIANT_ASSEMBLEDMAPPING_H
#define COMPLIANT_ASSEMBLEDMAPPING_H

#include <sofa/core/Mapping.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>

namespace sofa {
	namespace component {
		namespace mapping {

			// assembled mapping base class: derived classes only need to
			// implement apply and assemble
			template<class In, class Out>
			class AssembledMapping : public core::Mapping<In, Out> {

				typedef AssembledMapping self;
	
				typedef vector<sofa::defaulttype::BaseMatrix*> js_type;
				js_type js;
			public:

				SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE2(AssembledMapping,In,Out), SOFA_TEMPLATE2(core::Mapping,In,Out));
	
				// TODO make this final ?
				void init() {
					js.resize(1);
					js[0] = &jacobian;
					// assemble( in_pos() );
	  
					typedef typename core::Mapping<In, Out> base; // fixes g++-4.4
					base::init();
				}
	
				const helper::vector<sofa::defaulttype::BaseMatrix*>* getJs() {
					assert( !js.empty() );
					return &js;
				}

				const sofa::defaulttype::BaseMatrix* getJ() { return &jacobian; }
	
				virtual void apply(const core::MechanicalParams*,
				                   Data<typename self::OutVecCoord>& out, 
				                   const Data<typename self::InVecCoord>& in) {
					out_pos_type out_pos(out);
					in_pos_type in_pos(in);
	  
					apply(out_pos, in_pos);
					assemble( in_pos );
				}
	
				virtual void applyJ(const core::MechanicalParams*,
				                    Data<typename self::OutVecDeriv>& out, 
				                    const Data<typename self::InVecDeriv>& in) {
					if( jacobian.rowSize() > 0 ) jacobian.mult(out, in);
				}

				void debug() {
					std::cerr << this->getClassName() << std::endl;
					std::cerr << "from: " <<  this->getFromModel()->getContext()->getName() 
							  << "/" << this->getFromModel()->getName() << std::endl;
					std::cerr << "to: " <<  this->getToModel()->getContext()->getName() 
							  << "/" << this->getToModel()->getName() << std::endl;
					std::cerr << std::endl;
				}

				virtual void applyJT(const core::MechanicalParams*,			     
				                     Data<typename self::InVecDeriv>& in, 
				                     const Data<typename self::OutVecDeriv>& out) {
					// debug();
					if( jacobian.rowSize() > 0 ) jacobian.addMultTranspose(in, out);
				}

				virtual void applyJT(const core::ConstraintParams*,
				                     Data< typename self::InMatrixDeriv>& , 
				                     const Data<typename self::OutMatrixDeriv>& ) {
					// throw std::logic_error("not implemented");
					// if( jacobian.rowSize() > 0 ) jacobian.addMultTranspose(in, out);
				}


			protected:
				enum {Nin = In::deriv_total_size, Nout = Out::deriv_total_size };

				typedef helper::ReadAccessor< Data< typename self::InVecCoord > > in_pos_type;
				typedef helper::WriteAccessor< Data< typename self::OutVecCoord > > out_pos_type;
	
				in_pos_type in_pos() {

					const core::State<In>* fromModel = this->getFromModel();
					assert( fromModel );
					
					core::ConstMultiVecCoordId inPos = core::ConstVecCoordId::position();
	  
					const typename self::InDataVecCoord* in = inPos[fromModel].read();
					
					return *in;
				}
	

				virtual void assemble( const in_pos_type& in ) = 0;
				virtual void apply(out_pos_type& out, const in_pos_type& in ) = 0;
	
				typedef linearsolver::EigenSparseMatrix<In, Out> jacobian_type;
				jacobian_type jacobian;
			};

		}
	}
}



#endif
