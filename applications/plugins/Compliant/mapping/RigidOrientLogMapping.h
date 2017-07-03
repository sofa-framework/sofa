#ifndef COMPLIANT_RIGIDORIENTLOGMAPPING_H
#define COMPLIANT_RIGIDORIENTLOGMAPPING_H

#include <Compliant/mapping/CompliantMapping.h>
#include <Compliant/utils/se3.h>

namespace sofa {

namespace component {

namespace mapping {


namespace detail {
template<class U> using rigid_types = defaulttype::StdRigidTypes<3, U>;
template<class U> using vec3_types = defaulttype::StdVectorTypes< defaulttype::Vec<3, U>, 
                                                                  defaulttype::Vec<3, U>, U>;
}

template<class U>
class RigidOrientLogMapping 
    : public CompliantMapping< detail::vec3_types<U> (detail::rigid_types<U> ) > {
    
    using base = typename RigidOrientLogMapping::CompliantMapping;
    using output_types = typename base::output_types;
    
    using input_types = typename base::template input_types<0>;
    using jacobian_type = typename base::template jacobian_type<input_types>;        

    using typename base::error;
public:

    SOFA_CLASS(SOFA_TEMPLATE(RigidOrientLogMapping, U), 
			   SOFA_TEMPLATE(CompliantMapping, typename base::signature_type));

    Data<bool> use_dlog;    

    RigidOrientLogMapping() : use_dlog(initData(&use_dlog, false, "use_dlog", "use dlog")) { }
    
    using se3 = ::SE3<U>;
    
protected:

    void check(coord_view< output_types > out_pos, coord_view< const input_types > in_pos) {
        if( out_pos.size() != in_pos.size() ) {
            throw error("output dofs size error");
        }
    }
    
    virtual void apply(const core::MechanicalParams*,
                       coord_view< output_types > out_pos,
                       coord_view< const input_types > in_pos) {
        check(out_pos, in_pos);
        
        for(unsigned i = 0, n = in_pos.size(); i < n; ++i){
            // use maps to convert RigidDeriv to Vec6
            se3::map(out_pos[i]) = se3::log( se3::rotation(in_pos[i]) );
        }
    }

    
    virtual void assemble(jacobian_type& jacobian, 
                          coord_view< const input_types > in_pos) {

        auto& J = jacobian.compressedMatrix;
        J.resize( 3 * in_pos.size(), 6 * in_pos.size() );
        
        for(unsigned i = 0, n = in_pos.size(); i < n; ++i) {
            typename se3::mat36 block;

            if(use_dlog.getValue()) {
                block.template rightCols<3>().setIdentity();
            } else {
                block.template rightCols<3>() = se3::dlog( se3::rotation(in_pos[i]));
            }
                
            for(unsigned k = 0; k < 3; ++k) {
                const unsigned row = 3 * i + k;
                J.startVec(row);
                    
                for(unsigned j = 0; j < 6; ++j) {
                    const unsigned col = 6 * i + j;
                    if(block(k, j)) {
                        J.insertBack(row, col) = block(k, j);
                    }
                }
            }
        }

        J.finalize();
    }
    
};

    



}

}

}



#endif
