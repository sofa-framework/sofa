#ifndef COMPLIANT_RIGIDLOGMAPPING_H
#define COMPLIANT_RIGIDLOGMAPPING_H

#include <Compliant/mapping/CompliantMapping.h>
#include <Compliant/utils/se3.h>

namespace sofa {

namespace component {

namespace mapping {


namespace detail {
template<class U> using rigid_types = defaulttype::StdRigidTypes<3, U>;
template<class U> using vec6_types = defaulttype::StdVectorTypes< defaulttype::Vec<6, U>, 
                                                                  defaulttype::Vec<6, U>, U>;
}

template<class U>
class RigidLogMapping 
    : public CompliantMapping< detail::vec6_types<U> (detail::rigid_types<U> ) > {

    using base = typename RigidLogMapping::CompliantMapping;
    using output_types = typename base::output_types;
    
    using input_types = typename base::template input_types<0>;
    using jacobian_type = typename base::template jacobian_type<input_types>;        
    
public:

    SOFA_CLASS(SOFA_TEMPLATE(RigidLogMapping, U), 
			   SOFA_TEMPLATE(CompliantMapping, typename base::signature_type));
    
    using kind_type = int;
    enum : kind_type {
        SO3xR3 = 0,
        SE3 = 1,
    };

    
    struct data_type {
        
        Data<bool> use_dlog;
        Data<kind_type> kind;

        data_type(RigidLogMapping* owner)
            : use_dlog( owner->initData(&use_dlog, true, "use_dlog", "use exact logarithm derivative" ) ),
              kind( owner->initData(&kind, 0, "kind", "0: use SO(3)xR3 logarithm, 1: use SE(3) logarithm" ) ) {

        }
        
    } data;

    RigidLogMapping() : data(this) { }

    using se3 = ::SE3<U>;
    
protected:
    virtual void apply(const core::MechanicalParams*,
                       coord_view< output_types > out_pos,
                       coord_view< const input_types > in_pos) {
        
        if( out_pos.size() != in_pos.size() ) {
            msg_error() << "output dofs size error, aborting";
            return;
        }

        if( data.kind.getValue() == SO3xR3 ) {
            for(unsigned i = 0, n = in_pos.size(); i < n; ++i){
                // use maps to convert RigidDeriv to Vec6
                se3::map(out_pos[i]) = se3::map(se3::product_log(in_pos[i]));
            }
        } else {
            msg_error() << "SE3 log not implemented, aborting";
            return;
        }
    }

    
    virtual void assemble(jacobian_type& jacobian, 
                          coord_view< const input_types > in_pos) {

        auto& J = jacobian.compressedMatrix;
        J.resize( 6 * in_pos.size(), 6 * in_pos.size() );

        if(!data.use_dlog.getValue()) {
            J.setIdentity();
            return;
        }
        
        if( data.kind.getValue() == SO3xR3 ) { 
            for(unsigned i = 0, n = in_pos.size(); i < n; ++i) {
                const typename se3::mat66 block = se3::product_dlog(in_pos[i]);
                
                for(unsigned k = 0; k < 6; ++k) {
                    const unsigned row = 6 * i + k;
                    J.startVec(row);
                    
                    for(unsigned j = 0; j < 6; ++j) {
                        const unsigned col = 6 * i + j;
                        J.insertBack(row, col) = block(k, j);
                    }
                }
            }
        } else {
            msg_error() << "SE3 log not implemented, aborting";
            return;
        }
        
        J.finalize();
    }
    
};

    



}

}

}



#endif
