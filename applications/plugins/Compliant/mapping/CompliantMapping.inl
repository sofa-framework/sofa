#ifndef COMPLIANT_MAPPING_INL
#define COMPLIANT_MAPPING_INL

#include "CompliantMapping.h"

#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/template_name.h>

namespace sofa {

namespace component {

namespace mapping {


// this is the implementation class. you may jump straight to the end of it.
template<class TOut, class ... T>
template<std::size_t ... I>
struct CompliantMapping<TOut (T...) >::impl_type {

    static_assert( sizeof...(I) == sizeof...(T), "size mismatch");
    
    using self_type = CompliantMapping<TOut (T...) >;


    // construct links 
    static std::tuple< link_type<T>...> make_from_models(self_type* self) {

        static constexpr std::size_t N = sizeof...(I);

        // link names
        static std::string names[] = {
            (std::string("input") + (N == 1 ? std::string("") : std::to_string(I+1)))...
        };

        // link help messages
        static std::string help[] = {
            (std::string("mapping ") + std::to_string(I + 1)) + std::string("th input dofs")...
        };

        return std::tuple< link_type<T>... > { self->initLink(names[I].c_str(), help[I].c_str())... };
    }
    

    // construct input basestate vector
    static helper::vector<core::BaseState*> getFrom(self_type* self) {
        
        helper::vector<core::BaseState*> res;

        const int expand[] = {
            (res.push_back( std::get<I>(self->from_models)), 0) ... 
        };
        
        (void) expand;

        return res;
    }

    
    // construct input basemechanicalstate vector
    static helper::vector<core::behavior::BaseMechanicalState*> getMechFrom(self_type* self) {
        
        helper::vector<core::behavior::BaseMechanicalState*> res;

        const int expand[] = {
            (res.push_back( std::get<I>(self->from_models)->toBaseMechanicalState()), 0) ... 
        };
        
        (void) expand;

        return res;
    }

    
    struct not_implemented : std::runtime_error {
        not_implemented(const char* what)
            : std::runtime_error(std::string("not implemented: ") + what) {
        }
    };


    // init checks for ith input model
    template<std::size_t J>
    static void init_checks_from_model(self_type* self) {
        auto& from = std::get<J>(self->from_models);
        
        if( !from ) {
            msg_error(self) << "error: unset input " << J + 1 << " (wrong type?)";
            return;
        }

        if( !from->getSize() ) {
            msg_error(self) << "error: empty dofs for input " << J + 1;
            return;
        }

    };
    

    // init checks for input models
    static void init_checks(self_type* self) {

        const int expand[] = {
            (init_checks_from_model<I>(self), 0)...
        }; (void) expand;
        
    }



    template<class U>
    static view<U> vector_view(std::vector<U>& v) {
        return {v.data(), v.size()};
    }
    
    
    // a view on a Data
    template<class U>
    static view< const typename U::value_type > data_view(const Data<U>* data) {
        const std::size_t size = data->getValue().size();
        
        return {data->getValue().data(), size};
    }
    

    template<class U>
    static view< typename U::value_type > data_view(Data<U>* data) {
        const std::size_t size = data->getValue().size();

        // TODO do this cleanly without triggering data edition hell
        typename U::value_type* ptr = const_cast<typename U::value_type*>(data->getValue().data());
        return {ptr, size};
    }
    

    template<std::size_t J>
    static coord_view< const input_types<J> > in_coord_view(self_type* self, core::ConstMultiVecCoordId id) {
        return data_view( id[ std::get<J>(self->from_models).get() ].read() );
    }

    static coord_view< output_types > out_coord_view(self_type* self, core::MultiVecCoordId id) {
        return data_view( id[ self->to_model.get() ].write() );
    }

    static deriv_view< const output_types > out_deriv_view(self_type* self, core::ConstMultiVecDerivId id) {
        return data_view( id[ self->to_model.get() ].read() );
    }
    
    
    
    
    static void apply_assemble (self_type* self, const core::MechanicalParams* mparams,
                                core::MultiVecCoordId out_pos_id,
                                core::ConstMultiVecCoordId in_pos_id) {
        // apply
        self->apply(mparams,
                    out_coord_view(self, out_pos_id),
                    in_coord_view<I>(self, in_pos_id)...);


        // wtf is this nonsense
#ifdef SOFA_USE_MASK
        self->m_forceMaskNewStep = true;
#endif

        // assemble jacobians
        self->assemble(std::get<I>(self->jacobians)..., in_coord_view<I>(self, in_pos_id)...);
    }
    

    static void update_gs(self_type* self, core::ConstMultiVecDerivId out_force_id) {
        
        const core::ConstMultiVecCoordId in_pos_id = core::VecId::position();
        self->assemble_gs(self->geometric,
                          out_deriv_view(self, out_force_id),
                          in_coord_view<I>(self, in_pos_id)...);
    }


    template<std::size_t J>
    static void add_mult_ith_jacobian(self_type* self,
                                      Data< helper::vector< typename TOut::Deriv > >* out_vel_data,
                                      core::ConstMultiVecDerivId in_vel_id) {
        const auto in_vel_data = in_vel_id[std::get<J>(self->from_models).get()].read();
        std::get<J>(self->jacobians).addMult(*out_vel_data, *in_vel_data);
    }
    

    static void apply_jacobian(self_type* self, core::MultiVecDerivId out_vel_id,
                               core::ConstMultiVecDerivId in_vel_id) {

        auto* out_vel_data = out_vel_id[ self->to_model.get()].write();

        // set output to zero
        auto out_vel_view = data_view(out_vel_data);
        std::fill(out_vel_view.begin(), out_vel_view.end(), typename TOut::Deriv() );
        
        // multiplication
        const int expand[] = {
            (add_mult_ith_jacobian<I>(self, out_vel_data, in_vel_id), 0)...
        }; (void) expand;
    }
    

    template<std::size_t J>    
    static void add_mult_ith_jacobian_transpose(self_type* self,
                                                core::MultiVecDerivId in_force_id,
                                                core::ConstMultiVecDerivId out_force_id) {
        const auto* out_force_data = out_force_id[self->to_model.get()].read();
        auto* in_force_data = in_force_id[std::get<J>(self->from_models).get()].write();
        
        std::get<J>(self->jacobians).addMultTranspose(*in_force_data, *out_force_data);
    }

    

    static void apply_jacobian_transpose(self_type* self, core::MultiVecDerivId in_force_id,
                                         core::ConstMultiVecDerivId out_force_id) {
        // transpose multiplication
        const int expand[] = {
            (add_mult_ith_jacobian_transpose<I>(self, in_force_id, out_force_id), 0)...
        }; (void) expand;
        
    }
    



    // fill rhs segment with data from given vec id
    template<std::size_t J>
    static void fetch_rhs(self_type* self,
                          vec<SReal>& rhs, unsigned& off, core::ConstMultiVecDerivId in_vel_id, SReal kfactor) {

        const auto* in_vel_data = in_vel_id[std::get<J>(self->from_models).get()].read();
        auto in_vel_view = data_view(in_vel_data);
        
        using real_type = typename input_types<J>::Real;
        
        const real_type* ptr = reinterpret_cast<const real_type*>(in_vel_view.data());
        const std::size_t size = in_vel_view.size() * input_types<J>::deriv_total_size;
        
        // view mstate vector as eigen type            
        Eigen::Map<const vec<real_type>> map(ptr, size);
        
        rhs.segment(off, map.size()) = (kfactor * map).template cast<SReal>();
        off += map.size();
    }

    
    // fill back vec id from rhs segment
    template<std::size_t J>            
    static void dispatch_res(self_type* self,
                             core::MultiVecDerivId in_vel_id, unsigned& off, const vec<SReal>& res) {
        
        auto* in_vel_data = in_vel_id[std::get<J>(self->from_models).get()].write();
        auto in_vel_view = data_view(in_vel_data);
            
        using real_type = typename input_types<J>::Real;
            
        real_type* ptr = reinterpret_cast<real_type*>(in_vel_view.data());
        const std::size_t size = in_vel_view.size() * input_types<J>::deriv_total_size;

        // view mstate vector as eigen type
        Eigen::Map<vec<real_type>> map(ptr, size);
            
        map = res.segment(off, map.size()).template cast<real_type>();
        off += map.size();
        
    }


    // apply geometric stiffness
    static void apply_DJT(self_type* self, core::MultiVecDerivId in_vel_id, SReal kfactor) {

        const auto& cm = self->geometric.compressedMatrix;
        assert(cm.rows() == cm.cols());
        
        if( !cm.nonZeros() ) return;

        // resize temporary vectors
        self->rhs.resize( cm.rows() );
        self->res.resize( cm.rows() );
        
        // fill rhs with data from input vecids
        {
            unsigned off = 0;
            const int expand[] = {
                (fetch_rhs<I>(self, self->rhs, off, in_vel_id, kfactor), 0)...
            }; (void) expand;

            assert(off == (unsigned)cm.rows());
        }

        // actual multiplication
        self->res.noalias() = cm * self->rhs;


        // dispatch result back to vecids
        {
            unsigned off = 0;
            const int expand[] = {
                (dispatch_res<I>(self, in_vel_id, off, self->res), 0)...
            }; (void) expand;
            assert(off == (unsigned)cm.rows());
        }
        
        
    }

    template<std::size_t J>
    static void debug_jacobian(self_type* self, std::ostream& out = std::clog) {
        out << "debug jacobian: " << J << std::endl;
        out << std::get<J>(self->jacobians).compressedMatrix << std::endl;
    }

    static void update_js(self_type* self) {
        
        self->js.clear();
        
        const int expand [] = {
            (self->js.push_back( &std::get<I>(self->jacobians) ), 0)...
        }; (void) expand;
        
    }

    static void update_force_mask(self_type* self) {
        const int expand[] = {
            ( std::get<I>(self->from_models)->toBaseMechanicalState()->forceMask
              .assign(std::get<I>(self->from_models)->getSize(), true),
              0)...
        };
        (void) expand;
    }
    
    
};


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////


template<class TOut, class ... T>
CompliantMapping<TOut (T...) >::CompliantMapping()
    : from_models( impl::make_from_models(this) ),
      to_model(initLink("output", "output object")) {
    
}

namespace detail {
template<class ...> struct placeholder;
}

template<class TOut, class ... T>
std::string CompliantMapping<TOut (T...) >::templateName(const CompliantMapping* ) {
    const detail::placeholder<T..., TOut>* dummy = nullptr;
    return helper::template_name( dummy );
}

template<class TOut, class ... T>
std::string CompliantMapping<TOut (T...) >::getTemplateName() const {
    return templateName(this);
}


template<class TOut, class ... T>
helper::vector<core::BaseState*> CompliantMapping<TOut (T...) >::getFrom() {
    return impl::getFrom(this);
}

template<class TOut, class ... T>
helper::vector<core::behavior::BaseMechanicalState*> CompliantMapping<TOut (T...) >::getMechFrom() {
    return impl::getMechFrom(this);
}


template<class TOut, class ... T>
helper::vector<core::BaseState* > CompliantMapping<TOut (T...) >::getTo() {
    helper::vector<core::BaseState*> res(1);
    res[0] = to_model;
    return res;
}

template<class TOut, class ... T>
helper::vector<core::behavior::BaseMechanicalState* > CompliantMapping<TOut (T...) >::getMechTo() {
    helper::vector<core::behavior::BaseMechanicalState*> res(1);
    res[0] = to_model->toBaseMechanicalState();    
    return res;
}


#ifdef SOFA_USE_MASK
template<class TOut, class ... T>
void CompliantMapping<TOut (T...) >::updateForceMask() {
    impl::update_force_mask(this);
}
#endif


template<class TOut, class ... T>
void CompliantMapping<TOut (T...) >::disable() {
    // lolwat
}

template<class TOut, class ... T>
void CompliantMapping<TOut (T...) >::init() {
    assert( (this->getTo().size() == 1) && 
            "multi-mapping to multiple output dofs unimplemented" );
    
    core::BaseMapping::init();

    impl::init_checks(this);

    if(!to_model) {
        msg_error() << "output model not set, aborting (wrong type?)";
        return;
    }

    if(!to_model->getSize()) {
        msg_warning() << "empty output model";
    }

}


template<class TOut, class ... T>
void CompliantMapping<TOut (T...) >::applyJT(const core::ConstraintParams* /*mparams*/, 
                                             core::MultiMatrixDerivId /*inConst*/, 
                                             core::ConstMultiMatrixDerivId /*outConst*/) {
    throw typename impl::not_implemented(__func__);
}

template<class TOut, class ... T>
void CompliantMapping<TOut (T...) >::computeAccFromMapping(const core::MechanicalParams* /*mparams*/, 
                                                           core::MultiVecDerivId /*outAcc*/, 
                                                           core::ConstMultiVecDerivId /*inVel*/, 
                                                           core::ConstMultiVecDerivId /*inAcc*/ ) {
    throw typename impl::not_implemented(__func__);
}


template<class TOut, class ... T>
void CompliantMapping<TOut (T...) >::apply(const core::MechanicalParams* mparams,
                                           core::MultiVecCoordId out_pos_id,
                                           core::ConstMultiVecCoordId in_pos_id)  {
    try{
        impl::apply_assemble(this, mparams, out_pos_id, in_pos_id);
    } catch( error& e ) {
        msg_error() << e.what() << ", aborting";
    }

}

template<class TOut, class ... T>
void CompliantMapping<TOut (T...) >::updateK(const core::MechanicalParams* /*mparams*/,
                                             core::ConstMultiVecDerivId out_force_id ) {
    impl::update_gs(this, out_force_id);
}


template<class TOut, class ... T>
void CompliantMapping<TOut (T...) >::applyJ(const core::MechanicalParams* /*mparams*/,
                                            core::MultiVecDerivId out_vel_id,
                                            core::ConstMultiVecDerivId in_vel_id ) {
    impl::apply_jacobian(this, out_vel_id, in_vel_id );
}


template<class TOut, class ... T>
void CompliantMapping<TOut (T...) >::applyJT (const core::MechanicalParams* /*mparams*/,
                                              core::MultiVecDerivId in_force_id,
                                              core::ConstMultiVecDerivId out_force_id) {
    impl::apply_jacobian_transpose(this, in_force_id, out_force_id );
    
#ifdef SOFA_USE_MASK
    if( this->m_forceMaskNewStep ){
        this->m_forceMaskNewStep = false;
        updateForceMask();
    }
#endif
        
}


template<class TOut, class ... T>
void CompliantMapping<TOut (T...) >::applyDJT(const core::MechanicalParams* mparams,
                                              core::MultiVecDerivId in_vel_id,
                                              core::ConstMultiVecDerivId /*out_force_id*/) {
    impl::apply_DJT(this, in_vel_id, mparams->kFactor());
}



template<class TOut, class ... T>
const helper::vector<sofa::defaulttype::BaseMatrix*>* CompliantMapping<TOut (T...) >::getJs() {
    impl::update_js(this);
    return &js;
}


template<class TOut, class ... T>
const defaulttype::BaseMatrix* CompliantMapping<TOut (T...) >::getK() {
    if( geometric.compressedMatrix.nonZeros() ) return &geometric;
    else return nullptr;
};

template<class TOut, class ... T>
void CompliantMapping<TOut (T...) >::assemble_gs(geometric_type& /*gs*/,
                                                 deriv_view<const TOut> /*out_force*/,
                                                 coord_view<const T>... /*in_pos*/) {
    // default is no gs
}


}
}
}

#endif


