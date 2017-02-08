#ifndef COMPLIANT_MAPPING_INL
#define COMPLIANT_MAPPING_INL

#include "CompliantMapping.h"

namespace sofa {

namespace component {

namespace mapping {


template<class TOut, class ... T>
struct CompliantMapping<TOut (T...) >::impl {

    using self_type = CompliantMapping<TOut (T...) >;
    
    template<std::size_t ... I>
    static std::tuple< link_type<T>...> make_from_models(self_type* self, seq<I...>) {
        
        static std::string names[] = {
            (std::string("input") + std::to_string(I+1))...
        };

        static std::string help[] = {
            (std::string("help") + std::to_string(I+1))...
        };

        // TODO add alias 'input' for 'input1' when sizeof...(T) == 1
        return std::tuple< link_type<T>... > { self->initLink(names[I].c_str(), help[I].c_str())... };
    }



    template<std::size_t ...I>
    static helper::vector<core::BaseState*> getFrom(self_type* self, seq<I...> ) {
        
        helper::vector<core::BaseState*> res;

        const int expand[] = {
            (std::copy( std::get<I>(self->from_models).begin(),
                        std::get<I>(self->from_models).end(),
                        std::back_inserter(res) ), 0)...
        };

        (void) expand;

        return res;
    }


    struct to_mechanical_state {

        template<class U>
        core::behavior::BaseMechanicalState* operator()(const U& x) const {
            return x->toBaseMechanicalState();
        }
        
    };
    
    template<std::size_t ...I>
    static helper::vector<core::behavior::BaseMechanicalState*> getMechFrom(self_type* self, seq<I...> ) {
        
        helper::vector<core::behavior::BaseMechanicalState*> res;

        const int expand[] = {
            (std::transform( std::get<I>(self->from_models).begin(),
                             std::get<I>(self->from_models).end(),
                             std::back_inserter(res), to_mechanical_state() ), 0)...
        };
        (void) expand;

        // TODO filter out null values using std::remove_if
        
        return res;
    }


    struct not_implemented : std::runtime_error {
        not_implemented(const char* what)
            : std::runtime_error(std::string("not implemented: ") + what) {
        }
    };


    template<std::size_t  I>
    static void init_from_model(self_type* self) {
        auto& from = std::get<I>(self->from_models);
        
        if( from.size() == 0 ) {
            self->serr << "error: empty input " << I + 1 << self->sendl;
        }

        for(unsigned i = 0, n = from.size(); i < n; ++i) {
            if(!from[i]) {
                self->serr << "error: input " << i
                           << " for type " << I + 1
                           << " is invalid (wrong type?)" << self->sendl;
            }
        }
        
        
    };
    
    

    template<std::size_t ... I>
    static void init(self_type* self, seq<I...>) {

        const int expand[] = {
            (init_from_model<I>(self), 0)...
        }; (void) expand;
        
    }



    template<class U>
    static view<U> vector_view(std::vector<U>& v) {
        return {v.data(), v.size()};
    }


    template<std::size_t I>
    static void update_ith_in_pos(self_type* self, const core::ConstMultiVecCoordId& id) {
        auto& ith_in_pos = std::get<I>(self->in_pos);
        const auto& ith_from_models = std::get<I>(self->from_models);

        const unsigned n = ith_from_models.size();
        assert(n && "some input type has no model");
        
        ith_in_pos.clear();
        ith_in_pos.reserve(n);
        
        for(unsigned i = 0; i < n; ++i) {
            ith_in_pos.push_back( data_view( id[ith_from_models[i]].read() ) );
        }
        
    }
                       
    

    

    template<std::size_t ... I>
    static void update_in_pos(self_type* self, const core::ConstMultiVecCoordId& id, seq<I...> ) {

        // resize
        const int expand[] = {
            ( update_ith_in_pos<I>(self, id), 0)...
        };
        (void) expand;
        
    }


    

    template<class U>
    static view< typename U::value_type > data_view(const Data<U>* data) {
        std::size_t size = data->getValue().size();

        // mouhahahahahahaha !
        typename U::value_type* ptr = const_cast<typename U::value_type*>(data->getValue().data());
        
        return {ptr, size};
    }

    

    template<std::size_t ... I>
    static void apply (self_type* self, const core::MechanicalParams* mparams,
                core::MultiVecCoordId out_pos_id,
                core::ConstMultiVecCoordId in_pos_id,
                seq<I...> ) {

        update_in_pos(self, in_pos_id, make_sequence<T...>());

        coord_view<TOut> out_view = data_view( out_pos_id[ self->to_models[0] ].read() );
        
        std::tuple< view< coord_view<T> >... > in_view{
            vector_view(std::get<I>(self->in_pos)) ...
        };

        
        self->apply(mparams, out_view, std::get<I>(in_view)...);


        // resize jacobians
        const int expand[] = {
            (std::get<I>(self->jacobians).resize( std::get<I>(self->from_models).size() ), 0)...
        }; (void) expand;
        
        // build jacobian view
        std::tuple< view< jacobian_type<T> >... > jacobian_view{
            vector_view(std::get<I>(self->jacobians)) ...
                };
        
        // wtf is this nonsense
#ifdef SOFA_USE_MASK
        self->m_forceMaskNewStep = true;
#endif
        
        self->assemble(std::get<I>(jacobian_view)..., std::get<I>(in_view)...);
    }
    


    template<std::size_t ... I>
    static void update_gs(self_type* self, core::ConstMultiVecDerivId out_force_id, seq<I...>) {

        // TODO assert in_pos is set correctly
        auto* out_force_data = out_force_id[self->to_models[0]].read();
        auto out_force_view = data_view(out_force_data);
        
        std::tuple< view< coord_view<T> >... > in_view(
            vector_view(std::get<I>(self->in_pos)) ...
            );

        self->assemble_gs(self->geometric, out_force_view, std::get<I>(in_view)...);
    }



    template<std::size_t I>
    static void add_mult_ith_jacobian(self_type* self,
                                      Data< helper::vector< typename TOut::Deriv > >* out_vel_data,
                                      core::ConstMultiVecDerivId in_vel_id) {
        
        assert(std::get<I>(self->from_models).size() == std::get<I>(self->jacobians).size());
        
        for(unsigned i = 0, n = std::get<I>(self->jacobians).size(); i < n; ++i){
            auto in_vel_data = in_vel_id[std::get<I>(self->from_models)[i]].read();
            std::get<I>(self->jacobians)[i].addMult(*out_vel_data, *in_vel_data);
        }

    }
    

    template<std::size_t ... I>
    static void apply_jacobian(self_type* self, core::MultiVecDerivId out_vel_id,
                        core::ConstMultiVecDerivId in_vel_id,
                        seq<I...>) {

        auto* out_state = self->to_models[0];

        // TODO make sure this does not trigger funky stuff
        auto* out_vel_data = out_vel_id[out_state].write();
        
        // set output to zero
        auto out_vel_view = data_view(out_vel_data);
        assert(out_vel_view.size() == out_state->getSize());
        
        std::fill(out_vel_view.begin(), out_vel_view.end(), typename TOut::Deriv() );

        // multiplication
        const int expand[] = {
            (add_mult_ith_jacobian<I>(self, out_vel_data, in_vel_id), 0)...
        }; (void) expand;
        
    }
    

    template<std::size_t I>
    static void add_mult_ith_jacobian_transpose(self_type* self,
                                                core::MultiVecDerivId in_force_id,
                                                core::ConstMultiVecDerivId out_force_id) {
        assert(std::get<I>(self->from_models).size() == std::get<I>(self->jacobians).size());
        
        auto* out_force_data = out_force_id[self->to_models[0]].read();
        
        for(unsigned i = 0, n = std::get<I>(self->jacobians).size(); i < n; ++i){
            // TODO make sure 'write' does not trigger hell
            auto* in_force_data = in_force_id[std::get<I>(self->from_models)[i]].write();
            std::get<I>(self->jacobians)[i].addMultTranspose(*in_force_data, *out_force_data);
        }
        
    }

    

    template<std::size_t ... I>
    static void apply_jacobian_transpose(self_type* self, core::MultiVecDerivId in_force_id,
                                  core::ConstMultiVecDerivId out_force_id,
                                  seq<I...>) {
        // multiplication
        const int expand[] = {
            (add_mult_ith_jacobian_transpose<I>(self, in_force_id, out_force_id), 0)...
        }; (void) expand;
        
    }
    




    template<std::size_t I>
    static void fetch_rhs(self_type* self,
                          vec<SReal>& rhs, unsigned& off, core::ConstMultiVecDerivId in_vel_id, SReal kfactor) {

        for(unsigned i = 0, n = std::get<I>(self->from_models).size(); i < n; ++i) {
            auto* in_vel_data = in_vel_id[std::get<I>(self->from_models)[i]].read();
            auto in_vel_view = data_view(in_vel_data);

            using real_type = typename input_type<I>::Real;
            
            const real_type* ptr = reinterpret_cast<const real_type*>(in_vel_view.data());
            const std::size_t size = in_vel_view.size() * input_type<I>::deriv_total_size;

            // view mstate vector as eigen type            
            Eigen::Map<const vec<real_type>> map(ptr, size);
            
            rhs.segment(off, map.size()) = (kfactor * map).template cast<SReal>();
            off += map.size();
        }
        
    }

    template<std::size_t I>
    static void dispatch_res(self_type* self,
                             core::MultiVecDerivId in_vel_id, unsigned& off, const vec<SReal>& res) {

        for(unsigned i = 0, n = std::get<I>(self->from_models).size(); i < n; ++i) {
            auto* in_vel_data = in_vel_id[std::get<I>(self->from_models)[i]].write();
            auto in_vel_view = data_view(in_vel_data);

            using real_type = typename input_type<I>::Real;
            
            real_type* ptr = reinterpret_cast<real_type*>(in_vel_view.data());
            const std::size_t size = in_vel_view.size() * input_type<I>::deriv_total_size;

            // view mstate vector as eigen type
            Eigen::Map<vec<real_type>> map(ptr, size);
            
            map = res.segment(off, map.size()).template cast<real_type>();
            off += map.size();
        }
        
    }


    template<std::size_t ... I>
    static void apply_DJT(self_type* self, core::MultiVecDerivId in_vel_id, SReal kfactor, seq<I...> ) {

        const auto& cm = self->geometric.compressedMatrix;
        assert(cm.rows() == cm.cols());
        
        if( !cm.nonZeros() ) return;

        self->rhs.resize( cm.rows() );
        self->res.resize( cm.rows() );

        {
            unsigned off = 0;
            const int expand[] = {
                (fetch_rhs<I>(self, self->rhs, off, in_vel_id, kfactor), 0)...
            }; (void) expand;

            assert(off == (unsigned)cm.rows());
        }

        self->res.noalias() = cm * self->rhs;
        
        {
            unsigned off = 0;
            const int expand[] = {
                (dispatch_res<I>(self, in_vel_id, off, self->res), 0)...
            }; (void) expand;
            assert(off == (unsigned)cm.rows());
        }
        
        
    }



    struct js_helper {
        
        template<class U>
        sofa::defaulttype::BaseMatrix* operator()(U&& value) const {
            return &value;
        }
    };

    template<std::size_t I>
    static void debug_jacobians(self_type* self, std::ostream& out = std::clog) {
        for(unsigned i = 0, n = std::get<I>(self->jacobians).size(); i < n; ++i){        
            out << "debug jacobian type: " << I << " index: " << i << std::endl;
            out << std::get<I>(self->jacobians)[i].compressedMatrix << std::endl;
        }
    }
    
    template<std::size_t ... I>
    static void update_js(self_type* self, seq<I...> ) {
        
        self->js.clear();
        
        const int expand [] = {
            (std::transform(std::get<I>(self->jacobians).begin(), std::get<I>(self->jacobians).end(),
                            std::back_inserter(self->js), js_helper()),
             // debug_jacobians<I>(),
             0)...
        }; (void) expand;
        
    }


    
};




template<class TOut, class ... T>
CompliantMapping<TOut (T...) >::CompliantMapping()
    : from_models( impl::make_from_models(this, make_sequence<T...>()) ),
      to_models(initLink("output", "output object")) {
    
}

template<class TOut, class ... T>
helper::vector<core::BaseState*> CompliantMapping<TOut (T...) >::getFrom() {
    return impl::getFrom(this, make_sequence<T...>());
}

template<class TOut, class ... T>
helper::vector<core::behavior::BaseMechanicalState*> CompliantMapping<TOut (T...) >::getMechFrom() {
    return impl::getMechFrom(this, make_sequence<T...>());
}

template<class TOut, class ... T>
helper::vector<core::BaseState* > CompliantMapping<TOut (T...) >::getTo() {
    // TODO reserve
    helper::vector<core::BaseState*> res;
    std::copy(to_models.begin(), to_models.end(), std::back_inserter(res));
    return res;
}

template<class TOut, class ... T>
helper::vector<core::behavior::BaseMechanicalState* > CompliantMapping<TOut (T...) >::getMechTo() {
    // TODO reserve
    helper::vector<core::behavior::BaseMechanicalState*> res;
        
    std::transform(to_models.begin(), to_models.end(), std::back_inserter(res),
                   typename impl::to_mechanical_state());
    
    std::remove(res.begin(), res.end(), nullptr);
        
    return res;
}


#ifdef SOFA_USE_MASK
template<class TOut, class ... T>
void CompliantMapping<TOut (T...) >::updateForceMask() {
    // i have no idea what i'm doing lol
    helper::vector<core::behavior::BaseMechanicalState*> from = getMechFrom();
    for (unsigned i = 0; i < from.size(); ++i) {
        from[i]->forceMask.assign(from[i]->getSize(), true);
    }
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

    // TODO check that from_models all have size > 0
    impl::init(this, make_sequence<T...>() );

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
    impl::apply(this, mparams, out_pos_id, in_pos_id, make_sequence<T...>());
}

template<class TOut, class ... T>
void CompliantMapping<TOut (T...) >::updateK(const core::MechanicalParams* /*mparams*/,
                                             core::ConstMultiVecDerivId out_force_id ) {
    impl::update_gs(this, out_force_id, make_sequence<T...>());
}


template<class TOut, class ... T>
void CompliantMapping<TOut (T...) >::applyJ(const core::MechanicalParams* /*mparams*/,
                                            core::MultiVecDerivId out_vel_id,
                                            core::ConstMultiVecDerivId in_vel_id ) {
    impl::apply_jacobian(this, out_vel_id, in_vel_id, make_sequence<T...>() );
}


template<class TOut, class ... T>
void CompliantMapping<TOut (T...) >::applyJT (const core::MechanicalParams* /*mparams*/,
                                              core::MultiVecDerivId in_force_id,
                                              core::ConstMultiVecDerivId out_force_id) {
    impl::apply_jacobian_transpose(this, in_force_id, out_force_id, make_sequence<T...>() );
    
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
    impl::apply_DJT(this, in_vel_id, mparams->kFactor(), make_sequence<T...>() );
}



template<class TOut, class ... T>
const helper::vector<sofa::defaulttype::BaseMatrix*>* CompliantMapping<TOut (T...) >::getJs() {
    impl::update_js(this, make_sequence<T...>());
    return &js;
}


template<class TOut, class ... T>
const defaulttype::BaseMatrix* CompliantMapping<TOut (T...) >::getK() {
    if( geometric.compressedMatrix.nonZeros() ) return &geometric;
    else return nullptr;
};


}
}
}

#endif


