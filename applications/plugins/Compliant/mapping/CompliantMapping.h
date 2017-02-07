#ifndef SOFA_COMPONENT_MAPPING_COMPLIANTMAPPING_H
#define SOFA_COMPONENT_MAPPING_COMPLIANTMAPPING_H

#include <sofa/core/BaseMapping.h>
#include <sofa/core/core.h>
#include <sofa/core/VecId.h>


#include <SofaEigen2Solver/EigenSparseMatrix.h>

#include <Compliant/config.h>
#include <Compliant/utils/view.h>

namespace sofa
{

namespace component
{

namespace mapping
{


/**  
     Provides an implementation basis for assembled, sparse multi
     mappings with two input types

     TODO: add .inl to minimize bloat / compilation times
     
     @author: maxime.tournier@anatoscope.com
*/

template<class T>
class CompliantMapping;


template<std::size_t ... n>
struct seq {

    template<std::size_t x>
    using push = seq<n..., x>;
};

template<std::size_t i>
struct gen_seq {
    using type = typename gen_seq<i-1>::type::template push<i>;
};


template<>
struct gen_seq<0> {
    using type = seq<0>;
};

template<class ...T>
static typename gen_seq<sizeof...(T) - 1>::type make_sequence() { return {}; }




template<class TOut, class ... T>
class CompliantMapping< TOut (T...) > : public core::BaseMapping {

    using base_type = core::BaseMapping;
    using this_type = CompliantMapping;

protected:
    
    template<class U>
    using link_type = MultiLink<this_type, core::State< U >, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK>;
    
    std::tuple< link_type<T>...> from_models;
    link_type<TOut> to_models;

    template<class U>
	using jacobian_type = linearsolver::EigenSparseMatrix<U, TOut>;

    template<std::size_t I>
    using input_type = typename std::tuple_element<I, std::tuple<T...> >::type;

    using geometric_type = linearsolver::EigenBaseSparseMatrix<SReal>;
    geometric_type geometric;
    
private:
    std::tuple< std::vector<jacobian_type<T> >... > jacobians;
    
    template<std::size_t ... I>
    std::tuple< link_type<T>...> make_from_models(seq<I...>) {
        
        static std::string names[] = {
            (std::string("input") + std::to_string(I+1))...
        };

        static std::string help[] = {
            (std::string("help") + std::to_string(I+1))...
        };

        // TODO add alias 'input' for 'input1' when sizeof...(T) == 1
        return std::tuple< link_type<T>... > { initLink(names[I].c_str(), help[I].c_str())... };
    }
    
  public:

    CompliantMapping()
        : from_models( make_from_models(make_sequence<T...>()) ),
          to_models(initLink("output", "output object")) {
                  
    }
    
	SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE(CompliantMapping, TOut(T...) ),
                        core::BaseMapping);
    

    template<std::size_t I>
    auto getFromModels() const -> decltype( std::get<I>(this->from_models).getValue() ){
        return std::get<I>(from_models).getValue();
    }

    auto getToModels() const -> decltype(this->to_models.getValue()) {
        return to_models.getValue();
    }


    helper::vector<core::BaseState*> getFrom() {
        return getFrom(make_sequence<T...>());
    }


    helper::vector<core::behavior::BaseMechanicalState*> getMechFrom() {
        return getMechFrom(make_sequence<T...>());
    }
    

    struct not_implemented : std::runtime_error {
        not_implemented(const char* what)
            : std::runtime_error(std::string("not implemented: ") + what) {
        }
    };


    template<std::size_t ...I>
    helper::vector<core::BaseState*> getFrom( seq<I...> ) {
        
        helper::vector<core::BaseState*> res;

        const int expand[] = {
            (std::copy( std::get<I>(from_models).begin(),
                        std::get<I>(from_models).end(),
                        std::back_inserter(res) ), 0)...
        };

        (void) expand;

        return res;
    }


    template<class U>
    static core::behavior::BaseMechanicalState* to_mechanical_state(const typename link_type<U>::ValueType& x) {
        return x->toBaseMechanicalState();
    };
    

    template<std::size_t ...I>
    helper::vector<core::behavior::BaseMechanicalState*> getMechFrom( seq<I...> ) {
        
        helper::vector<core::behavior::BaseMechanicalState*> res;

        static auto transformers = std::make_tuple( &to_mechanical_state<T>...);
        
        const int expand[] = {
            (std::transform( std::get<I>(from_models).begin(),
                             std::get<I>(from_models).end(),
                             std::back_inserter(res), std::get<I>(transformers)), 0)...
        };
        (void) expand;

        return res;
    }
    
    
    helper::vector<core::BaseState* > getTo() {
        // TODO reserve
        helper::vector<core::BaseState*> res;
        std::copy(to_models.begin(), to_models.end(), std::back_inserter(res));
        return res;
    }


    helper::vector<core::behavior::BaseMechanicalState* > getMechTo() {
        // TODO reserve
        helper::vector<core::behavior::BaseMechanicalState*> res;
        
        std::transform(to_models.begin(), to_models.end(), std::back_inserter(res),
                       to_mechanical_state<TOut>);
        std::remove(res.begin(), res.end(), nullptr);
        
        return res;
    }

    
    void disable() {
        // lolwat
    }


#ifdef SOFA_USE_MASK   
    void updateForceMask() {
        // i have no idea what i'm doing lol
        helper::vector<core::behavior::BaseMechanicalState*> from = getMechFrom();
        for (unsigned i = 0; i < from.size(); ++i) {
            from[i]->forceMask.assign(from[i]->getSize(), true);
        }
    }
#endif

    template<std::size_t  I>
    void init_from_model() {
        auto& from = std::get<I>(from_models);
        
        if( from.size() == 0 ) {
            serr << "error: empty input " << I + 1 << sendl;
        }

        for(unsigned i = 0, n = from.size(); i < n; ++i) {
            if(!from[i]) {
                serr << "error: input " << i << " for type " << I + 1 << " is invalid (wrong type?)" << sendl;
            }
        }
        
        
    };
    

    
    template<std::size_t ... I>
    void init(seq<I...>) {

        const int expand[] = {
            (init_from_model<I>(), 0)...
        }; (void) expand;
        
    }


	virtual void init() {
		assert( (this->getTo().size() == 1) && 
		        "multi mapping to multiple output dofs unimplemented" );
        base_type::init();

        // TODO check that from_models all have size > 0
        init( make_sequence<T...>() );


        // TODO setup_masks
    //         maskFrom1.resize( this->fromModels1.size() );
    // for( unsigned i=0 ; i<this->fromModels1.size() ; ++i )
    //     if( core::behavior::BaseMechanicalState* stateFrom = this->fromModels1[i]->toBaseMechanicalState() ) maskFrom1[i] = &stateFrom->forceMask;
    // maskFrom2.resize( this->fromModels2.size() );
    // for( unsigned i=0 ; i<this->fromModels2.size() ; ++i )
    //     if( core::behavior::BaseMechanicalState* stateFrom = this->fromModels2[i]->toBaseMechanicalState() ) maskFrom2[i] = &stateFrom->forceMask;
    // maskTo.resize( this->toModels.size() );
    // for( unsigned i=0 ; i<this->toModels.size() ; ++i )
    //     if (core::behavior::BaseMechanicalState* stateTo = this->toModels[i]->toBaseMechanicalState()) maskTo[i] = &stateTo->forceMask;
    //     else this->setNonMechanical();

    // apply(MechanicalParams::defaultInstance() , VecCoordId::position(), ConstVecCoordId::position());
    // applyJ(MechanicalParams::defaultInstance() , VecDerivId::velocity(), ConstVecDerivId::velocity());
    // if (f_applyRestPosition.getValue())
    //     apply(MechanicalParams::defaultInstance(), VecCoordId::restPosition(), ConstVecCoordId::restPosition());

	}


protected:

    template<class U>
    using coord_view = view< typename U::Coord >;

    template<class U>
    using deriv_view = view< typename U::Deriv >;
    
    
    
public:
    std::tuple< std::vector< coord_view<T> >... > in_pos;

    template<std::size_t I>
    void update_ith_in_pos(const core::ConstMultiVecCoordId& id) {
        auto& ith_in_pos = std::get<I>(in_pos);
        const auto& ith_from_models = std::get<I>(from_models);

        const unsigned n = ith_from_models.size();
        assert(n && "some input type has no model");
        
        ith_in_pos.clear();
        ith_in_pos.reserve(n);
        
        for(unsigned i = 0; i < n; ++i) {
            ith_in_pos.push_back( data_view( id[ith_from_models[i]].read() ) );
        }
        
    }
                       
    
    template<std::size_t ... I>
    void update_in_pos(const core::ConstMultiVecCoordId& id, seq<I...> ) {

        // resize
        const int expand[] = {
            ( update_ith_in_pos<I>(id), 0)...
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

    

    virtual void apply(const core::MechanicalParams* mparams,
                       coord_view<TOut> out_pos,
                       view< coord_view<T> >... in_pos) = 0;
    
    virtual void assemble(view< jacobian_type<T> >...jacobians, 
                          view< coord_view<T> >... in_pos) = 0;

    virtual void assemble_gs(geometric_type& /*gs*/,
                             deriv_view<TOut> /*out_force*/,
                             view< coord_view<T> >... /*in_pos*/) {
        // default: no geometric stiffness
    };

    

    template<class U>
    static view<U> vector_view(std::vector<U>& v) {
        return {v.data(), v.size()};
    }
    
    template<std::size_t ... I>
    void apply (const core::MechanicalParams* mparams,
                core::MultiVecCoordId out_pos_id,
                core::ConstMultiVecCoordId in_pos_id,
                seq<I...> ) {

        update_in_pos(in_pos_id, make_sequence<T...>());

        coord_view<TOut> out_view = data_view( out_pos_id[ to_models[0] ].read() );
        
        std::tuple< view< coord_view<T> >... > in_view{
            vector_view(std::get<I>(in_pos)) ...
        };

        
        this->apply(mparams, out_view, std::get<I>(in_view)...);


        // resize jacobians
        const int expand[] = {
            (std::get<I>(jacobians).resize( std::get<I>(from_models).size() ), 0)...
        }; (void) expand;
        
        // build jacobian view
        std::tuple< view< jacobian_type<T> >... > jacobian_view{
            vector_view(std::get<I>(jacobians)) ...
         };

        // wtf is this nonsense
#ifdef SOFA_USE_MASK
        this->m_forceMaskNewStep = true;
#endif
        
        this->assemble(std::get<I>(jacobian_view)..., std::get<I>(in_view)...);
    }


    template<std::size_t ... I>
    void update_gs(core::ConstMultiVecDerivId out_force_id, seq<I...>) {
        // TODO assert in_pos is set correctly
        auto* out_force_data = out_force_id[to_models[0]].read();
        auto out_force_view = data_view(out_force_data);
        
        std::tuple< view< coord_view<T> >... > in_view(
            vector_view(std::get<I>(in_pos)) ...
            );

        this->assemble_gs(geometric, out_force_view, std::get<I>(in_view)...);
    }
    
    virtual void updateK(const core::MechanicalParams* /*mparams*/,
                         core::ConstMultiVecDerivId out_force_id ) {
        update_gs(out_force_id, make_sequence<T...>());
    }

    

    virtual void apply (const core::MechanicalParams* mparams,
                        core::MultiVecCoordId out_pos_id,
                        core::ConstMultiVecCoordId in_pos_id) {
        apply(mparams, out_pos_id, in_pos_id, make_sequence<T...>());
    }



    template<std::size_t I>
    void add_mult_ith_jacobian(Data< helper::vector< typename TOut::Deriv > >* out_vel_data,
                               core::ConstMultiVecDerivId in_vel_id) {
        assert(std::get<I>(from_models).size() == std::get<I>(jacobians).size());
        
        for(unsigned i = 0, n = std::get<I>(jacobians).size(); i < n; ++i){
            auto in_vel_data = in_vel_id[std::get<I>(from_models)[i]].read();
            std::get<I>(jacobians)[i].addMult(*out_vel_data, *in_vel_data);
        }

    }
    

    template<std::size_t ... I>
    void apply_jacobian(core::MultiVecDerivId out_vel_id,
                        core::ConstMultiVecDerivId in_vel_id,
                        seq<I...>) {

        auto* out_state = to_models[0];

        // TODO make sure this does not trigger funky stuff
        auto* out_vel_data = out_vel_id[out_state].write();
        
        // set output to zero
        auto out_vel_view = data_view(out_vel_data);
        assert(out_vel_view.size() == out_state->getSize());
        
        std::fill(out_vel_view.begin(), out_vel_view.end(), typename TOut::Deriv() );

        // multiplication
        const int expand[] = {
            (add_mult_ith_jacobian<I>(out_vel_data, in_vel_id), 0)...
        }; (void) expand;
        
    }
    
    
    virtual void applyJ (const core::MechanicalParams* /*mparams*/,
                         core::MultiVecDerivId out_vel_id,
                         core::ConstMultiVecDerivId in_vel_id ) {
        apply_jacobian(out_vel_id, in_vel_id, make_sequence<T...>() );
    }


    template<std::size_t I>
    void add_mult_ith_jacobian_transpose(core::MultiVecDerivId in_force_id,
                                         core::ConstMultiVecDerivId out_force_id) {
        assert(std::get<I>(from_models).size() == std::get<I>(jacobians).size());
        
        auto* out_force_data = out_force_id[to_models[0]].read();
        
        for(unsigned i = 0, n = std::get<I>(jacobians).size(); i < n; ++i){
            // TODO make sure 'write' does not trigger hell
            auto* in_force_data = in_force_id[std::get<I>(from_models)[i]].write();
            std::get<I>(jacobians)[i].addMultTranspose(*in_force_data, *out_force_data);
        }
        
    }

    

    template<std::size_t ... I>
    void apply_jacobian_transpose(core::MultiVecDerivId in_force_id,
                                  core::ConstMultiVecDerivId out_force_id,
                                  seq<I...>) {
        // multiplication
        const int expand[] = {
            (add_mult_ith_jacobian_transpose<I>(in_force_id, out_force_id), 0)...
        }; (void) expand;
        
    }
    
    
    virtual void applyJT (const core::MechanicalParams* /*mparams*/,
                          core::MultiVecDerivId in_force_id,
                          core::ConstMultiVecDerivId out_force_id) {
        apply_jacobian_transpose(in_force_id, out_force_id, make_sequence<T...>() );

#ifdef SOFA_USE_MASK
        if( this->m_forceMaskNewStep ){
            this->m_forceMaskNewStep = false;
            updateForceMask();
        }
#endif
        
    }

    
    virtual void applyDJT(const core::MechanicalParams* /*mparams*/,
                          core::MultiVecDerivId /*inForce*/,
                          core::ConstMultiVecDerivId /*outForce*/) {

        // TODO
        throw not_implemented(__func__);
    }

    
    
    helper::vector<sofa::defaulttype::BaseMatrix*> js;
    
    virtual const helper::vector<sofa::defaulttype::BaseMatrix*>* getJs() {
        update_js(make_sequence<T...>());
        return &js;
    }


    struct js_helper {
        
        template<class U>
        sofa::defaulttype::BaseMatrix* operator()(U&& value) const {
            return &value;
        }
    };

    template<std::size_t I>
    void debug_jacobians(std::ostream& out = std::clog) {
        for(unsigned i = 0, n = std::get<I>(jacobians).size(); i < n; ++i){        
            out << "debug jacobian type: " << I << " index: " << i << std::endl;
            out << std::get<I>(jacobians)[i].compressedMatrix << std::endl;
        }
    }
    
    template<std::size_t ... I>
    void update_js( seq<I...> ) {

        js.clear();
        
        const int expand [] = {
            (std::transform(std::get<I>(jacobians).begin(), std::get<I>(jacobians).end(),
                            std::back_inserter(js), js_helper()),
             // debug_jacobians<I>(),
             0)...
        }; (void) expand;
        
        
        
    }

    virtual const defaulttype::BaseMatrix* getK() {
        if( geometric.compressedMatrix.nonZeros() ) return &geometric;
        else return nullptr;
    };
    
    
    virtual void applyJT(const core::ConstraintParams* /*mparams*/, 
                         core::MultiMatrixDerivId /*inConst*/, 
                         core::ConstMultiMatrixDerivId /*outConst*/) {
        throw not_implemented(__func__);
    }

    
    virtual void computeAccFromMapping(const core::MechanicalParams* /*mparams*/, 
                                       core::MultiVecDerivId /*outAcc*/, 
                                       core::ConstMultiVecDerivId /*inVel*/, 
                                       core::ConstMultiVecDerivId /*inAcc*/ ) {
        throw not_implemented(__func__);
    }
    
    
    
};


}
}
}



#endif

