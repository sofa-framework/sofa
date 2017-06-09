#ifndef SOFA_COMPONENT_MAPPING_COMPLIANTMAPPING_H
#define SOFA_COMPONENT_MAPPING_COMPLIANTMAPPING_H

#include <sofa/core/BaseMapping.h>
#include <sofa/core/core.h>
#include <sofa/core/VecId.h>

#include <SofaEigen2Solver/EigenSparseMatrix.h>

#include <Compliant/config.h>
#include <Compliant/utils/view.h>
#include <Compliant/utils/seq.h>

namespace sofa {

namespace component {

namespace mapping {


/**  
     Provides an implementation basis for assembled, sparse multi
     mappings with any input types
     
     @author: maxime.tournier@anatoscope.com
*/

template<class T>
class CompliantMapping;




template<class TOut, class ... T>
class CompliantMapping< TOut (T...) > : public core::BaseMapping {
protected:


    using signature_type = TOut (T...);
    
    template<class U>
    using link_type = SingleLink<CompliantMapping, core::State< U >, 
                                 BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK>;
    
    // links
    std::tuple< link_type<T>...> from_models;
    link_type<TOut> to_model;

    // helper types
    template<std::size_t I>
    using input_types = typename std::tuple_element<I, std::tuple<T...> >::type;

    using output_types = TOut;
    
    template<class U>
	using jacobian_type = linearsolver::EigenSparseMatrix<U, TOut>;
    
    // jacobians
    std::tuple< jacobian_type<T> ... > jacobians;
    
    // geometric stiffness
    using geometric_type = linearsolver::EigenBaseSparseMatrix<SReal>;
    geometric_type geometric;
    
    // this is for getJs()
    helper::vector<sofa::defaulttype::BaseMatrix*> js;
    
    // dirty implementation stuff goes in there
    template<std::size_t ... I>
    struct impl_type;

    using sequence_type = typename make_sequence_type<0, sizeof...(T) - 1>::type;
    using impl = typename instantiate_sequence_type<impl_type, sequence_type>::type;
    
    // temporaries used by applyDJT
    template<class U>
    using vec = Eigen::Matrix<U, Eigen::Dynamic, 1>;
    mutable vec<SReal> rhs, res;

public:

    // yes we can
    CompliantMapping();
    
	SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE(CompliantMapping, TOut(T...) ),
                        core::BaseMapping);
    

    virtual helper::vector<core::BaseState*> getFrom();
    virtual helper::vector<core::behavior::BaseMechanicalState*> getMechFrom();
    
    virtual helper::vector<core::BaseState* > getTo();
    virtual helper::vector<core::behavior::BaseMechanicalState* > getMechTo();

	virtual void init();    
    virtual void disable();
    
#ifdef SOFA_USE_MASK   
    virtual void updateForceMask();
#endif

    // template representation
    static std::string templateName(const CompliantMapping* self);
    std::string getTemplateName() const;
    
protected:

    // derived classes need to implement this
    virtual void apply(const core::MechanicalParams* mparams,
                       coord_view<TOut> out_pos,
                       coord_view<const T>... in_pos) = 0;
    
    virtual void assemble(jacobian_type<T>&...jacobians, 
                          coord_view<const T>... in_pos) = 0;
    
    virtual void assemble_gs(geometric_type& gs,
                             deriv_view<const TOut> out_force,
                             coord_view<const T> ... in_pos);

public:
    
    virtual void updateK(const core::MechanicalParams* mparams,
                         core::ConstMultiVecDerivId out_force_id );



    virtual void apply (const core::MechanicalParams* mparams,
                        core::MultiVecCoordId out_pos_id,
                        core::ConstMultiVecCoordId in_pos_id);

    
    
    virtual void applyJ (const core::MechanicalParams* mparams,
                         core::MultiVecDerivId out_vel_id,
                         core::ConstMultiVecDerivId in_vel_id );


    
    virtual void applyJT (const core::MechanicalParams* mparams,
                          core::MultiVecDerivId in_force_id,
                          core::ConstMultiVecDerivId out_force_id);

    
    
    virtual void applyDJT(const core::MechanicalParams* mparams,
                          core::MultiVecDerivId in_vel_id,
                          core::ConstMultiVecDerivId out_force_id);

    
    virtual const helper::vector<sofa::defaulttype::BaseMatrix*>* getJs();
    virtual const defaulttype::BaseMatrix* getK();
    
    
    virtual void applyJT(const core::ConstraintParams* mparams, 
                         core::MultiMatrixDerivId inConst, 
                         core::ConstMultiMatrixDerivId outConst);

    
    virtual void computeAccFromMapping(const core::MechanicalParams* mparams, 
                                       core::MultiVecDerivId outAcc, 
                                       core::ConstMultiVecDerivId inVel, 
                                       core::ConstMultiVecDerivId inAcc );
    
    
    
};


}
}
}



#endif

