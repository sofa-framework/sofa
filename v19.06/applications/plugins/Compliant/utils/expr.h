#ifndef COMPLIANT_UTILS_EXPR_H
#define COMPLIANT_UTILS_EXPR_H

// expression templates for vops (as it should be)

namespace sofa {
namespace expr {


class visitor : public sofa::simulation::MechanicalVisitor {

public:


    typedef sofa::core::objectmodel::BaseContext context_type;
    typedef sofa::core::MechanicalParams mparams_type;
    
    struct delegate {
        virtual ~delegate() { }
        
        typedef sofa::core::behavior::BaseMechanicalState mstate;
        virtual void exec(mparams_type* params, mstate* mm) const = 0;
        
    };

    visitor(context_type* context,
            mparams_type* mparams)
        : sofa::simulation::MechanicalVisitor(mparams),
          context(context),
          mparams(mparams) {

        mapped = true;
        precomp = true;
        
        // TODO more
    }

    bool mapped;
    bool precomp;

    void exec( const delegate& del) {
        _delegate = &del;
        context->executeVisitor(this, precomp);
        _delegate = 0;
        
        // TODO options
    }

    
    template<class Result>
    struct setter;


    template<class Result>
    setter<Result> operator()(Result& result) {
        return setter<Result>(this, result);
    }

protected:
    
    virtual Result fwdMappedMechanicalState(sofa::simulation::Node* node,
                                            sofa::core::behavior::BaseMechanicalState* mm) {
        if(mapped) {
            _delegate->exec(mparams, mm);
            return RESULT_CONTINUE;
        } else return RESULT_PRUNE;
        
    }

    virtual Result fwdMechanicalState(sofa::simulation::Node* node,
                                      sofa::core::behavior::BaseMechanicalState* mm) {
        _delegate->exec(mparams, mm);
        return RESULT_CONTINUE;
    }
    
    
protected:
    context_type* context;
    mparams_type* mparams;

    const delegate* _delegate;
};


template<class LHS, class RHS>
struct sum {
    const LHS& lhs;
    const RHS& rhs;

    sum(const LHS& lhs,
        const RHS& rhs)
        : lhs(lhs),
          rhs(rhs) {

    }
    
};



template<class Expr>
struct prod {
    const Expr& expr;
    const SReal factor;

    prod(const Expr& expr,
         SReal factor = 1.0):
        expr(expr),
        factor(factor) {

    }
};

struct none  {

    struct id_type {
        sofa::core::ConstVecId getId(visitor::delegate::mstate* mm) {
            return sofa::core::ConstVecId::null();
        }
    };

    id_type id() const { return id_type(); }
    
};

template<class Check, class Return = Check>
struct enable_if {
    typedef Return type;
};

// operators
template<class LHS, class RHS>
sum<LHS, RHS> operator+(const LHS& lhs, const RHS& rhs) {
    return sum<LHS, RHS>(lhs, rhs);
}


template<class Expr>
typename enable_if<int Expr::*, prod<Expr> >::type operator*(const Expr& expr, SReal factor) {
    return prod<Expr>(expr, factor);
}

template<class Expr>
typename enable_if<int Expr::*, prod<Expr> >::type operator*(SReal factor, const Expr& expr ) {
    return prod<Expr>(expr, factor);
}

template<class Expr>
prod<Expr> operator/(const Expr& expr, SReal factor) {
    return prod<Expr>(expr, 1.0 / factor);
}


// evaluation
    
template<class Result, class A = none, class B = none>
struct vector_operation : visitor::delegate {
    Result& result;
    const A& a;
    const B& b;

    const SReal factor;

    vector_operation(Result& result,
                     const A& a = A(),
                     const B& b = B(),
                     const SReal factor = 1.0):
        result(result),
        a(a),
        b(b),
        factor(factor) { }

    virtual void exec(visitor::mparams_type* params,
                      visitor::delegate::mstate* mm) const {
        mm->vOp(params, result.id().getId(mm), a.id().getId(mm), b.id().getId(mm), factor);
    };
    
};


template<class Result, class LHS, class RHS>
vector_operation<Result, LHS, RHS> transform(Result& result,
                                             const sum<LHS, RHS>& expr) {
    return vector_operation<Result,
                            LHS, RHS>(result,
                                      expr.lhs,
                                      expr.rhs);
}


template<class Result, class LHS, class RHS>
vector_operation<Result, LHS, RHS> transform(Result& result,
                                             const sum<LHS, prod<RHS> >& expr) {
    return vector_operation<Result,
                            LHS, RHS>(result,
                                      expr.lhs,
                                      expr.rhs.expr,
                                      expr.rhs.factor);
}

template<class Result, class LHS, class RHS>
vector_operation<Result, LHS, RHS> transform(Result& result,
                                             const sum<prod<RHS>, LHS >& expr) {
    return vector_operation<Result,
                            LHS, RHS>(result,
                                      expr.lhs,
                                      expr.rhs.expr,
                                      expr.rhs.factor);
}


template<class Result, class Expr>
vector_operation<Result, none, Expr> transform(Result& result,
                                               const prod<Expr>& expr) {
    return vector_operation<Result,
                            none, Expr>(result,
                                        none(),
                                        expr.expr,
                                        expr.factor);
}



template<class Result, class Expr>
vector_operation<Result, Expr, none> transform(Result& result,
                                               const Expr& expr) {
    return vector_operation<Result,
                            Expr>(result,
                                  expr);
}





template<class Res>
struct visitor::setter {
    visitor* vis;
    Res& result;

    setter(visitor* vis,
           Res& result) : vis(vis), result(result) { }
        
    template<class Expr>
    void operator=(const Expr& expr) const {
        vis->exec( transform(result, expr) );
    }
        
};




}
}


#endif
