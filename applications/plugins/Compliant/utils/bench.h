#ifndef UTILS_BENCH_H
#define UTILS_BENCH_H

#include "../assembly/AssembledSystem.h"
#include "../numericalsolver/Response.h"

#include <boost/chrono.hpp>


namespace sofa {
namespace component {
namespace linearsolver {





struct BaseBenchmark {

    typedef AssembledSystem system_type;

    typedef system_type::real real;
    typedef system_type::vec vec;
    typedef system_type::mat mat;

    BaseBenchmark(){}

    /// must be called at the beginning of KKTSolver::factor (creating the bench file)
    virtual void beginFactor( double /*time*/ ) {}
    virtual void endFactor() {}
    virtual void beginSolve( bool /*lcp*/ ) {}

    /// must be called after each KKTSolver iteration (writing in the bench file)
    virtual void operator()(const vec& /*v*/) const {}
    virtual void operator()(const vec& /*v*/, const vec& /*lambda*/) const {}

    /// must be called at the beginning of KKTSolver::solve for lcp solvers
    virtual void setLCP( const AssembledSystem* /*system*/,
            const Response* /*response*/,
            const vec& /*unconstrained*/,
            const vec& /*b*/) {}

    /// must be called at the beginning of KKTSolver::solve for qp solvers
    virtual void setQP( const AssembledSystem* /*system*/,
           const vec& /*rhs*/) {}

};


struct Benchmark : BaseBenchmark {

    Benchmark( const std::string& path ) : _path(path)
    {
        std::cerr<<"Benchmark: created for "<<path<<std::endl;
    }

    virtual void beginFactor( double time )
    {
        if( _file.is_open() ) _file.close();

        std::stringstream stream;
        stream << _path << "/" << time;

        _file.open( stream.str().c_str() );
        _counter = 0;


        _clockStart = clock_type::now(); // let's start counting time
    }

    virtual void endFactor()
    {
        _factorDuration = ( boost::chrono::duration_cast<boost::chrono::microseconds> (clock_type::now() - _clockStart) ).count();
    }

    virtual void beginSolve( bool lcp )
    {
        _isLCP = lcp;
        if( lcp )
            _file << "# it us err_total err_primal err_dual err_compl\n";
        else
            _file << "# it us err_total err_primal err_dual err_compl err_opt\n";

        _clockStart = clock_type::now(); // let's start counter for first iteration
    }

    virtual void operator()(const vec& v) const
    {
        unsigned duration = ( boost::chrono::duration_cast<boost::chrono::microseconds> (clock_type::now() - _clockStart) ).count();

        if( _counter==0 ) duration += _factorDuration; // let's include the factor duration in the first iteration

        assert( _file.is_open() );
        if( _isLCP ) LCP(v,duration);
        else QP(v,duration);

        _clockStart = clock_type::now(); // let's start counter again for new iteration
    }

    virtual void operator()(const vec& v, const vec& lambda) const
    {
        assert( !_isLCP );
        assert( _file.is_open() );

        unsigned duration = ( boost::chrono::duration_cast<boost::chrono::microseconds> (clock_type::now() - _clockStart) ).count();

        if( _counter==0 ) duration += _factorDuration; // let's include the factor duration in the first iteration

        vec x(_system->size());
        x.head(_system->m) = v;
        if( _system->n ) x.tail(_system->n) = lambda;

        QP(x,duration);

        _clockStart = clock_type::now(); // let's start counter again for new iteration
    }

  protected:

    std::string _path;
    mutable std::ofstream _file;
    mutable unsigned _counter;
    typedef boost::chrono::high_resolution_clock clock_type;
    mutable clock_type::time_point _clockStart;
    unsigned _factorDuration;


    virtual ~Benchmark()
    {
        if( _file.is_open() ) _file.close();
    }


// common
    typedef BaseBenchmark::vec vec;
    typedef BaseBenchmark::real real;
    const AssembledSystem* _system;


// LCP

    const Response* _response;
    vec _unconstrained;
    vec _b;
    bool _isLCP;

public:
    virtual void setLCP( const AssembledSystem* system,
            const Response* response,
            const vec& unconstrained,
            const vec& b)
    {
        assert( _isLCP );

        _system  = system;
        _response = response;
        _unconstrained = unconstrained;
        _b = b;
    }

protected:
    void LCP(const vec& lambda, unsigned duration) const
    {
        assert( _system->n );

        vec tmp = lambda;

        vec v(_system->m);
        _response->solve( v, _system->J.transpose() * tmp );
        v += _unconstrained;

        vec primal = _system->J * v - _b;
        vec dual = tmp;

        real err_primal = primal.cwiseMin( vec::Zero(_system->n) ).norm() / _system->n;
        real err_dual = dual.cwiseMin( vec::Zero(_system->n) ).norm() / _system->n;
        real err_compl = std::abs(primal.dot(dual)) / _system->n;

        real total = err_primal + err_dual + err_compl;

        _file << (_counter++) << " " << duration << " " << total << " " << err_primal << " " << err_dual << " " << err_compl <<std::endl;
    }

// QP

    vec _rhs;

public:
    virtual void setQP( const AssembledSystem* system,
           const vec& rhs)
    {
          assert( !_isLCP );

          _system = system;
          _rhs = rhs;
    }
protected:
    void QP(const vec& x, unsigned duration) const
    {
        assert( x.size() == _system->size() );

        if( ! _system->n ) {
            real total = (_system->H * x.head(_system->m) - _rhs.head(_system->m)).norm() / _system->m;
            _file << (_counter++) << " " << duration << " " << total << std::endl;
        }

        vec tmp = x;

        vec primal = _system->J * tmp.head(_system->m) - _rhs.tail(_system->n);
        vec dual = tmp.tail(_system->n);

        real err_primal = primal.cwiseMin( vec::Zero(_system->n) ).norm() / _system->n;
        real err_dual = dual.cwiseMin( vec::Zero(_system->n) ).norm() / _system->n;
        real err_compl = std::abs(primal.dot(dual)) / _system->n;

        real err_opt = (_system->H * tmp.head(_system->m) - _system->J.transpose() * dual - _rhs.head(_system->m)).norm() / _system->m;

        real total = err_primal + err_dual + err_compl + err_opt;

        _file << (_counter++) << " " << duration << " " << total << " " << err_primal << " " << err_dual << " " << err_compl << " " << err_opt << std::endl;
    }
};


}
}
}

#endif
