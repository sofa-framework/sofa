#include "AnalysisSolver.h"
#include <fstream>
#include <sofa/core/ObjectFactory.h>

#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(AnalysisSolver)
int AnalysisSolverClass = core::RegisterObject("Analysis solver: runs other KKTSolvers successively on a given problem and performs extra analysis on KKT system").add< AnalysisSolver >();

AnalysisSolver::AnalysisSolver()
    : condest(initData(&condest, false, "condest", "compute condition number with svd"))
    , eigenvaluesign(initData(&eigenvaluesign, false, "eigenvaluesign", "computing the sign of the eigenvalues (of the implicit matrix H)"))
    , dump_qp(initData(&dump_qp, "dump_qp", "dump qp to file if non-empty"))

{}

void AnalysisSolver::init() {

    solvers.clear();
    this->getContext()->get<KKTSolver>( &solvers, core::objectmodel::BaseContext::Local );

    if( solvers.size() < 2 ) {
        std::cerr << "warning: no other kkt solvers found" << std::endl;
    } else {
        std::cout << "AnalysisSolver: dynamics/correction will use "
                  << solvers.back()->getName()
                  << std::endl;
    }
}

void AnalysisSolver::factor(const system_type& system) {

    // start at 1 since 0 is current solver
    for( unsigned i = 1, n = solvers.size() ; i < n; ++i ) {
        solvers[i]->factor(system);
    }



    typedef system_type::dmat dmat;

    if( condest.getValue() ) {

        dmat kkt;

        kkt.setZero(system.size(), system.size());

        kkt <<
            dmat(system.H), dmat(-system.J.transpose()),
            dmat(-system.J), dmat(-system.C);

        system_type::vec eigen = kkt.jacobiSvd().singularValues();

        real min = eigen.tail<1>()(0);
        real max = eigen(0);

        if( min < std::numeric_limits<real>::epsilon() ) std::cout << "AnalysisSolver: singular KKT system"<<std::endl;
        else
        {
            const real cond = max/min;
            std::cout << "condition number (KKT system): " << cond << " ("<<max<<"/"<<min<<")"<<std::endl;
            std::cout << "required precision (KKT system):  "<<ceil(log(cond))<<" bits"<<std::endl;
        }


        eigen = dmat(system.H).jacobiSvd().singularValues();

        min = eigen.tail<1>()(0);
        max = eigen(0);

        if( min < std::numeric_limits<real>::epsilon() ) std::cout << "AnalysisSolver: singular implicit system"<<std::endl;
        else
        {
            const real cond = max/min;
            std::cout << "condition number (implicit system): " << cond << " ("<<max<<"/"<<min<<")"<<std::endl;
            std::cout << "required precision (implicit system):  "<<ceil(log(cond))<<" bits"<<std::endl;
        }
    }


     if( eigenvaluesign.getValue() ) {


        Eigen::EigenSolver<dmat> es( dmat(system.H) );
//        Eigen::MatrixXcd D = es.eigenvalues().asDiagonal();
        unsigned positive = 0, negative = 0;
        for( dmat::Index i =0 ; i< es.eigenvalues().rows() ; ++i )
            if( es.eigenvalues()[i].real() < 0 ) negative++;
            else positive++;


        std::cout << "eigenvalues H: neg="<<negative<<" pos="<<positive<<std::endl;
        if( negative )
        {
            getContext()->getRootContext()->setAnimate(false);

            std::cerr<<es.eigenvalues().array().abs().minCoeff()<<std::endl;
            std::cerr<<" $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
        }

    }


    // TODO add more as needed
}


static void write_qp(std::ofstream& out,
                     const AssembledSystem& sys,
                     const AssembledSystem::vec& rhs) {

    const char endl = '\n';

    out << sys.m << ' ' << sys.n << endl;

    out << sys.H << endl;
    out << -rhs.head(sys.m).transpose() << endl;
    out << sys.P << endl;

    if( sys.n ) {
        out << sys.J << endl;
        out << rhs.tail(sys.n).transpose() << endl;

        // TODO unilateral mask !
    }

}

// solution is that of the first solver
void AnalysisSolver::correct(vec& res,
                             const system_type& sys,
                             const vec& rhs,
                             real damping) const {
    assert( solvers.size() > 1 );
    solvers.back()->correct(res, sys, rhs, damping);

    if(!dump_qp.getValue().empty() ) {
        const std::string filename = dump_qp.getValue() + ".correction";
        std::ofstream out(filename.c_str());

        write_qp(out, sys, rhs);
    }
}

// solution is that of the last solver
void AnalysisSolver::solve(vec& res,
                            const system_type& sys,
                            const vec& rhs) const {

    vec backup = res; // backup initial solution

    // start at 1 since 0 is current solver
    for( unsigned i = 1, n = solvers.size(); i < n; ++i ) {
        res = backup;
        solvers[i]->solve(res, sys, rhs);
    }

    if(!dump_qp.getValue().empty() ) {
        const std::string filename = dump_qp.getValue() + ".correction";
        std::ofstream out(filename.c_str());

        write_qp(out, sys, rhs);
    }

}






}
}
}
