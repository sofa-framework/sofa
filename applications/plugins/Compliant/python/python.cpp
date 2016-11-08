

#include <Eigen/Sparse>


typedef Eigen::SparseMatrix<double, Eigen::RowMajor> csr_matrix;

extern "C" {

    void eigen_sparse_matrix_assign(csr_matrix* lvalue, const csr_matrix* rvalue) {
        *lvalue = *rvalue;
        lvalue->makeCompressed();
    }
    
}
