

#include <Eigen/Sparse>
#include <cstddef>

template<class U>
struct scipy_csr_matrix {
    
    // SparseMatrix
    std::size_t rows, cols;
    int* outer_index;
    int* inner_nonzero;

    // CompressedStorage
    struct storage_type {
        U* values;
        int* indices;
        std::size_t size;
    } storage;

};


template<class U>
struct eigen_csr_matrix : Eigen::SparseMatrix<U, Eigen::RowMajor> {

private:
    // that's right. use placement new to make sure the destructor is never
    // called since memory is owned by scipy
    ~eigen_csr_matrix() { }
public:

    eigen_csr_matrix( const scipy_csr_matrix<U>* source ) {
        this->m_outerSize = source->rows;
        this->m_innerSize = source->cols;

        this->m_outerIndex = source->outer_index;
        this->m_innerNonZeros = source->inner_nonzero;

        new (&this->m_data) eigen_compressed_storage(source->storage);
    }
        
    struct eigen_compressed_storage : eigen_csr_matrix::Storage {

        eigen_compressed_storage(const typename scipy_csr_matrix<U>::storage_type& source) {
            this->m_values = source.values;
            this->m_indices = source.indices;
            this->m_size = source.size;
        }
        
        typename scipy_csr_matrix<U>::storage_type to_scipy() const {
            typename scipy_csr_matrix<U>::storage_type res;

            res.values = this->m_values;
            res.indices = this->m_indices;
            res.size = this->m_size;

            return res;
        }
            
    };

    scipy_csr_matrix<U> to_scipy() const {
        scipy_csr_matrix<U> res;

        res.rows = this->m_outerSize;
        res.cols = this->m_innerSize;

        res.outer_index = this->m_outerIndex;
        res.inner_nonzero = this->m_innerNonZeros;

        res.storage = static_cast<const eigen_compressed_storage&>(this->m_data).to_scipy();

        return res;
    }
};



using real = double;
#include <iostream>
extern "C" {


    std::size_t eigen_sizeof() {
        return sizeof(eigen_csr_matrix<real>);
    }
    
    void eigen_to_scipy(scipy_csr_matrix<real>* dst, const eigen_csr_matrix<real>* src) {
        *dst = src->to_scipy();
    }
    
    void eigen_from_scipy(eigen_csr_matrix<real>* lvalue,
                          const scipy_csr_matrix<real>* rvalue) {
        char storage[sizeof(eigen_csr_matrix<real>)];
        eigen_csr_matrix<real>* alias = new (storage) eigen_csr_matrix<real>(rvalue);
        
        *lvalue = *alias;
        lvalue->makeCompressed();
    }
    
}
