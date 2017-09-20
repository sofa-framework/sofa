// a bunch of ctypes-bound functions

#include <Eigen/Sparse>
#include <cstddef>
#include <type_traits>

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

    // this is a dummy class that provides glue between scipy matrices and eigen
    // matrices (mostly adapting pointers/shapes)

    
private:
    // that's right: use placement new to make sure the destructor is never
    // called since memory is owned by scipy.
    ~eigen_csr_matrix() { }
    
public:

    eigen_csr_matrix( const scipy_csr_matrix<U>* source ) {
        this->m_outerSize = source->rows;
        this->m_innerSize = source->cols;

        this->m_outerIndex = source->outer_index;
        this->m_innerNonZeros = source->inner_nonzero;

        // same here
        new (&this->m_data) eigen_compressed_storage(source->storage);
    }


    using eigen_compressed_storage_base = typename eigen_csr_matrix::Storage;
    
    struct eigen_compressed_storage : eigen_compressed_storage_base {

        eigen_compressed_storage(const typename scipy_csr_matrix<U>::storage_type& source) {
            this->m_values = source.values;
            this->m_indices = source.indices;
            this->m_size = source.size;
        }
        
        typename scipy_csr_matrix<U>::storage_type to_scipy() const {
            typename scipy_csr_matrix<U>::storage_type res;

            res.size = this->m_size;

            if( res.size ) {
                res.values = this->m_values;
                res.indices = this->m_indices;
            } else {
                // this keeps ctypes for yelling a warning
                res.values = nullptr;
                res.indices = nullptr;
            }

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


template<class U>
static void eigen_from_scipy_impl(eigen_csr_matrix<U>* lvalue,
                                  const scipy_csr_matrix<U>* rvalue) {
    // note: we use placement new to make sure destructor is never called
    // since memory is owned by scipy

    // note: damn you clang-3.4 you're supposed to be a c++11 compiler ffs
    // typename std::aligned_union<0, eigen_csr_matrix<U> >::type storage;

    union storage_type {
        eigen_csr_matrix<U> matrix;
        char bytes[0]; // ye olde c trick ahoy
        storage_type() { }
        ~storage_type() { }
    } storage;

    const eigen_csr_matrix<U>* alias = new (storage.bytes) eigen_csr_matrix<U>(rvalue);
        
    *lvalue = *alias;
    lvalue->makeCompressed();
}



extern "C" {

    std::size_t eigen_sizeof_double() {
        return sizeof(eigen_csr_matrix<double>);
    }

    void eigen_to_scipy_double(scipy_csr_matrix<double>* dst, const eigen_csr_matrix<double>* src) {
        *dst = src->to_scipy();
    }

    void eigen_from_scipy_double(eigen_csr_matrix<double>* lvalue,
                                 const scipy_csr_matrix<double>* rvalue) {    
        eigen_from_scipy_impl<double>(lvalue, rvalue);
    }


    std::size_t eigen_sizeof_float() {
        return sizeof(eigen_csr_matrix<float>);
    }


    void eigen_to_scipy_float(scipy_csr_matrix<float>* dst, const eigen_csr_matrix<float>* src) {
        *dst = src->to_scipy();
    }

    void eigen_from_scipy_float(eigen_csr_matrix<float>* lvalue,
                                const scipy_csr_matrix<float>* rvalue) {    
        eigen_from_scipy_impl<float>(lvalue, rvalue);
    }
    
}




