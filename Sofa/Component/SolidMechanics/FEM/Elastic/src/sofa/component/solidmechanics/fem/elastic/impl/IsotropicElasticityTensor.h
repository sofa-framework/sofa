#pragma once
#include <sofa/core/trait/DataTypes.h>
#include <sofa/type/Mat.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template <sofa::Size N, class real>
constexpr sofa::type::Vec<N, real> isotropicElasticityTensorProduct(const sofa::type::Mat<N, N, real>& tensor, const sofa::type::Vec<N, real>& v)
{
    return tensor * v;
}

/**
 * Specialization for 3D where the operations containing known zeros in the elasticity tensor are
 * omitted.
 */
template <class real>
constexpr sofa::type::Vec<6, real> isotropicElasticityTensorProduct(const sofa::type::Mat<6, 6, real>& tensor, const sofa::type::Vec<6, real>& v)
{
    sofa::type::Vec<6, real> result { sofa::type::NOINIT };

    result[0] = tensor(0, 0) * v[0] + tensor(0, 1) * v[1] + tensor(0, 2) * v[2];
    result[1] = tensor(1, 0) * v[0] + tensor(1, 1) * v[1] + tensor(1, 2) * v[2];
    result[2] = tensor(2, 0) * v[0] + tensor(2, 1) * v[1] + tensor(2, 2) * v[2];
    result[3] = tensor(3, 3) * v[3];
    result[4] = tensor(4, 4) * v[4];
    result[5] = tensor(5, 5) * v[5];

    return result;
}

template<class DataType>
struct IsotropicElasticityTensor
{
    static constexpr auto spatial_dimensions = DataType::spatial_dimensions;
    static constexpr auto NbIndependentElements = symmetric_tensor::NumberOfIndependentElements<spatial_dimensions>;

    explicit IsotropicElasticityTensor(const sofa::type::Mat<NbIndependentElements, NbIndependentElements, sofa::Real_t<DataType>>& mat) : C(mat) {}
    IsotropicElasticityTensor() = default;

    sofa::type::Vec<NbIndependentElements, sofa::Real_t<DataType>> operator*(const sofa::type::Vec<NbIndependentElements, sofa::Real_t<DataType>>& v) const
    {
        return isotropicElasticityTensorProduct(C, v);
    }

    template<sofa::Size C>
    sofa::type::Mat<NbIndependentElements, C, sofa::Real_t<DataType>> operator*(const sofa::type::Mat<NbIndependentElements, C, sofa::Real_t<DataType>>& v) const noexcept
    {
        return C * v;
    }

    sofa::Real_t<DataType> operator()(sofa::Size i, sofa::Size j) const
    {
        return C(i, j);
    }

    sofa::Real_t<DataType>& operator()(sofa::Size i, sofa::Size j)
    {
        return C(i, j);
    }

    const sofa::type::Mat<NbIndependentElements, NbIndependentElements, sofa::Real_t<DataType>>& toMat() const
    {
        return C;
    }

private:
    sofa::type::Mat<NbIndependentElements, NbIndependentElements, sofa::Real_t<DataType>> C;

};

}
