/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <sofa/core/config.h>
#include <sofa/type/vector.h>
#include <sofa/type/fwd.h>
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa::core
{

class SOFA_CORE_API MatrixAccumulatorInterface
{
public:
    virtual ~MatrixAccumulatorInterface() = default;

    virtual void add(sofa::SignedIndex /*row*/, sofa::SignedIndex /*col*/, float /*value*/) {}
    virtual void add(sofa::SignedIndex /*row*/, sofa::SignedIndex /*col*/, double /*value*/) {}

    virtual void add(sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<1, 1, float> & value);
    virtual void add(sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<1, 1, double>& value);
    virtual void add(sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<2, 2, float> & value);
    virtual void add(sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<2, 2, double>& value);
    virtual void add(sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<3, 3, float> & value);
    virtual void add(sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<3, 3, double>& value);
    virtual void add(sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<6, 6, float> & value);
    virtual void add(sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<6, 6, double>& value);

    virtual void clear() {}

    template<sofa::Size L, sofa::Size C, class real>
    void matAdd(sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<L, C, real>& value);
};

template <sofa::Size L, sofa::Size C, class real>
void MatrixAccumulatorInterface::matAdd(sofa::SignedIndex row, sofa::SignedIndex col,
    const sofa::type::Mat<L, C, real>& value)
{
    for (sofa::SignedIndex i = 0; i < sofa::SignedIndex(L); ++i)
    {
        for (sofa::SignedIndex j = 0; j < sofa::SignedIndex(C); ++j)
        {
            add(row + i, col + j, value(i, j));
        }
    }
}

namespace matrixaccumulator
{
class no_check_policy {};
inline constexpr no_check_policy no_check {};


struct SOFA_CORE_API IndexVerificationStrategy
{
    virtual ~IndexVerificationStrategy() = default;
    using verify_index = std::true_type;

    virtual void checkRowIndex(sofa::SignedIndex row) = 0;
    virtual void checkColIndex(sofa::SignedIndex col) = 0;
};

struct SOFA_CORE_API NoIndexVerification : IndexVerificationStrategy
{
    using verify_index = std::false_type;
private:
    void checkRowIndex(sofa::SignedIndex /* row */) override {}
    void checkColIndex(sofa::SignedIndex /* col */) override {}
};

struct SOFA_CORE_API RangeVerification : IndexVerificationStrategy
{
    using verify_index = std::true_type;

    sofa::SignedIndex minRowIndex { 0 };
    sofa::SignedIndex maxRowIndex { std::numeric_limits<sofa::SignedIndex>::max() };

    sofa::SignedIndex minColIndex { 0 };
    sofa::SignedIndex maxColIndex { std::numeric_limits<sofa::SignedIndex>::max() };

    sofa::core::objectmodel::BaseObject* m_messageComponent { nullptr };

    [[nodiscard]]
    helper::logging::MessageDispatcher::LoggerStream logger() const;

    void checkRowIndex(sofa::SignedIndex row) override;
    void checkColIndex(sofa::SignedIndex col) override;
};

}

/**
 * Decorator allowing to check the row and column indices before the matrix accumulation
 */
template<class TBaseMatrixAccumulator, class TStrategy>
class MatrixAccumulatorIndexChecker : public TBaseMatrixAccumulator
{
public:
    static_assert(std::is_base_of_v<MatrixAccumulatorInterface, TBaseMatrixAccumulator>, "Template argument must be a MatrixAccumulatorInterface");
    static_assert(std::is_base_of_v<objectmodel::BaseObject, TBaseMatrixAccumulator>, "Template argument must be a BaseObject");
    static_assert(std::is_base_of_v<matrixaccumulator::IndexVerificationStrategy, TStrategy>, "Template argument must be a IndexVerificationStrategy");

    SOFA_CLASS(MatrixAccumulatorIndexChecker, TBaseMatrixAccumulator);

    [[maybe_unused]]
    std::shared_ptr<TStrategy> indexVerificationStrategy;

    void add(const sofa::SignedIndex row, const sofa::SignedIndex col, const float value) override final
    {
        indexCheckedAdd(row, col, value);
    }

    void add(const sofa::SignedIndex row, const sofa::SignedIndex col, double value) override final
    {
        indexCheckedAdd(row, col, value);
    }

    void add(const sofa::SignedIndex row, const sofa::SignedIndex col, const sofa::type::Mat<2, 2, float>& value) override final
    {
        indexCheckedAdd(row, col, value);
    }

    void add(const sofa::SignedIndex row, const sofa::SignedIndex col, const sofa::type::Mat<2, 2, double>& value) override final
    {
        indexCheckedAdd(row, col, value);
    }

    void add(const sofa::SignedIndex row, const sofa::SignedIndex col, const sofa::type::Mat<3, 3, float>& value) override final
    {
        indexCheckedAdd(row, col, value);
    }

    void add(const sofa::SignedIndex row, const sofa::SignedIndex col, const sofa::type::Mat<3, 3, double>& value) override final
    {
        indexCheckedAdd(row, col, value);
    }

    void add(const sofa::SignedIndex row, const sofa::SignedIndex col, const sofa::type::Mat<6, 6, float>& value) override final
    {
        indexCheckedAdd(row, col, value);
    }

    void add(const sofa::SignedIndex row, const sofa::SignedIndex col, const sofa::type::Mat<6, 6, double>& value) override final
    {
        indexCheckedAdd(row, col, value);
    }

protected:

    using TBaseMatrixAccumulator::add;

    virtual void add(const matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, float value)
    {
        TBaseMatrixAccumulator::add(row, col, value);
    }
    virtual void add(const matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, double value)
    {
        TBaseMatrixAccumulator::add(row, col, value);
    }
    virtual void add(const matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<3, 3, float>& value)
    {
        TBaseMatrixAccumulator::add(row, col, value);
    }
    virtual void add(const matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<3, 3, double>& value)
    {
        TBaseMatrixAccumulator::add(row, col, value);
    }

    virtual void add(const matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<2, 2, float>& value)
    {
        TBaseMatrixAccumulator::add(row, col, value);
    }
    virtual void add(const matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<2, 2, double>& value)
    {
        TBaseMatrixAccumulator::add(row, col, value);
    }

    virtual void add(const matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<6, 6, float>& value)
    {
        TBaseMatrixAccumulator::add(row, col, value);
    }
    virtual void add(const matrixaccumulator::no_check_policy&, sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<6, 6, double>& value)
    {
        TBaseMatrixAccumulator::add(row, col, value);
    }

    template <typename T>
    void indexCheckedAdd(sofa::SignedIndex row, sofa::SignedIndex col, const T& value)
    {
        if constexpr (TStrategy::verify_index::value)
        {
            if (indexVerificationStrategy)
            {
                indexVerificationStrategy->checkRowIndex(row);
                indexVerificationStrategy->checkColIndex(col);
            }
        }
        add(matrixaccumulator::no_check, row, col, value);
    }
};

/**
 * Composite class of MatrixAccumulatorInterface
 */
template<class TMatrixAccumulator>
class ListMatrixAccumulator : public TMatrixAccumulator
{
    static_assert(std::is_base_of_v<MatrixAccumulatorInterface, TMatrixAccumulator>, "Invalid template argument");
    using InternalListMatrixAccumulator = sofa::type::vector<TMatrixAccumulator*>;

public:
    void push_back(TMatrixAccumulator* m)
    {
        m_list.push_back(m);
    }

    void remove(TMatrixAccumulator* m)
    {
        m_list.erase(std::remove(m_list.begin(), m_list.end(), m), m_list.end());
    }

    [[nodiscard]]
    bool empty() const
    {
        return m_list.empty();
    }

    void clear() override
    {
        for (auto* mat : m_list)
        {
            mat->clear();
        }
    }

    [[nodiscard]]
    typename InternalListMatrixAccumulator::size_type size() const
    {
        return m_list.size();
    }

    void add(sofa::SignedIndex i, sofa::SignedIndex j, float value) override
    {
        for (auto* mat : m_list)
        {
            mat->add(i, j, value);
        }
    }
    void add(sofa::SignedIndex i, sofa::SignedIndex j, double value) override
    {
        for (auto* mat : m_list)
        {
            mat->add(i, j, value);
        }
    }
    void add(sofa::SignedIndex i, sofa::SignedIndex j, const sofa::type::Mat3x3f& value) override
    {
        for (auto* mat : m_list)
        {
            mat->add(i, j, value);
        }
    }
    void add(sofa::SignedIndex i, sofa::SignedIndex j, const sofa::type::Mat3x3d& value) override
    {
        for (auto* mat : m_list)
        {
            mat->add(i, j, value);
        }
    }

    [[nodiscard]]
    const InternalListMatrixAccumulator& getAccumulators() const
    {
        return m_list;
    }

private:
    InternalListMatrixAccumulator m_list;
};

namespace matrixaccumulator
{
    /**
     * Type of contribution added into the global matrix
     * Since they are not treated equivalently, they need to be distinguished
     */
    enum class Contribution : sofa::Size
    {
        STIFFNESS,
        MASS,
        DAMPING,
        GEOMETRIC_STIFFNESS
    };

    template<Contribution c>
    struct ContributionName {};

    /**
     * Example: GetContributionName<Contribution::STIFFNESS>() returns "Stiffness"
     */
    template<Contribution c>
    constexpr std::string_view GetContributionName() { return ContributionName<c>::Name(); }

    template<> struct ContributionName<Contribution::STIFFNESS>
    {
        static constexpr std::string_view Name() { return "Stiffness"; }
    };
    template<> struct ContributionName<Contribution::MASS>
    {
        static constexpr std::string_view Name() { return "Mass"; }
    };
    template<> struct ContributionName<Contribution::DAMPING>
    {
        static constexpr std::string_view Name() { return "Damping"; }
    };
    template<> struct ContributionName<Contribution::GEOMETRIC_STIFFNESS>
    {
        static constexpr std::string_view Name() { return "GeometricStiffness"; }
    };

    /**
     * Provides member typedef @type for known Contribution using SFINAE
     *
     * Typedef @type is an abstract strong type derived from @MatrixAccumulatorInterface and depending on @c
     * Typedef @ComponentType is the type of object associated to the Contribution
     */
    template<Contribution c>
    struct get_abstract_strong
    {
    };

    /**
     * Helper alias
     *
     * Example: get_abstract_strong_type<Contribution::STIFFNESS>
     */
    template<Contribution c>
    using get_abstract_strong_type = typename get_abstract_strong<c>::type;

    /**
     * Helper alias
     *
     * Example: get_component_type<Contribution::STIFFNESS>
     */
    template<Contribution c>
    using get_component_type = typename get_abstract_strong<c>::ComponentType;

    template<Contribution c>
    using get_matrix_builder_type = typename get_abstract_strong<c>::MatrixBuilderType;


    /**
     * Provides member typedef @type for known Contribution using SFINAE
     *
     * Typedef @type is an abstract strong type derived from @ListMatrixAccumulator and depending on @c
     */
    template<Contribution c>
    struct get_list_abstract_strong
    {
    };

    template<Contribution c>
    using get_list_abstract_strong_type = typename get_list_abstract_strong<c>::type;
}

} //namespace sofa::core
