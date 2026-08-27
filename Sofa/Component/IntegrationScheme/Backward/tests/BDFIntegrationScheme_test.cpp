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

#include <gtest/gtest.h>
#include <sofa/component/integrationscheme/backward/BDFIntegrationScheme.h>
#include <sofa/testing/NumericTest.h>

namespace sofa
{

/// BDFIntegrationScheme::computeFactors() is a protected instance method operating on the
/// protected m_samples/m_dt/m_aFactors/m_bFactors members (unlike the former BDFOdeSolver,
/// which exposed a static computeLinearMultiStepCoefficients helper). This subclass only
/// exists to give the test access to that machinery without going through a full scene.
class BDFIntegrationSchemeTest : public component::integrationscheme::backward::BDFIntegrationScheme
{
public:
    void computeLinearMultiStepCoefficients(const std::deque<SReal>& samples, SReal dt,
                                             sofa::type::vector<SReal>& a_coef,
                                             sofa::type::vector<SReal>& b_coef)
    {
        m_samples = samples;
        m_dt = dt;
        computeFactors();
        a_coef.assign(m_aFactors.begin(), m_aFactors.end());
        b_coef.assign(m_bFactors.begin(), m_bFactors.end());
    }
};

// Unlike the former BDFOdeSolver::computeLinearMultiStepCoefficients, computeFactors() no longer
// normalizes the coefficients by a_coef[order] (that factor is mathematically ~1 already), so the
// result is only equal to the exact fractions below up to floating-point rounding: use EXPECT_NEAR
// rather than the ULP-strict EXPECT_FLOATINGPOINT_EQ.
constexpr SReal coefTolerance = 1e-9_sreal;

void testOrder1(SReal start, SReal end)
{
    sofa::type::vector<SReal> a_coef;
    sofa::type::vector<SReal> b_coef;
    std::deque<SReal> samples { start, end};
    BDFIntegrationSchemeTest solver;
    solver.computeLinearMultiStepCoefficients(samples, end - start, a_coef, b_coef);

    ASSERT_EQ(a_coef.size(), b_coef.size());
    ASSERT_EQ(a_coef.size(), 2);

    EXPECT_NEAR(a_coef[0], -1_sreal, coefTolerance);
    EXPECT_NEAR(a_coef[1], 1_sreal, coefTolerance);

    EXPECT_NEAR(b_coef[0], 0_sreal, coefTolerance);
    EXPECT_NEAR(b_coef[1], 1_sreal, coefTolerance);
}

TEST(BDFIntegrationSchemeTestSuite, Order1)
{
    for (const auto [start, end] : sofa::type::vector<std::pair<SReal, SReal>>{ {0,1}, {0, 12}, {12, 13}, {100, 900}, {900, 100}})
    {
        testOrder1(start, end);
    }
}

void testOrder2(SReal start, SReal dt)
{
    sofa::type::vector<SReal> a_coef;
    sofa::type::vector<SReal> b_coef;
    std::deque<SReal> samples { start, start + dt, start + 2 * dt};
    BDFIntegrationSchemeTest solver;
    solver.computeLinearMultiStepCoefficients(samples, dt, a_coef, b_coef);

    ASSERT_EQ(a_coef.size(), b_coef.size());
    ASSERT_EQ(a_coef.size(), 3);

    EXPECT_NEAR(a_coef[0], 1_sreal / 3_sreal, coefTolerance);
    EXPECT_NEAR(a_coef[1], -4_sreal / 3_sreal, coefTolerance);
    EXPECT_NEAR(a_coef[2], 1_sreal, coefTolerance);

    EXPECT_NEAR(b_coef[0], 0_sreal, coefTolerance);
    EXPECT_NEAR(b_coef[1], 0_sreal, coefTolerance);
    EXPECT_NEAR(b_coef[2], 2_sreal / 3_sreal, coefTolerance);
}

TEST(BDFIntegrationSchemeTestSuite, Order2)
{
    for (const auto [start, end] : sofa::type::vector<std::pair<SReal, SReal>>{ {0,1}, {0, 12}, {12, 1}, {100, 1}, {900, -1}})
    {
        testOrder2(start, end);
    }
}

void testOrder3(SReal start, SReal dt)
{
    sofa::type::vector<SReal> a_coef;
    sofa::type::vector<SReal> b_coef;
    std::deque<SReal> samples { start, start + dt, start + 2 * dt, start + 3 * dt};
    BDFIntegrationSchemeTest solver;
    solver.computeLinearMultiStepCoefficients(samples, dt, a_coef, b_coef);

    ASSERT_EQ(a_coef.size(), b_coef.size());
    ASSERT_EQ(a_coef.size(), 4);

    EXPECT_NEAR(a_coef[0], -2_sreal / 11_sreal, coefTolerance);
    EXPECT_NEAR(a_coef[1], 9_sreal / 11_sreal, coefTolerance);
    EXPECT_NEAR(a_coef[2], -18_sreal / 11_sreal, coefTolerance);
    EXPECT_NEAR(a_coef[3], 1_sreal, coefTolerance);

    EXPECT_NEAR(b_coef[0], 0_sreal, coefTolerance);
    EXPECT_NEAR(b_coef[1], 0_sreal, coefTolerance);
    EXPECT_NEAR(b_coef[2], 0_sreal, coefTolerance);
    EXPECT_NEAR(b_coef[3], 6_sreal / 11_sreal, coefTolerance);
}

TEST(BDFIntegrationSchemeTestSuite, Order3)
{
    for (const auto [start, end] : sofa::type::vector<std::pair<SReal, SReal>>{ {0,1}, {0, 12}, {12, 1}, {100, 1}, {900, -1}})
    {
        testOrder3(start, end);
    }
}

}
