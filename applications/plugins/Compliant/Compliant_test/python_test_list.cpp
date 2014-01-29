#include "Python_test.h"

// TODO build a list of .py files in Python_test::path() by static
// initialization in a std::vector, then launch tests on them

// in the meantime, this:

const char* test_files[] = {

	"Example.py",
	"LambdaPropagation.py",
	"UniformCompliance.py",
	"AffineMultiMapping.py"

	// add yours here
};

namespace sofa {

INSTANTIATE_TEST_CASE_P(Batch,
						Python_test,
						::testing::ValuesIn(test_files));

}
