
#include <gtest/gtest.h>
#include <sofa/simulation/common/Node.h>
#include <string>

namespace sofa {

namespace simulation {
class SceneLoader;
}

class Python_test : public ::testing::TestWithParam<const char*> {

protected:

	simulation::SceneLoader* loader;

public:

	static std::string path();

	struct result {
		result(bool value) : value( value ) { }
		bool value;
	};
	
	void run(const char* );

	Python_test();
	~Python_test();

};



}
