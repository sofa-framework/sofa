
#include <gtest/gtest.h>
#include <sofa/simulation/common/Node.h>
#include <string>
#include <plugins/SofaPython/SceneLoaderPY.h>

namespace sofa {


/// a Python_test is defined by a python filepath and optional arguments
struct Python_test_data
{
    Python_test_data( const std::string& filepath, const std::vector<std::string>& arguments ) : filepath(filepath), arguments(arguments) {}
    std::string filepath;
    std::vector<std::string> arguments; // argc/argv in the python script
};

/// utility to build a static list of Python_test_data
struct Python_test_list
{
    std::vector<Python_test_data> list;
protected:
    /// add a Python_test_data with given path
    void addTest( const std::string& filename, const std::string& path="", const std::vector<std::string>& arguments=std::vector<std::string>(0) )
    {
        list.push_back( Python_test_data( filepath(path,filename), arguments ) );
    }
private:
    /// concatenate path and filename
    static std::string filepath( const std::string& path, const std::string& filename )
    {
        if( path!="" )
            return path+"/"+filename;
        else
            return filename;
    }
};



class Python_test : public ::testing::TestWithParam<Python_test_data> {

protected:

    simulation::SceneLoaderPY loader;

public:

	struct result {
		result(bool value) : value( value ) { }
		bool value;
	};
	
    void run( const Python_test_data& );

	Python_test();
	~Python_test();


};


}
