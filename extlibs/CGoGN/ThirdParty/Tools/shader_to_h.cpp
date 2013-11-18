#include <iostream>
#include <fstream>
#include <sstream>

int main(int argc, char **argv)
{
	char buffer[512];
	for (int i=1; i< argc; ++i)
	{
		std::string filename(argv[i]);

		std::ifstream fs(filename.c_str(), std::ios::in);
		if (!fs.good())
		{
			std::cerr << "Unable to open file " << filename << std::endl;
			return 1;
		}

		size_t last_slash = filename.rfind('/');
		if (last_slash == std::string::npos)
		{
			last_slash = filename.rfind('\\');
			if (last_slash == std::string::npos)
				last_slash = 0;
			else
				++last_slash;
		}
		else
			++last_slash;
		
		std::string outName = filename.substr(last_slash,filename.size()-last_slash);
		
		std::stringstream ssi;
		std::stringstream sso;			
		
		std::ifstream fsi(outName.c_str(),std::ios::in);
		if (fsi.good())
		{
			while (!fsi.eof())
			{
				fsi.getline(buffer,512);
				if (!fsi.eof())
					ssi << buffer << std::endl ;
			}
			fsi.close();
		}
		
		
		// fist line
		fs.getline(buffer,512);				
		char *sub=buffer;
		while ((*sub=='/') || (*sub==' '))
			++sub;
		sso << "std::string "<<sub<< " =";

		// text of shader		
		unsigned int nbbl=0;
		while (!fs.eof())
		{
			fs.getline(buffer,512);
			//std::cout << buffer << std::endl;
			if (*buffer!=0)
			{
				for (unsigned int i=0; i<nbbl;++i)
					sso << std::endl <<"\"\\n\"";
				nbbl=0;
				sso << std::endl << "\"" << buffer <<"\\n\"";
			}
			else
				nbbl++;
		};
		sso << ";"<< std::endl<< std::endl;	
		
		std::string ssostr = sso.str();
		if (ssostr != ssi.str())
		{
			std::ofstream fso(outName.c_str(),std::ios::out);
			fso << ssostr;
			std::cout << "Shader_to_h: "<< outName << " copy"<< std::endl;
			fso.close();
		}
		else
			std::cout << "Shader_to_h: "<< outName << " ok"<< std::endl;
		
		fs.close();
		
	}
	
	
	return 0;
	
}
		
