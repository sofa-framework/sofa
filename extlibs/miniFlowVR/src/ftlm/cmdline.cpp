/******* COPYRIGHT ************************************************
*                                                                 *
*                             FlowVR                              *
*                       Template Library                          *
*                                                                 *
*-----------------------------------------------------------------*
* COPYRIGHT (C) 20054 by                                          *
* Laboratoire Informatique et Distribution (UMR5132) and          *
* INRIA Project MOVI. ALL RIGHTS RESERVED.                        *
*                                                                 *
* This source is covered by the GNU LGPL, please refer to the     *
* COPYING file for further information.                           *
*                                                                 *
*-----------------------------------------------------------------*
*                                                                 *
*  Original Contributors:                                         *
*    Jeremie Allard,                                              *
*    Clement Menier.                                              *
*                                                                 * 
*******************************************************************
*                                                                 *
* File: src/ftlm/cmdline.cpp                                      *
*                                                                 *
* Contacts: 10/26/2005 Jeremie Allard <Jeremie.Allard@imag.fr>    *
*                                                                 *
******************************************************************/
#include <ftl/cmdline.h>
#if defined(WIN32) || defined(_XBOX) || defined(PS3)
//short getopt_long_only replacement
struct option
{
	const char *name;
	int has_arg;
	int *flag;
	int val;
};
char* optarg = NULL;
int optind = 1;
int optopt = 0;
int getopt_long_only(int argc,char **argv, const char* shortoptions, option* options, int* option_index)
{
	int nopts = 0;
	while (options[nopts].name != NULL) ++nopts;
	optarg = NULL;
	if (optind >= argc) return -1;
	char* opt = argv[optind];
	if (opt[0] != '-') return -1;
	++optind;
	++opt;
	if (opt[0] == '-') ++opt; // -- is same as -
	if (opt[0] == '\0') return -1; // end of parsing
	if (opt[1] == '\0') // short option
	{
		optopt = opt[0];
		const char* shorto = strchr(shortoptions, optopt);
		if (shorto == NULL)
		{
			std::cerr << "ERROR: option "<<opt<<" not found.\n";
			return '?';
		}
		if (shorto[1] == ':')
		{
			if (optind < argc)
			{
				optarg = argv[optind++];
			}
			else if (shorto[1] != ':')
			{
				std::cerr << "ERROR: option "<<opt<<" requires an argument.\n";
				return ':';
			}
		}
		return shorto[0];
	}
	else
	{
		// check long options
		for(int i=0;i<nopts;i++)
		{
			if (!strcmp(opt,options[i].name))
			{
				*option_index = i;
				if (options[i].has_arg && optind < argc)
				{
					optarg = argv[optind++];
				}
				else if (options[i].has_arg == 1)
				{
					std::cerr << "ERROR: option "<<opt<<" requires an argument.\n";
					return ':';
				}
				if (options[i].flag)
				{
					*options[i].flag = options[i].val;
					return 0;
				}
				else
				{
					return options[i].val;
				}
			}
		}
		std::cerr << "ERROR: option "<<opt<<" not found.\n";
		return '?';
	}
	return -1;
}
#else
#include <getopt.h>
#endif
#include <iostream>

namespace ftl
{

FlagOption* helpoption = NULL;

BaseOption::BaseOption(const char* _longname, char _shortname, const char* _description, ArgType _arg)
: longname(_longname), shortname(_shortname), description(_description), arg(_arg), hasdefault(false), count(0)
{
  std::vector<BaseOption*>& opts = CmdLine::opts();
  opts.push_back(this);
}

BaseOption::~BaseOption()
{
  std::vector<BaseOption*>& opts = CmdLine::opts();
  std::vector<BaseOption*>::iterator it;
  for (it = opts.begin(); it != opts.end(); ++it)
  {
    opts.erase(it);
    break;
  }
}

CmdLine::CmdLine(const char* _description)
: description(_description)
{
}

bool CmdLine::parse(int argc, char** argv, bool* error)
{
  // First create the list of options
  int nopts = (int)opts().size();
  struct option* options = new struct option[nopts+1];
  std::string shortoptions;
  for (int i=0;i<nopts;i++)
  {
    options[i].name = opts()[i]->longname;
    options[i].has_arg = opts()[i]->arg;
    options[i].flag = NULL;
    options[i].val = 256+i;
    if (opts()[i]->shortname!='\0')
    {
      shortoptions+=opts()[i]->shortname;
      if (opts()[i]->arg == BaseOption::REQ_ARG)
	shortoptions+=':';
      else if (opts()[i]->arg == BaseOption::OPT_ARG)
	shortoptions+="::";
    }
  }
  options[nopts].name = NULL;
  options[nopts].has_arg = 0;
  options[nopts].flag = NULL;
  options[nopts].val = 0;

  //std::cerr << "ShortOptions: "<<shortoptions<<std::endl;

  int option_index = -1;
  int c;

  const char* program = strrchr(argv[0],'/');
  if (program != NULL) ++program;
  else program = argv[0];

#if !defined(__APPLE__)
  while( (c = getopt_long_only(argc,argv,shortoptions.data(),options,&option_index)) != -1)
#else
  while( (c = getopt_long(argc,argv,shortoptions.data(),options,&option_index)) != -1)
#endif
  {
    if (c=='?' || c==':')
    {
      if (error!=NULL) *error = true;
      delete [] options;
      return false;
    }
    bool longopt = true;
    if (c!=0 && (unsigned)option_index>=(unsigned)nopts)
    {
      longopt = false;
      for (int i=0;i<nopts;i++)
        if (opts()[i]->shortname==c)
	{
	  option_index = i;
	  break;
	}
    }
    if ((unsigned)option_index>=(unsigned)nopts)
    {
      std::cerr<<"Error parsing options"<<std::endl;
      if (error!=NULL) *error = true;
      delete [] options;
      return false;
    }

    BaseOption* opt = opts()[option_index];

    if (optarg && *optarg)
    {
      if (opt->arg == BaseOption::NO_ARG)
      {
	std::cerr << program << ": option ";
	if (longopt) std::cerr << "--"<<opt->longname;
	else         std::cerr << '-'<<opt->shortname;
	std::cerr << " requires no value while \""<<optarg<<"\" was specified."<<std::endl;
	if (error!=NULL) *error = true;
    delete [] options;
	return false;
      }
      else
      {
	if (!opt->set(optarg))
	{
	  std::cerr << program << ": incorrect value for option ";
	  if (longopt) std::cerr << "--"<<opt->longname;
	  else         std::cerr << "-"<<opt->shortname;
	  std::cerr << " .\n";
	  if (error!=NULL) *error = true;
      delete [] options;
	  return false;
	}
      }
    }
    else
    {
      if (opt->arg == BaseOption::REQ_ARG)
      {
	std::cerr << program << ": option ";
	if (longopt) std::cerr << "--"<<opt->longname;
	else         std::cerr << '-'<<opt->shortname;
	std::cerr << " requires a value."<<std::endl;
	if (error!=NULL) *error = true;
    delete [] options;
	return false;
      }
      else
      {
	if (!opt->set())
	{
	  std::cerr << program << ": error while treating option ";
	  if (longopt) std::cerr << "--"<<opt->longname;
	  else         std::cerr << '-'<<opt->shortname;
	  std::cerr << " .\n";
	  if (error!=NULL) *error = true;
      delete [] options;
	  return false;
	}
      }
    }
    option_index=-1;
  }
  for (int i=optind;i<argc;i++)
  {
    args.push_back(argv[i]);
  }
  if (error!=NULL) *error = false;
  if (helpoption!=NULL && helpoption->count>0)
  {
    if (description==NULL) std::cout << program<< "\n";
    std::cout << help() << std::endl;
    delete [] options;
    return false;
  }
  delete [] options;
  return true;
}

std::string CmdLine::help()
{
  std::ostringstream ss;
  if (description!=NULL)
    ss << description << std::endl;
  ss << "Available options:\n";
  for (unsigned int i=0;i<opts().size();i++)
    ss << opts()[i]->help() << std::endl;
  return ss.str();
}


std::vector<BaseOption*>& CmdLine::opts()
{
  static std::vector<BaseOption*> _opts;

  if (_opts.empty())
  {
    // Default options are added first
    _opts.push_back(NULL); // make sure the vector is not empty anymore

    helpoption = new FlagOption("help",'h',"Display available options");

    // Then remove the bogus first entry
    _opts.erase(_opts.begin());
  }
  // Now I can add the current option

  return _opts;
}

BaseOption* CmdLine::getOpt(const std::string longname_)
  {
    for( std::vector<BaseOption*>::const_iterator it =opts().begin(); it <opts().end(); ++it)
      {
	if ( strcmp((*it)->longname,longname_.c_str()) == 0) return (*it);
      }
    return NULL;
  }

  void CmdLine::setDesc(const std::string // desc
			)
  {
    //    strcpy(description,desc.c_str()); 
  }

} // namespace ftl
