
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Comment.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Parse/ParseAST.h"
#include "clang/AST/Mangle.h"
#include "clang/Rewrite/Frontend/Rewriters.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "utilsllvm.h"
#include "fileutils.h"

#include <iostream>
#include <vector>
#include <string>
#include <locale>

using namespace clang;
using namespace clang::ast_matchers ;
using namespace clang::comments ;
using namespace clang::tooling ;

using namespace llvm ;

using namespace std ;

vector<string> systemexcluded={ "extlibs/",
                                "/usr/include/",
                                "/usr/lib/",
                                "/lib/clang/"} ;

vector<string> excludedPathPatterns={};


static cl::OptionCategory MyToolCategory("Stylecheck.exe");
cl::list<string> userexcluded("E", llvm::cl::Prefix, llvm::cl::desc("Specify path pattern to exclude"), cl::cat(MyToolCategory)) ;
cl::list<string> userincluded("L", llvm::cl::Prefix, llvm::cl::desc("Specify path pattern to be restricted in"), cl::cat(MyToolCategory)) ;
cl::opt<bool> verbose("v", cl::desc("Set verbose mode"), cl::init(false), cl::cat(MyToolCategory));
cl::opt<bool> shouldAdvice("a", cl::desc("Add advice mode"), cl::init(false), cl::cat(MyToolCategory));
cl::opt<int> numberofincludes("n", cl::desc("Number of include files before a warning is emited [default is 20]"), cl::init(20), cl::cat(MyToolCategory));
cl::opt<bool> isLinking("W", cl::desc("This option is to detect that stylecheck is called in linking mode...do not use it"), cl::cat(MyToolCategory));

bool isInExcludedPath(const string& path, const vector<string>& excludedPaths){
    if(userincluded.size()!=0){
        for(auto pattern : userincluded)
        {
            if( path.find(pattern) != string::npos )
            {
                return false ;
            }
        }
        return true;
    }else{
        for(auto pattern : excludedPaths)
        {
            if( path.find(pattern) != string::npos )
            {
                return true ;
            }
        }
        return false ;
    }
    return false ;
}

bool isLowerCamlCase(const string& name)
{
    if(name.size()==0)
        return true ;

    if(!islower(name[0]))
        return false ;

    for( auto c : name ){
        if(!std::isalnum(c))
            return false;
    }
    return true ;
}

bool isUpperCamlCase(const string& name)
{
    if(name.size()==0)
        return true ;

    if(!isupper(name[0]))
        return false ;

    for( auto c :  name ){
        if(!std::isalnum(c))
            return false;
    }
    return true ;
}

bool islower(const std::string& name)
{
    for( auto c : name )
        if( ! std::islower(c) )
            return false ;
    return true ;
}

void printErrorV1(const string& filename, const int line, const int col, const string& varname){
    cerr << filename << ":" << line << ":" << col <<  ": warning: initialization of [" << varname << "] is violating the sofa coding style rules V1. " << endl ;
    cerr << " Variables should always have initializer.  Built-in data types (int, float, char, pointers...) have no default values, " << endl ;
    cerr << " so they're undefined until you give them one and without having been properly initialized, hard-to-track bugs can occur. " << endl ;
    cerr << " In addition, if they're initialized after having been declared, there is a risk that someone later inadvertently deletes " << endl ;
    cerr << " or moves the line where they're given a value. " << endl ;
    cerr << " Finally when instantiating a class or structure, you pay the cost of a constructor call, whether it is the default one or user-provided. " << endl ;
    cerr << " You can found the complete Sofa coding guidelines at: http://www.sofa-framework.com/codingstyle/coding-guide.html" << endl  << endl ;
}


void printErrorN1(const string& filename, const int line, const int col, const string& nsname){
    cerr << filename << ":" << line << ":" << col <<  ": warning: namespace [" << nsname << "] is violating the sofa coding style rules N1. " << endl ;
    cerr << " By convention, all namespaces must be in lowercase.' " << endl;
    cerr << " You can found the complete Sofa coding guidelines at: http://www.sofa-framework.com/codingstyle/coding-guide.html" << endl  << endl ;
}


void printErrorM1(const string& filename, const int line, const int col, const string& classname, const string& name){
    cerr << filename << ":" << line << ":" << col <<  ": warning: member [" << classname << ":" << name << "] is violating the sofa coding style rule M1. " << endl ;
    cerr << " Data fields are importants concept in Sofa, to emphasize this fact that they are not simple membre variable they should all be prefixed with d_" << endl;
    cerr << " You can found the complete Sofa coding guidelines at: http://www.sofa-framework.com/codingstyle/coding-guide.html" << endl ;
    cerr << " Suggested replacement: d_" << name << endl << endl ;
}

void printErrorM2(const string& filename, const int line, const int col, const string& classname, const string& name){
    cerr << filename << ":" << line << ":" << col <<  ": warning: member [" << classname << ":" << name << "] is violating the sofa coding style rule M2. " << endl ;
    cerr << " DataLink are importants concept in Sofa, to emphasize this fact that they are not simple membre variable they should all be prefixed with l_" << endl;
    cerr << " You can found the complete Sofa coding guidelines at: http://www.sofa-framework.com/codingstyle/coding-guide.html" << endl ;
    cerr << " Suggested replacement: s_" << name << endl << endl ;
}

void printErrorM3(const string& filename, const int line, const int col, const string& classname, const string& name){
    cerr << filename << ":" << line << ":" << col <<  ": warning: member [" << classname << ":" << name << "] is violating the sofa coding style rule M3. " << endl ;
    cerr << " To emphasize attributes membership of a class they should all be prefixed with m_" << endl;
    cerr << " You can found the complete Sofa coding guidelines at: http://www.sofa-framework.com/codingstyle/coding-guide.html" << endl ;
    cerr << " Suggested replacement: m_" << name << endl << endl ;
}

void printErrorM4(const string& filename, const int line, const int col, const string& classname){
    cerr << filename << ":" << line << ":" << col <<  ": warning: class [" << classname << "] is violating the sofa coding style rules M4. " << endl ;
    cerr << " y convention, all classes name must be in UpperCamlCase without any underscores '_'.' " << endl;
    cerr << " You can found the complete Sofa coding guidelines at: http://www.sofa-framework.com/codingstyle/coding-guide.html" << endl  << endl ;
}

void printErrorR1(const string& filename, const int sofacode, const int allcodes){
    cerr << filename << ":1:1: info: too much file are included. " << endl ;
    cerr << " To decrease compilation time as well as improving interfaces/ABI it is recommanded to include as few as possible files. " << endl ;
    cerr << " The current .cpp file finally ended in including and thus compiling " << allcodes << " other files. " << endl ;
    cerr << " There is " << sofacode << " sofa files among this "<< allcodes <<" other files. " << endl ;
    cerr << " To help fixing this issue you could use PIMPL or InterfaceBaseDesigned, more details " << endl ;
    cerr << " at http://www.sofa-framework.com/codingstyle/opaqueincludes.html " << endl << endl ;
}

class StyleChecker : public RecursiveASTVisitor<StyleChecker> {
public:

    void setContext(const ASTContext* ctx){
        Context=ctx;
    }

    bool VisitStmt(Stmt* stmt){
	if(Context == NULL )
	    return true ;

	if( stmt == NULL )
            return true ;

        FullSourceLoc FullLocation = Context->getFullLoc(stmt->getLocStart()) ;
        if ( !FullLocation.isValid() || exclude(FullLocation.getManager() , stmt) )
            return true ;

        // If we are on a declaration statement, check that we
        if(stmt->getStmtClass() == Stmt::DeclStmtClass) {
            auto& smanager=Context->getSourceManager() ;

            DeclStmt* declstmt=dyn_cast<DeclStmt>(stmt) ;
            for(auto cs=declstmt->decl_begin(); cs!=declstmt->decl_end();++cs) {
                Decl* decl = *cs;
                VarDecl* vardecl = dyn_cast<VarDecl>(decl) ;

                if(vardecl){
		    auto tmp=vardecl->getMostRecentDecl() ; 
		    if(tmp)
			decl=tmp ; 

                    SourceRange declsr=decl->getSourceRange() ;
                    SourceLocation sl=declsr.getBegin();

		    auto fileinfo=smanager.getFileEntryForID(smanager.getFileID(sl)) ;

      	            if(fileinfo==NULL || isInExcludedPath(fileinfo->getName(), excludedPathPatterns))
			continue ; 

                    if( vardecl->getAnyInitializer() == NULL ){
                        printErrorV1(fileinfo->getName(),
                                     smanager.getPresumedLineNumber(sl),
                                     smanager.getPresumedColumnNumber(sl),
                                     vardecl->getNameAsString()) ;

                    }
                }
            }
            //stmt->dumpColor() ;
        }else{
            // NOTHING TO DO
        }

        return true ;
    }

    bool VisitDecl(Decl* decl)
    {
        if(Context==NULL)
	    return true ;

	if(decl==NULL)
 	    return true ;


	return true ; 

        FullSourceLoc FullLocation = Context->getFullLoc(decl->getLocStart());
        if ( !FullLocation.isValid() || exclude(FullLocation.getManager() , decl) )
            return true ;

        /// Implement the different check on namespace naming.
        NamespaceDecl* nsdecl= dyn_cast<NamespaceDecl>(decl) ;
        if( nsdecl ){
	    string nsname=nsdecl->getNameAsString() ;
            if( islower(nsname) )
                return true ;

            auto& smanager = Context->getSourceManager() ;

	    Decl* mrdecl=decl->getMostRecentDecl() ;
	    if(mrdecl!=NULL)
		decl=mrdecl ; 

            SourceRange sr=decl->getSourceRange() ;
            SourceLocation sl=sr.getBegin();
	    auto fileinfo=smanager.getFileEntryForID(smanager.getFileID(sl)) ;

		 
            if(fileinfo==NULL || isInExcludedPath(fileinfo->getName(), excludedPathPatterns))
		return true ; 	

            printErrorN1(fileinfo->getName(),
                         smanager.getPresumedLineNumber(sl),
                         smanager.getPresumedColumnNumber(sl),
                         nsname) ;
	    return true; 
        }
        return RecursiveASTVisitor<StyleChecker>::VisitDecl(decl) ;
    }

    // http://clang.llvm.org/doxygen/classclang_1_1Stmt.html
    // For each declaration
    // http://clang.llvm.org/doxygen/classclang_1_1Decl.html
    // http://clang.llvm.org/doxygen/classclang_1_1CXXRecordDecl.html
    // and
    // http://clang.llvm.org/doxygen/classclang_1_1RecursiveASTVisitor.html
    bool VisitCXXRecordDecl(CXXRecordDecl *record) {
        if(Context==NULL)
            return true ;

	if(record==NULL)
 	    return true ;

        auto& smanager = Context->getSourceManager() ;

        FullSourceLoc FullLocation = Context->getFullLoc(record->getLocStart());
        // Check this declaration is not in the system headers...
        if ( FullLocation.isValid() && !exclude(FullLocation.getManager() , record) )
        {

            // Check the class name.
            // it should be in writtent in UpperCamlCase
            string classname=record->getNameAsString();

            if(!isUpperCamlCase(classname)){
                SourceRange declsr=record->getMostRecentDecl()->getSourceRange() ;
                SourceLocation sl=declsr.getBegin();

		auto fileinfo = smanager.getFileEntryForID(smanager.getFileID(sl)) ;

                if(fileinfo && !isInExcludedPath(fileinfo->getName(), excludedPathPatterns)){
	                printErrorM4(fileinfo->getName(),
	                             smanager.getPresumedLineNumber(sl),
	                             smanager.getPresumedColumnNumber(sl),
	                             classname) ;
		
		}
            }

            // Now check the attributes...
            RecordDecl::field_iterator it=record->field_begin() ;
            for(;it!=record->field_end();it++){
                clang::FieldDecl* ff=*it;

                SourceRange declsr=ff->getMostRecentDecl()->getSourceRange() ;
                SourceLocation sl=declsr.getBegin();
                std::string name=ff->getName() ;

		auto fileinfo = smanager.getFileEntryForID(smanager.getFileID(sl)) ; 

                if( fileinfo == NULL ){
                    continue ;
                }

                if(isInExcludedPath(fileinfo->getName(), excludedPathPatterns)){
                    continue ;
                }


                if(name.size()==0){
                    continue ;
                }

                const std::string filename=fileinfo->getName() ;
                const int line = smanager.getPresumedLineNumber(sl) ;
                const int col = smanager.getPresumedColumnNumber(sl) ;

                // RULES NUMBER 1: The name of members cannot be terminated by an underscore.
                if(name.rfind("_")!=name.size()-1){
                }else{
                    cerr << filename << ":" << line << ":" << col
                         << ": warning: member [" << classname << ":" <<name << "] is violating the sofa coding style http://www.sofa.../codingstyle.html...member's name cannot be terminated with an underscore.' " << std::endl;
                }

                /// THESES TWO RULES ARE NOW DEPRECATED BUT I KEEP THEM FOR HISTORY REASON
                // THE FOLLOWING RULES ARE ONLY CHECK ON PRIVATE & PROTECTED FIELDS
                if(ff->getAccess()==AS_public){
                    continue ;
                }

                /// THESES TWO RULES ARE NOW DEPRECATED BUT I KEEP THEM FOR HISTORY REASON
                /*
                if(ff->getType()->isPointerType()){
                    if(name.size() > 2 && name[0] == 'p' && isupper(name[1]) ){}
                    else{
                        std::cerr << smanager.getFileEntryForID(smanager.getFileID(sl))->getName()
                                  << ":" << smanager.getPresumedLineNumber(sl)
                                  << ":" << smanager.getPresumedColumnNumber(sl)
                                  << ": warning: member [" << record->getNameAsString() << ":" <<name << "] is violating the sofa coding style http://www.sofa-framework.com/codingstyle/codingstyle.html... it should prefixed with pUpperCased " << std::endl;
                    }
                    continue ;
                }

                if(ff->getType()->isBooleanType()){
                    if(name.size() > 2 && name[0] == 'b' && isupper(name[1]) ){}
                    else{
                        std::cerr << smanager.getFileEntryForID(smanager.getFileID(sl))->getName()
                                  << ":" << smanager.getPresumedLineNumber(sl)
                                  << ":" << smanager.getPresumedColumnNumber(sl)
                                  << ": warning: member [" << record->getNameAsString() << ":" <<name << "] is violating the sofa coding style http://www.sofa-framework.com/codingstyle/codingstyle.html... it should prefixed with bUpperCased " << std::endl;
                    }
                    continue ;
                }*/


                CXXRecordDecl* rd=ff->getType()->getAsCXXRecordDecl() ;
                if(rd){
                    std::string type=rd->getNameAsString() ;
                    if(type.find("Data")!=std::string::npos){
                        if(name.find("d_")==0){
                        }else{
                            printErrorM1(filename, line, col,
                                         classname, name) ;

                        }
                    }else if(type.find("SingleLink")!=std::string::npos || type.find("DualLink")!=std::string::npos){
                        if(name.find("d_")==0){
                        }else{
                            printErrorM2(filename, line, col, classname, name) ;
                        }
                    }
                }else{
                    if(name.find("m_")==0){
                    }else{
                        printErrorM3(filename, line, col, classname, name) ;
                    }
                }
            }
        }
        return true;
    }
private:
    const ASTContext *Context;
    MangleContext* mctx;
};


int main(int argc, const char** argv){
    CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);


    ClangTool Tool(OptionsParser.getCompilations(),
                   OptionsParser.getSourcePathList());

    if(isLinking){
    	return 0 ; 
    }

    std::vector<std::string> localFilename;
    for(unsigned int i=1;i<argc;i++){
        localFilename.push_back(argv[i]);
    }

    for(auto epath : systemexcluded){
	if(verbose)	
		cout << "SYSTEM PATH EXCLUDED: " << epath << endl ;
	excludedPathPatterns.push_back(epath) ;
    }

    for(auto epath : userexcluded){
	if(verbose)				
		cout << "USER PATH EXCLUDED:: " << epath << endl ;
	excludedPathPatterns.push_back(epath) ;
    }

    for(auto epath : userincluded){
	if(verbose)	
		cout << "PATH RESTRICTED TO:: " << epath << endl ;
    }

    // Build the ast for each file given as arguments
    std::vector<std::unique_ptr<ASTUnit> > asts ;
    Tool.buildASTs(asts) ;

    // Create a StyleChecker visitor
    StyleChecker* sr=new StyleChecker() ;

    // For each file...
    for(unsigned int i=0;i<asts.size();i++){
        auto& ctx=asts[i]->getASTContext() ;

        sr->setContext(&ctx) ;
        sr->TraverseDecl(ctx.getTranslationUnitDecl()) ;

        /// Now check other rules as the one trying to keep as few as possible include files.
        ///
	if(shouldAdvice){
        int j=0 ;
        int sofacode=-1 ;
        auto it=ctx.getSourceManager().fileinfo_begin() ;
        for(;it!=ctx.getSourceManager().fileinfo_end();++it){
            j++ ;

            string filepathname = it->first->getName() ;
            if(!isInExcludedPath(filepathname, systemexcluded)){
                sofacode++ ;
            }

        }
        auto& smanager=ctx.getSourceManager() ;
        if( sofacode > numberofincludes || j > 300 )
        {
            printErrorR1(smanager.getFileEntryForID(smanager.getMainFileID())->getName(), sofacode, j) ;
        }
	}
    }
}
