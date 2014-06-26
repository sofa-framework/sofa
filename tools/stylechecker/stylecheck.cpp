
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
using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace llvm;

using namespace std;

std::vector<std::string> excludedPathPatterns={"extlibs/",
                                               "/usr/include/qt4/",
                                               "framework/sofa/helper",
                                               "framework/sofa/defaulttype",
                                               "framework/sofa/core",
                                               "simulation/common"};

cl::list<string> listofpath("L", cl::Prefix, cl::desc("Specify path pattern"), cl::value_desc("directory")) ;

bool isInExcludedPath(const std::string& path){
    if(listofpath.size()!=0){
        for(auto pattern : listofpath)
        {
            if( path.find(pattern) != std::string::npos )
            {
                return false ;
            }
        }
        return true;
    }else{
        for(auto pattern : excludedPathPatterns)
        {
            if( path.find(pattern) != std::string::npos )
            {
                return true ;
            }
        }
        return false ;
    }
}

class StyleChecker : public RecursiveASTVisitor<StyleChecker> {
public:

    void setContext(const ASTContext* ctx){
        Context=ctx;
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

        FullSourceLoc FullLocation = Context->getFullLoc(record->getLocStart());
        // Check this declaration is not in the system headers...
        if ( FullLocation.isValid() && !exclude(FullLocation.getManager() , record) )
        {
            const SourceManager& smanager=Context->getSourceManager();

            // Now check the attributes...
            RecordDecl::field_iterator it=record->field_begin() ;
            for(;it!=record->field_end();it++){
                clang::FieldDecl* ff=*it;

                SourceRange declsr=ff->getMostRecentDecl()->getSourceRange() ;
                SourceLocation sl=declsr.getBegin();
                std::string name=ff->getName() ;

                if( smanager.getFileEntryForID(smanager.getFileID(sl)) == NULL ){
                    continue ;
                }

                if(isInExcludedPath(smanager.getFileEntryForID(smanager.getFileID(sl))->getName())){
                    continue ;
                }


                if(name.size()==0){
                    continue ;
                }

                // RULES NUMBER 1: The name of members cannot be terminated by an underscore.
                if(name.rfind("_")!=name.size()-1){
                }else{
                    std::cerr << smanager.getFileEntryForID(smanager.getFileID(sl))->getName()
                              << ":" << smanager.getPresumedLineNumber(sl)
                              << ":" << smanager.getPresumedColumnNumber(sl)
                              << ": warning: member [" << record->getNameAsString() << ":" <<name << "] is violating the sofa coding style http://www.sofa.../codingstyle.html...member's name cannot be terminated with an underscore.' " << std::endl;
                }

                // THE FOLLOWING RULES ARE ONLY CHECK ON PRIVATE & PROTECTED FIELDS
                if(ff->getAccess()==AS_public){
                    continue ;
                }

                if(ff->getType()->isPointerType()){
                    if(name.size() > 2 && name[0] == 'p' && isupper(name[1]) ){}
                    else{
                        std::cerr << smanager.getFileEntryForID(smanager.getFileID(sl))->getName()
                                  << ":" << smanager.getPresumedLineNumber(sl)
                                  << ":" << smanager.getPresumedColumnNumber(sl)
                                  << ": warning: member [" << record->getNameAsString() << ":" <<name << "] is violating the sofa coding style http://www.sofa.../codingstyle.html... it should prefixed with pUpperCased " << std::endl;
                    }
                    continue ;
                }

                if(ff->getType()->isBooleanType()){
                    if(name.size() > 2 && name[0] == 'b' && isupper(name[1]) ){}
                    else{
                        std::cerr << smanager.getFileEntryForID(smanager.getFileID(sl))->getName()
                                  << ":" << smanager.getPresumedLineNumber(sl)
                                  << ":" << smanager.getPresumedColumnNumber(sl)
                                  << ": warning: member [" << record->getNameAsString() << ":" <<name << "] is violating the sofa coding style http://www.sofa.../codingstyle.html... it should prefixed with bUpperCased " << std::endl;
                    }
                    continue ;
                }

                CXXRecordDecl* rd=ff->getType()->getAsCXXRecordDecl() ;
                if(rd){
                    std::string type=rd->getNameAsString() ;
                    if(type.find("Data")!=std::string::npos){
                        if(name.find("d_")==0){
                        }else{
                            std::cerr << smanager.getFileEntryForID(smanager.getFileID(sl))->getName()
                                      << ":" << smanager.getPresumedLineNumber(sl)
                                      << ":" << smanager.getPresumedColumnNumber(sl)
                                      << ": warning: member [" << record->getNameAsString() << ":" <<name << "] is violating the sofa coding style http://www.sofa.../codingstyle.html...all Data<> members should start with d_ " << std::endl;
                        }
                    }else if(type.find("SingleLink")!=std::string::npos || type.find("DualLink")!=std::string::npos){
                        if(name.find("d_")==0){
                        }else{
                            std::cerr << smanager.getFileEntryForID(smanager.getFileID(sl))->getName()
                                      << ":" << smanager.getPresumedLineNumber(sl)
                                      << ":" << smanager.getPresumedColumnNumber(sl)
                                      << ": warning: member [" << record->getNameAsString() << ":" <<name << "] is violating the sofa coding style http://www.sofa.../codingstyle.html...all Link<> members should start with l_ " << std::endl;
                        }
                    }
                }else{
                    if(name.find("m_")==0){
                    }else{
                        std::cerr << smanager.getFileEntryForID(smanager.getFileID(sl))->getName()
                                  << ":" << smanager.getPresumedLineNumber(sl)
                                  << ":" << smanager.getPresumedColumnNumber(sl)
                                  << ": warning: member [" << record->getNameAsString() << ":" << name << "] is violating the sofa coding style http://www.sofa.../codingstyle.html...all non public attributes should start with m_ " << std::endl;
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
    static cl::OptionCategory MyToolCategory("StyleChecker");
    CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);


    ClangTool Tool(OptionsParser.getCompilations(),
                   OptionsParser.getSourcePathList());

    std::vector<std::string> localFilename;
    for(unsigned int i=1;i<argc;i++){
        localFilename.push_back(argv[i]);
    }

    // Build the ast for each file given as arguments
    std::vector<std::unique_ptr<ASTUnit> > asts;
    Tool.buildASTs(asts);

    // Create a StyleChecker visitor
    StyleChecker* sr=new StyleChecker();

    // For each file...
    for(unsigned int i=0;i<asts.size();i++){
        ASTContext& ctx=asts[i]->getASTContext();

        int j=0;
        auto it=ctx.getSourceManager().fileinfo_begin() ;
        for(;it!=ctx.getSourceManager().fileinfo_end();++it){
            j++ ;
        }
        auto& smanager=ctx.getSourceManager();
        std::cerr << smanager.getFileEntryForID(smanager.getMainFileID())->getName()
                  << ":1:1: info: number of loaded include files: " << j << std::endl ;


        //ésr->setContext(&ctx);
        //sr->TraverseDecl(ctx.getTranslationUnitDecl());
    }

}
