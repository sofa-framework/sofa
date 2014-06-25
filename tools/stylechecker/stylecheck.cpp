
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

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace llvm;

std::vector<std::string> excludedPathPatterns={"extlibs/", "/usr/include/qt4/", "framework/sofa/helper", "framework/sofa/defaulttype", "framework/sofa/core"};

bool isInExcludedPath(const std::string& path){
    for(auto pattern : excludedPathPatterns)
    {
        if( path.find(pattern) != std::string::npos )
        {
            return true ;
        }
    }
    return false ;
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
        return true;
        FullSourceLoc FullLocation = Context->getFullLoc(record->getLocStart());


        // Check this declaration is not in the system headers...
        if ( FullLocation.isValid() && !exclude(FullLocation.getManager() , record) )
        {
            const SourceManager& smanager=Context->getSourceManager();

            // Now check the attributes...
            RecordDecl::field_iterator it=record->field_begin() ;
            for(;it!=record->field_end();it++){
                clang::FieldDecl* ff=*it;
                ff->isTemplateDecl();

                SourceRange declsr=(*it)->getMostRecentDecl()->getSourceRange() ;
                SourceLocation sl=declsr.getBegin();
                std::string name=(*it)->getName() ;

                if(isInExcludedPath(smanager.getFileEntryForID(smanager.getFileID(sl))->getName())){
                    continue ;
                }

                if((*it)->getAccess()==AS_public)
                    continue ;


                CXXRecordDecl* rd=(*it)->getType()->getAsCXXRecordDecl() ;
                if(rd){

                    std::string type=rd->getNameAsString() ;
                    if(type.find("Data")!=std::string::npos){
                        if(name.find("d_")==0){
                        }else{
                            std::cerr << smanager.getFileEntryForID(smanager.getFileID(sl))->getName()
                                      << ":" << smanager.getPresumedLineNumber(sl)
                                      << ":" << smanager.getPresumedColumnNumber(sl)
                                      << ": warning: variable [" << record->getNameAsString() << ":" <<name << "] is violating the sofa coding style http://www.sofa.../codingstyle.html...all Data attributes should start with d_ " << std::endl;
                        }
                    }
                    else if(type.find("SingleLink")!=std::string::npos || type.find("DualLink")!=std::string::npos){
                        if(name.find("d_")==0){
                        }else{
                            std::cerr << smanager.getFileEntryForID(smanager.getFileID(sl))->getName()
                                      << ":" << smanager.getPresumedLineNumber(sl)
                                      << ":" << smanager.getPresumedColumnNumber(sl)
                                      << ": warning: variable [" << record->getNameAsString() << ":" <<name << "] is violating the sofa coding style http://www.sofa.../codingstyle.html...all Link attributes should start with l_ " << std::endl;
                        }
                    }
                }else{
                    if(name.find("m_")==0){
                    }else{
                        std::cerr << smanager.getFileEntryForID(smanager.getFileID(sl))->getName()
                                  << ":" << smanager.getPresumedLineNumber(sl)
                                  << ":" << smanager.getPresumedColumnNumber(sl)
                                  << ": warning: variable [" << record->getNameAsString() << ":" << name << "] is violating the sofa coding style http://www.sofa.../codingstyle.html...all private attributes should start with m_ " << std::endl;
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

        sr->setContext(&ctx);
        sr->TraverseDecl(ctx.getTranslationUnitDecl());
    }

}
