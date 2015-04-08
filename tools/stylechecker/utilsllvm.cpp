#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
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

#include "clang/Rewrite/Frontend/Rewriters.h"
#include "clang/Rewrite/Core/Rewriter.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace llvm;


#include "utilsllvm.h"

bool exclude(const SourceManager& srcMgr, const Decl *decl){
    if(decl){
        SourceLocation loc=decl->getLocStart();

        //const FileEntry* file_entry =
        //    srcMgr.getFileEntryForID(srcMgr.getFileID(decl->getLocStart()));
        if(srcMgr.isInSystemHeader(loc)){
            return true;
        }
        return false;
   }
    return false;
}

bool exclude(const SourceManager& srcMgr, const Stmt *decl){
    if(decl){
        SourceLocation loc=decl->getLocStart();

        const FileEntry* file_entry =
            srcMgr.getFileEntryForID(srcMgr.getFileID(decl->getLocStart()));
        if(srcMgr.isInSystemHeader(loc)){
            return true;
        }
        return false;
   }
    return false;
}
