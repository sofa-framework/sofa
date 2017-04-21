
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

vector<pair<string, string>> oldccode={pair<string,string>("printf",  "msg_info(\"emitting point\") << operator //from #include <sofa/core/helper/Messaging.h>" ),
                                       pair<string,string>("fprintf", "std::ofstream << operator //from #include <iostream> "),
                                       pair<string,string>("sprintf", "std::stringstream <<  operator //from #include <iostream> "),
                                       pair<string,string>("atoi",    "strtol from <cstdlib>"),
                                       pair<string,string>("atof",    "strtod from <cstdlib>"),
                                       pair<string,string>("malloc",  "new"),
                                       pair<string,string>("free",    "delete") } ;


static cl::OptionCategory MyToolCategory("Stylecheck.exe");
cl::list<string> userexcluded("E", llvm::cl::Prefix, llvm::cl::desc("Specify path pattern to exclude"), cl::cat(MyToolCategory)) ;
cl::list<string> userincluded("L", llvm::cl::Prefix, llvm::cl::desc("Specify path pattern to be restricted in"), cl::cat(MyToolCategory)) ;
cl::opt<bool> verbose("v", cl::desc("Set verbose mode"), cl::init(false), cl::cat(MyToolCategory));
cl::opt<int> numberofincludes("n", cl::desc("Number of include files before a warning is emited [default is 40]"), cl::init(40), cl::cat(MyToolCategory));

enum QualityLevel {
  Q0, Q1, Q2
};

cl::opt<QualityLevel> qualityLevel(cl::desc("Choose the level of conformance stylecheck will use (default is Q0):"),
  cl::values(
    clEnumVal(Q0, "Emits warnings about style violation from the mandatory guidelines."),
    clEnumVal(Q1, "Emits warnings about style violation from the recommanded guidelines."),
    clEnumVal(Q2, "Emits warnings about style quality and advices.")),
   cl::init(Q0), cl::cat(MyToolCategory));

bool isAnExecParam(const string& path)
{
    if(path.find("class sofa::core::ExecParams")!=string::npos)
        return true ;
    if(path.find("class sofa::core::MechanicalParams")!=string::npos)
        return true ;

    return false ;
}

bool isInHeader(const string& path)
{
    if(path.rfind(".h")!=string::npos)
        return true ;
    if(path.rfind(".inl")!=string::npos)
        return true ;

    return false ;
}

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
    if(qualityLevel < Q0)
        return ;
    cerr << filename << ":" << line << ":" << col <<  ": warning: initialization of [" << varname << "] is violating the sofa coding style rules V1. " << endl ;
    cerr << " Variables should always have initializer.  Built-in data types (int, float, char, pointers...) have no default values, " << endl ;
    cerr << " so they're undefined until you give them one and without having been properly initialized, hard-to-track bugs can occur. " << endl ;
    cerr << " In addition, if they're initialized after having been declared, there is a risk that someone later inadvertently deletes " << endl ;
    cerr << " or moves the line where they're given a value. " << endl ;
    cerr << " Finally when instantiating a class or structure, you pay the cost of a constructor call, whether it is the default one or user-provided. " << endl ;
    cerr << " You can found the complete Sofa coding guidelines at: https://github.com/sofa-framework/sofa/blob/master/GUIDELINES.md" << endl  << endl ;
}


void printErrorN12(const string& filename, const int line, const int col, const string& nsname){
    if(qualityLevel< Q0)
        return ;
    cerr << filename << ":" << line << ":" << col <<  ": warning: namespace [" << nsname << "] is violating the sofa coding style rules N12. " << endl ;
    cerr << " By convention, all namespaces must be in lowercase.' " << endl;
    cerr << " You can found the complete Sofa coding guidelines at: https://github.com/sofa-framework/sofa/blob/master/GUIDELINES.md" << endl  << endl ;
}


void printErrorC1(const string& filename, const int line, const int col, const string& classname, const string& name){
    if(qualityLevel< Q0)
        return ;
    cerr << filename << ":" << line << ":" << col <<  ": warning: function member [" << classname << ":" << name << "] is violating the sofa coding style rule C1. " << endl ;
    cerr << " To keep compilation time between acceptable limits it is adviced that headers contains only declaration (i.e.: no body)" <<  endl ;
    cerr << " You can found the complete Sofa coding guidelines at: https://github.com/sofa-framework/sofa/blob/master/GUIDELINES.md" << endl ;
    cerr << " Suggested replacements to remove this warning: " << endl ;
    cerr << "     - if the function's body can be rewritten in a single line of source code, then do it " << endl;
    cerr << "     - otherwise move the body of the function "<< name << " into a a .cpp file" << endl << endl ;
}

void printErrorN2(const string& filename, const int line, const int col, const string& classname, const string& name){
    if(qualityLevel < Q0)
        return ;
    cerr << filename << ":" << line << ":" << col <<  ": warning: function member [" << classname << ":" << name << "] is violating the sofa coding style rule N2. " << endl ;
    cerr << " By convention, all functions names should use lowerCamlCase without underscore '_' " << endl ;
    cerr << " You can found the complete Sofa coding guidelines at: https://github.com/sofa-framework/sofa/blob/master/GUIDELINES.md" << endl << endl ;
}

void printErrorN5(const string& filename, const int line, const int col, const string& classname, const string& name){
    if(qualityLevel < Q0)
        return ;
    cerr << filename << ":" << line << ":" << col <<  ": warning: member [" << classname << ":" << name << "] is violating the sofa coding style rule N5. " << endl ;
    cerr << " Data fields are importants concept in Sofa, to emphasize this fact that they are not simple membre variable they should all be prefixed with d_" << endl;
    cerr << " You can found the complete Sofa coding guidelines at: https://github.com/sofa-framework/sofa/blob/master/GUIDELINES.md" << endl ;
    cerr << " Suggested replacement: d_" << name << endl << endl ;
}

void printErrorN6(const string& filename, const int line, const int col, const string& classname, const string& name){
    if(qualityLevel < Q0)
        return ;
    cerr << filename << ":" << line << ":" << col <<  ": warning: member [" << classname << ":" << name << "] is violating the sofa coding style rule N6. " << endl ;
    cerr << " DataLink are importants concept in Sofa, to emphasize this fact that they are not simple membre variable they should all be prefixed with l_" << endl;
    cerr << " You can found the complete Sofa coding guidelines at: https://github.com/sofa-framework/sofa/blob/master/GUIDELINES.md" << endl << endl  ;
    cerr << " Suggested replacement: s_" << name << endl << endl ;
}

void printErrorN7(const string& filename, const int line, const int col, const string& classname, const string& name){
    if(qualityLevel < Q0)
        return ;
    cerr << filename << ":" << line << ":" << col <<  ": warning: member [" << classname << ":" << name << "] is violating the sofa coding style rule N7. " << endl ;
    cerr << " To emphasize attributes membership of a class the private and protected members must be prefixed with m_" << endl;
    cerr << " You can found the complete Sofa coding guidelines at: https://github.com/sofa-framework/sofa/blob/master/GUIDELINES.md" << endl  ;
    cerr << " Suggested replacement: m_" << name << endl << endl ;
}

void printErrorN1(const string& filename, const int line, const int col, const string& classname){
    if(qualityLevel < Q0)
        return ;
    cerr << filename << ":" << line << ":" << col <<  ": warning: class [" << classname << "] is violating the sofa coding style rules N1. " << endl ;
    cerr << " [N1] By convention, all classes name must be in UpperCamlCase without any underscores '_'.' " << endl;
    cerr << " You can found the complete Sofa coding guidelines at: https://github.com/sofa-framework/sofa/blob/master/GUIDELINES.md" << endl  << endl ;
}

void printErrorM5(const string& filename, const int line, const int col, const string& classname, const string& name){
    if(qualityLevel < Q0)
        return ;
    cerr << filename << ":" << line << ":" << col <<  ": warning: member [" << classname << ": " << name << "] is violating the sofa coding style rules M5. " << endl ;
    cerr << " To avoid confusion with other coding-style a member's name cannot by terminated by an underscore '_'. " << endl;
    cerr << " You can found the complete Sofa coding guidelines at: https://github.com/sofa-framework/sofa/blob/master/GUIDELINES.md" << endl  << endl ;
}

void printErrorW1(const string& filename, const int line, const int col){
    if(qualityLevel < Q1)
        return ;
    cerr << filename << ":" << line << ":" << col <<  ": warning: use of the goto statement violates the sofa coding style rules W1. " << endl ;
    cerr << " Using the goto statement is controversial and it is higly recommended not to use it. " << endl ;
    cerr << " Recommendation: use a break statement or a continue statement as much as possible." << endl ;
    cerr << " If the goto is used to exit mutliple nested loop and if using a non-goto version leads to a large " << endl ;
    cerr << " increase of code complexity/number of line then it can be kept this way. " << endl ;
    cerr << " You can found the complete Sofa coding guidelines at: https://github.com/sofa-framework/sofa/blob/master/GUIDELINES.md" << endl  << endl ;
}

void printErrorW2(const string& filename, const int line, const int col, const std::string& oldfct, const std::string& newfct){
    if(qualityLevel < Q1)
        return ;
    cerr << filename << ":" << line << ":" << col <<  ": warning: using the C function ["<< oldfct << "] is violates the sofa coding style rules. " << endl ;
    cerr << " Sofa is a C++ project and thus the C++ standard library should be used. Nevertheless not being strictly forbidden " << endl ;
    cerr << " mixing C and C++ libraries or coding style is considered a poor practice so please avoid it." << endl ;
    cerr << " Suggestion: instead of ["<< oldfct <<"] you should use [" << newfct << "]" << endl ;
    cerr << " You can found the complete Sofa coding guidelines at: https://github.com/sofa-framework/sofa/blob/master/GUIDELINES.md" << endl  << endl ;
}

void printErrorW3(const string& filename, const int line, const int col, const std::string& nsname){
    if(qualityLevel < Q0)
        return ;
    cerr << filename << ":" << line << ":" << col <<  ": warning: using namespace ["<< nsname << "] in headers violates the sofa coding style. " << endl ;
    cerr << " Importing a namespace in an header may lead to name collisions. Consequently ait is stricly forbiden to import/using a namespace in a header file. " << endl ;
    cerr << " Suggestion to remove this warning: remove the line 'using namespace " << nsname << ";'' and fix all subsequent problems by compiling sofa." << endl ;
    cerr << " If namespaces are long and impact readability please consider employ using a private namespace to import the needed names with the using keywords" << endl ;
    cerr << " eg in a file MyObject.h: " << endl ;
    cerr << " namespace sofa {                            \n" 
            "     namespace constraint {                  \n"
            "         namespace myobject_h {              \n" 
            "               using sofa::core::Base ;      \n" 
            "               class MyObject {}             \n"
            "         }                                   \n"
            "         using myobject_h::MyObject; //export the object in the public namespace \n"
            "    } \n"
            " } \n"; 
    cerr << " You can found the complete Sofa coding guidelines at: https://github.com/sofa-framework/sofa/blob/master/GUIDELINES.md" << endl  << endl ;
}

void printErrorW4(const string& filename, const int line, const int col, const std::string& type, const std::string& nsname){
    if(qualityLevel < Q2)
        return ;
    cerr << filename << ":" << line << ":" << col <<  ": warning: parameter ["<< type << " " << nsname << "] is violating the sofa coding style W4. " << endl ;
    cerr << " To avoid problems when implementing visitors, it has been decieded that ExecParams and MechanicalParams cannot have a default value. " << endl ;
    cerr << " Suggestion: remove the default value for the parameter ["<< type << " " << nsname << "] ;" << endl ;
    cerr << " You can found the complete Sofa coding guidelines at: https://github.com/sofa-framework/sofa/blob/master/GUIDELINES.md" << endl  << endl ;
}

void printErrorW5(const string& filename, const int line, const int col, const std::string& type, const std::string& nsname){
    if(qualityLevel < Q2)
        return ;
    cerr << filename << ":" << line << ":" << col <<  ": warning: the parameter ["<< type << " " << nsname << "] is violating the sofa coding style W5 " << endl ;
    cerr << " To avoid problems when implementing visitors, it has been decieded that ExecParams and MechanicalParams must be the first parameter of methods. " << endl ;
    cerr << " Suggestion: move the parameter [" << type << " " << nsname << "] to make it the first parameter;" << endl ;
    cerr << " You can found the complete Sofa coding guidelines at: https://github.com/sofa-framework/sofa/blob/master/GUIDELINES.md" << endl  << endl ;
}


void printErrorR1(const string& filename, const int sofacode, const int allcodes){
    if(qualityLevel < Q2)
        return ;
    cerr << filename << ":1:1: info: too much file are included. " << endl ;
    cerr << " To decrease compilation time as well as improving interfaces/ABI it is recommanded to include as few as possible files. " << endl ;
    cerr << " The current .cpp file finally ended in including and thus compiling " << allcodes << " other files. " << endl ;
    cerr << " There is " << sofacode << " sofa files among these "<< allcodes <<" other files. " << endl ;
    //todo(dmarchal) interface based design. 
    cerr << " To help fixing this issue you could use PIMPL" ;
    //    cerr << " at http://www.sofa-framework.com/codingstyle/opaqueincludes.html " << endl << endl ;
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
        // correctly provide a default value.
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

        }else if(stmt->getStmtClass() == Stmt::GotoStmtClass){
            SourceRange sr=stmt->getSourceRange() ;
            SourceLocation sl=sr.getBegin() ;
            auto& smanager=Context->getSourceManager() ;
            auto fileinfo=smanager.getFileEntryForID(smanager.getFileID(sl)) ;

            if(fileinfo==NULL || isInExcludedPath(fileinfo->getName(), excludedPathPatterns))
                return true ;

            printErrorW1(fileinfo->getName(),
                         smanager.getPresumedLineNumber(sl),
                         smanager.getPresumedColumnNumber(sl));
        }else if(stmt->getStmtClass() == Stmt::CallExprClass){
            CallExpr* callexpr=dyn_cast<CallExpr>(stmt) ;
            FunctionDecl* fctdecl=callexpr->getDirectCallee() ;
            if(fctdecl){
                const string& fctname = fctdecl->getNameAsString() ;
                for(auto p : oldccode)
                {
                    if(fctname == p.first){
                        SourceRange sr=stmt->getSourceRange() ;
                        SourceLocation sl=sr.getBegin() ;
                        auto& smanager=Context->getSourceManager() ;
                        auto fileinfo=smanager.getFileEntryForID(smanager.getFileID(sl)) ;

                        if(fileinfo==NULL || isInExcludedPath(fileinfo->getName(), excludedPathPatterns))
                            return true ;

                        printErrorW2(fileinfo->getName(),
                                     smanager.getPresumedLineNumber(sl),
                                     smanager.getPresumedColumnNumber(sl),
                                     p.first, p.second);
                        break ;
                    }
                }
            }
        }

        return true ;
    }

    bool VisitDecl(Decl* decl)
    {
        if(Context==NULL)
            return true ;

        if(decl==NULL)
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

            printErrorN12(fileinfo->getName(),
                         smanager.getPresumedLineNumber(sl),
                         smanager.getPresumedColumnNumber(sl),
                         nsname) ;
            return true;
        }

        UsingDirectiveDecl* udecl = dyn_cast<UsingDirectiveDecl>(decl) ;
        if(udecl){
             auto& smanager = Context->getSourceManager() ;
             SourceRange sr=decl->getSourceRange() ;
             SourceLocation sl=sr.getBegin();
             auto fileinfo=smanager.getFileEntryForID(smanager.getFileID(sl)) ;
             string nsname = udecl->getNominatedNamespaceAsWritten()->getName() ;

             if(fileinfo==NULL || isInExcludedPath(fileinfo->getName(), excludedPathPatterns))
                 return true ;

             if(isInHeader(fileinfo->getName()) ){
                printErrorW3(fileinfo->getName(),
                             smanager.getPresumedLineNumber(sl),
                             smanager.getPresumedColumnNumber(sl),
                             nsname);
             }
             return true ;

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
                    printErrorN1(fileinfo->getName(),
                                 smanager.getPresumedLineNumber(sl),
                                 smanager.getPresumedColumnNumber(sl),
                                 classname) ;

                }
            }

            // Check the function definitions
            //
            if(qualityLevel>=Q2){
                for(auto f=record->method_begin();f!=record->method_end();++f){

                    SourceRange declsr=(*f)->getSourceRange() ;
                    SourceLocation sl=declsr.getBegin();
                    auto fileinfo = smanager.getFileEntryForID(smanager.getFileID(sl)) ;

                    if(fileinfo!=NULL && !isInExcludedPath(fileinfo->getName(), excludedPathPatterns))
                    {
                        // Rules C1: check that a function definition is not in a body.
                        if((*f)->hasBody())
                        {
                            Stmt* body=(*f)->getBody();

                            SourceRange bodysr=body->getSourceRange() ;
                            SourceLocation bodysl=bodysr.getBegin();
                            SourceLocation bodyend=bodysr.getEnd();
                            auto fileinfobody = smanager.getFileEntryForID(smanager.getFileID(bodysl)) ;

                            if(fileinfobody
                                    && isInHeader(fileinfobody->getName())
                                    && smanager.getPresumedLineNumber(bodyend)- smanager.getPresumedLineNumber(bodysl) > 1 ){
                               printErrorC1(fileinfo->getName(), smanager.getPresumedLineNumber(sl), smanager.getPresumedColumnNumber(sl),
                                           record->getNameAsString(), f->getNameAsString());
                            }
                        }

                        // Rules N2: check that a method name is following a LowerCamlCase mode
                        if(!isLowerCamlCase(f->getNameAsString())
                           && !f->isCopyAssignmentOperator()
                           && !f->isMoveAssignmentOperator()
                           && !CXXConstructorDecl::classof(*f)
                           && !CXXDestructorDecl::classof(*f)
                           && !f->isOverloadedOperator())
                        {
                            printErrorN2(fileinfo->getName(), smanager.getPresumedLineNumber(sl), smanager.getPresumedColumnNumber(sl),
                                         record->getNameAsString(), f->getNameAsString());
                        }

                        // Rules: check that all ExecParam are the first param and that they have no default arg.
                        for(auto p=(*f)->param_begin();p!=(*f)->param_end();++p)
                        {
                            string fullname=(*p)->getOriginalType().getCanonicalType().getAsString() ;
                            if(isAnExecParam(fullname)){
                                if( (*p)->hasUnparsedDefaultArg() ||
                                    (*p)->hasDefaultArg() ){
                                    printErrorW4(fileinfo->getName(), smanager.getPresumedLineNumber(sl), smanager.getPresumedColumnNumber(sl),
                                                 fullname, (*p)->getName());
                                }
                                if( p != (*f)->param_begin() ){
                                    printErrorW5(fileinfo->getName(), smanager.getPresumedLineNumber(sl), smanager.getPresumedColumnNumber(sl),
                                                 fullname, (*p)->getName());
                                }

                            }
                        }
                    }
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

                // The name of members cannot be terminated by an underscore.
                if(name.rfind("_")!=name.size()-1){
                }else{
                    printErrorM5(filename, line, col, classname, name);
                }

                CXXRecordDecl* rd=ff->getType()->getAsCXXRecordDecl() ;
                if(rd){
                    std::string type=rd->getNameAsString() ;
                    if(type.find("Data")!=std::string::npos){
                        if(name.find("d_")==0){
                        }else{
                            printErrorN5(filename, line, col,
                                         classname, name) ;

                     	    
                        }
                        continue;
                    }else if(type.find("SingleLink")!=std::string::npos || type.find("DualLink")!=std::string::npos){
                        if(name.find("d_")==0){
                        }else{
                            printErrorN6(filename, line, col, classname, name) ;
                        }
                        continue;
                    }
                    if(rd->isPOD())
                       continue; 	
                }
                
                /// THESES TWO RULES ARE NOW DEPRECATED BUT I KEEP THEM FOR HISTORY REASON
                // THE FOLLOWING RULES ARE ONLY CHECKED ON PRIVATE & PROTECTED FIELDS
                if(ff->getAccess()==AS_public){
                    continue ;
                }
                
                if(name.find("m_")==0 ){
                }else{
                	printErrorN7(filename, line, col, classname, name) ;
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
        if(qualityLevel>=Q2){
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
