#include "Algebra.h"
#include "NestedList.h"
#include "QueryProcessor.h"
#include "StandardTypes.h"
#include "Algebras/FText/FTextAlgebra.h"
#include <iostream>
#include <string>
#include <cstring> 
#include <map>
// Call Python Modules
#include "/home/lmy/anaconda3/include/python3.8/Python.h"

extern NestedList     *nl;
extern QueryProcessor *qp;
// The maximum number of characters for NLQ
const int MaxCharNum=500;
PyObject* pModule3;

using namespace std;


/****************************************************************

operator Spatial_NL

***************************************************************/
ListExpr Spatial_NLTypeMap( ListExpr args )
{
	// error message;
    string msg = "string expected";
	// check the number of arguments
    if( nl->ListLength(args) != 1){
        ErrorReporter::ReportError(msg + " (invalid number of arguments)");//无效的参数目
        return nl->TypeError();
    }
    // check type of the argument
    ListExpr question = nl->First(args);
    if(nl->SymbolValue(question) != "string"){
        ErrorReporter::ReportError(msg + " (first args is not a string)");
        return listutils::typeError();
    }
    // return the result type: nl->SymbolAtom(CcString::BasicType());
    return NList(FText::BasicType()).listExpr();
}

//convert const char *c to wchar_t *
 wchar_t *GetWC3(const char *c)
{
    const size_t cSize = strlen(c) + 1;
    wchar_t* wc = new wchar_t[cSize];
    mbstowcs(wc, c, cSize);
    return wc;
}

int Spatial_NLValueMap(Word *args, Word &result, int message, Word &local, Supplier s)
{
    // a: NLQ
    char a[MaxCharNum];
    string nl = ((CcString*)args[0].addr)->GetValue();
    strcpy(a,nl.c_str());
    cout<<"****************************"<<endl;
    cout<<"NLQ: "<<nl<<endl;
    
    // initialization
    Py_SetPythonHome(GetWC3("/home/lmy/anaconda3"));
    Py_Initialize();
	
    // Switch the path to the directory of the module to be called
    string path = "/home/lmy/secondo/Algebras/SpatialNLQ";
    string chdir_cmd = string("sys.path.append(\"") + path + "\")";
    const char* cstr_cmd = chdir_cmd.c_str();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString(cstr_cmd);
    
    // Load module
    pModule3 = PyImport_ImportModule("NLG");
    if (!pModule3) // Failed to load module
    {
        cout<<"[ERROR] Python get module failed."<<endl;
        return 0;
    }
    cout<<"[INFO] Python get module succeed."<<endl;
    
    // Load function
    PyObject* pv = PyObject_GetAttrString(pModule3, "secondo");
    if (!pv || !PyCallable_Check(pv))  // Failed to load function
    {
        cout<<"[ERROR] Can't find funftion (secondo)"<<endl;
        return 0;
    }
    cout<<"[INFO] Get function (secondo) succeed."<<endl;
    
    // Set parameters
    PyObject* pArgs = PyTuple_New(1); 
    PyTuple_SetItem(pArgs, 0, Py_BuildValue("s", a));
	
    // call function
    PyObject* pRet = PyObject_CallObject(pv, pArgs);

    // nlresult: result of the transformation
    string nlresult = PyUnicode_AsUTF8(pRet);

    cout<<"****************************"<<endl;
    
    result = qp->ResultStorage(s);
    FText* res = (FText*)(result.addr);
    res->Set(true,nlresult);
    
    return 0;
}

struct Spatial_NLInfo : OperatorInfo {
    Spatial_NLInfo()
    {
        name = "spatial_nl";
        signature = "string -> string";
        syntax = "spatial_nl( )";
        meaning = "Natural language to structured language";
    }
};


/****************************************************************

Algebra Monitor

***************************************************************/
class SpatialNLQAlgebra : public Algebra
{
public:
	SpatialNLQAlgebra() : Algebra()
	{
		AddOperator(Spatial_NLInfo(), Spatial_NLValueMap, Spatial_NLTypeMap);
	}

	~SpatialNLQAlgebra() {
		// Release resources
		Py_Finalize();
	}
};


// Initialization of the Algebra
extern "C"
Algebra* InitializeSpatialNLQAlgebra(NestedList *nlRef, QueryProcessor *qpRef)
{
	nl = nlRef;
	qp = qpRef;
	// cout << "program is here: InitializeSpatialNLQAlgebra()~" << endl;
	return (new SpatialNLQAlgebra());
}
