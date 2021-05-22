#include "pch.h"
#include "TfLiteNetWrapper.h"
#include <tensorflow/lite/c/c_api.h>
//#using <mscorlib.dll>

using namespace System::Runtime::InteropServices;

namespace TfLiteNetWrapper {

	ModelWrapper::ModelWrapper(System::String^ modelFilePath) {

		char* strPath = (char*)Marshal::StringToHGlobalAnsi(modelFilePath).ToPointer();
		TfLiteModel* model = TfLiteModelCreateFromFile(strPath);
		Marshal::FreeHGlobal((IntPtr)strPath);

		if (model == NULL)
			throw gcnew System::Exception("Model cannot be load");
	}

}
