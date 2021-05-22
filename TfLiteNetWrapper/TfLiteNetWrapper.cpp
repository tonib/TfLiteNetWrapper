#include "pch.h"
#include "TfLiteNetWrapper.h"
#include <tensorflow/lite/c/c_api.h>

namespace TfLiteNetWrapper {

	ModelWrapper::ModelWrapper(const char* modelFilePath) {
		TfLiteModel* model = TfLiteModelCreateFromFile(modelFilePath);
	}

}
