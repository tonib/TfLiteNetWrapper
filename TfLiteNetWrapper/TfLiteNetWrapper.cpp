#include "pch.h"
#include "TfLiteNetWrapper.h"
//#using <mscorlib.dll>

using namespace System::Runtime::InteropServices;

namespace TfLiteNetWrapper {

	TensorWrapper::TensorWrapper(TfLiteTensor* tensor) {
		Tensor = tensor;

		Name = gcnew String( TfLiteTensorName(tensor) );

		Type = (TensorType) TfLiteTensorType(tensor);

		int32_t nDims = TfLiteTensorNumDims(tensor);
		Dimensions = gcnew List<int>(nDims);
		for (int i = 0; i < nDims; i++) {
			int dimSize = TfLiteTensorDim(tensor, i);
			Dimensions->Add(dimSize);
		}

		BytesSize = TfLiteTensorByteSize(Tensor);
	}

	generic <typename T>
	void TensorWrapper::SetValues(array<T>^ values) {
		// Check array size is ok
		int arrayBytesSize = values->Length * sizeof(T);
		if (arrayBytesSize != BytesSize) {
			throw gcnew Exception("Array is expected to have a size in bytes equal to BytesSize");
		}
		if (BytesSize == 0) {
			return;
		}

		pin_ptr<T> valuesPtr = &values[0];
		TfLiteTensorCopyFromBuffer(Tensor, valuesPtr, BytesSize);
	}

	generic <typename T>
	void TensorWrapper::GetValues(array<T>^ values) {
		// Check array size is ok
		int arrayBytesSize = values->Length * sizeof(T);
		if (arrayBytesSize != BytesSize) {
			throw gcnew Exception("Array is expected to have a size in bytes equal to BytesSize");
		}
		if (BytesSize == 0) {
			return;
		}

		pin_ptr<T> valuesPtr = &values[0];
		TfLiteTensorCopyToBuffer(Tensor, valuesPtr, BytesSize);
	}

// ---------------------------------------------------------
// ---------------------------------------------------------

	ModelWrapper::ModelWrapper(System::String^ modelFilePath, int nThreads) {

		// Create the model
		Model = NULL;
		Interpreter = NULL;
		Options = NULL;

		char* strPath = (char*)Marshal::StringToHGlobalAnsi(modelFilePath).ToPointer();
		Model = TfLiteModelCreateFromFile(strPath);
		Marshal::FreeHGlobal((IntPtr)strPath);
		if (Model == NULL)
			throw gcnew System::Exception("Model cannot be load");

		// Set interpreter options
		Options = TfLiteInterpreterOptionsCreate();
		TfLiteInterpreterOptionsSetNumThreads(Options, 2);

		// Create the interpreter
		Interpreter = TfLiteInterpreterCreate(Model, Options);
		TfLiteInterpreterAllocateTensors(Interpreter);

		// Populate tensors information
		int32_t nInputs = TfLiteInterpreterGetInputTensorCount(Interpreter);
		InputTensors = gcnew List<TensorWrapper^>(nInputs);
		for (int i = 0; i < nInputs; i++) {
			TfLiteTensor* inputTensor = TfLiteInterpreterGetInputTensor(Interpreter, i);
			InputTensors->Add(gcnew TensorWrapper(inputTensor));
		}

		int32_t nOutputs = TfLiteInterpreterGetOutputTensorCount(Interpreter);
		OutputTensors = gcnew List<TensorWrapper^>(nOutputs);
		for (int i = 0; i < nOutputs; i++) {
			const TfLiteTensor* outputTensor = TfLiteInterpreterGetOutputTensor(Interpreter, i);
			OutputTensors->Add(gcnew TensorWrapper((TfLiteTensor *)outputTensor));
		}
	}

	void ModelWrapper::InvokeInterpreter() {
		TfLiteInterpreterInvoke(Interpreter);
	}

	ModelWrapper::~ModelWrapper() {
		InputTensors->Clear();
		OutputTensors->Clear();

		// Dispose of the model and interpreter objects.
		if (Interpreter != NULL) {
			TfLiteInterpreterDelete(Interpreter);
			Interpreter = NULL;
		}
		if (Options != NULL) {
			TfLiteInterpreterOptionsDelete(Options);
			Options = NULL;
		}
		if (Model != NULL) {
			TfLiteModelDelete(Model);
			Model = NULL;
		}
	}

}
