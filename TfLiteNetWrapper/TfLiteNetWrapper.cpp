#include "pch.h"
#include "TfLiteNetWrapper.h"
//#using <mscorlib.dll>

using namespace System::Runtime::InteropServices;

namespace TfLiteNetWrapper {

	void TestOpResult(TfLiteStatus status, String^ errorMsg) 
	{
		if (status == kTfLiteOk)
			return;

		String^ txtStatus;
		switch (status)
		{
		case kTfLiteOk:
			txtStatus = "kTfLiteOk";
			break;
		case kTfLiteError:
			txtStatus = "kTfLiteError";
			break;
		case kTfLiteDelegateError:
			txtStatus = "kTfLiteDelegateError";
			break;
		case kTfLiteApplicationError:
			txtStatus = "kTfLiteApplicationError";
			break;
		default:
			txtStatus = "Unknown";
			break;
		}

		throw gcnew System::Exception(errorMsg + ": " + txtStatus);
	}

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

		pin_ptr<T> valuesPtr = &values[values->GetLowerBound(0)];
		TestOpResult(TfLiteTensorCopyFromBuffer(Tensor, valuesPtr, BytesSize), "Error calling TfLiteTensorCopyFromBuffer");
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

		pin_ptr<T> valuesPtr = &values[values->GetLowerBound(0)];
		TestOpResult(TfLiteTensorCopyToBuffer(Tensor, valuesPtr, BytesSize), "Error calling TfLiteTensorCopyToBuffer");
	}

// ---------------------------------------------------------
// ---------------------------------------------------------

	ModelWrapper::ModelWrapper(System::String^ modelFilePath, int nThreads) {
		Model = NULL;
		Interpreter = NULL;
		Options = NULL;

		// Load model from file
		char* strPath = (char*)Marshal::StringToHGlobalAnsi(modelFilePath).ToPointer();
		Model = TfLiteModelCreateFromFile(strPath);
		Marshal::FreeHGlobal((IntPtr)strPath);
		if (Model == NULL)
			throw gcnew System::Exception("Model cannot be load");

		SetupModel(nThreads);
	}

	ModelWrapper::ModelWrapper(array<Byte>^ modelContent, int nThreads) {
		Model = NULL;
		Interpreter = NULL;
		Options = NULL;

		pin_ptr<Byte> modelContentPointer = &modelContent[modelContent->GetLowerBound(0)];
		Model = TfLiteModelCreate(modelContentPointer, modelContent->Length);
		if (Model == NULL)
			throw gcnew System::Exception("Model cannot be load");

		SetupModel(nThreads);
	}

	void ModelWrapper::SetupModel(int nThreads) {
		// Set interpreter options
		Options = TfLiteInterpreterOptionsCreate();
		TfLiteInterpreterOptionsSetNumThreads(Options, nThreads);

		// Create the interpreter
		Interpreter = TfLiteInterpreterCreate(Model, Options);
		TestOpResult(TfLiteInterpreterAllocateTensors(Interpreter), "Error calling TfLiteInterpreterAllocateTensors");

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
			OutputTensors->Add(gcnew TensorWrapper((TfLiteTensor*)outputTensor));
		}
	}

	void ModelWrapper::InvokeInterpreter() {
		TestOpResult(TfLiteInterpreterInvoke(Interpreter), "Error calling InvokeInterpreter");
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
