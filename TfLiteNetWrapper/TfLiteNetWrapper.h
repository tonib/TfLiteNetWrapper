#pragma once

#include <tensorflow/lite/c/c_api.h>

using namespace System;
using namespace System::Collections::Generic;

namespace TfLiteNetWrapper {

	public enum class TensorType {
		kTfLiteNoType = 0,
		kTfLiteFloat32 = 1,
		kTfLiteInt32 = 2,
		kTfLiteUInt8 = 3,
		kTfLiteInt64 = 4,
		kTfLiteString = 5,
		kTfLiteBool = 6,
		kTfLiteInt16 = 7,
		kTfLiteComplex64 = 8,
		kTfLiteInt8 = 9,
		kTfLiteFloat16 = 10,
		kTfLiteFloat64 = 11,
		kTfLiteComplex128 = 12,
		kTfLiteUInt64 = 13,
		kTfLiteResource = 14,
		kTfLiteVariant = 15,
		kTfLiteUInt32 = 16
	};

	void TestOpResult(TfLiteStatus status, String^ errorMsg);

	public ref class TensorWrapper {
		public:
			String^ Name;

			List<int>^ Dimensions;

			TensorType Type;

			TensorWrapper(TfLiteTensor*);

			generic <typename T>
			void SetValues(array<T>^ values);

			generic <typename T>
			void GetValues(array<T>^ values);

			int BytesSize;

		private:
			TfLiteTensor* Tensor;
	};

	public ref class ModelWrapper
	{
		public:

			ModelWrapper(String^ modelFilePath, int nThreads);

			ModelWrapper(array<Byte>^ modelContent, int nThreads);

			List<TensorWrapper^>^ InputTensors;

			List<TensorWrapper^>^ OutputTensors;

			void InvokeInterpreter();

			~ModelWrapper();

		private:
			TfLiteModel* Model;
			TfLiteInterpreter* Interpreter;
			TfLiteInterpreterOptions* Options;

			void SetupModel(int nThreads);
	};
}
