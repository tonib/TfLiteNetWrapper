#pragma once

using namespace System;

namespace TfLiteNetWrapper {

	public ref class ModelWrapper
	{
	public:
		ModelWrapper(const char* modelFilePath);
	};
}
