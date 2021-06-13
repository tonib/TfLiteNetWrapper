﻿using System;
using System.Collections.Generic;
using System.IO;
using TfLiteNetWrapper;

namespace TestApplication
{
	class Program
	{
		const int N_THREADS = 2;

		static HashSet<string> SeqColumns;

		static List<string> OutputNames;

		static string ModelPath;

		static private void TestModel(ModelWrapper model)
		{
			foreach (TensorWrapper tensor in model.InputTensors)
			{
				int dim = tensor.Dimensions.Count > 0 ? tensor.Dimensions[0] : 1;
				Int32[] input = new Int32[dim];
				for (int i = 0; i < dim; i++)
					input[i] = -1;
				if (!SeqColumns.Contains(tensor.Name))
					input[0] = 0;
				tensor.SetValues(input);
			}

			model.InvokeInterpreter();

			// Get output with rigth names
			Dictionary<string, float[]> result = new Dictionary<string, float[]>();
			for (int i = 0; i < model.OutputTensors.Count; i++)
			{
				TensorWrapper tensor = model.OutputTensors[i];
				int dim = tensor.Dimensions[0];
				float[] output = new float[dim];
				tensor.GetValues(output);
				result.Add(OutputNames[i], output);
			}

			// Print output
			foreach (string key in result.Keys)
			{
				Console.WriteLine(key + ": " + string.Join(", ", result[key]));
			}

			Console.WriteLine("Done");
		}

		static void SetupGptModel()
		{
			string[] sequenceInputs = {
					"wordType","keywordIdx","kbObjectTypeIdx","dataTypeIdx","dataTypeExtTypeHash","isCollection","lengthBucket","decimalsBucket","textHash0","textHash1",
					"textHash2","textHash3","controlType" };
			string[] outputNamesOriginal = { "isCollection", "lengthBucket", "decimalsBucket", "outputTypeIdx", "outputExtTypeHash", "textHash0", "textHash1", "textHash2",
					"textHash3", "isControl" };
			ModelPath = "model-gpt.tflite";
			SetupModelColumnsColumns(sequenceInputs, outputNamesOriginal);
		}

		static void SetupRnnModel()
		{
			string[] sequenceInputs = { "Type", "DataType", "Collection", "Length", "Decimals", "NameHash0", "NameHash1", "NameHash2", "ControlType" };
			string[] outputNamesOriginal = { "Type", "DataType", "Collection", "Length", "Decimals", "NameHash0", "NameHash1", "NameHash2" };
			ModelPath = "model-rnn.tflite";
			SetupModelColumnsColumns(sequenceInputs, outputNamesOriginal);
		}

		static void SetupModelColumnsColumns(string[] sequenceInputs, string[] outputNamesOriginal)
		{
			SeqColumns = new HashSet<string>(sequenceInputs);
			// Output names for TF lite are wrong. They keep a pattern: Order is the same as the original names sorted alphabetically:
			List<string> outputNames = new List<string>(outputNamesOriginal);
			outputNames.Sort();
			OutputNames = outputNames;
		}

		static void TestFileModel()
		{
			ModelWrapper model = new ModelWrapper(ModelPath, N_THREADS);
			TestModel(model);
		}

		static void TestContentModel()
		{
			byte[] content = File.ReadAllBytes(ModelPath);
			ModelWrapper model = new ModelWrapper(content, N_THREADS);
			TestModel(model);
		}

		static void ReportError(string errorMsg)
		{
			Console.WriteLine("ReportError: " + errorMsg);
		}

		static void Main(string[] args)
		{
			try
			{
				// Set delegate to report errors:
				ModelWrapper.ReportErrorsToConsole = false;
				ModelWrapper.ErrorReporter = ReportError;

				// Test GPT
				SetupGptModel();
				TestFileModel();
				TestContentModel();

				// Test RNN
				SetupRnnModel();
				// It's expected failure for this (Flex unsupported)
				try
				{
					TestFileModel();
				}
				catch{ }
				try
				{
					TestContentModel();
				}
				catch { }
			}
			catch(Exception ex)
			{
				Console.WriteLine(ex.ToString());
			}
		}
	}
}
