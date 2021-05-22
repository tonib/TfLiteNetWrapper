using System;
using System.Collections.Generic;
using TfLiteNetWrapper;

namespace TestApplication
{
	class Program
	{
		static void Main(string[] args)
		{
			try
			{

				string[] sequenceInputs = {
					"wordType","keywordIdx","kbObjectTypeIdx","dataTypeIdx","dataTypeExtTypeHash","isCollection","lengthBucket","decimalsBucket","textHash0","textHash1",
					"textHash2","textHash3","controlType" };
				HashSet<string> seqColumns = new HashSet<string>(sequenceInputs);

				string[] outputNamesOriginal = { "isCollection", "lengthBucket", "decimalsBucket", "outputTypeIdx", "outputExtTypeHash", "textHash0", "textHash1", "textHash2", 
					"textHash3", "isControl" };
				// Output names for TF lite are wrong. They keep a pattern: Order is the same as the original names sorted alphabetically:
				List<string> outputNames = new List<string>(outputNamesOriginal);
				outputNames.Sort();

				ModelWrapper model = new ModelWrapper(@"D:\kbases\subversion\TfLiteNetWrapper\TfLiteNetWrapper\TestApplication\model.tflite", 2);

				foreach(TensorWrapper tensor in model.InputTensors)
				{
					int dim = tensor.Dimensions[0];
					Int32[] input = new Int32[dim];
					for (int i = 0; i < dim; i++)
						input[i] = -1;
					if (!seqColumns.Contains(tensor.Name))
						input[0] = 0;
					tensor.SetValues(input);
				}

				model.InvokeInterpreter();

				Dictionary<string, float[]> result = new Dictionary<string, float[]>();
				for(int i=0; i<model.OutputTensors.Count; i++)
				{
					TensorWrapper tensor = model.OutputTensors[i];
					int dim = tensor.Dimensions[0];
					float[] output = new float[dim];
					tensor.GetValues(output);
					result.Add(outputNames[i], output);
				}

				Console.WriteLine("Done");
			}
			catch(Exception ex)
			{
				Console.WriteLine(ex.ToString());
			}
		}
	}
}
