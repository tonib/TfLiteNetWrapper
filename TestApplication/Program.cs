using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using TfLiteNetWrapper;

namespace TestApplication
{
    class Program
    {
        const int N_THREADS = 2;

        static HashSet<string> SeqColumns;
        static List<string> OutputNames;

        static void SetupModelColumnsColumns()
        {
            string[] sequenceInputs = { "Type", "DataType", "Collection", "Length", "Decimals", "NameHash0", "NameHash1", "NameHash2", "ControlType" };
            string[] outputNamesOriginal = { "Type", "DataType", "Collection", "Length", "Decimals", "NameHash0", "NameHash1", "NameHash2" };

            SeqColumns = new HashSet<string>(sequenceInputs);
            // Output names for TF lite are wrong. They keep a pattern: Order is the same as the original names sorted alphabetically:
            List<string> outputNames = new List<string>(outputNamesOriginal);
            outputNames.Sort();
            OutputNames = outputNames;
        }

        static private Dictionary<string, float[]> RunPrediction(Interpreter interpreter)
        {
            for(int inputIdx=0; inputIdx < interpreter.GetInputTensorCount(); inputIdx++)
            {
                Interpreter.TensorInfo inputTensor = interpreter.GetInputTensorInfo(inputIdx);
                
                int dim = inputTensor.dimensions.Length > 0 ? inputTensor.dimensions[0] : 1;
                Int32[] input = new Int32[dim];
                for (int i = 0; i < dim; i++)
                    input[i] = -1;
                if (!SeqColumns.Contains(inputTensor.name))
                    input[0] = 0;
                interpreter.SetInputTensorData(inputIdx, input);
            }

            interpreter.Invoke();

            // Get output with rigth names
            Dictionary<string, float[]> result = new Dictionary<string, float[]>();
            for(int outputIdx=0; outputIdx < interpreter.GetOutputTensorCount(); outputIdx++)
            {
                Interpreter.TensorInfo outputTensor = interpreter.GetOutputTensorInfo(outputIdx);
                int dim = outputTensor.dimensions[0];
                float[] output = new float[dim];
                interpreter.GetOutputTensorData(outputIdx, output);
                result.Add(OutputNames[outputIdx], output);
            }

            return result;
        }

        static void PrintResults(Dictionary<string, float[]> result)
        {
            // Print output
            foreach (string key in result.Keys)
            {
                Console.WriteLine(key + ": " + string.Join(", ", result[key].Select(x => x.ToString()).ToArray()));
            }
        }

        static public void LogCallback(string message)
        {
            Console.WriteLine("[TFLOG]: " + message);
        }

        static void TestModel(string fileModelPath, bool loadFromBuffer)
        {
            Interpreter.Options options = new Interpreter.Options();
            options.threads = N_THREADS;
            options.LogCallback = LogCallback;

            Interpreter interpreter;
            if (loadFromBuffer)
            {
                byte[] content = File.ReadAllBytes(fileModelPath);
                interpreter = new Interpreter(content, options);
            }
            else
                interpreter = new Interpreter(fileModelPath, options);

            interpreter.AllocateTensors();

            // Test performance
            /*for(int i=0; i<1000; i++)
                RunPrediction(interpreter);*/

            Dictionary<string, float[]> result = RunPrediction(interpreter);
            PrintResults(result);

            interpreter.Dispose();
        }

        static void Main(string[] args)
        {
            SetupModelColumnsColumns();

            TestModel("model-gpt.tflite", true);
            TestModel("model-gpt.tflite", false);

            // This is expected to fail
            try { TestModel("model-rnn.tflite", true); } catch { }
            try { TestModel("model-rnn.tflite", false); } catch { }

            Console.WriteLine("Done");
        }
    }
}
