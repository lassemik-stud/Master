import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import sys
import os
import glob

def convert_json_booleans(obj):
    """
    Recursively converts lowercase "true" and "false" in a JSON object to Python booleans.
    """
    if isinstance(obj, dict):
        return {k: convert_json_booleans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_json_booleans(elem) for elem in obj]
    elif obj == 'false': 
        return False
    elif obj == 'true':  
        return True
    else:
        return obj

def save_confusion_matrix(input_file, output_dir="../../results/plot/"):
  """
  Reads confusion matrix data from a JSON file, generates a confusion matrix plot,
  and saves it to a specified output directory with a filename based on the input file.

  Args:
    input_file: Path to the JSON file containing the confusion matrix data.
    output_dir: Directory where the plot will be saved (defaults to "plot").
  """

  try:
    with open(input_file, 'r') as f:
      data = json.load(f, object_hook=convert_json_booleans)
      confusion_matrix_data = data["confusion_matrix"]

      # Extract values from the confusion matrix dictionary
      tn = confusion_matrix_data["true_negative"]
      fp = confusion_matrix_data["false_positive"]
      fn = confusion_matrix_data["false_negative"]
      tp = confusion_matrix_data["true_positive"]

      # Create the confusion matrix
      cm = np.array([[tn, fp], [fn, tp]])

      # Generate the plot
      plt.figure(figsize=(8, 6))
      sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 30})
      plt.xlabel("Predicted label")
      plt.ylabel("True label")
      plt.title("Confusion Matrix")

      # Generate output filename and path
      input_filename = os.path.splitext(os.path.basename(input_file))[0]
      output_filename = input_filename + "_CM.png"
      output_path = os.path.join(output_dir, output_filename)

      # Create the output directory if it doesn't exist
      os.makedirs(output_dir, exist_ok=True)

      # Save the plot
      plt.savefig(output_path)

  except FileNotFoundError:
    print(f"Error: Input file '{input_file}' not found.")
  except KeyError:
    print(f"Error: 'confusion_matrix' key not found in the JSON file.")
  except Exception as e:
    print(f"An error occurred: {e}")

if __name__ == "__main__":
  print(len(sys.argv))
  if len(sys.argv) != 3:  # Corrected the number of arguments to 3
    print("Usage: python script_name.py <input_glob> <output_dir>")
    print('Example: python3 confusion_matrix.py "../../results/pan*_b0_*.jsonl" "../../results/plot/CM/"') 
  else:
    input_glob = sys.argv[1]
    output_dir = sys.argv[2]

    for input_file in glob.glob(input_glob):
      print(f"Processing file: {input_file}")
      save_confusion_matrix(input_file, output_dir)