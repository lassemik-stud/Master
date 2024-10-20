import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import sys
import os

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
      data = json.load(f)
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
      sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
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
  if len(sys.argv) != 2:
    print("Usage: python script_name.py <input_file>")
  else:
    input_file = sys.argv[1]
    save_confusion_matrix(input_file)