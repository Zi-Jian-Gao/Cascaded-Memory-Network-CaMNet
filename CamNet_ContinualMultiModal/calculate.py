import yaml
import argparse


def calculate_cl_metrics(file_path):
    try:
        # Read the YAML file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return

    num_tasks = len(data)
    if num_tasks == 0:
        print("No task data found.")
        return

    # 1. Calculate Final Accuracy
    # Get the last accuracy for each task (tested after training all tasks)
    final_accuracies = [task_acc[-1] for task_acc in data]
    avg_final_accuracy = sum(final_accuracies) / num_tasks

    # 2. Calculate Average Retroactive Interference (RI)
    # RI is calculated only on historical tasks (the first N-1 tasks)
    ri_list = []
    for i in range(num_tasks - 1):
        initial_acc = data[i][0]  # Accuracy right after training on this specific task
        final_acc = data[i][-1]  # Final accuracy on this task after training all tasks

        # Retroactive Interference calculation: Final Accuracy - Initial Accuracy
        ri = final_acc - initial_acc
        ri_list.append(ri)

    avg_ri = sum(ri_list) / len(ri_list)

    # Print overall results
    print(f"Total number of tasks: {num_tasks}")
    print("-" * 50)
    print(f"Final Accuracy: {avg_final_accuracy:.4f} ({avg_final_accuracy * 100:.2f}%)")
    print(f"Retroactive Interference: {avg_ri:.4f} ({avg_ri * 100:.2f}%)")
    print("-" * 50)

    # Print specific changes for each historical task
    print("Retroactive Interference details for historical tasks (Final - Initial):")
    for i, ri in enumerate(ri_list):
        print(f"  Task {i + 1}: {data[i][-1]:.4f} - {data[i][0]:.4f} = {ri:.4f} ({ri * 100:.2f}%)")


if __name__ == "__main__":
    # Setup argparse to handle command-line arguments
    parser = argparse.ArgumentParser(
        description="Calculate Continual Learning metrics (Final Accuracy and Retroactive Interference).")

    # Add the --log argument
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Path to the YAML log file (e.g., cl_matrix.yaml)"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Run the calculation with the provided file path
    calculate_cl_metrics(args.log)