import subprocess
import tempfile
import os
import random
import re
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict, Tuple, Callable, Optional

# --- Mermaid Diagram & Q/A Definitions ---

MERMAID_SAMPLES = [
    {
        "type": "flowchart",
        "code": """
graph TD
    A[Start] --> B{Decision?};
    B -- Yes --> C[Process 1];
    B -- No --> D[Process 2];
    C --> E[End];
    D --> E;
""",
        "qa_generators": [
            lambda code: (
                "How many nodes are in this flowchart?",
                str(
                    len(
                        set(
                            re.findall(
                                r"\b([A-Za-z0-9]+)(?:\[|\{|\(|\(|\<|\/|\\|\{|\(|>|\)|\})?",
                                code.split("graph")[1],
                            )
                        )
                    )
                ),
            ),
            lambda code: (
                "What node follows the 'Yes' path from 'Decision?'?",
                "Process 1",
            ),
            lambda code: ("What is the starting node of this flowchart?", "Start"),
            lambda code: ("Which node represents a decision point?", "Decision?"),
            lambda code: (
                "What processes can lead to the 'End' node?",
                "Process 1, Process 2",
            ),  # Manual, harder to automate generically
        ],
    },
    {
        "type": "flowchart_simple",
        "code": """
graph LR
    Start --> Step1;
    Step1 --> Step2;
    Step2 --> End;
""",
        "qa_generators": [
            lambda code: (
                "How many steps are between 'Start' and 'End' (exclusive)?",
                "2",
            ),
            lambda code: ("What is the node after 'Step1'?", "Step2"),
            lambda code: ("What is the direction of this flowchart?", "Left to Right"),
        ],
    },
    {
        "type": "sequence",
        "code": """
sequenceDiagram
    participant Alice
    participant Bob
    Alice->>Bob: Hello Bob, how are you?
    Bob-->>Alice: I am good, thanks!
    Alice->>John: Hi John!
""",
        "qa_generators": [
            lambda code: (
                "How many participants are in this sequence diagram?",
                str(len(set(re.findall(r"participant (\w+)", code)))),
            ),
            lambda code: ("Who sends the first message?", "Alice"),
            lambda code: (
                "To whom does Alice send the message 'Hello Bob, how are you?'?",
                "Bob",
            ),
            lambda code: ("What message does Bob send to Alice?", "I am good, thanks!"),
            lambda code: ("Does Alice interact with John?", "Yes"),
        ],
    },
    {
        "type": "pie_chart",
        "code": """
pie title Sales Distribution
    "Books" : 42.96
    "Electronics" : 25.03
    "Apparel" : 15.00
    "Home Goods" : 17.01
""",
        "qa_generators": [
            lambda code: (
                "What is the title of the pie chart?",
                re.search(r"title (.*)", code).group(1).strip()
                if re.search(r"title (.*)", code)
                else "N/A",
            ),
            lambda code: (
                "What is the sales percentage for 'Books'?",
                re.search(r'"Books"\s*:\s*([\d.]+)', code).group(1)
                if re.search(r'"Books"\s*:\s*([\d.]+)', code)
                else "N/A",
            ),
            lambda code: (
                "Which category has the highest sales percentage?",
                "Books",
            ),  # Harder to make generic
            lambda code: (
                "How many categories are shown in the pie chart?",
                str(len(re.findall(r'"(.*?)"\s*:', code))),
            ),
        ],
    },
    {
        "type": "class_diagram",
        "code": """
classDiagram
    Animal <|-- Duck
    Animal <|-- Fish
    Animal <|-- Zebra
    Animal : +int age
    Animal : +String gender
    Animal: +isMammal()
    Animal: +mate()
    class Duck{
        +String beakColor
        +swim()
        +quack()
    }
    class Fish{
        -int sizeInFt
        -canEat()
    }
    class Zebra{
        +bool is_wild
        +run()
    }
""",
        "qa_generators": [
            lambda code: ("How many classes directly inherit from 'Animal'?", "3"),
            lambda code: (
                "What is a public attribute of the 'Animal' class mentioned in the diagram?",
                "age or gender (age/gender)",
            ),  # or "gender"
            lambda code: ("Does the 'Duck' class have a 'swim' method?", "Yes"),
            lambda code: (
                "Is 'sizeInFt' a public or private attribute of 'Fish'?",
                "Private",
            ),
            lambda code: ("Which class has a method 'run()'?", "Zebra"),
        ],
    },
]

# --- Helper Functions ---


def generate_mermaid_image(
    mermaid_code: str, output_format: str = "png"
) -> Optional[bytes]:
    """
    Generates an image from Mermaid code using mmdc.
    Returns image bytes or None if generation fails.
    """
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".mmd"
        ) as mmd_file:
            mmd_file.write(mermaid_code)
            mmd_file_path = mmd_file.name

        img_file_path = tempfile.mktemp(suffix=f".{output_format}")

        # Ensure mmdc can be found. Add to PATH or provide full path if necessary.
        # Example: command = ["/path/to/node_modules/.bin/mmdc", "-i", mmd_file_path, "-o", img_file_path]
        command = [
            "mmdc",
            "-i",
            mmd_file_path,
            "-o",
            img_file_path,
            "-w",
            "800",
            "-H",
            "600",
        ]

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(timeout=30)  # Add timeout

        if process.returncode != 0:
            print(f"Error generating Mermaid diagram:")
            print(f"Stdout: {stdout.decode(errors='ignore')}")
            print(f"Stderr: {stderr.decode(errors='ignore')}")
            print(f"Mermaid code:\n{mermaid_code}")
            return None

        if not os.path.exists(img_file_path) or os.path.getsize(img_file_path) == 0:
            print(f"Generated image file is missing or empty: {img_file_path}")
            print(f"Mermaid code:\n{mermaid_code}")
            return None

        with open(img_file_path, "rb") as f:
            image_bytes = f.read()

        return image_bytes

    except subprocess.TimeoutExpired:
        print(f"mmdc command timed out for code:\n{mermaid_code}")
        return None
    except FileNotFoundError:
        print(
            "Error: mmdc (mermaid-cli) not found. Make sure it's installed and in your PATH."
        )
        print("Try: npm install -g @mermaid-js/mermaid-cli")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during image generation: {e}")
        print(f"Mermaid code:\n{mermaid_code}")
        return None
    finally:
        if "mmd_file_path" in locals() and os.path.exists(mmd_file_path):
            os.remove(mmd_file_path)
        if "img_file_path" in locals() and os.path.exists(img_file_path):
            os.remove(img_file_path)


def generate_dataset(
    num_records: int = 100, output_filename: str = "mermaid_dataset.parquet"
) -> None:
    """
    Generates a synthetic dataset and saves it as a Parquet file.
    """
    dataset = []
    generated_count = 0

    print(f"Attempting to generate {num_records} records...")

    while generated_count < num_records:
        sample_def = random.choice(MERMAID_SAMPLES)
        mermaid_code = sample_def["code"].strip()

        # Try to pick a QA generator, if multiple exist for this sample
        if sample_def["qa_generators"]:
            qa_generator = random.choice(sample_def["qa_generators"])
            try:
                question, answer = qa_generator(mermaid_code)
            except Exception as e:
                print(f"Error generating Q/A for code type {sample_def['type']}: {e}")
                # Potentially skip this iteration or use a default Q/A
                continue
        else:
            # This case should ideally not happen if all samples have qa_generators
            print(f"Warning: No Q/A generator for code type {sample_def['type']}")
            question, answer = (
                "Default question: What is this diagram about?",
                "A diagram.",
            )

        print(
            f"Generating record {generated_count + 1}/{num_records} (Type: {sample_def['type']})"
        )
        image_bytes = generate_mermaid_image(mermaid_code)

        if image_bytes:
            dataset.append(
                {
                    "diagram_image": image_bytes,
                    "mermaid_code": mermaid_code,
                    "question": question,
                    "answer": answer,
                }
            )
            generated_count += 1
        else:
            print(f"Failed to generate image for a sample. Skipping.")
            # Optionally, you might want to retry with a different sample or stop
            # For simplicity, we just skip and the loop will try to make up for it

        if generated_count % 10 == 0 and generated_count > 0:
            print(f"Successfully generated {generated_count} records so far.")

    if not dataset:
        print("No data was generated. Exiting.")
        return

    # Create PyArrow Table
    # Infer schema or define explicitly for more control
    # pa.schema([('diagram_image', pa.binary()), ('mermaid_code', pa.string()), ...])
    try:
        table = pa.Table.from_pylist(dataset)
        pq.write_table(table, output_filename)
        print(f"\nSuccessfully generated {len(dataset)} records.")
        print(f"Dataset saved to {output_filename}")
    except Exception as e:
        print(f"Error writing Parquet file: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    # Check if mmdc is available
    try:
        subprocess.run(
            ["mmdc", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("mermaid-cli (mmdc) found.")
        # Generate more records for a more diverse dataset
        generate_dataset(
            num_records=50, output_filename="synthetic_mermaid_vqa_dataset.parquet"
        )
    except FileNotFoundError:
        print("Error: mmdc (mermaid-cli) not found. Cannot generate diagram images.")
        print("Please install it using: npm install -g @mermaid-js/mermaid-cli")
        print("And ensure it's in your system's PATH.")
    except subprocess.CalledProcessError:
        print(
            "Error: mmdc (mermaid-cli) found but is not working correctly. Check mmdc installation."
        )
