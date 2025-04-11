# PTM Prediction using Modified Casanovo

Adapts [Casanovo](https://casanovo.readthedocs.io/en/latest/) for *de novo* sequencing with prediction of specific Post-Translational Modifications (PTMs).

**Note:** Identifies **only known, predefined PTMs** listed in `config.yaml`. **Cannot perform open modification searches.**

## Key Modifications

* Predicts PTMs defined in `config.yaml` using a fixed vocabulary (each AA+PTM is a unique token).
* Includes input annotation standardization (`standardize_sequence.py`) for robust training.
* Generates detailed mzTab output with PTM locations, scores, and total mass shifts.
* Built upon the Casanovo (`depthcharge`) Transformer architecture.

## Approach

PTMs are handled by defining all target AA+PTM combinations (e.g., `C+57.02146`) with their masses in the `config.yaml` `residues` section. This creates a fixed model vocabulary. The `standardize_sequence.py` utility maps input annotations to these exact tokens for training. The model predicts sequences using these tokens. The `MztabWriter` then parses these predicted tokens to generate mzTab PTM information, including calculated total mass shifts.

## Limitations

* **No Open Modification Search:** Only identifies PTMs explicitly defined in `config.yaml`.
* **Vocabulary Size:** Large numbers of defined PTMs increase vocabulary size and potentially training difficulty.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2. **Install dependencies:**
    *(Assuming you have a requirements.txt file)*
    ```bash
    pip install -r requirements.txt
    ```
3. **Change the run.sh script for desired function:**
   ```bash
    bash run.sh
    ```
    

## Configuration

* **Modify the `residues` section** to include all standard amino acids and the specific AA+PTM combinations you want the model to be able to predict. Ensure the masses are accurate.
* You can copy the default configuration and modify it:
    ```bash
    cp src/config.yaml my_custom_config.yaml
    # Now edit my_custom_config.yaml
    ```
* Specify your custom config file using the `-c` or `--config` option when running commands.

## Usage

The main entry point is `src/denovo_ptm/denovo.py`.

```bash
# Show help message and commands
python -m src.denovo_ptm.denovo --help
