import tarfile
import urllib.request


library_file = "human_hcd_tryp_best.msp"


## Downloading the dataset
url = "https://chemdata.nist.gov/download/peptide_library/libraries/human/HCD/2020_05_19/human_hcd_tryp_best.msp.tar.gz"

# Download file
_ = urllib.request.urlretrieve(url, f"{library_file}.tar.gz")

with tarfile.open(f"{library_file}.tar.gz") as f:
    f.extractall("./datasets/")