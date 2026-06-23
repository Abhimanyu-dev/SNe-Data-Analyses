import pickle
from pathlib import Path
import gzip
import pandas as pd

union3_file = Path("../data/union3/union3_release/inputs_Amanullah10_CNIa02_CSP_CalanTololo_CfA1_CfA2_CfA3_CfA4_DES3_Deep_DES3_Shallow_ESSENCE_Foundation_LOSS_MCT_NB99_Pan-STARRS_Riess07_SDSS_SNLS_SuzukiRubin_Tonry03_LSQ+LCO_LSQ_knop03_Krisciunas.pickle")

pickle_in = gzip.open(union3_file, "rb")
union3_data = pickle.load(pickle_in)

print(union3_data[0].keys())

print(len(union3_data[0]["mB_list"]))