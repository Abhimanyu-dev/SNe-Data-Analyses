import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import minimize
from pathlib import Path

C = 299792.458  # km/s
DATA_PATH = Path("../data/pantheon/DataRelease/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat")
COV_PATH = Path("../data/pantheon/DataRelease/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov")

def load_covariance_matrix(path: Path = COV_PATH) -> np.ndarray:
    with open(path) as f:
        N = int(f.readline())

    cov = np.loadtxt(path, skiprows=1)
    cov = cov.reshape(N, N)
    return cov

def load_pantheon_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", comment="#")
    df["idx"] = np.arange(len(df))
    # print(df.columns)
    # print(df.head())
    # print(f"Total rows: {len(df)}")
    # print(f"Unique CIDs: {df['CID'].nunique()}")
    # print(df[["RA", "DEC", "zHD", "MU_SH0ES"]].describe())
    return df


def add_galactic_coords(df: pd.DataFrame) -> pd.DataFrame:
    coords = SkyCoord(ra=df["RA"].values * u.deg, dec=df["DEC"].values * u.deg, frame="icrs")
    df = df.copy()
    df["l"] = coords.galactic.l.deg
    df["b"] = coords.galactic.b.deg
    # print(df[["RA", "DEC", "l", "b"]].head())
    # print(df[["RA", "DEC", "l", "b"]].describe())
    return df


def survey_fraction_in_arc(df: pd.DataFrame) -> None:
    arc = df[(df["b"] < -40) & (df["b"] > -70)]
    arc_frac = arc["IDSURVEY"].value_counts(normalize=True)
    full_frac = df["IDSURVEY"].value_counts(normalize=True)
    result = (
        pd.DataFrame({"arc": arc_frac, "full": full_frac})
        .fillna(0)
        .sort_values("arc", ascending=False)
        .head(10)
    )
    # print(result)


def filter_redshift(df: pd.DataFrame, z_min: float = 0.07, z_max: float = 0.8) -> pd.DataFrame:
    df_cut = df[(df["zHD"] > z_min) & (df["zHD"] < z_max)].copy()
    df_cut = df_cut.reset_index(drop=False)
    # print(f"Original: {len(df)} | Cut ({z_min} < z < {z_max}): {len(df_cut)}")
    return df_cut


def add_cartesian_coords(df: pd.DataFrame) -> pd.DataFrame:
    ra_rad = np.radians(df["RA"].values)
    dec_rad = np.radians(df["DEC"].values)
    df = df.copy()
    df["x"] = np.cos(dec_rad) * np.cos(ra_rad)
    df["y"] = np.cos(dec_rad) * np.sin(ra_rad)
    df["z"] = np.sin(dec_rad)
    return df


def split_hemisphere(df: pd.DataFrame, normal: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    dot = df[["x","y","z"]].values @ normal
    return df[dot > 0], df[dot <= 0]


def m_theory(z: np.ndarray, q0: float, M: float) -> np.ndarray:
    series = 1 + 0.5 * (1 - q0) * z - (1 / 6) * (2 - q0 - 3 * q0 * q0) * z * z
    return 5 * np.log10(C * z * series) + M


def chi2(params: np.ndarray, data: pd.DataFrame, Cinv: np.ndarray) -> float:
    q0, M = params
    model = m_theory(data["zHD"].values, q0, M)
    delta = ( data["m_b_corr"].values - model )
    return delta @ Cinv @ delta


def fit_cosmology(data: pd.DataFrame, Cinv: np.ndarray) -> np.ndarray:
    result = minimize(chi2, x0=[-0.5, 24], args=(data, Cinv))
    # print(result.x)
    return result.x


def main() -> None:
    df = load_pantheon_data()
    df = add_galactic_coords(df)
    # survey_fraction_in_arc(df)

    df_cut = filter_redshift(df)
    df_cut = add_cartesian_coords(df_cut)
    
    # plt.scatter(df_cut["RA"], df_cut["DEC"], s=3)
    # plt.xlabel("RA")
    # plt.ylabel("DEC")
    # plt.show()
    
    mask = (df["zHD"] > 0.07) & (df["zHD"] < 0.8)
    indices = np.where(mask)[0]
    
    cov = load_covariance_matrix()
    cov_cut = cov[np.ix_(indices, indices)]
    
    ra_grid = np.arange(0, 360, 5)  
    dec_grid = np.arange(-90, 91, 5)
    
    results = []
    
    for ra in ra_grid:
        for dec in dec_grid:
            normal = np.array([np.cos(np.radians(dec)) * np.cos(np.radians(ra)),
                               np.cos(np.radians(dec)) * np.sin(np.radians(ra)),
                               np.sin(np.radians(dec))])
            north, south = split_hemisphere(df_cut, normal=normal)
            north_idx = north.index.values
            south_idx = south.index.values
            
            cov_north = cov_cut[np.ix_(north_idx, north_idx)]
            cov_south = cov_cut[np.ix_(south_idx, south_idx)]
            
            if len(north) > 0 and len(south) > 0:
                q0_north, _ = fit_cosmology(north, np.linalg.inv(cov_north))
                q0_south, _ = fit_cosmology(south, np.linalg.inv(cov_south))
                
                delta_q0 = q0_north - q0_south
                
                results.append((ra, dec, len(north), len(south), q0_north, q0_south, delta_q0))

    # north, south = split_hemisphere(df_cut, normal=np.array([1, 0, 0]))
    
    # north_idx = north.index.values
    # south_idx = south.index.values
    
    # cov_north = cov_cut[np.ix_(north_idx, north_idx)]
    # cov_south = cov_cut[np.ix_(south_idx, south_idx)]
    # plt.figure(figsize=(10, 5))
    # plt.scatter(df_cut["RA"], df_cut["DEC"], s=2, label="All")
    # plt.show()

    # plt.figure(figsize=(10, 5))
    # plt.scatter(north["l"], north["b"], s=2, label="North")
    # plt.scatter(south["l"], south["b"], s=2, label="South")
    # plt.legend()
    # plt.show()
    print(f"Total results: {len(results)}")
    for r in results:
        print(f"RA: {r[0]:.1f}, DEC: {r[1]:.1f}, N_north: {r[2]}, N_south: {r[3]}, q0_north: {r[4]:.4f}, q0_south: {r[5]:.4f}, delta_q0: {r[6]:.4f}")

    print("MAX DELTA Q0:", max(results, key=lambda x: abs(x[6])))
    print("MIN DELTA Q0:", min(results, key=lambda x: abs(x[6])))

if __name__ == "__main__":
    main()
