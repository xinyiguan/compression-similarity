from pymer4.models import lmer
import pandas as pd
from model.utils import load_json, save_json


def run_lme(json_fpath: str, save_json_path: str = None):
    df = transform_to_long_form_df(json_fpath=json_fpath)

    model = lmer('D_stitch ~ D_ppm + (1|piece_i) + (1|piece_j)', data=df)
    result = model.fit(summary=True)
    print(result.fixef.head())

    # if save_json_path:
    #     save_json(fpath=save_json_path, res=result_dict)
    return result


def transform_to_long_form_df(json_fpath: str, save_csv_path:str=None) -> pd.DataFrame:
    raw_data = load_json(json_fpath)
    df_raw = pd.DataFrame(raw_data)

    df = pd.DataFrame()

    df["piece_i"] = df_raw["nameX"].astype(str)
    df["piece_j"] = df_raw["nameY"].astype(str)

    df["D_ppm"] = df_raw["D_ppm"]
    df["D_stitch"] = df_raw["D_stitch"]

    if save_csv_path:
        df.to_csv(path_or_buf=save_csv_path, columns=["piece_i", "piece_j", "D_ppm", "D_stitch"])
    return df
