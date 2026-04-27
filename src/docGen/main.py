

import json
from typing import Dict
import pandas as pd
import vl_convert


def format_and_save_to_tex_csv(output_folder,filename,table):
    table.to_csv(f'{output_folder}/{filename}.tsv',sep='\t',float_format = "{:,.2f}",index=False)
    table.rename(
            columns={"nameX":"$piece_1$",
                     "nameY":"$piece_2$",
                     'abs_dist_diff':'$|\Delta z|$',
                     'dist_diff':'$\Delta z$',
                     "D_stitch(z-score)":"$z_{stitch}$",
                     "D_ppm(z-score)":"$z_{ppm}$"
                     }
            ).to_latex(
                f'{output_folder}/{filename}.tex',
                index=False,
                float_format = "{:,.2f}".format,
                formatters = {"$piece_1$": lambda x: "\\textit{" + x[:10] + "\dots}",
                              "$piece_2$":  lambda x: "\\textit{" + x[:10] + "\dots}",
                }

                )


def to_be_discarded(z_score1 : float,z_score2: float ) -> bool:
    all_outliers = abs(z_score1) > 2 and abs(z_score2) > 2
    same_direction = z_score1 * z_score2 >= 0
    return all_outliers and same_direction
    

def zscore_result_to_tsv(zscore_result:Dict,output_folder:str):
    df = pd.DataFrame.from_records(zscore_result["Top k disagree"])
    
    for model_name in ['stitch','ppm']:
        for ascending in [True,False]:
            _df = df[["nameX","nameY",f"D_{model_name}(z-score)"]].sort_values(f"D_{model_name}(z-score)",ascending = ascending).head(10)
            format_and_save_to_tex_csv(output_folder,f"{model_name}(sorted z-score)(ascending={ascending})",_df)
    
    # for the diff z-score, filter out entry that: both pieces are outliers (|z| > 2) in the same direction
    df = df[df.apply(lambda entry: not to_be_discarded(entry["D_stitch(z-score)"],entry["D_ppm(z-score)"]),axis=1)]
    
    table_largestDiff = df[["nameX","nameY","D_stitch(z-score)","D_ppm(z-score)","dist_diff"]]
    table_smallestDiff = df[["nameX","nameY","D_stitch(z-score)","D_ppm(z-score)","abs_dist_diff"]]
    name1 = "largestDiff(pos)"
    name2 = "largestDiff(neg)"
    name3 = "smallestDiff(abs)"
    table1 = table_largestDiff.sort_values("dist_diff",ascending = False).head(10)
    
    table2 = table_largestDiff.sort_values("dist_diff",ascending = True).head(10)
    table3 = table_smallestDiff.sort_values("abs_dist_diff",ascending = True).head(10)
    
    for name,table in [(name1,table1),(name2,table2),(name3,table3)]:
        format_and_save_to_tex_csv(output_folder,name,table)
        # table.to_csv(f'{output_folder}/{name}.tsv',sep='\t',float_format = "{:,.2f}",index=False)
        # table.rename(
        #     columns={"nameX":"$piece_1$",
        #              "nameY":"$piece_2$",
        #              'abs_dist_diff':'$|\Delta z|$',
        #              'dist_diff':'$\Delta z$',
        #              "D_stitch(z-score)":"$z_{stitch}$",
        #              "D_ppm(z-score)":"$z_{ppm}$"
        #              }
        #     ).to_latex(
        #         f'{output_folder}/{name}.tex',
        #         index=False,
        #         float_format = "{:,.2f}".format,
        #         formatters = {"$piece_1$": lambda x: "\\textit{" + x[:10] + "\dots}",
        #                       "$piece_2$":  lambda x: "\\textit{" + x[:10] + "\dots}",
        #         }

        #         )

def heatmap_to_pdf(vega_path,output_folder):
    with open(vega_path) as f:
        vega = f.read()
        with open(f'{output_folder}/heatmap.png',"w") as out:
            out.write(vl_convert.vega_to_png(vega))

def main(table_json_path:str,heatmap_vega_path,output_folder:str):
    with open(table_json_path) as f:
        zscore_result = json.load(f)
    
    # heatmap_to_pdf(heatmap_vega_path,output_folder)
    zscore_result_to_tsv(zscore_result,output_folder)
    

# if __name__ == "__main__":
    
#     main(table_json_path="data/result/report(stitch_ppm).json",
#          heatmap_vega_path="docs/vega/pairwise-similarity.vega",
#          out_doc_path = "docs/latex/figs")
    
#     main(table_json_path="data/result/report(stitch_ppm)_ctx.json",
#          heatmap_vega_path="docs/vega/pairwise-similarity-ctx.vega",
#          out_doc_path = "docs/latex/figs")