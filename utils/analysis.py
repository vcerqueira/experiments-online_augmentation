import pandas as pd


def to_latex_tab(df, round_to_n):
    annotated_res = []
    for i, r in df.round(round_to_n).iterrows():
        top_2 = r.sort_values().unique()[:2]
        if len(top_2) < 2:
            raise ValueError('only one score')

        best1 = r[r == top_2[0]].values[0]
        best2 = r[r == top_2[1]].values[0]

        r[r == top_2[0]] = f'\\textbf{{{best1}}}'
        r[r == top_2[1]] = f'\\underline{{{best2}}}'

        annotated_res.append(r)

    annotated_res = pd.DataFrame(annotated_res).astype(str)

    text_tab = annotated_res.to_latex(caption='CAPTION', label='tab:scores_by_ds')

    return text_tab
