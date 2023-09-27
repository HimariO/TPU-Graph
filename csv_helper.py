import os
import glob

import torch
import fire
import pandas as pd
from tqdm import tqdm


def strip_table(ckpt):
    checkpoint = torch.load(ckpt)
    # print(checkpoint.keys())
    # breakpoint()
    checkpoint['model_state'].pop('history.emb')
    output = ckpt.replace('.ckpt', '.strip.ckpt')
    torch.save(checkpoint, output)


def int8_config_feat(dir):
    files = glob.glob(os.path.join(dir, '*.pt'))
    for f in tqdm(files):
        if "_data" in f:
            data = torch.load(f)
            if -2**7 <= data.config_feats.min() and data.config_feats.min() <= 2**7:
                data.config_feats = data.config_feats.to(torch.int8)
                torch.save(data, f)


def overwrite(csv_a, csv_b, out_path):
    # Step 1: Read both CSV files into DataFrames
    a_df = pd.read_csv(csv_a)
    b_df = pd.read_csv(csv_b)

    # Step 2: Identify the rows to be overwritten (e.g., based on a common column)
    # For example, if you have a common column 'ID' to match rows:
    common_column = 'ID'
    rows_to_overwrite = a_df[common_column].isin(b_df[common_column])

    # Step 3: Overwrite the selected rows in a_df with corresponding rows from b_df
    a_df.loc[rows_to_overwrite] = b_df
    print(f'Overwrite {sum(rows_to_overwrite)} rows')

    # Step 4: Save the modified a_df to 'a.csv'
    a_df.to_csv(out_path, index=False)


if __name__ == '__main__':
    fire.Fire({
        overwrite.__name__: overwrite,
        strip_table.__name__: strip_table,
        int8_config_feat.__name__: int8_config_feat,
    })