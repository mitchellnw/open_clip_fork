import os
import numpy as np
import pandas as pd


if __name__ == '__main__':
    # for i in range(32):
    #     print(i)
    #     try:
    #         df = pd.read_csv(f'/checkpoint/mitchellw/experiments/open_clip/b32-400m-w0-opt-0.001-0.9-0.98-1e-06-bs-4096-amp-v1/data/{i}/features-module.transformer.resblocks.0.csv')
    #     except:
    #         print('EMPTY')
        
    # exit()
    names = os.listdir(f'/checkpoint/mitchellw/experiments/open_clip/b32-400m-w0-opt-0.001-0.9-0.98-1e-06-bs-4096-amp-v1/data/0')
    for name in names:
        print(name)
        dfls = []
        
        is_empty = False
        for i in range(32):
            try:
                df = pd.read_csv(f'/checkpoint/mitchellw/experiments/open_clip/b32-400m-w0-opt-0.001-0.9-0.98-1e-06-bs-4096-amp-v1/data/{i}/{name}')
            except:
                print('EMPTY')
                is_empty = True
                break
            dfl = df.to_numpy().flatten()
            dfls.append(dfl)
        
        if not is_empty:
            for i in range(32):
                A = dfls[i % 32]
                B = dfls[(i + 1) % 32]
                k = min(A.shape[0], B.shape[0])
                m = np.array_equal(A[:k] , B[:k], equal_nan=True)
                if not m:
                    print('BROKEN!!!')
                    is_empty = True
                    break
            
            if not is_empty:
                print('SUCCESS')