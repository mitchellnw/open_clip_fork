import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

if __name__ == '__main__':
    df = pd.read_csv('plots/clip_table_3.csv')
    openai_vit_results = {
        'ViT-B-32': 63.2,
        'ViT-B-16': 68.6,
        'ViT-L-14': 75.3,
    #    'ViT-L-14-336': 76.2,  # this model was fine-tuned from L-14 @ higher res for one extra epoch, excluding as it throws off trend
    }

    openai_rn_results = {
        'RN50': 59.6,
        'RN101': 62.2,
        'RN50x4': 65.8,
        'RN50x16': 70.5,
        #'RN50x64': 73.6,
    }

    laion400m_results = {
        'ViT-B-32': 62.96,
        'ViT-B-16': 67.07,
        'ViT-B-16-plus-240': 69.21,
        'ViT-L-14': 72.77,
        #'ViT-g-14': 76.6,
    }

    laion2b_results = {
        'ViT-B-32': 65.62,
        'ViT-L-14': 75.28,
        'ViT-H-14': 77.97,
        'ViT-G-14': 80.12,
    }

    result_series = dict(
        openai_rn=openai_rn_results,
        openai_vit=openai_vit_results,
        laion400m_vit=laion400m_results,
        laion2b_vit=laion2b_results,
    )

    by_model = df.set_index('model').to_dict(orient='index')
    remaining = set(by_model.keys())
    #df = df['in1k_zero_shot'] = np.nan

    out = []
    for key, series in result_series.items():
        for m, top1 in series.items():
            remaining.discard(m)
            entry = {}  
            entry['model'] = m
            entry['series'] = key
            entry['in1k_zero_shot'] = top1
            entry.update(by_model[m])
            out.append(entry)

    for m in remaining:
        entry = {}  
        entry['model'] = m
        entry['series'] = 'untrained' + ('_vit' if 'ViT' in m else '_rn')
        entry['in1k_zero_shot'] = np.nan    
        entry.update(by_model[m])
        out.append(entry)

    out_df = pd.DataFrame(out, columns=out[0].keys())
    out_df = out_df.sort_values('macts')

    #from pandas.core.resample import f
    df_openai_vit = out_df[out_df.series == 'openai_vit']
    df_laion400m_vit = out_df[out_df.series == 'laion400m_vit']
    df_laion2b_vit = out_df[out_df.series == 'laion2b_vit']
    df_untrained_vit = out_df[out_df.series == 'untrained_vit']

    log_x_openai = np.log(df_openai_vit.macts)
    y_openai = np.array(df_openai_vit.in1k_zero_shot)
    fit_openai = np.polyfit(log_x_openai, y_openai, 1)
    print(fit_openai)

    log_x_laion400m = np.log(df_laion400m_vit.macts)
    y_laion400m = np.array(df_laion400m_vit.in1k_zero_shot)
    fit_laion400m = np.polyfit(log_x_laion400m, y_laion400m, 1)
    print(fit_laion400m)


    log_x_laion2b = np.log(df_laion2b_vit.macts)
    y_laion2b = np.array(df_laion2b_vit.in1k_zero_shot)
    fit_laion2b = np.polyfit(log_x_laion2b, y_laion2b, 1)
    print(fit_laion2b)

    xe = np.log(df_untrained_vit.macts)
    out = fit_openai[0] * xe + fit_openai[1]
    out_df['estimate_openai'] = out

    filt_openai = out_df.loc[out.index]
    filt_openai['series'] = 'estimate_openai'
    filt_openai['in1k_zero_shot'] = out.values

    xe = np.log(df_untrained_vit.macts)
    out = fit_laion400m[0] * xe + fit_laion400m[1]
    out_df['estimate_laion400m'] = out
    filt_laion400m = out_df.loc[out.index]
    filt_laion400m['series'] = 'estimate_laion400m'
    filt_laion400m['in1k_zero_shot'] = out.values

    xe = np.log(df_untrained_vit.macts)
    out = fit_laion2b[0] * xe + fit_laion2b[1]
    out_df['estimate_laion2b'] = out
    filt_laion2b = out_df.loc[out.index]
    filt_laion2b['series'] = 'estimate_laion2bm'
    filt_laion2b['in1k_zero_shot'] = out.values

    out_combined = pd.concat([out_df, filt_openai, filt_laion400m, filt_laion2b]).sort_values('macts')


    import kaleido
    # layout = go.Layout(
    #     autosize=False,
    #     width=500,
    #     height=500
    # )
    out_df = out_df[out_df.series != 'untrained_vit']
    figa = px.scatter(
        data_frame=out_df, x='macts', y='in1k_zero_shot', text="model", log_x=True, log_y=False,
        hover_name=out_df.index, color='series',
        trendline='ols',trendline_options=dict(log_x=True, log_y=False)
    )
    figa.update_layout(title_text='In1k zero-shot vs Activation count', title_x=0.5, xaxis_title='Activations (M)', yaxis_title='In1k zero-shot top-1',)
    #figa.add_trace(go.Scatter(x=out_df['macts'], y=out_df['estimate_openai'], mode='markers+text', text=out_df['model'], name='OpenAI Estimates'))
    #figa.add_trace(go.Scatter(x=out_df['macts'], y=out_df['estimate_laion400m'], mode='markers+text', text=out_df['model'], name='OpenAI Estimates'))
    figa.update_traces(textposition='top center')
    #figa.show()
    pio.write_image(figa, "plots/scaling_plot.pdf",  width=1200, height=500)