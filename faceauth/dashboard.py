"""Read-only Streamlit evaluation board for FaceAuth.

Only aggregate LFW JSON reports are loaded. No user images, templates, SQLite,
or network requests are performed by this module.
"""
from __future__ import annotations

from pathlib import Path

from .dashboard_data import ReportDataError, load_pair_test, load_ident_test, locate_reports


def _pct(x: float) -> str:
    return f'{100.0 * x:.2f}%'


def _round_ms(value: float | None) -> str:
    return '-' if value is None else f'{value:.3f}'


def render_experiment_dashboard(report_dir: Path | str) -> None:
    import pandas as pd
    import streamlit as st

    report_dir = Path(report_dir)
    st.subheader('LFW \u5b9e\u9a8c\u7ed3\u679c\u770b\u677f')
    st.caption('\u8bfb\u53d6\u672c\u673a reports \u4e0b\u7684\u771f\u5b9e\u8bc4\u4f30\u7ed3\u679c\uff1b\u6570\u636e\u4e0d\u4f1a\u4e0a\u4f20\u5230\u7f51\u7edc\u3002')
    pair_path, ident_path = locate_reports(report_dir)

    st.markdown('#### 1:1 \u4eba\u8138\u9a8c\u8bc1')
    if pair_path is None:
        st.info('\u7f3a\u5c11 1:1 \u6d4b\u8bd5\u7ed3\u679c (reports/lfw_metrics_test_lfwprimary.json).')
    else:
        try:
            pair = load_pair_test(pair_path)
        except (ReportDataError, OSError) as exc:
            st.warning(f'1:1 JSON: {exc}')
        else:
            cols = st.columns(5)
            cols[0].metric('\u9a8c\u8bc1\u51c6\u786e\u7387', _pct(pair['accuracy_e2e']))
            cols[1].metric('\u8bc4\u4f30\u8986\u76d6\u7387', _pct(pair['coverage']))
            cols[2].metric('FAR', _pct(pair['far_e2e']))
            cols[3].metric('FRR', _pct(pair['frr_e2e']))
            cols[4].metric('ROC-AUC', f"{pair['roc_auc_valid']:.4f}")
            st.caption(
                f"LFW dev test | {pair['processed']}/{pair['total']} pairs | "
                f"face selection: {pair['policy']} | threshold: "
                f"{pair['threshold']:.4f}" if pair['threshold'] is not None else
                f"LFW dev test | {pair['processed']}/{pair['total']} pairs | {pair['policy']}"
            )
            if pair['skipped']:
                st.warning(
                    f"{pair['skipped']}/{pair['total']} pairs failed extraction. "
                    "Accuracy counts failures as authentication rejection; "
                    "AUC is conditional on extracted pairs."
                )
            with st.expander('1:1 confusion matrix (counts)', expanded=False):
                cm = pd.DataFrame([
                    {'Actual': 'same person', 'Predicted same': pair['tp'],
                     'Predicted different': pair['fn']},
                    {'Actual': 'different people', 'Predicted same': pair['fp'],
                     'Predicted different': pair['tn']},
                ])
                st.dataframe(cm, hide_index=True, use_container_width=True)
            st.caption(f"Source: {pair_path.name}")

    st.divider()
    st.markdown('#### 1:N \u8eab\u4efd\u68c0\u7d22\u4e0e\u672a\u77e5\u8eab\u4efd\u62d2\u8bc6')
    if ident_path is None:
        st.info('\u7f3a\u5c11 1:N \u6d4b\u8bd5\u7ed3\u679c (reports/lfw_1n_test.json).')
    else:
        try:
            ident = load_ident_test(ident_path)
        except (ReportDataError, OSError) as exc:
            st.warning(f'1:N JSON: {exc}')
        else:
            rows = ident['rows']
            largest = rows[-1]
            a, b, c, d = st.columns(4)
            a.metric('\u6ce8\u518c\u4eba\u6570', f"{largest['gallery_people']}")
            b.metric('Rank-1', _pct(largest['rank1']))
            c.metric('DIR @ threshold', _pct(largest['dir']))
            d.metric('Unknown FPIR', _pct(largest['fpir']))
            st.caption(
                f"{largest['gallery_people']}-person gallery: {largest['known_correct_accept']}/"
                f"{largest['known_queries']} known correctly accepted; "
                f"{largest['unknown_false_accept']}/{largest['unknown_queries']} unknown falsely accepted."
            )
            table = pd.DataFrame([{
                '\u6ce8\u518c\u4eba\u6570': x['gallery_people'],
                '\u5df2\u77e5\u8eab\u4efd\u67e5\u8be2\u6570': x['known_queries'],
                '\u672a\u77e5\u8eab\u4efd\u67e5\u8be2\u6570': x['unknown_queries'],
                'Rank-1 (%)': round(100 * x['rank1'], 4),
                'DIR (%)': round(100 * x['dir'], 4),
                'FPIR (%)': round(100 * x['fpir'], 4),
                '\u6821\u51c6\u9608\u503c': round(x['threshold'], 6),
                '\u5df2\u77e5\u8986\u76d6\u7387 (%)': round(100*x['known_coverage'], 3),
                '\u672a\u77e5\u8986\u76d6\u7387 (%)': round(100*x['unknown_coverage'], 3),
            } for x in rows])
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.download_button(
                '\u4e0b\u8f7d\u805a\u5408\u6307\u6807 CSV',
                data=table.to_csv(index=False).encode('utf-8-sig'),
                file_name='lfw_1n_summary_v16.csv', mime='text/csv',
                key='download_lfw_1n_summary',
            )
            left, right = st.columns(2, gap='large')
            bars = table.set_index('\u6ce8\u518c\u4eba\u6570')
            with left:
                st.markdown('**Rank-1 / DIR (%)**')
                st.bar_chart(bars[['Rank-1 (%)', 'DIR (%)']])
                st.caption('Rank-1 is before rejection; DIR requires correct ID and score >= threshold.')
            with right:
                st.markdown('**Unknown FPIR (%)**')
                st.bar_chart(bars[['FPIR (%)']])
                st.caption('Separate 0-based axis; 0/300 errors is not a population risk of zero.')
            st.caption(
                '\u9a8c\u8bc1\u4e0e\u6d4b\u8bd5\u8eab\u4efd\u4e0d\u91cd\u53e0\uff1b'
                '\u6bcf\u4e2a\u5e93\u89c4\u6a21\u5728\u9a8c\u8bc1\u9636\u6bb5\u72ec\u7acb\u6821\u51c6\u9608\u503c\u3002'
            )

            st.markdown('#### \u6bcf\u6b21\u67e5\u8be2\u79bb\u7ebf\u8017\u65f6\uff08ms\uff09')
            timing = pd.DataFrame([{
                '\u6ce8\u518c\u4eba\u6570': x['gallery_people'],
                '\u7eaf\u7279\u5f81\u5411\u91cf\u68c0\u7d22 mean (ms)': x['match_mean_ms'],
                '\u7eaf\u7279\u5f81\u5411\u91cf\u68c0\u7d22 P95 (ms)': x['match_p95_ms'],
                '\u6a21\u578b+\u68c0\u7d22\u4f30\u7b97 mean (ms)': x['model_search_est_mean_ms'],
                '\u6a21\u578b+\u68c0\u7d22\u4f30\u7b97 P95 (ms)': x['model_search_est_p95_ms'],
            } for x in rows])
            st.dataframe(timing, hide_index=True, use_container_width=True)
            st.caption(
                '\u4f30\u7b97\u503c\u4e0d\u542b\u56fe\u50cf\u8bfb\u53d6\u3001'
                '\u6d4f\u89c8\u5668\u6e32\u67d3\u3001\u6570\u636e\u5e93\u64cd\u4f5c\uff0c'
                '\u4e0d\u662f Web \u7aef\u5230\u7aef\u8017\u65f6\u3002'
            )
            st.caption(
                f"Protocol: identity-disjoint custom LFW | policy: {ident['face_selection']} "
                f"| enrollment={ident['enroll_per_identity']}/person "
                f"| seed={ident['seed']} | source={ident_path.name}"
            )

    st.divider()
    st.info(
        'LFW \u91c7\u7528\u4e3b\u4f53\u8138\u9009\u62e9\u7b56\u7565\uff0c'
        '\u7ebf\u4e0a\u4ecd\u662f\u4e25\u683c\u5355\u4eba\u8138\u62d2\u7edd\u89c4\u5219\uff1b'
        '\u672c\u7ed3\u679c\u5e76\u975e LFW \u5b98\u65b9\u5341\u6298\u6210\u7ee9\uff0c'
        '\u4e5f\u4e0d\u4ee3\u8868\u6297\u7167\u7247\u653b\u51fb\u80fd\u529b\u3002'
    )
    st.caption(
        '\u62a5\u544a\u53ea\u8bfb\u53d6\u805a\u5408\u6307\u6807\uff0c'
        '\u4e0d\u8bfb\u53d6\u4eba\u8138\u7167\u7247\u6216 SQLite \u751f\u7269\u7279\u5f81\u5e93\u3002'
    )
