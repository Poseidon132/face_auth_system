"""Standalone read-only dashboard; run: streamlit run experiment_dashboard.py."""
from pathlib import Path

import streamlit as st

from faceauth.dashboard import render_experiment_dashboard

st.set_page_config(page_title='FaceAuth LFW | Experiment Dashboard', page_icon="\U0001f4ca", layout='wide')
render_experiment_dashboard(Path(__file__).resolve().parent / 'reports')
