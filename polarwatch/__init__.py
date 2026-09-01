"""PolarWatch AI — simulation package.

Every module here stands in for one stage of the real pipeline described in
the project proposal (YOLO detection, CNN health assessment, RNN behavioural
analysis, the multimodal risk engine, and alerting). No models are trained or
loaded — outputs are synthetic but shaped the way the real pipeline's outputs
would be, so the dashboard in app.py exercises the same data flow the
finished system will use.
"""
