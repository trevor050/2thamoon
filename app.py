"""2thamoon: Streamlit entry point.

Streamlit Community Cloud runs this file from the repo root. The simulation itself lives in
the ``moonsim`` package; this file only starts the UI in ``moonsim_app``.
"""

from moonsim_app.main import main

main()
